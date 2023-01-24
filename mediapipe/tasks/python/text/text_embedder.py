# Copyright 2022 The MediaPipe Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""MediaPipe text embedder task."""

import dataclasses
from typing import Optional

from mediapipe.python import packet_creator
from mediapipe.python import packet_getter
from mediapipe.tasks.cc.components.containers.proto import embeddings_pb2
from mediapipe.tasks.cc.components.processors.proto import embedder_options_pb2
from mediapipe.tasks.cc.text.text_embedder.proto import text_embedder_graph_options_pb2
from mediapipe.tasks.python.components.containers import embedding_result as embedding_result_module
from mediapipe.tasks.python.components.utils import cosine_similarity
from mediapipe.tasks.python.core import base_options as base_options_module
from mediapipe.tasks.python.core import task_info as task_info_module
from mediapipe.tasks.python.core.optional_dependencies import doc_controls
from mediapipe.tasks.python.text.core import base_text_task_api

TextEmbedderResult = embedding_result_module.EmbeddingResult
_BaseOptions = base_options_module.BaseOptions
_TextEmbedderGraphOptionsProto = text_embedder_graph_options_pb2.TextEmbedderGraphOptions
_EmbedderOptionsProto = embedder_options_pb2.EmbedderOptions
_TaskInfo = task_info_module.TaskInfo

_EMBEDDINGS_OUT_STREAM_NAME = 'embeddings_out'
_EMBEDDINGS_TAG = 'EMBEDDINGS'
_TEXT_IN_STREAM_NAME = 'text_in'
_TEXT_TAG = 'TEXT'
_TASK_GRAPH_NAME = 'mediapipe.tasks.text.text_embedder.TextEmbedderGraph'


@dataclasses.dataclass
class TextEmbedderOptions:
  """Options for the text embedder task.

  Attributes:
    base_options: Base options for the text embedder task.
    l2_normalize: Whether to normalize the returned feature vector with L2 norm.
      Use this option only if the model does not already contain a native
      L2_NORMALIZATION TF Lite Op. In most cases, this is already the case and
      L2 norm is thus achieved through TF Lite inference.
    quantize: Whether the returned embedding should be quantized to bytes via
      scalar quantization. Embeddings are implicitly assumed to be unit-norm and
      therefore any dimension is guaranteed to have a value in [-1.0, 1.0]. Use
      the l2_normalize option if this is not the case.
  """
  base_options: _BaseOptions
  l2_normalize: Optional[bool] = None
  quantize: Optional[bool] = None

  @doc_controls.do_not_generate_docs
  def to_pb2(self) -> _TextEmbedderGraphOptionsProto:
    """Generates an TextEmbedderOptions protobuf object."""
    base_options_proto = self.base_options.to_pb2()
    embedder_options_proto = _EmbedderOptionsProto(
        l2_normalize=self.l2_normalize, quantize=self.quantize)

    return _TextEmbedderGraphOptionsProto(
        base_options=base_options_proto,
        embedder_options=embedder_options_proto)


class TextEmbedder(base_text_task_api.BaseTextTaskApi):
  """Class that performs embedding extraction on text.

  This API expects a TFLite model with TFLite Model Metadata that contains the
  mandatory (described below) input tensors and output tensors. Metadata should
  contain the input process unit for the model's Tokenizer as well as input /
  output tensor metadata.

  Input tensors:
    (kTfLiteInt32)
    - 3 input tensors of size `[batch_size x bert_max_seq_len]` with names
      "ids", "mask", and "segment_ids" representing the input ids, mask ids, and
      segment ids respectively.
    - or 1 input tensor of size `[batch_size x max_seq_len]` representing the
      input ids.

  At least one output tensor with:
    (kTfLiteFloat32)
    - `N` components corresponding to the `N` dimensions of the returned
      feature vector for this output layer.
    - Either 2 or 4 dimensions, i.e. `[1 x N]` or `[1 x 1 x 1 x N]`.
  """

  @classmethod
  def create_from_model_path(cls, model_path: str) -> 'TextEmbedder':
    """Creates an `TextEmbedder` object from a TensorFlow Lite model and the default `TextEmbedderOptions`.

    Args:
      model_path: Path to the model.

    Returns:
      `TextEmbedder` object that's created from the model file and the default
      `TextEmbedderOptions`.

    Raises:
      ValueError: If failed to create `TextEmbedder` object from the provided
        file such as invalid file path.
      RuntimeError: If other types of error occurred.
    """
    base_options = _BaseOptions(model_asset_path=model_path)
    options = TextEmbedderOptions(base_options=base_options)
    return cls.create_from_options(options)

  @classmethod
  def create_from_options(cls, options: TextEmbedderOptions) -> 'TextEmbedder':
    """Creates the `TextEmbedder` object from text embedder options.

    Args:
      options: Options for the text embedder task.

    Returns:
      `TextEmbedder` object that's created from `options`.

    Raises:
      ValueError: If failed to create `TextEmbedder` object from
        `TextEmbedderOptions` such as missing the model.
      RuntimeError: If other types of error occurred.
    """
    task_info = _TaskInfo(
        task_graph=_TASK_GRAPH_NAME,
        input_streams=[':'.join([_TEXT_TAG, _TEXT_IN_STREAM_NAME])],
        output_streams=[
            ':'.join([_EMBEDDINGS_TAG, _EMBEDDINGS_OUT_STREAM_NAME])
        ],
        task_options=options)
    return cls(task_info.generate_graph_config())

  def embed(
      self,
      text: str,
  ) -> TextEmbedderResult:
    """Performs text embedding extraction on the provided text.

    Args:
      text: The input text.

    Returns:
      An embedding result object that contains a list of embeddings.

    Raises:
      ValueError: If any of the input arguments is invalid.
      RuntimeError: If text embedder failed to run.
    """
    output_packets = self._runner.process(
        {_TEXT_IN_STREAM_NAME: packet_creator.create_string(text)})

    embedding_result_proto = embeddings_pb2.EmbeddingResult()
    embedding_result_proto.CopyFrom(
        packet_getter.get_proto(output_packets[_EMBEDDINGS_OUT_STREAM_NAME]))

    return TextEmbedderResult.create_from_pb2(embedding_result_proto)

  @classmethod
  def cosine_similarity(cls, u: embedding_result_module.Embedding,
                        v: embedding_result_module.Embedding) -> float:
    """Utility function to compute cosine similarity between two embedding entries.

    May return an InvalidArgumentError if e.g. the feature vectors are
    of different types (quantized vs. float), have different sizes, or have a
    an L2-norm of 0.

    Args:
      u: An embedding entry.
      v: An embedding entry.

    Returns:
      The cosine similarity for the two embeddings.

    Raises:
      ValueError: May return an error if e.g. the feature vectors are of
        different types (quantized vs. float), have different sizes, or have
        an L2-norm of 0.
    """
    return cosine_similarity.cosine_similarity(u, v)
