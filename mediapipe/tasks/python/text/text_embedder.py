# Copyright 2022 The MediaPipe Authors.
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
import ctypes
import dataclasses
from typing import Optional

from mediapipe.tasks.python.components.containers import embedding_result as embedding_result_module
from mediapipe.tasks.python.components.containers import embedding_result_c as embedding_result_c_module
from mediapipe.tasks.python.components.utils import cosine_similarity
from mediapipe.tasks.python.core import base_options as base_options_module
from mediapipe.tasks.python.core import base_options_c as base_options_c_module
from mediapipe.tasks.python.core import mediapipe_c_bindings
from mediapipe.tasks.python.core import mediapipe_c_utils
from mediapipe.tasks.python.core import serial_dispatcher

TextEmbedderResult = embedding_result_module.EmbeddingResult


class _EmbedderOptionsC(ctypes.Structure):
  """C struct for embedder options."""

  _fields_ = [
      ('l2_normalize', ctypes.c_bool),
      ('quantize', ctypes.c_bool),
  ]


class _TextEmbedderOptionsC(ctypes.Structure):
  """C struct for text embedder options."""

  _fields_ = [
      ('base_options', base_options_c_module.BaseOptionsC),
      ('embedder_options', _EmbedderOptionsC),
  ]


_CTYPES_SIGNATURES = (
    mediapipe_c_utils.CStatusFunction(
        'MpTextEmbedderCreate',
        [
            ctypes.POINTER(_TextEmbedderOptionsC),
            ctypes.POINTER(ctypes.c_void_p),
        ],
    ),
    mediapipe_c_utils.CStatusFunction(
        'MpTextEmbedderEmbed',
        [
            ctypes.c_void_p,
            ctypes.c_char_p,
            ctypes.POINTER(embedding_result_c_module.EmbeddingResultC),
        ],
    ),
    mediapipe_c_utils.CStatusFunction(
        'MpTextEmbedderClose',
        [
            ctypes.c_void_p,
        ],
    ),
    mediapipe_c_utils.CFunction(
        'MpTextEmbedderCloseResult',
        [ctypes.POINTER(embedding_result_c_module.EmbeddingResultC)],
        None,
    ),
)


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
  base_options: base_options_module.BaseOptions
  l2_normalize: Optional[bool] = None
  quantize: Optional[bool] = None

  def to_ctypes(self) -> _TextEmbedderOptionsC:
    """Generates a ctypes TextEmbedderOptionsC object."""
    base_options_c = self.base_options.to_ctypes()
    embedder_options_c = _EmbedderOptionsC(
        l2_normalize=self.l2_normalize
        if self.l2_normalize is not None
        else False,
        quantize=self.quantize if self.quantize is not None else False,
    )
    return _TextEmbedderOptionsC(
        base_options=base_options_c, embedder_options=embedder_options_c
    )


class TextEmbedder:
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
  _lib: serial_dispatcher.SerialDispatcher
  _handle: ctypes.c_void_p

  def __init__(
      self,
      lib: serial_dispatcher.SerialDispatcher,
      handle: ctypes.c_void_p,
  ):
    self._lib = lib
    self._embedder_handle = handle

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
    base_options = base_options_module.BaseOptions(model_asset_path=model_path)
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
    lib = mediapipe_c_bindings.load_shared_library(_CTYPES_SIGNATURES)
    ctypes_options = options.to_ctypes()
    embedder_handle = ctypes.c_void_p()
    lib.MpTextEmbedderCreate(
        ctypes.byref(ctypes_options), ctypes.byref(embedder_handle)
    )
    return TextEmbedder(lib=lib, handle=embedder_handle)

  def embed(
      self,
      text: str,
  ) -> embedding_result_module.EmbeddingResult:
    """Performs text embedding extraction on the provided text.

    Args:
      text: The input text.

    Returns:
      An embedding result object that contains a list of embeddings.

    Raises:
      ValueError: If any of the input arguments is invalid.
      RuntimeError: If text embedder failed to run.
    """
    ctypes_result = embedding_result_c_module.EmbeddingResultC()
    self._lib.MpTextEmbedderEmbed(
        self._embedder_handle,
        text.encode('utf-8'),
        ctypes.byref(ctypes_result),
    )
    python_result = (
        embedding_result_module.EmbeddingResult.from_ctypes(
            ctypes_result
        )
    )
    self._lib.MpTextEmbedderCloseResult(ctypes.byref(ctypes_result))
    return python_result

  @classmethod
  def cosine_similarity(
      cls,
      u: embedding_result_module.Embedding,
      v: embedding_result_module.Embedding,
  ) -> float:
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

  def close(self):
    """Shuts down the MediaPipe task instance."""
    if self._embedder_handle:
      self._lib.MpTextEmbedderClose(self._embedder_handle)
      self._embedder_handle = None
      self._lib.close()

  def __enter__(self):
    """Returns `self` upon entering the runtime context."""
    return self

  def __exit__(self, unused_exc_type, unused_exc_value, unused_traceback):
    """Shuts down the MediaPipe task instance on exit of the context manager."""
    self.close()

  def __del__(self):
    self.close()
