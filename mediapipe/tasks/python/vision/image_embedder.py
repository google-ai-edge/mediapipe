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
"""MediaPipe image embedder task."""

import dataclasses
from typing import Callable, Mapping, Optional

from mediapipe.python import packet_creator
from mediapipe.python import packet_getter
from mediapipe.python._framework_bindings import image as image_module
from mediapipe.python._framework_bindings import packet as packet_module
from mediapipe.tasks.cc.components.containers.proto import embeddings_pb2
from mediapipe.tasks.cc.components.processors.proto import embedder_options_pb2
from mediapipe.tasks.cc.vision.image_embedder.proto import image_embedder_graph_options_pb2
from mediapipe.tasks.python.components.containers import embedding_result as embedding_result_module
from mediapipe.tasks.python.components.utils import cosine_similarity
from mediapipe.tasks.python.core import base_options as base_options_module
from mediapipe.tasks.python.core import task_info as task_info_module
from mediapipe.tasks.python.core.optional_dependencies import doc_controls
from mediapipe.tasks.python.vision.core import base_vision_task_api
from mediapipe.tasks.python.vision.core import image_processing_options as image_processing_options_module
from mediapipe.tasks.python.vision.core import vision_task_running_mode as running_mode_module

ImageEmbedderResult = embedding_result_module.EmbeddingResult
_BaseOptions = base_options_module.BaseOptions
_ImageEmbedderGraphOptionsProto = image_embedder_graph_options_pb2.ImageEmbedderGraphOptions
_EmbedderOptionsProto = embedder_options_pb2.EmbedderOptions
_RunningMode = running_mode_module.VisionTaskRunningMode
_TaskInfo = task_info_module.TaskInfo
_ImageProcessingOptions = image_processing_options_module.ImageProcessingOptions

_EMBEDDINGS_OUT_STREAM_NAME = 'embeddings_out'
_EMBEDDINGS_TAG = 'EMBEDDINGS'
_IMAGE_IN_STREAM_NAME = 'image_in'
_IMAGE_OUT_STREAM_NAME = 'image_out'
_IMAGE_TAG = 'IMAGE'
_NORM_RECT_STREAM_NAME = 'norm_rect_in'
_NORM_RECT_TAG = 'NORM_RECT'
_TASK_GRAPH_NAME = 'mediapipe.tasks.vision.image_embedder.ImageEmbedderGraph'
_MICRO_SECONDS_PER_MILLISECOND = 1000


@dataclasses.dataclass
class ImageEmbedderOptions:
  """Options for the image embedder task.

  Attributes:
    base_options: Base options for the image embedder task.
    running_mode: The running mode of the task. Default to the image mode. Image
      embedder task has three running modes: 1) The image mode for embedding
      image on single image inputs. 2) The video mode for embedding image on the
      decoded frames of a video. 3) The live stream mode for embedding image on
      a live stream of input data, such as from camera.
    l2_normalize: Whether to normalize the returned feature vector with L2 norm.
      Use this option only if the model does not already contain a native
      L2_NORMALIZATION TF Lite Op. In most cases, this is already the case and
      L2 norm is thus achieved through TF Lite inference.
    quantize: Whether the returned embedding should be quantized to bytes via
      scalar quantization. Embeddings are implicitly assumed to be unit-norm and
      therefore any dimension is guaranteed to have a value in [-1.0, 1.0]. Use
      the l2_normalize option if this is not the case.
    result_callback: The user-defined result callback for processing live stream
      data. The result callback should only be specified when the running mode
      is set to the live stream mode.
  """
  base_options: _BaseOptions
  running_mode: _RunningMode = _RunningMode.IMAGE
  l2_normalize: Optional[bool] = None
  quantize: Optional[bool] = None
  result_callback: Optional[Callable[
      [ImageEmbedderResult, image_module.Image, int], None]] = None

  @doc_controls.do_not_generate_docs
  def to_pb2(self) -> _ImageEmbedderGraphOptionsProto:
    """Generates an ImageEmbedderOptions protobuf object."""
    base_options_proto = self.base_options.to_pb2()
    base_options_proto.use_stream_mode = False if self.running_mode == _RunningMode.IMAGE else True
    embedder_options_proto = _EmbedderOptionsProto(
        l2_normalize=self.l2_normalize, quantize=self.quantize)

    return _ImageEmbedderGraphOptionsProto(
        base_options=base_options_proto,
        embedder_options=embedder_options_proto)


class ImageEmbedder(base_vision_task_api.BaseVisionTaskApi):
  """Class that performs embedding extraction on images.

  The API expects a TFLite model with optional, but strongly recommended,
  TFLite Model Metadata.

  Input tensor:
    (kTfLiteUInt8/kTfLiteFloat32)
    - image input of size `[batch x height x width x channels]`.
    - batch inference is not supported (`batch` is required to be 1).
    - only RGB inputs are supported (`channels` is required to be 3).
    - if type is kTfLiteFloat32, NormalizationOptions are required to be
      attached to the metadata for input normalization.
  At least one output tensor with:
    (kTfLiteUInt8/kTfLiteFloat32)
    - `N` components corresponding to the `N` dimensions of the returned
      feature vector for this output layer.
    - Either 2 or 4 dimensions, i.e. `[1 x N]` or `[1 x 1 x 1 x N]`.
  """

  @classmethod
  def create_from_model_path(cls, model_path: str) -> 'ImageEmbedder':
    """Creates an `ImageEmbedder` object from a TensorFlow Lite model and the default `ImageEmbedderOptions`.

    Note that the created `ImageEmbedder` instance is in image mode, for
    embedding image on single image inputs.

    Args:
      model_path: Path to the model.

    Returns:
      `ImageEmbedder` object that's created from the model file and the default
      `ImageEmbedderOptions`.

    Raises:
      ValueError: If failed to create `ImageEmbedder` object from the provided
        file such as invalid file path.
      RuntimeError: If other types of error occurred.
    """
    base_options = _BaseOptions(model_asset_path=model_path)
    options = ImageEmbedderOptions(
        base_options=base_options, running_mode=_RunningMode.IMAGE)
    return cls.create_from_options(options)

  @classmethod
  def create_from_options(cls,
                          options: ImageEmbedderOptions) -> 'ImageEmbedder':
    """Creates the `ImageEmbedder` object from image embedder options.

    Args:
      options: Options for the image embedder task.

    Returns:
      `ImageEmbedder` object that's created from `options`.

    Raises:
      ValueError: If failed to create `ImageEmbedder` object from
        `ImageEmbedderOptions` such as missing the model.
      RuntimeError: If other types of error occurred.
    """

    def packets_callback(output_packets: Mapping[str, packet_module.Packet]):
      if output_packets[_IMAGE_OUT_STREAM_NAME].is_empty():
        return

      embedding_result_proto = embeddings_pb2.EmbeddingResult()
      embedding_result_proto.CopyFrom(
          packet_getter.get_proto(output_packets[_EMBEDDINGS_OUT_STREAM_NAME]))

      image = packet_getter.get_image(output_packets[_IMAGE_OUT_STREAM_NAME])
      timestamp = output_packets[_IMAGE_OUT_STREAM_NAME].timestamp
      options.result_callback(
          ImageEmbedderResult.create_from_pb2(embedding_result_proto), image,
          timestamp.value // _MICRO_SECONDS_PER_MILLISECOND)

    task_info = _TaskInfo(
        task_graph=_TASK_GRAPH_NAME,
        input_streams=[
            ':'.join([_IMAGE_TAG, _IMAGE_IN_STREAM_NAME]),
            ':'.join([_NORM_RECT_TAG, _NORM_RECT_STREAM_NAME]),
        ],
        output_streams=[
            ':'.join([_EMBEDDINGS_TAG, _EMBEDDINGS_OUT_STREAM_NAME]),
            ':'.join([_IMAGE_TAG, _IMAGE_OUT_STREAM_NAME])
        ],
        task_options=options)
    return cls(
        task_info.generate_graph_config(
            enable_flow_limiting=options.running_mode ==
            _RunningMode.LIVE_STREAM), options.running_mode,
        packets_callback if options.result_callback else None)

  def embed(
      self,
      image: image_module.Image,
      image_processing_options: Optional[_ImageProcessingOptions] = None
  ) -> ImageEmbedderResult:
    """Performs image embedding extraction on the provided MediaPipe Image.

     Extraction is performed on the region of interest specified by the `roi`
     argument if provided, or on the entire image otherwise.

    Args:
      image: MediaPipe Image.
      image_processing_options: Options for image processing.

    Returns:
      An embedding result object that contains a list of embeddings.

    Raises:
      ValueError: If any of the input arguments is invalid.
      RuntimeError: If image embedder failed to run.
    """
    normalized_rect = self.convert_to_normalized_rect(image_processing_options)
    output_packets = self._process_image_data({
        _IMAGE_IN_STREAM_NAME:
            packet_creator.create_image(image),
        _NORM_RECT_STREAM_NAME:
            packet_creator.create_proto(normalized_rect.to_pb2())
    })

    embedding_result_proto = embeddings_pb2.EmbeddingResult()
    embedding_result_proto.CopyFrom(
        packet_getter.get_proto(output_packets[_EMBEDDINGS_OUT_STREAM_NAME]))

    return ImageEmbedderResult.create_from_pb2(embedding_result_proto)

  def embed_for_video(
      self,
      image: image_module.Image,
      timestamp_ms: int,
      image_processing_options: Optional[_ImageProcessingOptions] = None
  ) -> ImageEmbedderResult:
    """Performs image embedding extraction on the provided video frames.

    Extraction is performed on the region of interested specified by the `roi`
    argument if provided, or on the entire image otherwise.

    Only use this method when the ImageEmbedder is created with the video
    running mode. It's required to provide the video frame's timestamp (in
    milliseconds) along with the video frame. The input timestamps should be
    monotonically increasing for adjacent calls of this method.

    Args:
      image: MediaPipe Image.
      timestamp_ms: The timestamp of the input video frame in milliseconds.
      image_processing_options: Options for image processing.

    Returns:
      An embedding result object that contains a list of embeddings.

    Raises:
      ValueError: If any of the input arguments is invalid.
      RuntimeError: If image embedder failed to run.
    """
    normalized_rect = self.convert_to_normalized_rect(image_processing_options)
    output_packets = self._process_video_data({
        _IMAGE_IN_STREAM_NAME:
            packet_creator.create_image(image).at(
                timestamp_ms * _MICRO_SECONDS_PER_MILLISECOND),
        _NORM_RECT_STREAM_NAME:
            packet_creator.create_proto(normalized_rect.to_pb2()).at(
                timestamp_ms * _MICRO_SECONDS_PER_MILLISECOND)
    })
    embedding_result_proto = embeddings_pb2.EmbeddingResult()
    embedding_result_proto.CopyFrom(
        packet_getter.get_proto(output_packets[_EMBEDDINGS_OUT_STREAM_NAME]))

    return ImageEmbedderResult.create_from_pb2(embedding_result_proto)

  def embed_async(
      self,
      image: image_module.Image,
      timestamp_ms: int,
      image_processing_options: Optional[_ImageProcessingOptions] = None
  ) -> None:
    """Sends live image data to embedder.

    The results will be available via the "result_callback" provided in the
    ImageEmbedderOptions. Embedding extraction is performed on the region of
    interested specified by the `roi` argument if provided, or on the entire
    image otherwise.

    Only use this method when the ImageEmbedder is created with the live
    stream running mode. The input timestamps should be monotonically increasing
    for adjacent calls of this method. This method will return immediately after
    the input image is accepted. The results will be available via the
    `result_callback` provided in the `ImageEmbedderOptions`. The
    `embed_async` method is designed to process live stream data such as
    camera input. To lower the overall latency, image embedder may drop the
    input images if needed. In other words, it's not guaranteed to have output
    per input image.

    The `result_callback` provides:
      - An embedding result object that contains a list of embeddings.
      - The input image that the image embedder runs on.
      - The input timestamp in milliseconds.

    Args:
      image: MediaPipe Image.
      timestamp_ms: The timestamp of the input image in milliseconds.
      image_processing_options: Options for image processing.

    Raises:
      ValueError: If the current input timestamp is smaller than what the image
        embedder has already processed.
    """
    normalized_rect = self.convert_to_normalized_rect(image_processing_options)
    self._send_live_stream_data({
        _IMAGE_IN_STREAM_NAME:
            packet_creator.create_image(image).at(
                timestamp_ms * _MICRO_SECONDS_PER_MILLISECOND),
        _NORM_RECT_STREAM_NAME:
            packet_creator.create_proto(normalized_rect.to_pb2()).at(
                timestamp_ms * _MICRO_SECONDS_PER_MILLISECOND)
    })

  @classmethod
  def cosine_similarity(cls, u: embedding_result_module.Embedding,
                        v: embedding_result_module.Embedding) -> float:
    """Utility function to compute cosine similarity between two embedding entries.

    May return an InvalidArgumentError if e.g. the feature vectors are of
    different types (quantized vs. float), have different sizes, or have an
    L2-norm of 0.

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
