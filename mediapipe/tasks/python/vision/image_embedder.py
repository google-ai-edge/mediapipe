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
from mediapipe.python._framework_bindings import task_runner as task_runner_module
from mediapipe.tasks.cc.vision.image_embedder.proto import image_embedder_graph_options_pb2
from mediapipe.tasks.python.components.proto import embedder_options
from mediapipe.tasks.python.components.containers import embeddings as embeddings_module
from mediapipe.tasks.python.components.containers import rect as rect_module
from mediapipe.tasks.python.core import base_options as base_options_module
from mediapipe.tasks.python.core import task_info as task_info_module
from mediapipe.tasks.python.core.optional_dependencies import doc_controls
from mediapipe.tasks.python.vision.core import base_vision_task_api
from mediapipe.tasks.python.vision.core import vision_task_running_mode as running_mode_module

_NormalizedRect = rect_module.NormalizedRect
_BaseOptions = base_options_module.BaseOptions
_ImageEmbedderGraphOptionsProto = image_embedder_graph_options_pb2.ImageEmbedderGraphOptions
_EmbedderOptions = embedder_options.EmbedderOptions
_RunningMode = running_mode_module.VisionTaskRunningMode
_TaskInfo = task_info_module.TaskInfo
_TaskRunner = task_runner_module.TaskRunner

_EMBEDDING_RESULT_OUT_STREAM_NAME = 'embedding_result_out'
_EMBEDDING_RESULT_TAG = 'EMBEDDING_RESULT'
_IMAGE_IN_STREAM_NAME = 'image_in'
_IMAGE_OUT_STREAM_NAME = 'image_out'
_IMAGE_TAG = 'IMAGE'
_NORM_RECT_NAME = 'norm_rect_in'
_NORM_RECT_TAG = 'NORM_RECT'
_TASK_GRAPH_NAME = 'mediapipe.tasks.vision.image_embedder.ImageEmbedderGraph'
_MICRO_SECONDS_PER_MILLISECOND = 1000


def _build_full_image_norm_rect() -> _NormalizedRect:
  # Builds a NormalizedRect covering the entire image.
  return _NormalizedRect(x_center=0.5, y_center=0.5, width=1, height=1)


@dataclasses.dataclass
class ImageEmbedderOptions:
  """Options for the image embedder task.

  Attributes:
    base_options: Base options for the image embedder task.
    running_mode: The running mode of the task. Default to the image mode.
      Image embedder task has three running modes:
      1) The image mode for embedding image on single image inputs.
      2) The video mode for embedding image on the decoded frames of a
         video.
      3) The live stream mode for embedding image on a live stream of input
         data, such as from camera.
    embedder_options: Options for the image embedder task.
    result_callback: The user-defined result callback for processing live stream
      data. The result callback should only be specified when the running mode
      is set to the live stream mode.
  """
  base_options: _BaseOptions
  running_mode: _RunningMode = _RunningMode.IMAGE
  embedder_options: _EmbedderOptions = _EmbedderOptions()
  result_callback: Optional[
      Callable[[embeddings_module.EmbeddingResult, image_module.Image,
                int], None]] = None

  @doc_controls.do_not_generate_docs
  def to_pb2(self) -> _ImageEmbedderGraphOptionsProto:
    """Generates an ImageEmbedderOptions protobuf object."""
    base_options_proto = self.base_options.to_pb2()
    base_options_proto.use_stream_mode = False if self.running_mode == _RunningMode.IMAGE else True
    embedder_options_proto = self.embedder_options.to_pb2()

    return _ImageEmbedderGraphOptionsProto(
        base_options=base_options_proto,
        embedder_options=embedder_options_proto
    )


class ImageEmbedder(base_vision_task_api.BaseVisionTaskApi):
  """Class that performs embedding extraction on images."""

  @classmethod
  def create_from_model_path(cls, model_path: str) -> 'ImageEmbedder':
    """Creates an `ImageEmbedder` object from a TensorFlow Lite model and the
      default `ImageEmbedderOptions`.

    Note that the created `ImageEmbedder` instance is in image mode, for
    embedding image on single image inputs.

    Args:
      model_path: Path to the model.

    Returns:
      `ImageEmbedder` object that's created from the model file and the default
      `ImageEmbedderOptions`.

    Raises:
      ValueError: If failed to create `ImageClassifier` object from the provided
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
      embedding_result_proto = packet_getter.get_proto(
          output_packets[_EMBEDDING_RESULT_OUT_STREAM_NAME])

      embedding_result = embeddings_module.EmbeddingResult([
          embeddings_module.Embeddings.create_from_pb2(embedding)
          for embedding in embedding_result_proto.embeddings
      ])
      image = packet_getter.get_image(output_packets[_IMAGE_OUT_STREAM_NAME])
      timestamp = output_packets[_IMAGE_OUT_STREAM_NAME].timestamp
      options.result_callback(embedding_result, image,
                              timestamp.value // _MICRO_SECONDS_PER_MILLISECOND)

    task_info = _TaskInfo(
        task_graph=_TASK_GRAPH_NAME,
        input_streams=[
            ':'.join([_IMAGE_TAG, _IMAGE_IN_STREAM_NAME]),
            ':'.join([_NORM_RECT_TAG, _NORM_RECT_NAME]),
        ],
        output_streams=[
            ':'.join([_EMBEDDING_RESULT_TAG,
                      _EMBEDDING_RESULT_OUT_STREAM_NAME]),
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
      roi: Optional[_NormalizedRect] = None
  ) -> embeddings_module.EmbeddingResult:
    """Performs image embedding extraction on the provided MediaPipe Image.
     Extraction is performed on the region of interest specified by the `roi`
     argument if provided, or on the entire image otherwise.

    Args:
      image: MediaPipe Image.
      roi: The region of interest.

    Returns:
      A embedding result object that contains a list of embeddings.

    Raises:
      ValueError: If any of the input arguments is invalid.
      RuntimeError: If image embedder failed to run.
    """
    norm_rect = roi if roi is not None else _build_full_image_norm_rect()
    output_packets = self._process_image_data({
        _IMAGE_IN_STREAM_NAME: packet_creator.create_image(image),
        _NORM_RECT_NAME: packet_creator.create_proto(norm_rect.to_pb2())})
    embedding_result_proto = packet_getter.get_proto(
        output_packets[_EMBEDDING_RESULT_OUT_STREAM_NAME])

    return embeddings_module.EmbeddingResult([
        embeddings_module.Embeddings.create_from_pb2(embedding)
        for embedding in embedding_result_proto.embeddings
    ])

  def embed_for_video(
      self, image: image_module.Image,
      timestamp_ms: int,
      roi: Optional[_NormalizedRect] = None
  ) -> embeddings_module.EmbeddingResult:
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
      roi: The region of interest.

    Returns:
      A embedding result object that contains a list of embeddings.

    Raises:
      ValueError: If any of the input arguments is invalid.
      RuntimeError: If image embedder failed to run.
    """
    norm_rect = roi if roi is not None else _build_full_image_norm_rect()
    output_packets = self._process_video_data({
        _IMAGE_IN_STREAM_NAME: packet_creator.create_image(image).at(
            timestamp_ms * _MICRO_SECONDS_PER_MILLISECOND),
        _NORM_RECT_NAME: packet_creator.create_proto(norm_rect.to_pb2()).at(
            timestamp_ms * _MICRO_SECONDS_PER_MILLISECOND)
    })
    embedding_result_proto = packet_getter.get_proto(
      output_packets[_EMBEDDING_RESULT_OUT_STREAM_NAME])

    return embeddings_module.EmbeddingResult([
        embeddings_module.Embeddings.create_from_pb2(embedding)
        for embedding in embedding_result_proto.embeddings
    ])

  def embed_async(
      self,
      image: image_module.Image,
      timestamp_ms: int,
      roi: Optional[_NormalizedRect] = None
  ) -> None:
    """ Sends live image data to embedder, and the results will be available via
    the "result_callback" provided in the ImageEmbedderOptions. Embedding
    extraction is performed on the region of interested specified by the `roi`
    argument if provided, or on the entire image otherwise.

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
      - A embedding result object that contains a list of embeddings.
      - The input image that the image embedder runs on.
      - The input timestamp in milliseconds.

    Args:
      image: MediaPipe Image.
      timestamp_ms: The timestamp of the input image in milliseconds.
      roi: The region of interest.

    Raises:
      ValueError: If the current input timestamp is smaller than what the image
        embedder has already processed.
    """
    norm_rect = roi if roi is not None else _build_full_image_norm_rect()
    self._send_live_stream_data({
        _IMAGE_IN_STREAM_NAME: packet_creator.create_image(image).at(
            timestamp_ms * _MICRO_SECONDS_PER_MILLISECOND),
        _NORM_RECT_NAME: packet_creator.create_proto(norm_rect.to_pb2()).at(
            timestamp_ms * _MICRO_SECONDS_PER_MILLISECOND)
    })
