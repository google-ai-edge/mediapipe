# Copyright 2023 The MediaPipe Authors.
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
"""MediaPipe face stylizer task."""

import dataclasses
from typing import Callable, Mapping, Optional

from mediapipe.python import packet_creator
from mediapipe.python import packet_getter
from mediapipe.python._framework_bindings import image as image_module
from mediapipe.python._framework_bindings import packet as packet_module
from mediapipe.tasks.cc.vision.face_stylizer.proto import face_stylizer_graph_options_pb2
from mediapipe.tasks.python.core import base_options as base_options_module
from mediapipe.tasks.python.core import task_info as task_info_module
from mediapipe.tasks.python.core.optional_dependencies import doc_controls
from mediapipe.tasks.python.vision.core import base_vision_task_api
from mediapipe.tasks.python.vision.core import image_processing_options as image_processing_options_module
from mediapipe.tasks.python.vision.core import vision_task_running_mode as running_mode_module

_BaseOptions = base_options_module.BaseOptions
_FaceStylizerGraphOptionsProto = (
    face_stylizer_graph_options_pb2.FaceStylizerGraphOptions
)
_RunningMode = running_mode_module.VisionTaskRunningMode
_ImageProcessingOptions = image_processing_options_module.ImageProcessingOptions
_TaskInfo = task_info_module.TaskInfo

_STYLIZED_IMAGE_NAME = 'stylized_image'
_STYLIZED_IMAGE_TAG = 'STYLIZED_IMAGE'
_NORM_RECT_STREAM_NAME = 'norm_rect_in'
_NORM_RECT_TAG = 'NORM_RECT'
_IMAGE_IN_STREAM_NAME = 'image_in'
_IMAGE_OUT_STREAM_NAME = 'image_out'
_IMAGE_TAG = 'IMAGE'
_TASK_GRAPH_NAME = 'mediapipe.tasks.vision.face_stylizer.FaceStylizerGraph'
_MICRO_SECONDS_PER_MILLISECOND = 1000


@dataclasses.dataclass
class FaceStylizerOptions:
  """Options for the face stylizer task.

  Attributes:
    base_options: Base options for the face stylizer task.
    running_mode: The running mode of the task. Default to the image mode. Face
      stylizer task has three running modes: 1) The image mode for stylizing one
      face on a single image input. 2) The video mode for stylizing one face per
      frame on the decoded frames of a video. 3) The live stream mode for
      stylizing one face on a live stream of input data, such as from camera.
    result_callback: The user-defined result callback for processing live stream
      data. The result callback should only be specified when the running mode
      is set to the live stream mode.
  """

  base_options: _BaseOptions
  running_mode: _RunningMode = _RunningMode.IMAGE
  result_callback: Optional[
      Callable[[image_module.Image, image_module.Image, int], None]
  ] = None

  @doc_controls.do_not_generate_docs
  def to_pb2(self) -> _FaceStylizerGraphOptionsProto:
    """Generates an FaceStylizerOptions protobuf object."""
    base_options_proto = self.base_options.to_pb2()
    base_options_proto.use_stream_mode = (
        False if self.running_mode == _RunningMode.IMAGE else True
    )
    return _FaceStylizerGraphOptionsProto(base_options=base_options_proto)


class FaceStylizer(base_vision_task_api.BaseVisionTaskApi):
  """Class that performs face stylization on images."""

  @classmethod
  def create_from_model_path(cls, model_path: str) -> 'FaceStylizer':
    """Creates an `FaceStylizer` object from a TensorFlow Lite model and the default `FaceStylizerOptions`.

    Note that the created `FaceStylizer` instance is in image mode, for
    stylizing one face on a single image input.

    Args:
      model_path: Path to the model.

    Returns:
      `FaceStylizer` object that's created from the model file and the default
      `FaceStylizerOptions`.

    Raises:
      ValueError: If failed to create `FaceStylizer` object from the provided
        file such as invalid file path.
      RuntimeError: If other types of error occurred.
    """
    base_options = _BaseOptions(model_asset_path=model_path)
    options = FaceStylizerOptions(
        base_options=base_options, running_mode=_RunningMode.IMAGE
    )
    return cls.create_from_options(options)

  @classmethod
  def create_from_options(cls, options: FaceStylizerOptions) -> 'FaceStylizer':
    """Creates the `FaceStylizer` object from face stylizer options.

    Args:
      options: Options for the face stylizer task.

    Returns:
      `FaceStylizer` object that's created from `options`.

    Raises:
      ValueError: If failed to create `FaceStylizer` object from
        `FaceStylizerOptions` such as missing the model.
      RuntimeError: If other types of error occurred.
    """

    def packets_callback(output_packets: Mapping[str, packet_module.Packet]):
      if output_packets[_IMAGE_OUT_STREAM_NAME].is_empty():
        return
      image = packet_getter.get_image(output_packets[_IMAGE_OUT_STREAM_NAME])
      stylized_image_packet = output_packets[_STYLIZED_IMAGE_NAME]
      if stylized_image_packet.is_empty():
        options.result_callback(
            None,
            image,
            stylized_image_packet.timestamp.value
            // _MICRO_SECONDS_PER_MILLISECOND,
        )

      stylized_image = packet_getter.get_image(stylized_image_packet)

      options.result_callback(
          stylized_image,
          image,
          stylized_image_packet.timestamp.value
          // _MICRO_SECONDS_PER_MILLISECOND,
      )

    task_info = _TaskInfo(
        task_graph=_TASK_GRAPH_NAME,
        input_streams=[
            ':'.join([_IMAGE_TAG, _IMAGE_IN_STREAM_NAME]),
            ':'.join([_NORM_RECT_TAG, _NORM_RECT_STREAM_NAME]),
        ],
        output_streams=[
            ':'.join([_STYLIZED_IMAGE_TAG, _STYLIZED_IMAGE_NAME]),
            ':'.join([_IMAGE_TAG, _IMAGE_OUT_STREAM_NAME]),
        ],
        task_options=options,
    )
    return cls(
        task_info.generate_graph_config(
            enable_flow_limiting=options.running_mode
            == _RunningMode.LIVE_STREAM
        ),
        options.running_mode,
        packets_callback if options.result_callback else None,
    )

  def stylize(
      self,
      image: image_module.Image,
      image_processing_options: Optional[_ImageProcessingOptions] = None,
  ) -> image_module.Image:
    """Performs face stylization on the provided MediaPipe Image.

    Only use this method when the FaceStylizer is created with the image
    running mode.

    Args:
      image: MediaPipe Image.
      image_processing_options: Options for image processing.

    Returns:
      The stylized image of the most visible face. The stylized output image
      size is the same as the model output size. None if no face is detected
      on the input image.

    Raises:
      ValueError: If any of the input arguments is invalid.
      RuntimeError: If face stylization failed to run.
    """
    normalized_rect = self.convert_to_normalized_rect(
        image_processing_options, image)
    output_packets = self._process_image_data({
        _IMAGE_IN_STREAM_NAME: packet_creator.create_image(image),
        _NORM_RECT_STREAM_NAME: packet_creator.create_proto(
            normalized_rect.to_pb2()
        ),
    })
    if output_packets[_STYLIZED_IMAGE_NAME].is_empty():
      return None
    return packet_getter.get_image(output_packets[_STYLIZED_IMAGE_NAME])

  def stylize_for_video(
      self,
      image: image_module.Image,
      timestamp_ms: int,
      image_processing_options: Optional[_ImageProcessingOptions] = None,
  ) -> image_module.Image:
    """Performs face stylization on the provided video frames.

    Only use this method when the FaceStylizer is created with the video
    running mode. It's required to provide the video frame's timestamp (in
    milliseconds) along with the video frame. The input timestamps should be
    monotonically increasing for adjacent calls of this method.

    Args:
      image: MediaPipe Image.
      timestamp_ms: The timestamp of the input video frame in milliseconds.
      image_processing_options: Options for image processing.

    Returns:
      The stylized image of the most visible face. The stylized output image
      size is the same as the model output size. None if no face is detected
      on the input image.

    Raises:
      ValueError: If any of the input arguments is invalid.
      RuntimeError: If face stylization failed to run.
    """
    normalized_rect = self.convert_to_normalized_rect(
        image_processing_options, image)
    output_packets = self._process_video_data({
        _IMAGE_IN_STREAM_NAME: packet_creator.create_image(image).at(
            timestamp_ms * _MICRO_SECONDS_PER_MILLISECOND
        ),
        _NORM_RECT_STREAM_NAME: packet_creator.create_proto(
            normalized_rect.to_pb2()
        ).at(timestamp_ms * _MICRO_SECONDS_PER_MILLISECOND),
    })
    if output_packets[_STYLIZED_IMAGE_NAME].is_empty():
      return None
    return packet_getter.get_image(output_packets[_STYLIZED_IMAGE_NAME])

  def stylize_async(
      self,
      image: image_module.Image,
      timestamp_ms: int,
      image_processing_options: Optional[_ImageProcessingOptions] = None,
  ) -> None:
    """Sends live image data (an Image with a unique timestamp) to perform face stylization.

    Only use this method when the FaceStylizer is created with the live stream
    running mode. The input timestamps should be monotonically increasing for
    adjacent calls of this method. This method will return immediately after the
    input image is accepted. The results will be available via the
    `result_callback` provided in the `FaceStylizerOptions`. The
    `stylize_async` method is designed to process live stream data such as
    camera input. To lower the overall latency, face stylizer may drop the input
    images if needed. In other words, it's not guaranteed to have output per
    input image.

    The `result_callback` provides:
      - The stylized image of the most visible face. The stylized output image
        size is the same as the model output size. None if no face is detected
        on the input image.
      - The input image that the face stylizer runs on.
      - The input timestamp in milliseconds.

    Args:
      image: MediaPipe Image.
      timestamp_ms: The timestamp of the input image in milliseconds.
      image_processing_options: Options for image processing.

    Raises:
      ValueError: If the current input timestamp is smaller than what the face
        stylizer has already processed.
    """
    normalized_rect = self.convert_to_normalized_rect(
        image_processing_options, image)
    self._send_live_stream_data({
        _IMAGE_IN_STREAM_NAME: packet_creator.create_image(image).at(
            timestamp_ms * _MICRO_SECONDS_PER_MILLISECOND
        ),
        _NORM_RECT_STREAM_NAME: packet_creator.create_proto(
            normalized_rect.to_pb2()
        ).at(timestamp_ms * _MICRO_SECONDS_PER_MILLISECOND),
    })
