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
"""MediaPipe image segmenter task."""

import dataclasses
import enum
from typing import Callable, List, Mapping, Optional

from mediapipe.python import packet_creator
from mediapipe.python import packet_getter
from mediapipe.python._framework_bindings import image as image_module
from mediapipe.python._framework_bindings import packet
from mediapipe.tasks.cc.vision.image_segmenter.proto import image_segmenter_graph_options_pb2
from mediapipe.tasks.cc.vision.image_segmenter.proto import segmenter_options_pb2
from mediapipe.tasks.python.core import base_options as base_options_module
from mediapipe.tasks.python.core import task_info as task_info_module
from mediapipe.tasks.python.core.optional_dependencies import doc_controls
from mediapipe.tasks.python.vision.core import base_vision_task_api
from mediapipe.tasks.python.vision.core import vision_task_running_mode

_BaseOptions = base_options_module.BaseOptions
_SegmenterOptionsProto = segmenter_options_pb2.SegmenterOptions
_ImageSegmenterGraphOptionsProto = image_segmenter_graph_options_pb2.ImageSegmenterGraphOptions
_RunningMode = vision_task_running_mode.VisionTaskRunningMode
_TaskInfo = task_info_module.TaskInfo

_SEGMENTATION_OUT_STREAM_NAME = 'segmented_mask_out'
_SEGMENTATION_TAG = 'GROUPED_SEGMENTATION'
_IMAGE_IN_STREAM_NAME = 'image_in'
_IMAGE_OUT_STREAM_NAME = 'image_out'
_IMAGE_TAG = 'IMAGE'
_TASK_GRAPH_NAME = 'mediapipe.tasks.vision.image_segmenter.ImageSegmenterGraph'
_MICRO_SECONDS_PER_MILLISECOND = 1000


@dataclasses.dataclass
class ImageSegmenterOptions:
  """Options for the image segmenter task.

  Attributes:
    base_options: Base options for the image segmenter task.
    running_mode: The running mode of the task. Default to the image mode. Image
      segmenter task has three running modes: 1) The image mode for segmenting
      objects on single image inputs. 2) The video mode for segmenting objects
      on the decoded frames of a video. 3) The live stream mode for segmenting
      objects on a live stream of input data, such as from camera.
    output_type: The output mask type allows specifying the type of
      post-processing to perform on the raw model results.
    activation: Activation function to apply to input tensor.
    result_callback: The user-defined result callback for processing live stream
      data. The result callback should only be specified when the running mode
      is set to the live stream mode.
  """

  class OutputType(enum.Enum):
    UNSPECIFIED = 0
    CATEGORY_MASK = 1
    CONFIDENCE_MASK = 2

  class Activation(enum.Enum):
    NONE = 0
    SIGMOID = 1
    SOFTMAX = 2

  base_options: _BaseOptions
  running_mode: _RunningMode = _RunningMode.IMAGE
  output_type: Optional[OutputType] = OutputType.CATEGORY_MASK
  activation: Optional[Activation] = Activation.NONE
  result_callback: Optional[Callable[
      [List[image_module.Image], image_module.Image, int], None]] = None

  @doc_controls.do_not_generate_docs
  def to_pb2(self) -> _ImageSegmenterGraphOptionsProto:
    """Generates an ImageSegmenterOptions protobuf object."""
    base_options_proto = self.base_options.to_pb2()
    base_options_proto.use_stream_mode = False if self.running_mode == _RunningMode.IMAGE else True
    segmenter_options_proto = _SegmenterOptionsProto(
        output_type=self.output_type.value, activation=self.activation.value)
    return _ImageSegmenterGraphOptionsProto(
        base_options=base_options_proto,
        segmenter_options=segmenter_options_proto)


class ImageSegmenter(base_vision_task_api.BaseVisionTaskApi):
  """Class that performs image segmentation on images.

  The API expects a TFLite model with mandatory TFLite Model Metadata.

  Input tensor:
    (kTfLiteUInt8/kTfLiteFloat32)
    - image input of size `[batch x height x width x channels]`.
    - batch inference is not supported (`batch` is required to be 1).
    - RGB and greyscale inputs are supported (`channels` is required to be
      1 or 3).
    - if type is kTfLiteFloat32, NormalizationOptions are required to be
      attached to the metadata for input normalization.
  Output tensors:
    (kTfLiteUInt8/kTfLiteFloat32)
    - list of segmented masks.
    - if `output_type` is CATEGORY_MASK, uint8 Image, Image vector of size 1.
    - if `output_type` is CONFIDENCE_MASK, float32 Image list of size
      `channels`.
    - batch is always 1

  An example of such model can be found at:
  https://tfhub.dev/tensorflow/lite-model/deeplabv3/1/metadata/2
  """

  @classmethod
  def create_from_model_path(cls, model_path: str) -> 'ImageSegmenter':
    """Creates an `ImageSegmenter` object from a TensorFlow Lite model and the default `ImageSegmenterOptions`.

    Note that the created `ImageSegmenter` instance is in image mode, for
    performing image segmentation on single image inputs.

    Args:
      model_path: Path to the model.

    Returns:
      `ImageSegmenter` object that's created from the model file and the default
      `ImageSegmenterOptions`.

    Raises:
      ValueError: If failed to create `ImageSegmenter` object from the provided
        file such as invalid file path.
      RuntimeError: If other types of error occurred.
    """
    base_options = _BaseOptions(model_asset_path=model_path)
    options = ImageSegmenterOptions(
        base_options=base_options, running_mode=_RunningMode.IMAGE)
    return cls.create_from_options(options)

  @classmethod
  def create_from_options(cls,
                          options: ImageSegmenterOptions) -> 'ImageSegmenter':
    """Creates the `ImageSegmenter` object from image segmenter options.

    Args:
      options: Options for the image segmenter task.

    Returns:
      `ImageSegmenter` object that's created from `options`.

    Raises:
      ValueError: If failed to create `ImageSegmenter` object from
        `ImageSegmenterOptions` such as missing the model.
      RuntimeError: If other types of error occurred.
    """

    def packets_callback(output_packets: Mapping[str, packet.Packet]):
      if output_packets[_IMAGE_OUT_STREAM_NAME].is_empty():
        return
      segmentation_result = packet_getter.get_image_list(
          output_packets[_SEGMENTATION_OUT_STREAM_NAME])
      image = packet_getter.get_image(output_packets[_IMAGE_OUT_STREAM_NAME])
      timestamp = output_packets[_SEGMENTATION_OUT_STREAM_NAME].timestamp
      options.result_callback(segmentation_result, image,
                              timestamp.value // _MICRO_SECONDS_PER_MILLISECOND)

    task_info = _TaskInfo(
        task_graph=_TASK_GRAPH_NAME,
        input_streams=[':'.join([_IMAGE_TAG, _IMAGE_IN_STREAM_NAME])],
        output_streams=[
            ':'.join([_SEGMENTATION_TAG, _SEGMENTATION_OUT_STREAM_NAME]),
            ':'.join([_IMAGE_TAG, _IMAGE_OUT_STREAM_NAME])
        ],
        task_options=options)
    return cls(
        task_info.generate_graph_config(
            enable_flow_limiting=options.running_mode ==
            _RunningMode.LIVE_STREAM), options.running_mode,
        packets_callback if options.result_callback else None)

  def segment(self, image: image_module.Image) -> List[image_module.Image]:
    """Performs the actual segmentation task on the provided MediaPipe Image.

    Args:
      image: MediaPipe Image.

    Returns:
      If the output_type is CATEGORY_MASK, the returned vector of images is
      per-category segmented image mask.
      If the output_type is CONFIDENCE_MASK, the returned vector of images
      contains only one confidence image mask. A segmentation result object that
      contains a list of segmentation masks as images.

    Raises:
      ValueError: If any of the input arguments is invalid.
      RuntimeError: If image segmentation failed to run.
    """
    output_packets = self._process_image_data(
        {_IMAGE_IN_STREAM_NAME: packet_creator.create_image(image)})
    segmentation_result = packet_getter.get_image_list(
        output_packets[_SEGMENTATION_OUT_STREAM_NAME])
    return segmentation_result

  def segment_for_video(self, image: image_module.Image,
                        timestamp_ms: int) -> List[image_module.Image]:
    """Performs segmentation on the provided video frames.

    Only use this method when the ImageSegmenter is created with the video
    running mode. It's required to provide the video frame's timestamp (in
    milliseconds) along with the video frame. The input timestamps should be
    monotonically increasing for adjacent calls of this method.

    Args:
      image: MediaPipe Image.
      timestamp_ms: The timestamp of the input video frame in milliseconds.

    Returns:
      If the output_type is CATEGORY_MASK, the returned vector of images is
      per-category segmented image mask.
      If the output_type is CONFIDENCE_MASK, the returned vector of images
      contains only one confidence image mask. A segmentation result object that
      contains a list of segmentation masks as images.

    Raises:
      ValueError: If any of the input arguments is invalid.
      RuntimeError: If image segmentation failed to run.
    """
    output_packets = self._process_video_data({
        _IMAGE_IN_STREAM_NAME:
            packet_creator.create_image(image).at(
                timestamp_ms * _MICRO_SECONDS_PER_MILLISECOND)
    })
    segmentation_result = packet_getter.get_image_list(
        output_packets[_SEGMENTATION_OUT_STREAM_NAME])
    return segmentation_result

  def segment_async(self, image: image_module.Image, timestamp_ms: int) -> None:
    """Sends live image data (an Image with a unique timestamp) to perform image segmentation.

    Only use this method when the ImageSegmenter is created with the live stream
    running mode. The input timestamps should be monotonically increasing for
    adjacent calls of this method. This method will return immediately after the
    input image is accepted. The results will be available via the
    `result_callback` provided in the `ImageSegmenterOptions`. The
    `segment_async` method is designed to process live stream data such as
    camera input. To lower the overall latency, image segmenter may drop the
    input images if needed. In other words, it's not guaranteed to have output
    per input image.

    The `result_callback` prvoides:
      - A segmentation result object that contains a list of segmentation masks
        as images.
      - The input image that the image segmenter runs on.
      - The input timestamp in milliseconds.

    Args:
      image: MediaPipe Image.
      timestamp_ms: The timestamp of the input image in milliseconds.

    Raises:
      ValueError: If the current input timestamp is smaller than what the image
        segmenter has already processed.
    """
    self._send_live_stream_data({
        _IMAGE_IN_STREAM_NAME:
            packet_creator.create_image(image).at(
                timestamp_ms * _MICRO_SECONDS_PER_MILLISECOND)
    })
