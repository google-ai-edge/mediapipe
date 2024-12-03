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
"""MediaPipe object detector task."""

import dataclasses
from typing import Callable, List, Mapping, Optional

from mediapipe.python import packet_creator
from mediapipe.python import packet_getter
from mediapipe.python._framework_bindings import image as image_module
from mediapipe.python._framework_bindings import packet as packet_module
from mediapipe.tasks.cc.vision.object_detector.proto import object_detector_options_pb2
from mediapipe.tasks.python.components.containers import detections as detections_module
from mediapipe.tasks.python.components.containers import rect
from mediapipe.tasks.python.core import base_options as base_options_module
from mediapipe.tasks.python.core import task_info as task_info_module
from mediapipe.tasks.python.core.optional_dependencies import doc_controls
from mediapipe.tasks.python.vision.core import base_vision_task_api
from mediapipe.tasks.python.vision.core import image_processing_options as image_processing_options_module
from mediapipe.tasks.python.vision.core import vision_task_running_mode as running_mode_module

ObjectDetectorResult = detections_module.DetectionResult
_NormalizedRect = rect.NormalizedRect
_BaseOptions = base_options_module.BaseOptions
_ObjectDetectorOptionsProto = object_detector_options_pb2.ObjectDetectorOptions
_RunningMode = running_mode_module.VisionTaskRunningMode
_ImageProcessingOptions = image_processing_options_module.ImageProcessingOptions
_TaskInfo = task_info_module.TaskInfo

_DETECTIONS_OUT_STREAM_NAME = 'detections_out'
_DETECTIONS_TAG = 'DETECTIONS'
_IMAGE_IN_STREAM_NAME = 'image_in'
_IMAGE_OUT_STREAM_NAME = 'image_out'
_IMAGE_TAG = 'IMAGE'
_NORM_RECT_STREAM_NAME = 'norm_rect_in'
_NORM_RECT_TAG = 'NORM_RECT'
_TASK_GRAPH_NAME = 'mediapipe.tasks.vision.ObjectDetectorGraph'
_MICRO_SECONDS_PER_MILLISECOND = 1000


@dataclasses.dataclass
class ObjectDetectorOptions:
  """Options for the object detector task.

  Attributes:
    base_options: Base options for the object detector task.
    running_mode: The running mode of the task. Default to the image mode.
      Object detector task has three running modes: 1) The image mode for
      detecting objects on single image inputs. 2) The video mode for detecting
      objects on the decoded frames of a video. 3) The live stream mode for
      detecting objects on a live stream of input data, such as from camera.
    display_names_locale: The locale to use for display names specified through
      the TFLite Model Metadata.
    max_results: The maximum number of top-scored classification results to
      return.
    score_threshold: Overrides the ones provided in the model metadata. Results
      below this value are rejected.
    category_allowlist: Allowlist of category names. If non-empty, detection
      results whose category name is not in this set will be filtered out.
      Duplicate or unknown category names are ignored. Mutually exclusive with
      `category_denylist`.
    category_denylist: Denylist of category names. If non-empty, detection
      results whose category name is in this set will be filtered out. Duplicate
      or unknown category names are ignored. Mutually exclusive with
      `category_allowlist`.
    result_callback: The user-defined result callback for processing live stream
      data. The result callback should only be specified when the running mode
      is set to the live stream mode.
  """

  base_options: _BaseOptions
  running_mode: _RunningMode = _RunningMode.IMAGE
  display_names_locale: Optional[str] = None
  max_results: Optional[int] = None
  score_threshold: Optional[float] = None
  category_allowlist: Optional[List[str]] = None
  category_denylist: Optional[List[str]] = None
  result_callback: Optional[
      Callable[[ObjectDetectorResult, image_module.Image, int], None]
  ] = None

  @doc_controls.do_not_generate_docs
  def to_pb2(self) -> _ObjectDetectorOptionsProto:
    """Generates an ObjectDetectorOptions protobuf object."""
    base_options_proto = self.base_options.to_pb2()
    base_options_proto.use_stream_mode = (
        False if self.running_mode == _RunningMode.IMAGE else True
    )
    return _ObjectDetectorOptionsProto(
        base_options=base_options_proto,
        display_names_locale=self.display_names_locale,
        max_results=self.max_results,
        score_threshold=self.score_threshold,
        category_allowlist=self.category_allowlist,
        category_denylist=self.category_denylist,
    )


class ObjectDetector(base_vision_task_api.BaseVisionTaskApi):
  """Class that performs object detection on images.

  The API expects a TFLite model with mandatory TFLite Model Metadata.

  Input tensor:
    (kTfLiteUInt8/kTfLiteFloat32)
    - image input of size `[batch x height x width x channels]`.
    - batch inference is not supported (`batch` is required to be 1).
    - only RGB inputs are supported (`channels` is required to be 3).
    - if type is kTfLiteFloat32, NormalizationOptions are required to be
      attached to the metadata for input normalization.
  Output tensors must be the 4 outputs of a `DetectionPostProcess` op, i.e:
    (kTfLiteFloat32)
    - locations tensor of size `[num_results x 4]`, the inner array
      representing bounding boxes in the form [top, left, right, bottom].
    - BoundingBoxProperties are required to be attached to the metadata
      and must specify type=BOUNDARIES and coordinate_type=RATIO.
    (kTfLiteFloat32)
    - classes tensor of size `[num_results]`, each value representing the
      integer index of a class.
    - optional (but recommended) label map(s) can be attached as
      AssociatedFile-s with type TENSOR_VALUE_LABELS, containing one label per
      line. The first such AssociatedFile (if any) is used to fill the
      `class_name` field of the results. The `display_name` field is filled
      from the AssociatedFile (if any) whose locale matches the
      `display_names_locale` field of the `ObjectDetectorOptions` used at
      creation time ("en" by default, i.e. English). If none of these are
      available, only the `index` field of the results will be filled.
    (kTfLiteFloat32)
    - scores tensor of size `[num_results]`, each value representing the score
      of the detected object.
    - optional score calibration can be attached using ScoreCalibrationOptions
      and an AssociatedFile with type TENSOR_AXIS_SCORE_CALIBRATION. See
      metadata_schema.fbs [1] for more details.
    (kTfLiteFloat32)
    - integer num_results as a tensor of size `[1]`

  An example of such model can be found at:
  https://tfhub.dev/google/lite-model/object_detection/mobile_object_localizer_v1/1/metadata/1

  [1]:
  https://github.com/google/mediapipe/blob/6cdc6443b6a7ed662744e2a2ce2d58d9c83e6d6f/mediapipe/tasks/metadata/metadata_schema.fbs#L456
  """

  @classmethod
  def create_from_model_path(cls, model_path: str) -> 'ObjectDetector':
    """Creates an `ObjectDetector` object from a TensorFlow Lite model and the default `ObjectDetectorOptions`.

    Note that the created `ObjectDetector` instance is in image mode, for
    detecting objects on single image inputs.

    Args:
      model_path: Path to the model.

    Returns:
      `ObjectDetector` object that's created from the model file and the default
      `ObjectDetectorOptions`.

    Raises:
      ValueError: If failed to create `ObjectDetector` object from the provided
        file such as invalid file path.
      RuntimeError: If other types of error occurred.
    """
    base_options = _BaseOptions(model_asset_path=model_path)
    options = ObjectDetectorOptions(
        base_options=base_options, running_mode=_RunningMode.IMAGE
    )
    return cls.create_from_options(options)

  @classmethod
  def create_from_options(
      cls, options: ObjectDetectorOptions
  ) -> 'ObjectDetector':
    """Creates the `ObjectDetector` object from object detector options.

    Args:
      options: Options for the object detector task.

    Returns:
      `ObjectDetector` object that's created from `options`.

    Raises:
      ValueError: If failed to create `ObjectDetector` object from
        `ObjectDetectorOptions` such as missing the model.
      RuntimeError: If other types of error occurred.
    """

    def packets_callback(output_packets: Mapping[str, packet_module.Packet]):
      if output_packets[_IMAGE_OUT_STREAM_NAME].is_empty():
        return
      image = packet_getter.get_image(output_packets[_IMAGE_OUT_STREAM_NAME])
      if output_packets[_DETECTIONS_OUT_STREAM_NAME].is_empty():
        empty_packet = output_packets[_DETECTIONS_OUT_STREAM_NAME]
        options.result_callback(
            ObjectDetectorResult([]),
            image,
            empty_packet.timestamp.value // _MICRO_SECONDS_PER_MILLISECOND,
        )
        return
      detection_proto_list = packet_getter.get_proto_list(
          output_packets[_DETECTIONS_OUT_STREAM_NAME]
      )
      detection_result = ObjectDetectorResult(
          [
              detections_module.Detection.create_from_pb2(result)
              for result in detection_proto_list
          ]
      )
      timestamp = output_packets[_IMAGE_OUT_STREAM_NAME].timestamp
      options.result_callback(
          detection_result,
          image,
          timestamp.value // _MICRO_SECONDS_PER_MILLISECOND,
      )

    task_info = _TaskInfo(
        task_graph=_TASK_GRAPH_NAME,
        input_streams=[
            ':'.join([_IMAGE_TAG, _IMAGE_IN_STREAM_NAME]),
            ':'.join([_NORM_RECT_TAG, _NORM_RECT_STREAM_NAME]),
        ],
        output_streams=[
            ':'.join([_DETECTIONS_TAG, _DETECTIONS_OUT_STREAM_NAME]),
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

  # TODO: Create an Image class for MediaPipe Tasks.
  def detect(
      self,
      image: image_module.Image,
      image_processing_options: Optional[_ImageProcessingOptions] = None,
  ) -> ObjectDetectorResult:
    """Performs object detection on the provided MediaPipe Image.

    Only use this method when the ObjectDetector is created with the image
    running mode.

    Args:
      image: MediaPipe Image.
      image_processing_options: Options for image processing.

    Returns:
      A detection result object that contains a list of detections, each
      detection has a bounding box that is expressed in the unrotated input
      frame of reference coordinates system, i.e. in `[0,image_width) x [0,
      image_height)`, which are the dimensions of the underlying image data.

    Raises:
      ValueError: If any of the input arguments is invalid.
      RuntimeError: If object detection failed to run.
    """
    normalized_rect = self.convert_to_normalized_rect(
        image_processing_options, image, roi_allowed=False
    )
    output_packets = self._process_image_data({
        _IMAGE_IN_STREAM_NAME: packet_creator.create_image(image),
        _NORM_RECT_STREAM_NAME: packet_creator.create_proto(
            normalized_rect.to_pb2()
        ),
    })
    if output_packets[_DETECTIONS_OUT_STREAM_NAME].is_empty():
      return ObjectDetectorResult([])
    detection_proto_list = packet_getter.get_proto_list(
        output_packets[_DETECTIONS_OUT_STREAM_NAME]
    )
    return ObjectDetectorResult(
        [
            detections_module.Detection.create_from_pb2(result)
            for result in detection_proto_list
        ]
    )

  def detect_for_video(
      self,
      image: image_module.Image,
      timestamp_ms: int,
      image_processing_options: Optional[_ImageProcessingOptions] = None,
  ) -> ObjectDetectorResult:
    """Performs object detection on the provided video frames.

    Only use this method when the ObjectDetector is created with the video
    running mode. It's required to provide the video frame's timestamp (in
    milliseconds) along with the video frame. The input timestamps should be
    monotonically increasing for adjacent calls of this method.

    Args:
      image: MediaPipe Image.
      timestamp_ms: The timestamp of the input video frame in milliseconds.
      image_processing_options: Options for image processing.

    Returns:
      A detection result object that contains a list of detections, each
      detection has a bounding box that is expressed in the unrotated input
      frame of reference coordinates system, i.e. in `[0,image_width) x [0,
      image_height)`, which are the dimensions of the underlying image data.

    Raises:
      ValueError: If any of the input arguments is invalid.
      RuntimeError: If object detection failed to run.
    """
    normalized_rect = self.convert_to_normalized_rect(
        image_processing_options, image, roi_allowed=False
    )
    output_packets = self._process_video_data({
        _IMAGE_IN_STREAM_NAME: packet_creator.create_image(image).at(
            timestamp_ms * _MICRO_SECONDS_PER_MILLISECOND
        ),
        _NORM_RECT_STREAM_NAME: packet_creator.create_proto(
            normalized_rect.to_pb2()
        ).at(timestamp_ms * _MICRO_SECONDS_PER_MILLISECOND),
    })
    if output_packets[_DETECTIONS_OUT_STREAM_NAME].is_empty():
      return ObjectDetectorResult([])
    detection_proto_list = packet_getter.get_proto_list(
        output_packets[_DETECTIONS_OUT_STREAM_NAME]
    )
    return ObjectDetectorResult(
        [
            detections_module.Detection.create_from_pb2(result)
            for result in detection_proto_list
        ]
    )

  def detect_async(
      self,
      image: image_module.Image,
      timestamp_ms: int,
      image_processing_options: Optional[_ImageProcessingOptions] = None,
  ) -> None:
    """Sends live image data (an Image with a unique timestamp) to perform object detection.

    Only use this method when the ObjectDetector is created with the live stream
    running mode. The input timestamps should be monotonically increasing for
    adjacent calls of this method. This method will return immediately after the
    input image is accepted. The results will be available via the
    `result_callback` provided in the `ObjectDetectorOptions`. The
    `detect_async` method is designed to process live stream data such as camera
    input. To lower the overall latency, object detector may drop the input
    images if needed. In other words, it's not guaranteed to have output per
    input image.

    The `result_callback` prvoides:
      - A detection result object that contains a list of detections, each
        detection has a bounding box that is expressed in the unrotated input
        frame of reference coordinates system, i.e. in `[0,image_width) x [0,
        image_height)`, which are the dimensions of the underlying image data.
      - The input image that the object detector runs on.
      - The input timestamp in milliseconds.

    Args:
      image: MediaPipe Image.
      timestamp_ms: The timestamp of the input image in milliseconds.
      image_processing_options: Options for image processing.

    Raises:
      ValueError: If the current input timestamp is smaller than what the object
        detector has already processed.
    """
    normalized_rect = self.convert_to_normalized_rect(
        image_processing_options, image, roi_allowed=False
    )
    self._send_live_stream_data({
        _IMAGE_IN_STREAM_NAME: packet_creator.create_image(image).at(
            timestamp_ms * _MICRO_SECONDS_PER_MILLISECOND
        ),
        _NORM_RECT_STREAM_NAME: packet_creator.create_proto(
            normalized_rect.to_pb2()
        ).at(timestamp_ms * _MICRO_SECONDS_PER_MILLISECOND),
    })
