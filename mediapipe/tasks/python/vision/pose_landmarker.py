# Copyright 2023 The MediaPipe Authors. All Rights Reserved.
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
"""MediaPipe pose landmarker task."""

import dataclasses
from typing import Callable, Mapping, Optional, List

from mediapipe.framework.formats import landmark_pb2
from mediapipe.python import packet_creator
from mediapipe.python import packet_getter
from mediapipe.python._framework_bindings import image as image_module
from mediapipe.python._framework_bindings import packet as packet_module
from mediapipe.tasks.cc.vision.pose_landmarker.proto import pose_landmarker_graph_options_pb2
from mediapipe.tasks.python.components.containers import landmark as landmark_module
from mediapipe.tasks.python.core import base_options as base_options_module
from mediapipe.tasks.python.core import task_info as task_info_module
from mediapipe.tasks.python.core.optional_dependencies import doc_controls
from mediapipe.tasks.python.vision.core import base_vision_task_api
from mediapipe.tasks.python.vision.core import image_processing_options as image_processing_options_module
from mediapipe.tasks.python.vision.core import vision_task_running_mode as running_mode_module

_BaseOptions = base_options_module.BaseOptions
_PoseLandmarkerGraphOptionsProto = (
    pose_landmarker_graph_options_pb2.PoseLandmarkerGraphOptions
)
_RunningMode = running_mode_module.VisionTaskRunningMode
_ImageProcessingOptions = image_processing_options_module.ImageProcessingOptions
_TaskInfo = task_info_module.TaskInfo

_IMAGE_IN_STREAM_NAME = 'image_in'
_IMAGE_OUT_STREAM_NAME = 'image_out'
_IMAGE_TAG = 'IMAGE'
_NORM_RECT_STREAM_NAME = 'norm_rect_in'
_NORM_RECT_TAG = 'NORM_RECT'
_SEGMENTATION_MASK_STREAM_NAME = 'segmentation_mask'
_SEGMENTATION_MASK_TAG = 'SEGMENTATION_MASK'
_NORM_LANDMARKS_STREAM_NAME = 'norm_landmarks'
_NORM_LANDMARKS_TAG = 'NORM_LANDMARKS'
_POSE_WORLD_LANDMARKS_STREAM_NAME = 'world_landmarks'
_POSE_WORLD_LANDMARKS_TAG = 'WORLD_LANDMARKS'
_TASK_GRAPH_NAME = 'mediapipe.tasks.vision.pose_landmarker.PoseLandmarkerGraph'
_MICRO_SECONDS_PER_MILLISECOND = 1000


@dataclasses.dataclass
class PoseLandmarkerResult:
  """The pose landmarks detection result from PoseLandmarker, where each vector element represents a single pose detected in the image.

  Attributes:
    pose_landmarks: Detected pose landmarks in normalized image coordinates.
    pose_world_landmarks:  Detected pose landmarks in world coordinates.
    segmentation_masks: Optional segmentation masks for pose.
  """

  pose_landmarks: List[List[landmark_module.NormalizedLandmark]]
  pose_world_landmarks: List[List[landmark_module.Landmark]]
  segmentation_masks: Optional[List[image_module.Image]] = None


def _build_landmarker_result(
    output_packets: Mapping[str, packet_module.Packet]
) -> PoseLandmarkerResult:
  """Constructs a `PoseLandmarkerResult` from output packets."""
  pose_landmarker_result = PoseLandmarkerResult([], [])

  if _SEGMENTATION_MASK_STREAM_NAME in output_packets:
    pose_landmarker_result.segmentation_masks = packet_getter.get_image_list(
        output_packets[_SEGMENTATION_MASK_STREAM_NAME]
    )

  pose_landmarks_proto_list = packet_getter.get_proto_list(
      output_packets[_NORM_LANDMARKS_STREAM_NAME]
  )
  pose_world_landmarks_proto_list = packet_getter.get_proto_list(
      output_packets[_POSE_WORLD_LANDMARKS_STREAM_NAME]
  )

  for proto in pose_landmarks_proto_list:
    pose_landmarks = landmark_pb2.NormalizedLandmarkList()
    pose_landmarks.MergeFrom(proto)
    pose_landmarks_list = []
    for pose_landmark in pose_landmarks.landmark:
      pose_landmarks_list.append(
          landmark_module.NormalizedLandmark.create_from_pb2(pose_landmark)
      )
    pose_landmarker_result.pose_landmarks.append(pose_landmarks_list)

  for proto in pose_world_landmarks_proto_list:
    pose_world_landmarks = landmark_pb2.LandmarkList()
    pose_world_landmarks.MergeFrom(proto)
    pose_world_landmarks_list = []
    for pose_world_landmark in pose_world_landmarks.landmark:
      pose_world_landmarks_list.append(
          landmark_module.Landmark.create_from_pb2(pose_world_landmark)
      )
    pose_landmarker_result.pose_world_landmarks.append(
        pose_world_landmarks_list
    )

  return pose_landmarker_result


class PoseLandmarksConnections:
  """The connections between pose landmarks."""

  @dataclasses.dataclass
  class Connection:
    """The connection class for pose landmarks."""

    start: int
    end: int

  POSE_LANDMARKS: List[Connection] = [
      Connection(0, 1),
      Connection(1, 2),
      Connection(2, 3),
      Connection(3, 7),
      Connection(0, 4),
      Connection(4, 5),
      Connection(5, 6),
      Connection(6, 8),
      Connection(9, 10),
      Connection(11, 12),
      Connection(11, 13),
      Connection(13, 15),
      Connection(15, 17),
      Connection(15, 19),
      Connection(15, 21),
      Connection(17, 19),
      Connection(12, 14),
      Connection(14, 16),
      Connection(16, 18),
      Connection(16, 20),
      Connection(16, 22),
      Connection(18, 20),
      Connection(11, 23),
      Connection(12, 24),
      Connection(23, 24),
      Connection(23, 25),
      Connection(24, 26),
      Connection(25, 27),
      Connection(26, 28),
      Connection(27, 29),
      Connection(28, 30),
      Connection(29, 31),
      Connection(30, 32),
      Connection(27, 31),
      Connection(28, 32)
  ]


@dataclasses.dataclass
class PoseLandmarkerOptions:
  """Options for the pose landmarker task.

  Attributes:
    base_options: Base options for the pose landmarker task.
    running_mode: The running mode of the task. Default to the image mode.
      PoseLandmarker has three running modes: 1) The image mode for detecting
      pose landmarks on single image inputs. 2) The video mode for detecting
      pose landmarks on the decoded frames of a video. 3) The live stream mode
      for detecting pose landmarks on the live stream of input data, such as
      from camera. In this mode, the "result_callback" below must be specified
      to receive the detection results asynchronously.
    num_poses: The maximum number of poses can be detected by the
      PoseLandmarker.
    min_pose_detection_confidence: The minimum confidence score for the pose
      detection to be considered successful.
    min_pose_presence_confidence: The minimum confidence score of pose presence
      score in the pose landmark detection.
    min_tracking_confidence: The minimum confidence score for the pose tracking
      to be considered successful.
    output_segmentation_masks: whether to output segmentation masks.
    result_callback: The user-defined result callback for processing live stream
      data. The result callback should only be specified when the running mode
      is set to the live stream mode.
  """

  base_options: _BaseOptions
  running_mode: _RunningMode = _RunningMode.IMAGE
  num_poses: int = 1
  min_pose_detection_confidence: float = 0.5
  min_pose_presence_confidence: float = 0.5
  min_tracking_confidence: float = 0.5
  output_segmentation_masks: bool = False
  result_callback: Optional[
      Callable[[PoseLandmarkerResult, image_module.Image, int], None]
  ] = None

  @doc_controls.do_not_generate_docs
  def to_pb2(self) -> _PoseLandmarkerGraphOptionsProto:
    """Generates an PoseLandmarkerGraphOptions protobuf object."""
    base_options_proto = self.base_options.to_pb2()
    base_options_proto.use_stream_mode = (
        False if self.running_mode == _RunningMode.IMAGE else True
    )

    # Initialize the pose landmarker options from base options.
    pose_landmarker_options_proto = _PoseLandmarkerGraphOptionsProto(
        base_options=base_options_proto
    )
    pose_landmarker_options_proto.min_tracking_confidence = (
        self.min_tracking_confidence
    )
    pose_landmarker_options_proto.pose_detector_graph_options.num_poses = (
        self.num_poses
    )
    pose_landmarker_options_proto.pose_detector_graph_options.min_detection_confidence = (
        self.min_pose_detection_confidence
    )
    pose_landmarker_options_proto.pose_landmarks_detector_graph_options.min_detection_confidence = (
        self.min_pose_presence_confidence
    )
    return pose_landmarker_options_proto


class PoseLandmarker(base_vision_task_api.BaseVisionTaskApi):
  """Class that performs pose landmarks detection on images."""

  @classmethod
  def create_from_model_path(cls, model_path: str) -> 'PoseLandmarker':
    """Creates a `PoseLandmarker` object from a model bundle file and the default `PoseLandmarkerOptions`.

    Note that the created `PoseLandmarker` instance is in image mode, for
    detecting pose landmarks on single image inputs.

    Args:
      model_path: Path to the model.

    Returns:
      `PoseLandmarker` object that's created from the model file and the
      default `PoseLandmarkerOptions`.

    Raises:
      ValueError: If failed to create `PoseLandmarker` object from the
        provided file such as invalid file path.
      RuntimeError: If other types of error occurred.
    """
    base_options = _BaseOptions(model_asset_path=model_path)
    options = PoseLandmarkerOptions(
        base_options=base_options, running_mode=_RunningMode.IMAGE
    )
    return cls.create_from_options(options)

  @classmethod
  def create_from_options(
      cls, options: PoseLandmarkerOptions
  ) -> 'PoseLandmarker':
    """Creates the `PoseLandmarker` object from pose landmarker options.

    Args:
      options: Options for the pose landmarker task.

    Returns:
      `PoseLandmarker` object that's created from `options`.

    Raises:
      ValueError: If failed to create `PoseLandmarker` object from
        `PoseLandmarkerOptions` such as missing the model.
      RuntimeError: If other types of error occurred.
    """

    def packets_callback(output_packets: Mapping[str, packet_module.Packet]):
      if output_packets[_IMAGE_OUT_STREAM_NAME].is_empty():
        return

      image = packet_getter.get_image(output_packets[_IMAGE_OUT_STREAM_NAME])

      if output_packets[_NORM_LANDMARKS_STREAM_NAME].is_empty():
        empty_packet = output_packets[_NORM_LANDMARKS_STREAM_NAME]
        options.result_callback(
            PoseLandmarkerResult([], []),
            image,
            empty_packet.timestamp.value // _MICRO_SECONDS_PER_MILLISECOND,
        )
        return

      pose_landmarker_result = _build_landmarker_result(output_packets)
      timestamp = output_packets[_NORM_LANDMARKS_STREAM_NAME].timestamp
      options.result_callback(
          pose_landmarker_result,
          image,
          timestamp.value // _MICRO_SECONDS_PER_MILLISECOND,
      )

    output_streams = [
        ':'.join([_NORM_LANDMARKS_TAG, _NORM_LANDMARKS_STREAM_NAME]),
        ':'.join(
            [_POSE_WORLD_LANDMARKS_TAG, _POSE_WORLD_LANDMARKS_STREAM_NAME]
        ),
        ':'.join([_IMAGE_TAG, _IMAGE_OUT_STREAM_NAME]),
    ]

    if options.output_segmentation_masks:
      output_streams.append(
          ':'.join([_SEGMENTATION_MASK_TAG, _SEGMENTATION_MASK_STREAM_NAME])
      )

    task_info = _TaskInfo(
        task_graph=_TASK_GRAPH_NAME,
        input_streams=[
            ':'.join([_IMAGE_TAG, _IMAGE_IN_STREAM_NAME]),
            ':'.join([_NORM_RECT_TAG, _NORM_RECT_STREAM_NAME]),
        ],
        output_streams=output_streams,
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

  def detect(
      self,
      image: image_module.Image,
      image_processing_options: Optional[_ImageProcessingOptions] = None,
  ) -> PoseLandmarkerResult:
    """Performs pose landmarks detection on the given image.

    Only use this method when the PoseLandmarker is created with the image
    running mode.

    Args:
      image: MediaPipe Image.
      image_processing_options: Options for image processing.

    Returns:
      The pose landmarker detection results.

    Raises:
      ValueError: If any of the input arguments is invalid.
      RuntimeError: If pose landmarker detection failed to run.
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

    if output_packets[_NORM_LANDMARKS_STREAM_NAME].is_empty():
      return PoseLandmarkerResult([], [])

    return _build_landmarker_result(output_packets)

  def detect_for_video(
      self,
      image: image_module.Image,
      timestamp_ms: int,
      image_processing_options: Optional[_ImageProcessingOptions] = None,
  ) -> PoseLandmarkerResult:
    """Performs pose landmarks detection on the provided video frame.

    Only use this method when the PoseLandmarker is created with the video
    running mode.

    Only use this method when the PoseLandmarker is created with the video
    running mode. It's required to provide the video frame's timestamp (in
    milliseconds) along with the video frame. The input timestamps should be
    monotonically increasing for adjacent calls of this method.

    Args:
      image: MediaPipe Image.
      timestamp_ms: The timestamp of the input video frame in milliseconds.
      image_processing_options: Options for image processing.

    Returns:
      The pose landmarker detection results.

    Raises:
      ValueError: If any of the input arguments is invalid.
      RuntimeError: If pose landmarker detection failed to run.
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

    if output_packets[_NORM_LANDMARKS_STREAM_NAME].is_empty():
      return PoseLandmarkerResult([], [])

    return _build_landmarker_result(output_packets)

  def detect_async(
      self,
      image: image_module.Image,
      timestamp_ms: int,
      image_processing_options: Optional[_ImageProcessingOptions] = None,
  ) -> None:
    """Sends live image data to perform pose landmarks detection.

    The results will be available via the "result_callback" provided in the
    PoseLandmarkerOptions. Only use this method when the PoseLandmarker is
    created with the live stream running mode.

    Only use this method when the PoseLandmarker is created with the live
    stream running mode. The input timestamps should be monotonically increasing
    for adjacent calls of this method. This method will return immediately after
    the input image is accepted. The results will be available via the
    `result_callback` provided in the `PoseLandmarkerOptions`. The
    `detect_async` method is designed to process live stream data such as
    camera input. To lower the overall latency, pose landmarker may drop the
    input images if needed. In other words, it's not guaranteed to have output
    per input image.

    The `result_callback` provides:
      - The pose landmarker detection results.
      - The input image that the pose landmarker runs on.
      - The input timestamp in milliseconds.

    Args:
      image: MediaPipe Image.
      timestamp_ms: The timestamp of the input image in milliseconds.
      image_processing_options: Options for image processing.

    Raises:
      ValueError: If the current input timestamp is smaller than what the
      pose landmarker has already processed.
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
