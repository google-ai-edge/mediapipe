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
"""MediaPipe hand landmarker task."""

import dataclasses
import enum
from typing import Callable, Mapping, Optional, List

from mediapipe.framework.formats import classification_pb2
from mediapipe.framework.formats import landmark_pb2
from mediapipe.python import packet_creator
from mediapipe.python import packet_getter
from mediapipe.python._framework_bindings import image as image_module
from mediapipe.python._framework_bindings import packet as packet_module
from mediapipe.tasks.cc.vision.hand_landmarker.proto import hand_landmarker_graph_options_pb2
from mediapipe.tasks.python.components.containers import category as category_module
from mediapipe.tasks.python.components.containers import landmark as landmark_module
from mediapipe.tasks.python.core import base_options as base_options_module
from mediapipe.tasks.python.core import task_info as task_info_module
from mediapipe.tasks.python.core.optional_dependencies import doc_controls
from mediapipe.tasks.python.vision.core import base_vision_task_api
from mediapipe.tasks.python.vision.core import image_processing_options as image_processing_options_module
from mediapipe.tasks.python.vision.core import vision_task_running_mode as running_mode_module

_BaseOptions = base_options_module.BaseOptions
_HandLandmarkerGraphOptionsProto = (
    hand_landmarker_graph_options_pb2.HandLandmarkerGraphOptions
)
_RunningMode = running_mode_module.VisionTaskRunningMode
_ImageProcessingOptions = image_processing_options_module.ImageProcessingOptions
_TaskInfo = task_info_module.TaskInfo

_IMAGE_IN_STREAM_NAME = 'image_in'
_IMAGE_OUT_STREAM_NAME = 'image_out'
_IMAGE_TAG = 'IMAGE'
_NORM_RECT_STREAM_NAME = 'norm_rect_in'
_NORM_RECT_TAG = 'NORM_RECT'
_HANDEDNESS_STREAM_NAME = 'handedness'
_HANDEDNESS_TAG = 'HANDEDNESS'
_HAND_LANDMARKS_STREAM_NAME = 'landmarks'
_HAND_LANDMARKS_TAG = 'LANDMARKS'
_HAND_WORLD_LANDMARKS_STREAM_NAME = 'world_landmarks'
_HAND_WORLD_LANDMARKS_TAG = 'WORLD_LANDMARKS'
_TASK_GRAPH_NAME = 'mediapipe.tasks.vision.hand_landmarker.HandLandmarkerGraph'
_MICRO_SECONDS_PER_MILLISECOND = 1000


class HandLandmark(enum.IntEnum):
  """The 21 hand landmarks."""

  WRIST = 0
  THUMB_CMC = 1
  THUMB_MCP = 2
  THUMB_IP = 3
  THUMB_TIP = 4
  INDEX_FINGER_MCP = 5
  INDEX_FINGER_PIP = 6
  INDEX_FINGER_DIP = 7
  INDEX_FINGER_TIP = 8
  MIDDLE_FINGER_MCP = 9
  MIDDLE_FINGER_PIP = 10
  MIDDLE_FINGER_DIP = 11
  MIDDLE_FINGER_TIP = 12
  RING_FINGER_MCP = 13
  RING_FINGER_PIP = 14
  RING_FINGER_DIP = 15
  RING_FINGER_TIP = 16
  PINKY_MCP = 17
  PINKY_PIP = 18
  PINKY_DIP = 19
  PINKY_TIP = 20


class HandLandmarksConnections:
  """The connections between hand landmarks."""

  @dataclasses.dataclass
  class Connection:
    """The connection class for hand landmarks."""

    start: int
    end: int

  HAND_PALM_CONNECTIONS: List[Connection] = [
      Connection(0, 1),
      Connection(1, 5),
      Connection(9, 13),
      Connection(13, 17),
      Connection(5, 9),
      Connection(0, 17),
  ]

  HAND_THUMB_CONNECTIONS: List[Connection] = [
      Connection(1, 2),
      Connection(2, 3),
      Connection(3, 4),
  ]

  HAND_INDEX_FINGER_CONNECTIONS: List[Connection] = [
      Connection(5, 6),
      Connection(6, 7),
      Connection(7, 8),
  ]

  HAND_MIDDLE_FINGER_CONNECTIONS: List[Connection] = [
      Connection(9, 10),
      Connection(10, 11),
      Connection(11, 12),
  ]

  HAND_RING_FINGER_CONNECTIONS: List[Connection] = [
      Connection(13, 14),
      Connection(14, 15),
      Connection(15, 16),
  ]

  HAND_PINKY_FINGER_CONNECTIONS: List[Connection] = [
      Connection(17, 18),
      Connection(18, 19),
      Connection(19, 20),
  ]

  HAND_CONNECTIONS: List[Connection] = (
      HAND_PALM_CONNECTIONS +
      HAND_THUMB_CONNECTIONS +
      HAND_INDEX_FINGER_CONNECTIONS +
      HAND_MIDDLE_FINGER_CONNECTIONS +
      HAND_RING_FINGER_CONNECTIONS +
      HAND_PINKY_FINGER_CONNECTIONS
  )


@dataclasses.dataclass
class HandLandmarkerResult:
  """The hand landmarks result from HandLandmarker, where each vector element represents a single hand detected in the image.

  Attributes:
    handedness: Classification of handedness.
    hand_landmarks: Detected hand landmarks in normalized image coordinates.
    hand_world_landmarks: Detected hand landmarks in world coordinates.
  """

  handedness: List[List[category_module.Category]]
  hand_landmarks: List[List[landmark_module.NormalizedLandmark]]
  hand_world_landmarks: List[List[landmark_module.Landmark]]


def _build_landmarker_result(
    output_packets: Mapping[str, packet_module.Packet]
) -> HandLandmarkerResult:
  """Constructs a `HandLandmarksDetectionResult` from output packets."""
  handedness_proto_list = packet_getter.get_proto_list(
      output_packets[_HANDEDNESS_STREAM_NAME]
  )
  hand_landmarks_proto_list = packet_getter.get_proto_list(
      output_packets[_HAND_LANDMARKS_STREAM_NAME]
  )
  hand_world_landmarks_proto_list = packet_getter.get_proto_list(
      output_packets[_HAND_WORLD_LANDMARKS_STREAM_NAME]
  )

  handedness_results = []
  for proto in handedness_proto_list:
    handedness_categories = []
    handedness_classifications = classification_pb2.ClassificationList()
    handedness_classifications.MergeFrom(proto)
    for handedness in handedness_classifications.classification:
      handedness_categories.append(
          category_module.Category(
              index=handedness.index,
              score=handedness.score,
              display_name=handedness.display_name,
              category_name=handedness.label,
          )
      )
    handedness_results.append(handedness_categories)

  hand_landmarks_results = []
  for proto in hand_landmarks_proto_list:
    hand_landmarks = landmark_pb2.NormalizedLandmarkList()
    hand_landmarks.MergeFrom(proto)
    hand_landmarks_list = []
    for hand_landmark in hand_landmarks.landmark:
      hand_landmarks_list.append(
          landmark_module.NormalizedLandmark.create_from_pb2(hand_landmark)
      )
    hand_landmarks_results.append(hand_landmarks_list)

  hand_world_landmarks_results = []
  for proto in hand_world_landmarks_proto_list:
    hand_world_landmarks = landmark_pb2.LandmarkList()
    hand_world_landmarks.MergeFrom(proto)
    hand_world_landmarks_list = []
    for hand_world_landmark in hand_world_landmarks.landmark:
      hand_world_landmarks_list.append(
          landmark_module.Landmark.create_from_pb2(hand_world_landmark)
      )
    hand_world_landmarks_results.append(hand_world_landmarks_list)

  return HandLandmarkerResult(
      handedness_results, hand_landmarks_results, hand_world_landmarks_results
  )


@dataclasses.dataclass
class HandLandmarkerOptions:
  """Options for the hand landmarker task.

  Attributes:
    base_options: Base options for the hand landmarker task.
    running_mode: The running mode of the task. Default to the image mode.
      HandLandmarker has three running modes: 1) The image mode for detecting
      hand landmarks on single image inputs. 2) The video mode for detecting
      hand landmarks on the decoded frames of a video. 3) The live stream mode
      for detecting hand landmarks on the live stream of input data, such as
      from camera. In this mode, the "result_callback" below must be specified
      to receive the detection results asynchronously.
    num_hands: The maximum number of hands can be detected by the hand
      landmarker.
    min_hand_detection_confidence: The minimum confidence score for the hand
      detection to be considered successful.
    min_hand_presence_confidence: The minimum confidence score of hand presence
      score in the hand landmark detection.
    min_tracking_confidence: The minimum confidence score for the hand tracking
      to be considered successful.
    result_callback: The user-defined result callback for processing live stream
      data. The result callback should only be specified when the running mode
      is set to the live stream mode.
  """

  base_options: _BaseOptions
  running_mode: _RunningMode = _RunningMode.IMAGE
  num_hands: int = 1
  min_hand_detection_confidence: float = 0.5
  min_hand_presence_confidence: float = 0.5
  min_tracking_confidence: float = 0.5
  result_callback: Optional[
      Callable[[HandLandmarkerResult, image_module.Image, int], None]
  ] = None

  @doc_controls.do_not_generate_docs
  def to_pb2(self) -> _HandLandmarkerGraphOptionsProto:
    """Generates an HandLandmarkerGraphOptions protobuf object."""
    base_options_proto = self.base_options.to_pb2()
    base_options_proto.use_stream_mode = (
        False if self.running_mode == _RunningMode.IMAGE else True
    )

    # Initialize the hand landmarker options from base options.
    hand_landmarker_options_proto = _HandLandmarkerGraphOptionsProto(
        base_options=base_options_proto
    )
    hand_landmarker_options_proto.min_tracking_confidence = (
        self.min_tracking_confidence
    )
    hand_landmarker_options_proto.hand_detector_graph_options.num_hands = (
        self.num_hands
    )
    hand_landmarker_options_proto.hand_detector_graph_options.min_detection_confidence = (
        self.min_hand_detection_confidence
    )
    hand_landmarker_options_proto.hand_landmarks_detector_graph_options.min_detection_confidence = (
        self.min_hand_presence_confidence
    )
    return hand_landmarker_options_proto


class HandLandmarker(base_vision_task_api.BaseVisionTaskApi):
  """Class that performs hand landmarks detection on images."""

  @classmethod
  def create_from_model_path(cls, model_path: str) -> 'HandLandmarker':
    """Creates an `HandLandmarker` object from a TensorFlow Lite model and the default `HandLandmarkerOptions`.

    Note that the created `HandLandmarker` instance is in image mode, for
    detecting hand landmarks on single image inputs.

    Args:
      model_path: Path to the model.

    Returns:
      `HandLandmarker` object that's created from the model file and the
      default `HandLandmarkerOptions`.

    Raises:
      ValueError: If failed to create `HandLandmarker` object from the
        provided file such as invalid file path.
      RuntimeError: If other types of error occurred.
    """
    base_options = _BaseOptions(model_asset_path=model_path)
    options = HandLandmarkerOptions(
        base_options=base_options, running_mode=_RunningMode.IMAGE
    )
    return cls.create_from_options(options)

  @classmethod
  def create_from_options(
      cls, options: HandLandmarkerOptions
  ) -> 'HandLandmarker':
    """Creates the `HandLandmarker` object from hand landmarker options.

    Args:
      options: Options for the hand landmarker task.

    Returns:
      `HandLandmarker` object that's created from `options`.

    Raises:
      ValueError: If failed to create `HandLandmarker` object from
        `HandLandmarkerOptions` such as missing the model.
      RuntimeError: If other types of error occurred.
    """

    def packets_callback(output_packets: Mapping[str, packet_module.Packet]):
      if output_packets[_IMAGE_OUT_STREAM_NAME].is_empty():
        return

      image = packet_getter.get_image(output_packets[_IMAGE_OUT_STREAM_NAME])

      if output_packets[_HAND_LANDMARKS_STREAM_NAME].is_empty():
        empty_packet = output_packets[_HAND_LANDMARKS_STREAM_NAME]
        options.result_callback(
            HandLandmarkerResult([], [], []),
            image,
            empty_packet.timestamp.value // _MICRO_SECONDS_PER_MILLISECOND,
        )
        return

      hand_landmarks_detection_result = _build_landmarker_result(output_packets)
      timestamp = output_packets[_HAND_LANDMARKS_STREAM_NAME].timestamp
      options.result_callback(
          hand_landmarks_detection_result,
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
            ':'.join([_HANDEDNESS_TAG, _HANDEDNESS_STREAM_NAME]),
            ':'.join([_HAND_LANDMARKS_TAG, _HAND_LANDMARKS_STREAM_NAME]),
            ':'.join(
                [_HAND_WORLD_LANDMARKS_TAG, _HAND_WORLD_LANDMARKS_STREAM_NAME]
            ),
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

  def detect(
      self,
      image: image_module.Image,
      image_processing_options: Optional[_ImageProcessingOptions] = None,
  ) -> HandLandmarkerResult:
    """Performs hand landmarks detection on the given image.

    Only use this method when the HandLandmarker is created with the image
    running mode.

    The image can be of any size with format RGB or RGBA.
    TODO: Describes how the input image will be preprocessed after the yuv
    support is implemented.

    Args:
      image: MediaPipe Image.
      image_processing_options: Options for image processing.

    Returns:
      The hand landmarks detection results.

    Raises:
      ValueError: If any of the input arguments is invalid.
      RuntimeError: If hand landmarker detection failed to run.
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

    if output_packets[_HAND_LANDMARKS_STREAM_NAME].is_empty():
      return HandLandmarkerResult([], [], [])

    return _build_landmarker_result(output_packets)

  def detect_for_video(
      self,
      image: image_module.Image,
      timestamp_ms: int,
      image_processing_options: Optional[_ImageProcessingOptions] = None,
  ) -> HandLandmarkerResult:
    """Performs hand landmarks detection on the provided video frame.

    Only use this method when the HandLandmarker is created with the video
    running mode.

    Only use this method when the HandLandmarker is created with the video
    running mode. It's required to provide the video frame's timestamp (in
    milliseconds) along with the video frame. The input timestamps should be
    monotonically increasing for adjacent calls of this method.

    Args:
      image: MediaPipe Image.
      timestamp_ms: The timestamp of the input video frame in milliseconds.
      image_processing_options: Options for image processing.

    Returns:
      The hand landmarks detection results.

    Raises:
      ValueError: If any of the input arguments is invalid.
      RuntimeError: If hand landmarker detection failed to run.
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

    if output_packets[_HAND_LANDMARKS_STREAM_NAME].is_empty():
      return HandLandmarkerResult([], [], [])

    return _build_landmarker_result(output_packets)

  def detect_async(
      self,
      image: image_module.Image,
      timestamp_ms: int,
      image_processing_options: Optional[_ImageProcessingOptions] = None,
  ) -> None:
    """Sends live image data to perform hand landmarks detection.

    The results will be available via the "result_callback" provided in the
    HandLandmarkerOptions. Only use this method when the HandLandmarker is
    created with the live stream running mode.

    Only use this method when the HandLandmarker is created with the live
    stream running mode. The input timestamps should be monotonically increasing
    for adjacent calls of this method. This method will return immediately after
    the input image is accepted. The results will be available via the
    `result_callback` provided in the `HandLandmarkerOptions`. The
    `detect_async` method is designed to process live stream data such as
    camera input. To lower the overall latency, hand landmarker may drop the
    input images if needed. In other words, it's not guaranteed to have output
    per input image.

    The `result_callback` provides:
      - The hand landmarks detection results.
      - The input image that the hand landmarker runs on.
      - The input timestamp in milliseconds.

    Args:
      image: MediaPipe Image.
      timestamp_ms: The timestamp of the input image in milliseconds.
      image_processing_options: Options for image processing.

    Raises:
      ValueError: If the current input timestamp is smaller than what the
      hand landmarker has already processed.
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
