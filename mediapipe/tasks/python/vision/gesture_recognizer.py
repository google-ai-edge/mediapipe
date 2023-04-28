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
"""MediaPipe gesture recognizer task."""

import dataclasses
from typing import Callable, Mapping, Optional, List

from mediapipe.framework.formats import classification_pb2
from mediapipe.framework.formats import landmark_pb2
from mediapipe.python import packet_creator
from mediapipe.python import packet_getter
from mediapipe.python._framework_bindings import image as image_module
from mediapipe.python._framework_bindings import packet as packet_module
from mediapipe.tasks.cc.vision.gesture_recognizer.proto import gesture_recognizer_graph_options_pb2
from mediapipe.tasks.python.components.containers import category as category_module
from mediapipe.tasks.python.components.containers import landmark as landmark_module
from mediapipe.tasks.python.components.processors import classifier_options
from mediapipe.tasks.python.core import base_options as base_options_module
from mediapipe.tasks.python.core import task_info as task_info_module
from mediapipe.tasks.python.core.optional_dependencies import doc_controls
from mediapipe.tasks.python.vision.core import base_vision_task_api
from mediapipe.tasks.python.vision.core import image_processing_options as image_processing_options_module
from mediapipe.tasks.python.vision.core import vision_task_running_mode as running_mode_module

_BaseOptions = base_options_module.BaseOptions
_GestureRecognizerGraphOptionsProto = (
    gesture_recognizer_graph_options_pb2.GestureRecognizerGraphOptions
)
_ClassifierOptions = classifier_options.ClassifierOptions
_RunningMode = running_mode_module.VisionTaskRunningMode
_ImageProcessingOptions = image_processing_options_module.ImageProcessingOptions
_TaskInfo = task_info_module.TaskInfo

_IMAGE_IN_STREAM_NAME = 'image_in'
_IMAGE_OUT_STREAM_NAME = 'image_out'
_IMAGE_TAG = 'IMAGE'
_NORM_RECT_STREAM_NAME = 'norm_rect_in'
_NORM_RECT_TAG = 'NORM_RECT'
_HAND_GESTURE_STREAM_NAME = 'hand_gestures'
_HAND_GESTURE_TAG = 'HAND_GESTURES'
_HANDEDNESS_STREAM_NAME = 'handedness'
_HANDEDNESS_TAG = 'HANDEDNESS'
_HAND_LANDMARKS_STREAM_NAME = 'landmarks'
_HAND_LANDMARKS_TAG = 'LANDMARKS'
_HAND_WORLD_LANDMARKS_STREAM_NAME = 'world_landmarks'
_HAND_WORLD_LANDMARKS_TAG = 'WORLD_LANDMARKS'
_TASK_GRAPH_NAME = (
    'mediapipe.tasks.vision.gesture_recognizer.GestureRecognizerGraph'
)
_MICRO_SECONDS_PER_MILLISECOND = 1000
_GESTURE_DEFAULT_INDEX = -1


@dataclasses.dataclass
class GestureRecognizerResult:
  """The gesture recognition result from GestureRecognizer, where each vector element represents a single hand detected in the image.

  Attributes:
    gestures: Recognized hand gestures of detected hands. Note that the index of
      the gesture is always -1, because the raw indices from multiple gesture
      classifiers cannot consolidate to a meaningful index.
    handedness: Classification of handedness.
    hand_landmarks: Detected hand landmarks in normalized image coordinates.
    hand_world_landmarks: Detected hand landmarks in world coordinates.
  """

  gestures: List[List[category_module.Category]]
  handedness: List[List[category_module.Category]]
  hand_landmarks: List[List[landmark_module.NormalizedLandmark]]
  hand_world_landmarks: List[List[landmark_module.Landmark]]


def _build_recognition_result(
    output_packets: Mapping[str, packet_module.Packet]
) -> GestureRecognizerResult:
  """Constructs a `GestureRecognizerResult` from output packets."""
  gestures_proto_list = packet_getter.get_proto_list(
      output_packets[_HAND_GESTURE_STREAM_NAME]
  )
  handedness_proto_list = packet_getter.get_proto_list(
      output_packets[_HANDEDNESS_STREAM_NAME]
  )
  hand_landmarks_proto_list = packet_getter.get_proto_list(
      output_packets[_HAND_LANDMARKS_STREAM_NAME]
  )
  hand_world_landmarks_proto_list = packet_getter.get_proto_list(
      output_packets[_HAND_WORLD_LANDMARKS_STREAM_NAME]
  )

  gesture_results = []
  for proto in gestures_proto_list:
    gesture_categories = []
    gesture_classifications = classification_pb2.ClassificationList()
    gesture_classifications.MergeFrom(proto)
    for gesture in gesture_classifications.classification:
      gesture_categories.append(
          category_module.Category(
              index=_GESTURE_DEFAULT_INDEX,
              score=gesture.score,
              display_name=gesture.display_name,
              category_name=gesture.label,
          )
      )
    gesture_results.append(gesture_categories)

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

  return GestureRecognizerResult(
      gesture_results,
      handedness_results,
      hand_landmarks_results,
      hand_world_landmarks_results,
  )


@dataclasses.dataclass
class GestureRecognizerOptions:
  """Options for the gesture recognizer task.

  Attributes:
    base_options: Base options for the hand gesture recognizer task.
    running_mode: The running mode of the task. Default to the image mode.
      Gesture recognizer task has three running modes: 1) The image mode for
      recognizing hand gestures on single image inputs. 2) The video mode for
      recognizing hand gestures on the decoded frames of a video. 3) The live
      stream mode for recognizing hand gestures on a live stream of input data,
      such as from camera.
    num_hands: The maximum number of hands can be detected by the recognizer.
    min_hand_detection_confidence: The minimum confidence score for the hand
      detection to be considered successful.
    min_hand_presence_confidence: The minimum confidence score of hand presence
      score in the hand landmark detection.
    min_tracking_confidence: The minimum confidence score for the hand tracking
      to be considered successful.
    canned_gesture_classifier_options: Options for configuring the canned
      gestures classifier, such as score threshold, allow list and deny list of
      gestures. The categories for canned gesture classifiers are: ["None",
      "Closed_Fist", "Open_Palm", "Pointing_Up", "Thumb_Down", "Thumb_Up",
      "Victory", "ILoveYou"]. Note this option is subject to change.
    custom_gesture_classifier_options: Options for configuring the custom
      gestures classifier, such as score threshold, allow list and deny list of
      gestures. Note this option is subject to change.
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
  canned_gesture_classifier_options: _ClassifierOptions = dataclasses.field(
      default_factory=_ClassifierOptions
  )
  custom_gesture_classifier_options: _ClassifierOptions = dataclasses.field(
      default_factory=_ClassifierOptions
  )
  result_callback: Optional[
      Callable[[GestureRecognizerResult, image_module.Image, int], None]
  ] = None

  @doc_controls.do_not_generate_docs
  def to_pb2(self) -> _GestureRecognizerGraphOptionsProto:
    """Generates an GestureRecognizerOptions protobuf object."""
    base_options_proto = self.base_options.to_pb2()
    base_options_proto.use_stream_mode = (
        False if self.running_mode == _RunningMode.IMAGE else True
    )

    # Initialize gesture recognizer options from base options.
    gesture_recognizer_options_proto = _GestureRecognizerGraphOptionsProto(
        base_options=base_options_proto
    )
    # Configure hand detector and hand landmarker options.
    hand_landmarker_options_proto = (
        gesture_recognizer_options_proto.hand_landmarker_graph_options
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

    # Configure hand gesture recognizer options.
    hand_gesture_recognizer_options_proto = (
        gesture_recognizer_options_proto.hand_gesture_recognizer_graph_options
    )
    hand_gesture_recognizer_options_proto.canned_gesture_classifier_graph_options.classifier_options.CopyFrom(
        self.canned_gesture_classifier_options.to_pb2()
    )
    hand_gesture_recognizer_options_proto.custom_gesture_classifier_graph_options.classifier_options.CopyFrom(
        self.custom_gesture_classifier_options.to_pb2()
    )

    return gesture_recognizer_options_proto


class GestureRecognizer(base_vision_task_api.BaseVisionTaskApi):
  """Class that performs gesture recognition on images."""

  @classmethod
  def create_from_model_path(cls, model_path: str) -> 'GestureRecognizer':
    """Creates an `GestureRecognizer` object from a TensorFlow Lite model and the default `GestureRecognizerOptions`.

    Note that the created `GestureRecognizer` instance is in image mode, for
    recognizing hand gestures on single image inputs.

    Args:
      model_path: Path to the model.

    Returns:
      `GestureRecognizer` object that's created from the model file and the
      default `GestureRecognizerOptions`.

    Raises:
      ValueError: If failed to create `GestureRecognizer` object from the
        provided file such as invalid file path.
      RuntimeError: If other types of error occurred.
    """
    base_options = _BaseOptions(model_asset_path=model_path)
    options = GestureRecognizerOptions(
        base_options=base_options, running_mode=_RunningMode.IMAGE
    )
    return cls.create_from_options(options)

  @classmethod
  def create_from_options(
      cls, options: GestureRecognizerOptions
  ) -> 'GestureRecognizer':
    """Creates the `GestureRecognizer` object from gesture recognizer options.

    Args:
      options: Options for the gesture recognizer task.

    Returns:
      `GestureRecognizer` object that's created from `options`.

    Raises:
      ValueError: If failed to create `GestureRecognizer` object from
        `GestureRecognizerOptions` such as missing the model.
      RuntimeError: If other types of error occurred.
    """

    def packets_callback(output_packets: Mapping[str, packet_module.Packet]):
      if output_packets[_IMAGE_OUT_STREAM_NAME].is_empty():
        return

      image = packet_getter.get_image(output_packets[_IMAGE_OUT_STREAM_NAME])

      if output_packets[_HAND_GESTURE_STREAM_NAME].is_empty():
        empty_packet = output_packets[_HAND_GESTURE_STREAM_NAME]
        options.result_callback(
            GestureRecognizerResult([], [], [], []),
            image,
            empty_packet.timestamp.value // _MICRO_SECONDS_PER_MILLISECOND,
        )
        return

      gesture_recognizer_result = _build_recognition_result(output_packets)
      timestamp = output_packets[_HAND_GESTURE_STREAM_NAME].timestamp
      options.result_callback(
          gesture_recognizer_result,
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
            ':'.join([_HAND_GESTURE_TAG, _HAND_GESTURE_STREAM_NAME]),
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

  def recognize(
      self,
      image: image_module.Image,
      image_processing_options: Optional[_ImageProcessingOptions] = None,
  ) -> GestureRecognizerResult:
    """Performs hand gesture recognition on the given image.

    Only use this method when the GestureRecognizer is created with the image
    running mode.

    The image can be of any size with format RGB or RGBA.
    TODO: Describes how the input image will be preprocessed after the yuv
    support is implemented.

    Args:
      image: MediaPipe Image.
      image_processing_options: Options for image processing.

    Returns:
      The hand gesture recognition results.

    Raises:
      ValueError: If any of the input arguments is invalid.
      RuntimeError: If gesture recognition failed to run.
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

    if output_packets[_HAND_GESTURE_STREAM_NAME].is_empty():
      return GestureRecognizerResult([], [], [], [])

    return _build_recognition_result(output_packets)

  def recognize_for_video(
      self,
      image: image_module.Image,
      timestamp_ms: int,
      image_processing_options: Optional[_ImageProcessingOptions] = None,
  ) -> GestureRecognizerResult:
    """Performs gesture recognition on the provided video frame.

    Only use this method when the GestureRecognizer is created with the video
    running mode.

    Only use this method when the GestureRecognizer is created with the video
    running mode. It's required to provide the video frame's timestamp (in
    milliseconds) along with the video frame. The input timestamps should be
    monotonically increasing for adjacent calls of this method.

    Args:
      image: MediaPipe Image.
      timestamp_ms: The timestamp of the input video frame in milliseconds.
      image_processing_options: Options for image processing.

    Returns:
      The hand gesture recognition results.

    Raises:
      ValueError: If any of the input arguments is invalid.
      RuntimeError: If gesture recognition failed to run.
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

    if output_packets[_HAND_GESTURE_STREAM_NAME].is_empty():
      return GestureRecognizerResult([], [], [], [])

    return _build_recognition_result(output_packets)

  def recognize_async(
      self,
      image: image_module.Image,
      timestamp_ms: int,
      image_processing_options: Optional[_ImageProcessingOptions] = None,
  ) -> None:
    """Sends live image data to perform gesture recognition.

    The results will be available via the "result_callback" provided in the
    GestureRecognizerOptions. Only use this method when the GestureRecognizer
    is created with the live stream running mode.

    Only use this method when the GestureRecognizer is created with the live
    stream running mode. The input timestamps should be monotonically increasing
    for adjacent calls of this method. This method will return immediately after
    the input image is accepted. The results will be available via the
    `result_callback` provided in the `GestureRecognizerOptions`. The
    `recognize_async` method is designed to process live stream data such as
    camera input. To lower the overall latency, gesture recognizer may drop the
    input images if needed. In other words, it's not guaranteed to have output
    per input image.

    The `result_callback` provides:
      - The hand gesture recognition results.
      - The input image that the gesture recognizer runs on.
      - The input timestamp in milliseconds.

    Args:
      image: MediaPipe Image.
      timestamp_ms: The timestamp of the input image in milliseconds.
      image_processing_options: Options for image processing.

    Raises:
      ValueError: If the current input timestamp is smaller than what the
      gesture recognizer has already processed.
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
