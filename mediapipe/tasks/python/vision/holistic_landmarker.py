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
"""MediaPipe holistic landmarker task."""

import dataclasses
from typing import Callable, List, Mapping, Optional

from mediapipe.framework.formats import classification_pb2
from mediapipe.framework.formats import landmark_pb2
from mediapipe.python import packet_creator
from mediapipe.python import packet_getter
from mediapipe.python._framework_bindings import image as image_module
from mediapipe.python._framework_bindings import packet as packet_module
from mediapipe.tasks.cc.vision.holistic_landmarker.proto import holistic_landmarker_graph_options_pb2
from mediapipe.tasks.cc.vision.holistic_landmarker.proto import holistic_result_pb2
from mediapipe.tasks.python.components.containers import category as category_module
from mediapipe.tasks.python.components.containers import landmark as landmark_module
from mediapipe.tasks.python.core import base_options as base_options_module
from mediapipe.tasks.python.core import task_info as task_info_module
from mediapipe.tasks.python.core.optional_dependencies import doc_controls
from mediapipe.tasks.python.vision.core import base_vision_task_api
from mediapipe.tasks.python.vision.core import image_processing_options as image_processing_options_module
from mediapipe.tasks.python.vision.core import vision_task_running_mode as running_mode_module

_BaseOptions = base_options_module.BaseOptions
_HolisticResultProto = holistic_result_pb2.HolisticResult
_HolisticLandmarkerGraphOptionsProto = (
    holistic_landmarker_graph_options_pb2.HolisticLandmarkerGraphOptions
)
_RunningMode = running_mode_module.VisionTaskRunningMode
_ImageProcessingOptions = image_processing_options_module.ImageProcessingOptions
_TaskInfo = task_info_module.TaskInfo

_IMAGE_IN_STREAM_NAME = 'image_in'
_IMAGE_OUT_STREAM_NAME = 'image_out'
_IMAGE_TAG = 'IMAGE'

_POSE_LANDMARKS_STREAM_NAME = 'pose_landmarks'
_POSE_LANDMARKS_TAG_NAME = 'POSE_LANDMARKS'
_POSE_WORLD_LANDMARKS_STREAM_NAME = 'pose_world_landmarks'
_POSE_WORLD_LANDMARKS_TAG = 'POSE_WORLD_LANDMARKS'
_POSE_SEGMENTATION_MASK_STREAM_NAME = 'pose_segmentation_mask'
_POSE_SEGMENTATION_MASK_TAG = 'POSE_SEGMENTATION_MASK'
_FACE_LANDMARKS_STREAM_NAME = 'face_landmarks'
_FACE_LANDMARKS_TAG = 'FACE_LANDMARKS'
_FACE_BLENDSHAPES_STREAM_NAME = 'extra_blendshapes'
_FACE_BLENDSHAPES_TAG = 'FACE_BLENDSHAPES'
_LEFT_HAND_LANDMARKS_STREAM_NAME = 'left_hand_landmarks'
_LEFT_HAND_LANDMARKS_TAG = 'LEFT_HAND_LANDMARKS'
_LEFT_HAND_WORLD_LANDMARKS_STREAM_NAME = 'left_hand_world_landmarks'
_LEFT_HAND_WORLD_LANDMARKS_TAG = 'LEFT_HAND_WORLD_LANDMARKS'
_RIGHT_HAND_LANDMARKS_STREAM_NAME = 'right_hand_landmarks'
_RIGHT_HAND_LANDMARKS_TAG = 'RIGHT_HAND_LANDMARKS'
_RIGHT_HAND_WORLD_LANDMARKS_STREAM_NAME = 'right_hand_world_landmarks'
_RIGHT_HAND_WORLD_LANDMARKS_TAG = 'RIGHT_HAND_WORLD_LANDMARKS'

_TASK_GRAPH_NAME = (
    'mediapipe.tasks.vision.holistic_landmarker.HolisticLandmarkerGraph'
)
_MICRO_SECONDS_PER_MILLISECOND = 1000


@dataclasses.dataclass
class HolisticLandmarkerResult:
  """The holistic landmarks result from HolisticLandmarker, where each vector element represents a single holistic detected in the image.

  Attributes:
    face_landmarks: Detected face landmarks in normalized image coordinates.
    pose_landmarks: Detected pose landmarks in normalized image coordinates.
    pose_world_landmarks: Detected pose world landmarks in image coordinates.
    left_hand_landmarks: Detected left hand landmarks in normalized image
      coordinates.
    left_hand_world_landmarks: Detected left hand landmarks in image
      coordinates.
    right_hand_landmarks: Detected right hand landmarks in normalized image
      coordinates.
    right_hand_world_landmarks: Detected right hand landmarks in image
      coordinates.
    face_blendshapes: Optional face blendshapes.
    segmentation_mask: Optional segmentation mask for pose.
  """

  face_landmarks: List[landmark_module.NormalizedLandmark]
  pose_landmarks: List[landmark_module.NormalizedLandmark]
  pose_world_landmarks: List[landmark_module.Landmark]
  left_hand_landmarks: List[landmark_module.NormalizedLandmark]
  left_hand_world_landmarks: List[landmark_module.Landmark]
  right_hand_landmarks: List[landmark_module.NormalizedLandmark]
  right_hand_world_landmarks: List[landmark_module.Landmark]
  face_blendshapes: Optional[List[category_module.Category]] = None
  segmentation_mask: Optional[image_module.Image] = None

  @classmethod
  @doc_controls.do_not_generate_docs
  def create_from_pb2(
      cls, pb2_obj: _HolisticResultProto
  ) -> 'HolisticLandmarkerResult':
    """Creates a `HolisticLandmarkerResult` object from the given protobuf object."""
    face_blendshapes = None
    if hasattr(pb2_obj, 'face_blendshapes'):
      face_blendshapes = [
          category_module.Category(
              score=classification.score,
              index=classification.index,
              category_name=classification.label,
              display_name=classification.display_name,
          )
          for classification in pb2_obj.face_blendshapes.classification
      ]

    return HolisticLandmarkerResult(
        face_landmarks=[
            landmark_module.NormalizedLandmark.create_from_pb2(landmark)
            for landmark in pb2_obj.face_landmarks.landmark
        ],
        pose_landmarks=[
            landmark_module.NormalizedLandmark.create_from_pb2(landmark)
            for landmark in pb2_obj.pose_landmarks.landmark
        ],
        pose_world_landmarks=[
            landmark_module.Landmark.create_from_pb2(landmark)
            for landmark in pb2_obj.pose_world_landmarks.landmark
        ],
        left_hand_landmarks=[
            landmark_module.NormalizedLandmark.create_from_pb2(landmark)
            for landmark in pb2_obj.left_hand_landmarks.landmark
        ],
        left_hand_world_landmarks=[],
        right_hand_landmarks=[
            landmark_module.NormalizedLandmark.create_from_pb2(landmark)
            for landmark in pb2_obj.right_hand_landmarks.landmark
        ],
        right_hand_world_landmarks=[],
        face_blendshapes=face_blendshapes,
        segmentation_mask=None,
    )


def _build_landmarker_result(
    output_packets: Mapping[str, packet_module.Packet]
) -> HolisticLandmarkerResult:
  """Constructs a `HolisticLandmarksDetectionResult` from output packets."""
  holistic_landmarker_result = HolisticLandmarkerResult(
      [], [], [], [], [], [], []
  )

  face_landmarks_proto_list = packet_getter.get_proto(
      output_packets[_FACE_LANDMARKS_STREAM_NAME]
  )

  pose_landmarks_proto_list = packet_getter.get_proto(
      output_packets[_POSE_LANDMARKS_STREAM_NAME]
  )

  pose_world_landmarks_proto_list = packet_getter.get_proto(
      output_packets[_POSE_WORLD_LANDMARKS_STREAM_NAME]
  )

  left_hand_landmarks_proto_list = packet_getter.get_proto(
      output_packets[_LEFT_HAND_LANDMARKS_STREAM_NAME]
  )

  left_hand_world_landmarks_proto_list = packet_getter.get_proto(
      output_packets[_LEFT_HAND_WORLD_LANDMARKS_STREAM_NAME]
  )

  right_hand_landmarks_proto_list = packet_getter.get_proto(
      output_packets[_RIGHT_HAND_LANDMARKS_STREAM_NAME]
  )

  right_hand_world_landmarks_proto_list = packet_getter.get_proto(
      output_packets[_RIGHT_HAND_WORLD_LANDMARKS_STREAM_NAME]
  )

  face_landmarks = landmark_pb2.NormalizedLandmarkList()
  face_landmarks.MergeFrom(face_landmarks_proto_list)
  for face_landmark in face_landmarks.landmark:
    holistic_landmarker_result.face_landmarks.append(
        landmark_module.NormalizedLandmark.create_from_pb2(face_landmark)
    )

  pose_landmarks = landmark_pb2.NormalizedLandmarkList()
  pose_landmarks.MergeFrom(pose_landmarks_proto_list)
  for pose_landmark in pose_landmarks.landmark:
    holistic_landmarker_result.pose_landmarks.append(
        landmark_module.NormalizedLandmark.create_from_pb2(pose_landmark)
    )

  pose_world_landmarks = landmark_pb2.LandmarkList()
  pose_world_landmarks.MergeFrom(pose_world_landmarks_proto_list)
  for pose_world_landmark in pose_world_landmarks.landmark:
    holistic_landmarker_result.pose_world_landmarks.append(
        landmark_module.Landmark.create_from_pb2(pose_world_landmark)
    )

  left_hand_landmarks = landmark_pb2.NormalizedLandmarkList()
  left_hand_landmarks.MergeFrom(left_hand_landmarks_proto_list)
  for hand_landmark in left_hand_landmarks.landmark:
    holistic_landmarker_result.left_hand_landmarks.append(
        landmark_module.NormalizedLandmark.create_from_pb2(hand_landmark)
    )

  left_hand_world_landmarks = landmark_pb2.LandmarkList()
  left_hand_world_landmarks.MergeFrom(left_hand_world_landmarks_proto_list)
  for left_hand_world_landmark in left_hand_world_landmarks.landmark:
    holistic_landmarker_result.left_hand_world_landmarks.append(
        landmark_module.Landmark.create_from_pb2(left_hand_world_landmark)
    )

  right_hand_landmarks = landmark_pb2.NormalizedLandmarkList()
  right_hand_landmarks.MergeFrom(right_hand_landmarks_proto_list)
  for hand_landmark in right_hand_landmarks.landmark:
    holistic_landmarker_result.right_hand_landmarks.append(
        landmark_module.NormalizedLandmark.create_from_pb2(hand_landmark)
    )

  right_hand_world_landmarks = landmark_pb2.LandmarkList()
  right_hand_world_landmarks.MergeFrom(right_hand_world_landmarks_proto_list)
  for right_hand_world_landmark in right_hand_world_landmarks.landmark:
    holistic_landmarker_result.right_hand_world_landmarks.append(
        landmark_module.Landmark.create_from_pb2(right_hand_world_landmark)
    )

  if _FACE_BLENDSHAPES_STREAM_NAME in output_packets:
    face_blendshapes_proto_list = packet_getter.get_proto(
        output_packets[_FACE_BLENDSHAPES_STREAM_NAME]
    )
    face_blendshapes_classifications = classification_pb2.ClassificationList()
    face_blendshapes_classifications.MergeFrom(face_blendshapes_proto_list)
    holistic_landmarker_result.face_blendshapes = []
    for face_blendshapes in face_blendshapes_classifications.classification:
      holistic_landmarker_result.face_blendshapes.append(
          category_module.Category(
              index=face_blendshapes.index,
              score=face_blendshapes.score,
              display_name=face_blendshapes.display_name,
              category_name=face_blendshapes.label,
          )
      )

  if _POSE_SEGMENTATION_MASK_STREAM_NAME in output_packets:
    holistic_landmarker_result.segmentation_mask = packet_getter.get_image(
        output_packets[_POSE_SEGMENTATION_MASK_STREAM_NAME]
    )

  return holistic_landmarker_result


@dataclasses.dataclass
class HolisticLandmarkerOptions:
  """Options for the holistic landmarker task.

  Attributes:
    base_options: Base options for the holistic landmarker task.
    running_mode: The running mode of the task. Default to the image mode.
      HolisticLandmarker has three running modes: 1) The image mode for
      detecting holistic landmarks on single image inputs. 2) The video mode for
      detecting holistic landmarks on the decoded frames of a video. 3) The live
      stream mode for detecting holistic landmarks on the live stream of input
      data, such as from camera. In this mode, the "result_callback" below must
      be specified to receive the detection results asynchronously.
    min_face_detection_confidence: The minimum confidence score for the face
      detection to be considered successful.
    min_face_suppression_threshold: The minimum non-maximum-suppression
      threshold for face detection to be considered overlapped.
    min_face_landmarks_confidence: The minimum confidence score for the face
      landmark detection to be considered successful.
    min_pose_detection_confidence: The minimum confidence score for the pose
      detection to be considered successful.
    min_pose_suppression_threshold: The minimum non-maximum-suppression
      threshold for pose detection to be considered overlapped.
    min_pose_landmarks_confidence: The minimum confidence score for the pose
      landmark detection to be considered successful.
    min_hand_landmarks_confidence: The minimum confidence score for the hand
      landmark detection to be considered successful.
    output_face_blendshapes: Whether HolisticLandmarker outputs face blendshapes
      classification. Face blendshapes are used for rendering the 3D face model.
    output_segmentation_mask: whether to output segmentation masks.
    result_callback: The user-defined result callback for processing live stream
      data. The result callback should only be specified when the running mode
      is set to the live stream mode.
  """

  base_options: _BaseOptions
  running_mode: _RunningMode = _RunningMode.IMAGE
  min_face_detection_confidence: float = 0.5
  min_face_suppression_threshold: float = 0.5
  min_face_landmarks_confidence: float = 0.5
  min_pose_detection_confidence: float = 0.5
  min_pose_suppression_threshold: float = 0.5
  min_pose_landmarks_confidence: float = 0.5
  min_hand_landmarks_confidence: float = 0.5
  output_face_blendshapes: bool = False
  output_segmentation_mask: bool = False
  result_callback: Optional[
      Callable[[HolisticLandmarkerResult, image_module.Image, int], None]
  ] = None

  @doc_controls.do_not_generate_docs
  def to_pb2(self) -> _HolisticLandmarkerGraphOptionsProto:
    """Generates an HolisticLandmarkerGraphOptions protobuf object."""
    base_options_proto = self.base_options.to_pb2()
    base_options_proto.use_stream_mode = (
        False if self.running_mode == _RunningMode.IMAGE else True
    )

    # Initialize the holistic landmarker options from base options.
    holistic_landmarker_options_proto = _HolisticLandmarkerGraphOptionsProto(
        base_options=base_options_proto
    )
    # Configure face detector and face landmarks detector options.
    holistic_landmarker_options_proto.face_detector_graph_options.min_detection_confidence = (
        self.min_face_detection_confidence
    )
    holistic_landmarker_options_proto.face_detector_graph_options.min_suppression_threshold = (
        self.min_face_suppression_threshold
    )
    holistic_landmarker_options_proto.face_landmarks_detector_graph_options.min_detection_confidence = (
        self.min_face_landmarks_confidence
    )
    # Configure pose detector and pose landmarks detector options.
    holistic_landmarker_options_proto.pose_detector_graph_options.min_detection_confidence = (
        self.min_pose_detection_confidence
    )
    holistic_landmarker_options_proto.pose_detector_graph_options.min_suppression_threshold = (
        self.min_pose_suppression_threshold
    )
    holistic_landmarker_options_proto.pose_landmarks_detector_graph_options.min_detection_confidence = (
        self.min_pose_landmarks_confidence
    )
    # Configure hand landmarks detector options.
    holistic_landmarker_options_proto.hand_landmarks_detector_graph_options.min_detection_confidence = (
        self.min_hand_landmarks_confidence
    )
    return holistic_landmarker_options_proto


class HolisticLandmarker(base_vision_task_api.BaseVisionTaskApi):
  """Class that performs holistic landmarks detection on images."""

  @classmethod
  def create_from_model_path(cls, model_path: str) -> 'HolisticLandmarker':
    """Creates an `HolisticLandmarker` object from a TensorFlow Lite model and the default `HolisticLandmarkerOptions`.

    Note that the created `HolisticLandmarker` instance is in image mode, for
    detecting holistic landmarks on single image inputs.

    Args:
      model_path: Path to the model.

    Returns:
      `HolisticLandmarker` object that's created from the model file and the
      default `HolisticLandmarkerOptions`.

    Raises:
      ValueError: If failed to create `HolisticLandmarker` object from the
        provided file such as invalid file path.
      RuntimeError: If other types of error occurred.
    """
    base_options = _BaseOptions(model_asset_path=model_path)
    options = HolisticLandmarkerOptions(
        base_options=base_options, running_mode=_RunningMode.IMAGE
    )
    return cls.create_from_options(options)

  @classmethod
  def create_from_options(
      cls, options: HolisticLandmarkerOptions
  ) -> 'HolisticLandmarker':
    """Creates the `HolisticLandmarker` object from holistic landmarker options.

    Args:
      options: Options for the holistic landmarker task.

    Returns:
      `HolisticLandmarker` object that's created from `options`.

    Raises:
      ValueError: If failed to create `HolisticLandmarker` object from
        `HolisticLandmarkerOptions` such as missing the model.
      RuntimeError: If other types of error occurred.
    """

    def packets_callback(output_packets: Mapping[str, packet_module.Packet]):
      if output_packets[_IMAGE_OUT_STREAM_NAME].is_empty():
        return

      image = packet_getter.get_image(output_packets[_IMAGE_OUT_STREAM_NAME])

      if output_packets[_FACE_LANDMARKS_STREAM_NAME].is_empty():
        empty_packet = output_packets[_FACE_LANDMARKS_STREAM_NAME]
        options.result_callback(
            HolisticLandmarkerResult([], [], [], [], [], [], []),
            image,
            empty_packet.timestamp.value // _MICRO_SECONDS_PER_MILLISECOND,
        )
        return

      holistic_landmarks_detection_result = _build_landmarker_result(
          output_packets
      )
      timestamp = output_packets[_FACE_LANDMARKS_STREAM_NAME].timestamp
      options.result_callback(
          holistic_landmarks_detection_result,
          image,
          timestamp.value // _MICRO_SECONDS_PER_MILLISECOND,
      )

    output_streams = [
        ':'.join([_FACE_LANDMARKS_TAG, _FACE_LANDMARKS_STREAM_NAME]),
        ':'.join([_POSE_LANDMARKS_TAG_NAME, _POSE_LANDMARKS_STREAM_NAME]),
        ':'.join(
            [_POSE_WORLD_LANDMARKS_TAG, _POSE_WORLD_LANDMARKS_STREAM_NAME]
        ),
        ':'.join([_LEFT_HAND_LANDMARKS_TAG, _LEFT_HAND_LANDMARKS_STREAM_NAME]),
        ':'.join([
            _LEFT_HAND_WORLD_LANDMARKS_TAG,
            _LEFT_HAND_WORLD_LANDMARKS_STREAM_NAME,
        ]),
        ':'.join(
            [_RIGHT_HAND_LANDMARKS_TAG, _RIGHT_HAND_LANDMARKS_STREAM_NAME]
        ),
        ':'.join([
            _RIGHT_HAND_WORLD_LANDMARKS_TAG,
            _RIGHT_HAND_WORLD_LANDMARKS_STREAM_NAME,
        ]),
        ':'.join([_IMAGE_TAG, _IMAGE_OUT_STREAM_NAME]),
    ]

    if options.output_segmentation_mask:
      output_streams.append(
          ':'.join(
              [_POSE_SEGMENTATION_MASK_TAG, _POSE_SEGMENTATION_MASK_STREAM_NAME]
          )
      )

    if options.output_face_blendshapes:
      output_streams.append(
          ':'.join([_FACE_BLENDSHAPES_TAG, _FACE_BLENDSHAPES_STREAM_NAME])
      )

    task_info = _TaskInfo(
        task_graph=_TASK_GRAPH_NAME,
        input_streams=[
            ':'.join([_IMAGE_TAG, _IMAGE_IN_STREAM_NAME]),
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
  ) -> HolisticLandmarkerResult:
    """Performs holistic landmarks detection on the given image.

    Only use this method when the HolisticLandmarker is created with the image
    running mode.

    The image can be of any size with format RGB or RGBA.

    Args:
      image: MediaPipe Image.

    Returns:
      The holistic landmarks detection results.

    Raises:
      ValueError: If any of the input arguments is invalid.
      RuntimeError: If holistic landmarker detection failed to run.
    """
    output_packets = self._process_image_data({
        _IMAGE_IN_STREAM_NAME: packet_creator.create_image(image),
    })

    if output_packets[_FACE_LANDMARKS_STREAM_NAME].is_empty():
      return HolisticLandmarkerResult([], [], [], [], [], [], [])

    return _build_landmarker_result(output_packets)

  def detect_for_video(
      self,
      image: image_module.Image,
      timestamp_ms: int,
  ) -> HolisticLandmarkerResult:
    """Performs holistic landmarks detection on the provided video frame.

    Only use this method when the HolisticLandmarker is created with the video
    running mode.

    Only use this method when the HolisticLandmarker is created with the video
    running mode. It's required to provide the video frame's timestamp (in
    milliseconds) along with the video frame. The input timestamps should be
    monotonically increasing for adjacent calls of this method.

    Args:
      image: MediaPipe Image.
      timestamp_ms: The timestamp of the input video frame in milliseconds.

    Returns:
      The holistic landmarks detection results.

    Raises:
      ValueError: If any of the input arguments is invalid.
      RuntimeError: If holistic landmarker detection failed to run.
    """
    output_packets = self._process_video_data({
        _IMAGE_IN_STREAM_NAME: packet_creator.create_image(image).at(
            timestamp_ms * _MICRO_SECONDS_PER_MILLISECOND
        ),
    })

    if output_packets[_FACE_LANDMARKS_STREAM_NAME].is_empty():
      return HolisticLandmarkerResult([], [], [], [], [], [], [])

    return _build_landmarker_result(output_packets)

  def detect_async(
      self,
      image: image_module.Image,
      timestamp_ms: int,
  ) -> None:
    """Sends live image data to perform holistic landmarks detection.

    The results will be available via the "result_callback" provided in the
    HolisticLandmarkerOptions. Only use this method when the HolisticLandmarker
    is
    created with the live stream running mode.

    Only use this method when the HolisticLandmarker is created with the live
    stream running mode. The input timestamps should be monotonically increasing
    for adjacent calls of this method. This method will return immediately after
    the input image is accepted. The results will be available via the
    `result_callback` provided in the `HolisticLandmarkerOptions`. The
    `detect_async` method is designed to process live stream data such as
    camera input. To lower the overall latency, holistic landmarker may drop the
    input images if needed. In other words, it's not guaranteed to have output
    per input image.

    The `result_callback` provides:
      - The holistic landmarks detection results.
      - The input image that the holistic landmarker runs on.
      - The input timestamp in milliseconds.

    Args:
      image: MediaPipe Image.
      timestamp_ms: The timestamp of the input image in milliseconds.

    Raises:
      ValueError: If the current input timestamp is smaller than what the
      holistic landmarker has already processed.
    """
    self._send_live_stream_data({
        _IMAGE_IN_STREAM_NAME: packet_creator.create_image(image).at(
            timestamp_ms * _MICRO_SECONDS_PER_MILLISECOND
        ),
    })
