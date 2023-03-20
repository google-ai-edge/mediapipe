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
"""MediaPipe face landmarker task."""

import dataclasses
import enum
from typing import Callable, Mapping, Optional, List

import numpy as np

from mediapipe.framework.formats import classification_pb2
from mediapipe.framework.formats import landmark_pb2
from mediapipe.framework.formats import matrix_data_pb2
from mediapipe.python import packet_creator
from mediapipe.python import packet_getter
from mediapipe.python._framework_bindings import image as image_module
from mediapipe.python._framework_bindings import packet as packet_module
# pylint: disable=unused-import
from mediapipe.tasks.cc.vision.face_geometry.proto import face_geometry_pb2
# pylint: enable=unused-import
from mediapipe.tasks.cc.vision.face_landmarker.proto import face_landmarker_graph_options_pb2
from mediapipe.tasks.python.components.containers import category as category_module
from mediapipe.tasks.python.components.containers import landmark as landmark_module
from mediapipe.tasks.python.core import base_options as base_options_module
from mediapipe.tasks.python.core import task_info as task_info_module
from mediapipe.tasks.python.core.optional_dependencies import doc_controls
from mediapipe.tasks.python.vision.core import base_vision_task_api
from mediapipe.tasks.python.vision.core import image_processing_options as image_processing_options_module
from mediapipe.tasks.python.vision.core import vision_task_running_mode as running_mode_module

_BaseOptions = base_options_module.BaseOptions
_FaceLandmarkerGraphOptionsProto = (
    face_landmarker_graph_options_pb2.FaceLandmarkerGraphOptions
)
_LayoutEnum = matrix_data_pb2.MatrixData.Layout
_RunningMode = running_mode_module.VisionTaskRunningMode
_ImageProcessingOptions = image_processing_options_module.ImageProcessingOptions
_TaskInfo = task_info_module.TaskInfo

_IMAGE_IN_STREAM_NAME = 'image_in'
_IMAGE_OUT_STREAM_NAME = 'image_out'
_IMAGE_TAG = 'IMAGE'
_NORM_RECT_STREAM_NAME = 'norm_rect_in'
_NORM_RECT_TAG = 'NORM_RECT'
_NORM_LANDMARKS_STREAM_NAME = 'norm_landmarks'
_NORM_LANDMARKS_TAG = 'NORM_LANDMARKS'
_BLENDSHAPES_STREAM_NAME = 'blendshapes'
_BLENDSHAPES_TAG = 'BLENDSHAPES'
_FACE_GEOMETRY_STREAM_NAME = 'face_geometry'
_FACE_GEOMETRY_TAG = 'FACE_GEOMETRY'
_TASK_GRAPH_NAME = 'mediapipe.tasks.vision.face_landmarker.FaceLandmarkerGraph'
_MICRO_SECONDS_PER_MILLISECOND = 1000


class Blendshapes(enum.IntEnum):
  """The 52 blendshape coefficients."""

  NEUTRAL = 0
  BROW_DOWN_LEFT = 1
  BROW_DOWN_RIGHT = 2
  BROW_INNER_UP = 3
  BROW_OUTER_UP_LEFT = 4
  BROW_OUTER_UP_RIGHT = 5
  CHEEK_PUFF = 6
  CHEEK_SQUINT_LEFT = 7
  CHEEK_SQUINT_RIGHT = 8
  EYE_BLINK_LEFT = 9
  EYE_BLINK_RIGHT = 10
  EYE_LOOK_DOWN_LEFT = 11
  EYE_LOOK_DOWN_RIGHT = 12
  EYE_LOOK_IN_LEFT = 13
  EYE_LOOK_IN_RIGHT = 14
  EYE_LOOK_OUT_LEFT = 15
  EYE_LOOK_OUT_RIGHT = 16
  EYE_LOOK_UP_LEFT = 17
  EYE_LOOK_UP_RIGHT = 18
  EYE_SQUINT_LEFT = 19
  EYE_SQUINT_RIGHT = 20
  EYE_WIDE_LEFT = 21
  EYE_WIDE_RIGHT = 22
  JAW_FORWARD = 23
  JAW_LEFT = 24
  JAW_OPEN = 25
  JAW_RIGHT = 26
  MOUTH_CLOSE = 27
  MOUTH_DIMPLE_LEFT = 28
  MOUTH_DIMPLE_RIGHT = 29
  MOUTH_FROWN_LEFT = 30
  MOUTH_FROWN_RIGHT = 31
  MOUTH_FUNNEL = 32
  MOUTH_LEFT = 33
  MOUTH_LOWER_DOWN_LEFT = 34
  MOUTH_LOWER_DOWN_RIGHT = 35
  MOUTH_PRESS_LEFT = 36
  MOUTH_PRESS_RIGHT = 37
  MOUTH_PUCKER = 38
  MOUTH_RIGHT = 39
  MOUTH_ROLL_LOWER = 40
  MOUTH_ROLL_UPPER = 41
  MOUTH_SHRUG_LOWER = 42
  MOUTH_SHRUG_UPPER = 43
  MOUTH_SMILE_LEFT = 44
  MOUTH_SMILE_RIGHT = 45
  MOUTH_STRETCH_LEFT = 46
  MOUTH_STRETCH_RIGHT = 47
  MOUTH_UPPER_UP_LEFT = 48
  MOUTH_UPPER_UP_RIGHT = 49
  NOSE_SNEER_LEFT = 50
  NOSE_SNEER_RIGHT = 51


@dataclasses.dataclass
class FaceLandmarkerResult:
  """The face landmarks detection result from FaceLandmarker, where each vector element represents a single face detected in the image.

  Attributes:
    face_landmarks: Detected face landmarks in normalized image coordinates.
    face_blendshapes: Optional face blendshapes results.
    facial_transformation_matrixes: Optional facial transformation matrix.
  """

  face_landmarks: List[List[landmark_module.NormalizedLandmark]]
  face_blendshapes: List[List[category_module.Category]]
  facial_transformation_matrixes: List[np.ndarray]


def _build_landmarker_result(
    output_packets: Mapping[str, packet_module.Packet]
) -> FaceLandmarkerResult:
  """Constructs a `FaceLandmarkerResult` from output packets."""
  face_landmarks_proto_list = packet_getter.get_proto_list(
      output_packets[_NORM_LANDMARKS_STREAM_NAME]
  )

  face_landmarks_results = []
  for proto in face_landmarks_proto_list:
    face_landmarks = landmark_pb2.NormalizedLandmarkList()
    face_landmarks.MergeFrom(proto)
    face_landmarks_list = []
    for face_landmark in face_landmarks.landmark:
      face_landmarks_list.append(
          landmark_module.NormalizedLandmark.create_from_pb2(face_landmark)
      )
    face_landmarks_results.append(face_landmarks_list)

  face_blendshapes_results = []
  if _BLENDSHAPES_STREAM_NAME in output_packets:
    face_blendshapes_proto_list = packet_getter.get_proto_list(
        output_packets[_BLENDSHAPES_STREAM_NAME]
    )
    for proto in face_blendshapes_proto_list:
      face_blendshapes_categories = []
      face_blendshapes_classifications = classification_pb2.ClassificationList()
      face_blendshapes_classifications.MergeFrom(proto)
      for face_blendshapes in face_blendshapes_classifications.classification:
        face_blendshapes_categories.append(
            category_module.Category(
                index=face_blendshapes.index,
                score=face_blendshapes.score,
                display_name=face_blendshapes.display_name,
                category_name=face_blendshapes.label,
            )
        )
      face_blendshapes_results.append(face_blendshapes_categories)

  facial_transformation_matrixes_results = []
  if _FACE_GEOMETRY_STREAM_NAME in output_packets:
    facial_transformation_matrixes_proto_list = packet_getter.get_proto_list(
        output_packets[_FACE_GEOMETRY_STREAM_NAME]
    )
    for proto in facial_transformation_matrixes_proto_list:
      if hasattr(proto, 'pose_transform_matrix'):
        matrix_data = matrix_data_pb2.MatrixData()
        matrix_data.MergeFrom(proto.pose_transform_matrix)
        matrix = np.array(matrix_data.packed_data)
        matrix = matrix.reshape((matrix_data.rows, matrix_data.cols))
        matrix = (
            matrix if matrix_data.layout == _LayoutEnum.ROW_MAJOR else matrix.T
        )
        facial_transformation_matrixes_results.append(matrix)

  return FaceLandmarkerResult(
      face_landmarks_results,
      face_blendshapes_results,
      facial_transformation_matrixes_results,
  )


@dataclasses.dataclass
class FaceLandmarkerOptions:
  """Options for the face landmarker task.

  Attributes:
    base_options: Base options for the face landmarker task.
    running_mode: The running mode of the task. Default to the image mode.
      HandLandmarker has three running modes: 1) The image mode for detecting
      face landmarks on single image inputs. 2) The video mode for detecting
      face landmarks on the decoded frames of a video. 3) The live stream mode
      for detecting face landmarks on the live stream of input data, such as
      from camera. In this mode, the "result_callback" below must be specified
      to receive the detection results asynchronously.
    num_faces: The maximum number of faces that can be detected by the
      FaceLandmarker.
    min_face_detection_confidence: The minimum confidence score for the face
      detection to be considered successful.
    min_face_presence_confidence: The minimum confidence score of face presence
      score in the face landmark detection.
    min_tracking_confidence: The minimum confidence score for the face tracking
      to be considered successful.
    output_face_blendshapes: Whether FaceLandmarker outputs face blendshapes
      classification. Face blendshapes are used for rendering the 3D face model.
    output_facial_transformation_matrixes: Whether FaceLandmarker outputs facial
      transformation_matrix. Facial transformation matrix is used to transform
      the face landmarks in canonical face to the detected face, so that users
      can apply face effects on the detected landmarks.
    result_callback: The user-defined result callback for processing live stream
      data. The result callback should only be specified when the running mode
      is set to the live stream mode.
  """

  base_options: _BaseOptions
  running_mode: _RunningMode = _RunningMode.IMAGE
  num_faces: Optional[int] = 1
  min_face_detection_confidence: Optional[float] = 0.5
  min_face_presence_confidence: Optional[float] = 0.5
  min_tracking_confidence: Optional[float] = 0.5
  output_face_blendshapes: Optional[bool] = False
  output_facial_transformation_matrixes: Optional[bool] = False
  result_callback: Optional[
      Callable[[FaceLandmarkerResult, image_module.Image, int], None]
  ] = None

  @doc_controls.do_not_generate_docs
  def to_pb2(self) -> _FaceLandmarkerGraphOptionsProto:
    """Generates an FaceLandmarkerGraphOptions protobuf object."""
    base_options_proto = self.base_options.to_pb2()
    base_options_proto.use_stream_mode = (
        False if self.running_mode == _RunningMode.IMAGE else True
    )

    # Initialize the face landmarker options from base options.
    face_landmarker_options_proto = _FaceLandmarkerGraphOptionsProto(
        base_options=base_options_proto
    )

    # Configure face detector options.
    face_landmarker_options_proto.face_detector_graph_options.num_faces = (
        self.num_faces
    )
    face_landmarker_options_proto.face_detector_graph_options.min_detection_confidence = (
        self.min_face_detection_confidence
    )

    # Configure face landmark detector options.
    face_landmarker_options_proto.min_tracking_confidence = (
        self.min_tracking_confidence
    )
    face_landmarker_options_proto.face_landmarks_detector_graph_options.min_detection_confidence = (
        self.min_face_detection_confidence
    )
    return face_landmarker_options_proto


class FaceLandmarker(base_vision_task_api.BaseVisionTaskApi):
  """Class that performs face landmarks detection on images."""

  @classmethod
  def create_from_model_path(cls, model_path: str) -> 'FaceLandmarker':
    """Creates an `FaceLandmarker` object from a TensorFlow Lite model and the default `FaceLandmarkerOptions`.

    Note that the created `FaceLandmarker` instance is in image mode, for
    detecting face landmarks on single image inputs.

    Args:
      model_path: Path to the model.

    Returns:
      `FaceLandmarker` object that's created from the model file and the
      default `FaceLandmarkerOptions`.

    Raises:
      ValueError: If failed to create `FaceLandmarker` object from the
        provided file such as invalid file path.
      RuntimeError: If other types of error occurred.
    """
    base_options = _BaseOptions(model_asset_path=model_path)
    options = FaceLandmarkerOptions(
        base_options=base_options, running_mode=_RunningMode.IMAGE
    )
    return cls.create_from_options(options)

  @classmethod
  def create_from_options(
      cls, options: FaceLandmarkerOptions
  ) -> 'FaceLandmarker':
    """Creates the `FaceLandmarker` object from face landmarker options.

    Args:
      options: Options for the face landmarker task.

    Returns:
      `FaceLandmarker` object that's created from `options`.

    Raises:
      ValueError: If failed to create `FaceLandmarker` object from
        `FaceLandmarkerOptions` such as missing the model.
      RuntimeError: If other types of error occurred.
    """

    def packets_callback(output_packets: Mapping[str, packet_module.Packet]):
      if output_packets[_IMAGE_OUT_STREAM_NAME].is_empty():
        return

      image = packet_getter.get_image(output_packets[_IMAGE_OUT_STREAM_NAME])
      if output_packets[_IMAGE_OUT_STREAM_NAME].is_empty():
        return

      if output_packets[_NORM_LANDMARKS_STREAM_NAME].is_empty():
        empty_packet = output_packets[_NORM_LANDMARKS_STREAM_NAME]
        options.result_callback(
            FaceLandmarkerResult([], [], []),
            image,
            empty_packet.timestamp.value // _MICRO_SECONDS_PER_MILLISECOND,
        )
        return

      face_landmarks_result = _build_landmarker_result(output_packets)
      timestamp = output_packets[_NORM_LANDMARKS_STREAM_NAME].timestamp
      options.result_callback(
          face_landmarks_result,
          image,
          timestamp.value // _MICRO_SECONDS_PER_MILLISECOND,
      )

    output_streams = [
        ':'.join([_NORM_LANDMARKS_TAG, _NORM_LANDMARKS_STREAM_NAME]),
        ':'.join([_IMAGE_TAG, _IMAGE_OUT_STREAM_NAME]),
    ]

    if options.output_face_blendshapes:
      output_streams.append(
          ':'.join([_BLENDSHAPES_TAG, _BLENDSHAPES_STREAM_NAME])
      )
    if options.output_facial_transformation_matrixes:
      output_streams.append(
          ':'.join([_FACE_GEOMETRY_TAG, _FACE_GEOMETRY_STREAM_NAME])
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
  ) -> FaceLandmarkerResult:
    """Performs face landmarks detection on the given image.

    Only use this method when the FaceLandmarker is created with the image
    running mode.

    The image can be of any size with format RGB or RGBA.
    TODO: Describes how the input image will be preprocessed after the yuv
    support is implemented.

    Args:
      image: MediaPipe Image.
      image_processing_options: Options for image processing.

    Returns:
      The face landmarks detection results.

    Raises:
      ValueError: If any of the input arguments is invalid.
      RuntimeError: If face landmarker detection failed to run.
    """
    normalized_rect = self.convert_to_normalized_rect(
        image_processing_options, roi_allowed=False
    )
    output_packets = self._process_image_data({
        _IMAGE_IN_STREAM_NAME: packet_creator.create_image(image),
        _NORM_RECT_STREAM_NAME: packet_creator.create_proto(
            normalized_rect.to_pb2()
        ),
    })

    if output_packets[_NORM_LANDMARKS_STREAM_NAME].is_empty():
      return FaceLandmarkerResult([], [], [])

    return _build_landmarker_result(output_packets)

  def detect_for_video(
      self,
      image: image_module.Image,
      timestamp_ms: int,
      image_processing_options: Optional[_ImageProcessingOptions] = None,
  ) -> FaceLandmarkerResult:
    """Performs face landmarks detection on the provided video frame.

    Only use this method when the FaceLandmarker is created with the video
    running mode.

    Only use this method when the FaceLandmarker is created with the video
    running mode. It's required to provide the video frame's timestamp (in
    milliseconds) along with the video frame. The input timestamps should be
    monotonically increasing for adjacent calls of this method.

    Args:
      image: MediaPipe Image.
      timestamp_ms: The timestamp of the input video frame in milliseconds.
      image_processing_options: Options for image processing.

    Returns:
      The face landmarks detection results.

    Raises:
      ValueError: If any of the input arguments is invalid.
      RuntimeError: If face landmarker detection failed to run.
    """
    normalized_rect = self.convert_to_normalized_rect(
        image_processing_options, roi_allowed=False
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
      return FaceLandmarkerResult([], [], [])

    return _build_landmarker_result(output_packets)

  def detect_async(
      self,
      image: image_module.Image,
      timestamp_ms: int,
      image_processing_options: Optional[_ImageProcessingOptions] = None,
  ) -> None:
    """Sends live image data to perform face landmarks detection.

    The results will be available via the "result_callback" provided in the
    FaceLandmarkerOptions. Only use this method when the FaceLandmarker is
    created with the live stream running mode.

    Only use this method when the FaceLandmarker is created with the live
    stream running mode. The input timestamps should be monotonically increasing
    for adjacent calls of this method. This method will return immediately after
    the input image is accepted. The results will be available via the
    `result_callback` provided in the `FaceLandmarkerOptions`. The
    `detect_async` method is designed to process live stream data such as
    camera input. To lower the overall latency, face landmarker may drop the
    input images if needed. In other words, it's not guaranteed to have output
    per input image.

    The `result_callback` provides:
      - The face landmarks detection results.
      - The input image that the face landmarker runs on.
      - The input timestamp in milliseconds.

    Args:
      image: MediaPipe Image.
      timestamp_ms: The timestamp of the input image in milliseconds.
      image_processing_options: Options for image processing.

    Raises:
      ValueError: If the current input timestamp is smaller than what the
      face landmarker has already processed.
    """
    normalized_rect = self.convert_to_normalized_rect(
        image_processing_options, roi_allowed=False
    )
    self._send_live_stream_data({
        _IMAGE_IN_STREAM_NAME: packet_creator.create_image(image).at(
            timestamp_ms * _MICRO_SECONDS_PER_MILLISECOND
        ),
        _NORM_RECT_STREAM_NAME: packet_creator.create_proto(
            normalized_rect.to_pb2()
        ).at(timestamp_ms * _MICRO_SECONDS_PER_MILLISECOND),
    })
