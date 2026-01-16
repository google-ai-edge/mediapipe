# Copyright 2021 The MediaPipe Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless requi_RED by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""MediaPipe solution drawing styles."""

from typing import Mapping

from mediapipe.tasks.python.vision import drawing_utils
from mediapipe.tasks.python.vision import face_landmarker
from mediapipe.tasks.python.vision import hand_landmarker
from mediapipe.tasks.python.vision import pose_landmarker

_FaceLandmarksConnections = face_landmarker.FaceLandmarksConnections
_HandLandmark = hand_landmarker.HandLandmark
_PoseLandmark = pose_landmarker.PoseLandmark
_DrawingSpec = drawing_utils.DrawingSpec

_RADIUS = 5
_RED = (48, 48, 255)
_GREEN = (48, 255, 48)
_BLUE = (192, 101, 21)
_YELLOW = (0, 204, 255)
_GRAY = (128, 128, 128)
_PURPLE = (128, 64, 128)
_PEACH = (180, 229, 255)
_WHITE = (224, 224, 224)
_CYAN = (192, 255, 48)
_MAGENTA = (192, 48, 255)

# Hands
_THICKNESS_WRIST_MCP = 3
_THICKNESS_FINGER = 2
_THICKNESS_DOT = -1

# Hand landmarks
_PALM_LANDMARKS = (
    _HandLandmark.WRIST,
    _HandLandmark.THUMB_CMC,
    _HandLandmark.INDEX_FINGER_MCP,
    _HandLandmark.MIDDLE_FINGER_MCP,
    _HandLandmark.RING_FINGER_MCP,
    _HandLandmark.PINKY_MCP,
)
_THUMP_LANDMARKS = (
    _HandLandmark.THUMB_MCP,
    _HandLandmark.THUMB_IP,
    _HandLandmark.THUMB_TIP,
)
_INDEX_FINGER_LANDMARKS = (
    _HandLandmark.INDEX_FINGER_PIP,
    _HandLandmark.INDEX_FINGER_DIP,
    _HandLandmark.INDEX_FINGER_TIP,
)
_MIDDLE_FINGER_LANDMARKS = (
    _HandLandmark.MIDDLE_FINGER_PIP,
    _HandLandmark.MIDDLE_FINGER_DIP,
    _HandLandmark.MIDDLE_FINGER_TIP,
)
_RING_FINGER_LANDMARKS = (
    _HandLandmark.RING_FINGER_PIP,
    _HandLandmark.RING_FINGER_DIP,
    _HandLandmark.RING_FINGER_TIP,
)
_PINKY_FINGER_LANDMARKS = (
    _HandLandmark.PINKY_PIP,
    _HandLandmark.PINKY_DIP,
    _HandLandmark.PINKY_TIP,
)
_HAND_LANDMARK_STYLE = {
    _PALM_LANDMARKS: _DrawingSpec(
        color=_RED, thickness=_THICKNESS_DOT, circle_radius=_RADIUS
    ),
    _THUMP_LANDMARKS: _DrawingSpec(
        color=_PEACH, thickness=_THICKNESS_DOT, circle_radius=_RADIUS
    ),
    _INDEX_FINGER_LANDMARKS: _DrawingSpec(
        color=_PURPLE, thickness=_THICKNESS_DOT, circle_radius=_RADIUS
    ),
    _MIDDLE_FINGER_LANDMARKS: _DrawingSpec(
        color=_YELLOW, thickness=_THICKNESS_DOT, circle_radius=_RADIUS
    ),
    _RING_FINGER_LANDMARKS: _DrawingSpec(
        color=_GREEN, thickness=_THICKNESS_DOT, circle_radius=_RADIUS
    ),
    _PINKY_FINGER_LANDMARKS: _DrawingSpec(
        color=_BLUE, thickness=_THICKNESS_DOT, circle_radius=_RADIUS
    ),
}

# Hands connections
_HAND_CONNECTION_STYLE = [
    (
        hand_landmarker.HandLandmarksConnections.HAND_PALM_CONNECTIONS,
        _DrawingSpec(color=_GRAY, thickness=_THICKNESS_WRIST_MCP),
    ),
    (
        hand_landmarker.HandLandmarksConnections.HAND_THUMB_CONNECTIONS,
        _DrawingSpec(color=_PEACH, thickness=_THICKNESS_FINGER),
    ),
    (
        hand_landmarker.HandLandmarksConnections.HAND_INDEX_FINGER_CONNECTIONS,
        _DrawingSpec(color=_PURPLE, thickness=_THICKNESS_FINGER),
    ),
    (
        hand_landmarker.HandLandmarksConnections.HAND_MIDDLE_FINGER_CONNECTIONS,
        _DrawingSpec(color=_YELLOW, thickness=_THICKNESS_FINGER),
    ),
    (
        hand_landmarker.HandLandmarksConnections.HAND_RING_FINGER_CONNECTIONS,
        _DrawingSpec(color=_GREEN, thickness=_THICKNESS_FINGER),
    ),
    (
        hand_landmarker.HandLandmarksConnections.HAND_PINKY_FINGER_CONNECTIONS,
        _DrawingSpec(color=_BLUE, thickness=_THICKNESS_FINGER),
    ),
]

# FaceMesh connections
_THICKNESS_TESSELATION = 1
_THICKNESS_CONTOURS = 2
_FACE_LANDMARKER_CONTOURS_CONNECTION_STYLE = [
    (
        _FaceLandmarksConnections.FACE_LANDMARKS_LIPS,
        _DrawingSpec(color=_WHITE, thickness=_THICKNESS_CONTOURS),
    ),
    (
        _FaceLandmarksConnections.FACE_LANDMARKS_LEFT_EYE,
        _DrawingSpec(color=_GREEN, thickness=_THICKNESS_CONTOURS),
    ),
    (
        _FaceLandmarksConnections.FACE_LANDMARKS_LEFT_EYEBROW,
        _DrawingSpec(color=_GREEN, thickness=_THICKNESS_CONTOURS),
    ),
    (
        _FaceLandmarksConnections.FACE_LANDMARKS_RIGHT_EYE,
        _DrawingSpec(color=_RED, thickness=_THICKNESS_CONTOURS),
    ),
    (
        _FaceLandmarksConnections.FACE_LANDMARKS_RIGHT_EYEBROW,
        _DrawingSpec(color=_RED, thickness=_THICKNESS_CONTOURS),
    ),
    (
        _FaceLandmarksConnections.FACE_LANDMARKS_FACE_OVAL,
        _DrawingSpec(color=_WHITE, thickness=_THICKNESS_CONTOURS),
    ),
]

_FACE_LANDMARKER_CONTOURS_CONNECTION_STYLE_1 = [
    (
        _FaceLandmarksConnections.FACE_LANDMARKS_LIPS,
        _DrawingSpec(color=_BLUE, thickness=_THICKNESS_CONTOURS),
    ),
    (
        _FaceLandmarksConnections.FACE_LANDMARKS_LEFT_EYE,
        _DrawingSpec(color=_CYAN, thickness=_THICKNESS_CONTOURS),
    ),
    (
        _FaceLandmarksConnections.FACE_LANDMARKS_LEFT_EYEBROW,
        _DrawingSpec(color=_GREEN, thickness=_THICKNESS_CONTOURS),
    ),
    (
        _FaceLandmarksConnections.FACE_LANDMARKS_RIGHT_EYE,
        _DrawingSpec(color=_MAGENTA, thickness=_THICKNESS_CONTOURS),
    ),
    (
        _FaceLandmarksConnections.FACE_LANDMARKS_RIGHT_EYEBROW,
        _DrawingSpec(color=_RED, thickness=_THICKNESS_CONTOURS),
    ),
    (
        _FaceLandmarksConnections.FACE_LANDMARKS_FACE_OVAL,
        _DrawingSpec(color=_WHITE, thickness=_THICKNESS_CONTOURS),
    ),
    (
        _FaceLandmarksConnections.FACE_LANDMARKS_NOSE,
        _DrawingSpec(color=_YELLOW, thickness=_THICKNESS_CONTOURS),
    ),
]

# Pose
_THICKNESS_POSE_LANDMARKS = 2
_POSE_LANDMARKS_LEFT = frozenset([
    _PoseLandmark.LEFT_EYE_INNER,
    _PoseLandmark.LEFT_EYE,
    _PoseLandmark.LEFT_EYE_OUTER,
    _PoseLandmark.LEFT_EAR,
    _PoseLandmark.MOUTH_LEFT,
    _PoseLandmark.LEFT_SHOULDER,
    _PoseLandmark.LEFT_ELBOW,
    _PoseLandmark.LEFT_WRIST,
    _PoseLandmark.LEFT_PINKY,
    _PoseLandmark.LEFT_INDEX,
    _PoseLandmark.LEFT_THUMB,
    _PoseLandmark.LEFT_HIP,
    _PoseLandmark.LEFT_KNEE,
    _PoseLandmark.LEFT_ANKLE,
    _PoseLandmark.LEFT_HEEL,
    _PoseLandmark.LEFT_FOOT_INDEX,
])

_POSE_LANDMARKS_RIGHT = frozenset([
    _PoseLandmark.RIGHT_EYE_INNER,
    _PoseLandmark.RIGHT_EYE,
    _PoseLandmark.RIGHT_EYE_OUTER,
    _PoseLandmark.RIGHT_EAR,
    _PoseLandmark.MOUTH_RIGHT,
    _PoseLandmark.RIGHT_SHOULDER,
    _PoseLandmark.RIGHT_ELBOW,
    _PoseLandmark.RIGHT_WRIST,
    _PoseLandmark.RIGHT_PINKY,
    _PoseLandmark.RIGHT_INDEX,
    _PoseLandmark.RIGHT_THUMB,
    _PoseLandmark.RIGHT_HIP,
    _PoseLandmark.RIGHT_KNEE,
    _PoseLandmark.RIGHT_ANKLE,
    _PoseLandmark.RIGHT_HEEL,
    _PoseLandmark.RIGHT_FOOT_INDEX,
])


def get_default_hand_landmarks_style() -> Mapping[int, _DrawingSpec]:
  """Returns the default hand landmarks drawing style.

  Returns:
      A mapping from each hand landmark to its default drawing spec.
  """
  hand_landmark_style = {}
  for k, v in _HAND_LANDMARK_STYLE.items():
    for landmark in k:
      hand_landmark_style[landmark] = v
  return hand_landmark_style


def get_default_hand_connections_style() -> (
    Mapping[tuple[int, int], _DrawingSpec]
):
  """Returns the default hand connections drawing style.

  Returns:
      A mapping from each hand connection to its default drawing spec.
  """
  hand_connection_style = {}
  for connections, style in _HAND_CONNECTION_STYLE:
    for connection in connections:
      hand_connection_style[(connection.start, connection.end)] = style
  return hand_connection_style


def get_default_face_mesh_contours_style(
    i: int = 0,
) -> Mapping[tuple[int, int], _DrawingSpec]:
  """Returns the default face mesh contours drawing style.

  Args:
      i: The id for default style. Currently there are two default styles.

  Returns:
      A mapping from each face mesh contours connection to its default drawing
      spec.
  """
  default_style = (
      _FACE_LANDMARKER_CONTOURS_CONNECTION_STYLE_1
      if i == 1
      else _FACE_LANDMARKER_CONTOURS_CONNECTION_STYLE
  )
  face_mesh_contours_connection_style = {}
  for connections, style in default_style:
    for connection in connections:
      face_mesh_contours_connection_style[
          (connection.start, connection.end)
      ] = style
  return face_mesh_contours_connection_style


def get_default_face_mesh_tesselation_style() -> _DrawingSpec:
  """Returns the default face mesh tesselation drawing style.

  Returns:
      A DrawingSpec.
  """
  return _DrawingSpec(color=_GRAY, thickness=_THICKNESS_TESSELATION)


def get_default_face_mesh_iris_connections_style() -> (
    Mapping[tuple[int, int], _DrawingSpec]
):
  """Returns the default face mesh iris connections drawing style.

  Returns:
       A mapping from each iris connection to its default drawing spec.
  """
  face_mesh_iris_connections_style = {}
  left_spec = _DrawingSpec(color=_GREEN, thickness=_THICKNESS_CONTOURS)
  for connection in _FaceLandmarksConnections.FACE_LANDMARKS_LEFT_IRIS:
    face_mesh_iris_connections_style[(connection.start, connection.end)] = (
        left_spec
    )
  right_spec = _DrawingSpec(color=_RED, thickness=_THICKNESS_CONTOURS)
  for connection in _FaceLandmarksConnections.FACE_LANDMARKS_RIGHT_IRIS:
    face_mesh_iris_connections_style[(connection.start, connection.end)] = (
        right_spec
    )
  return face_mesh_iris_connections_style


def get_default_pose_landmarks_style() -> Mapping[int, _DrawingSpec]:
  """Returns the default pose landmarks drawing style.

  Returns:
      A mapping from each pose landmark to its default drawing spec.
  """
  pose_landmark_style = {}
  left_spec = _DrawingSpec(
      color=(0, 138, 255), thickness=_THICKNESS_POSE_LANDMARKS
  )
  right_spec = _DrawingSpec(
      color=(231, 217, 0), thickness=_THICKNESS_POSE_LANDMARKS
  )
  for landmark in _POSE_LANDMARKS_LEFT:
    pose_landmark_style[landmark] = left_spec
  for landmark in _POSE_LANDMARKS_RIGHT:
    pose_landmark_style[landmark] = right_spec
  pose_landmark_style[_PoseLandmark.NOSE] = _DrawingSpec(
      color=_WHITE, thickness=_THICKNESS_POSE_LANDMARKS
  )
  return pose_landmark_style
