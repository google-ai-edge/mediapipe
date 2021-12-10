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

from typing import Mapping, Tuple

from mediapipe.python.solutions import face_mesh_connections
from mediapipe.python.solutions import hands_connections
from mediapipe.python.solutions.drawing_utils import DrawingSpec
from mediapipe.python.solutions.hands import HandLandmark
from mediapipe.python.solutions.pose import PoseLandmark

_RADIUS = 5
_RED = (48, 48, 255)
_GREEN = (48, 255, 48)
_BLUE = (192, 101, 21)
_YELLOW = (0, 204, 255)
_GRAY = (128, 128, 128)
_PURPLE = (128, 64, 128)
_PEACH = (180, 229, 255)
_WHITE = (224, 224, 224)

# Hands
_THICKNESS_WRIST_MCP = 3
_THICKNESS_FINGER = 2
_THICKNESS_DOT = -1

# Hand landmarks
_PALM_LANMARKS = (HandLandmark.WRIST, HandLandmark.THUMB_CMC,
                  HandLandmark.INDEX_FINGER_MCP, HandLandmark.MIDDLE_FINGER_MCP,
                  HandLandmark.RING_FINGER_MCP, HandLandmark.PINKY_MCP)
_THUMP_LANDMARKS = (HandLandmark.THUMB_MCP, HandLandmark.THUMB_IP,
                    HandLandmark.THUMB_TIP)
_INDEX_FINGER_LANDMARKS = (HandLandmark.INDEX_FINGER_PIP,
                           HandLandmark.INDEX_FINGER_DIP,
                           HandLandmark.INDEX_FINGER_TIP)
_MIDDLE_FINGER_LANDMARKS = (HandLandmark.MIDDLE_FINGER_PIP,
                            HandLandmark.MIDDLE_FINGER_DIP,
                            HandLandmark.MIDDLE_FINGER_TIP)
_RING_FINGER_LANDMARKS = (HandLandmark.RING_FINGER_PIP,
                          HandLandmark.RING_FINGER_DIP,
                          HandLandmark.RING_FINGER_TIP)
_PINKY_FINGER_LANDMARKS = (HandLandmark.PINKY_PIP, HandLandmark.PINKY_DIP,
                           HandLandmark.PINKY_TIP)
_HAND_LANDMARK_STYLE = {
    _PALM_LANMARKS:
        DrawingSpec(
            color=_RED, thickness=_THICKNESS_DOT, circle_radius=_RADIUS),
    _THUMP_LANDMARKS:
        DrawingSpec(
            color=_PEACH, thickness=_THICKNESS_DOT, circle_radius=_RADIUS),
    _INDEX_FINGER_LANDMARKS:
        DrawingSpec(
            color=_PURPLE, thickness=_THICKNESS_DOT, circle_radius=_RADIUS),
    _MIDDLE_FINGER_LANDMARKS:
        DrawingSpec(
            color=_YELLOW, thickness=_THICKNESS_DOT, circle_radius=_RADIUS),
    _RING_FINGER_LANDMARKS:
        DrawingSpec(
            color=_GREEN, thickness=_THICKNESS_DOT, circle_radius=_RADIUS),
    _PINKY_FINGER_LANDMARKS:
        DrawingSpec(
            color=_BLUE, thickness=_THICKNESS_DOT, circle_radius=_RADIUS),
}

# Hands connections
_HAND_CONNECTION_STYLE = {
    hands_connections.HAND_PALM_CONNECTIONS:
        DrawingSpec(color=_GRAY, thickness=_THICKNESS_WRIST_MCP),
    hands_connections.HAND_THUMB_CONNECTIONS:
        DrawingSpec(color=_PEACH, thickness=_THICKNESS_FINGER),
    hands_connections.HAND_INDEX_FINGER_CONNECTIONS:
        DrawingSpec(color=_PURPLE, thickness=_THICKNESS_FINGER),
    hands_connections.HAND_MIDDLE_FINGER_CONNECTIONS:
        DrawingSpec(color=_YELLOW, thickness=_THICKNESS_FINGER),
    hands_connections.HAND_RING_FINGER_CONNECTIONS:
        DrawingSpec(color=_GREEN, thickness=_THICKNESS_FINGER),
    hands_connections.HAND_PINKY_FINGER_CONNECTIONS:
        DrawingSpec(color=_BLUE, thickness=_THICKNESS_FINGER)
}

# FaceMesh connections
_THICKNESS_TESSELATION = 1
_THICKNESS_CONTOURS = 2
_FACEMESH_CONTOURS_CONNECTION_STYLE = {
    face_mesh_connections.FACEMESH_LIPS:
        DrawingSpec(color=_WHITE, thickness=_THICKNESS_CONTOURS),
    face_mesh_connections.FACEMESH_LEFT_EYE:
        DrawingSpec(color=_GREEN, thickness=_THICKNESS_CONTOURS),
    face_mesh_connections.FACEMESH_LEFT_EYEBROW:
        DrawingSpec(color=_GREEN, thickness=_THICKNESS_CONTOURS),
    face_mesh_connections.FACEMESH_RIGHT_EYE:
        DrawingSpec(color=_RED, thickness=_THICKNESS_CONTOURS),
    face_mesh_connections.FACEMESH_RIGHT_EYEBROW:
        DrawingSpec(color=_RED, thickness=_THICKNESS_CONTOURS),
    face_mesh_connections.FACEMESH_FACE_OVAL:
        DrawingSpec(color=_WHITE, thickness=_THICKNESS_CONTOURS)
}

# Pose
_THICKNESS_POSE_LANDMARKS = 2
_POSE_LANDMARKS_LEFT = frozenset([
    PoseLandmark.LEFT_EYE_INNER, PoseLandmark.LEFT_EYE,
    PoseLandmark.LEFT_EYE_OUTER, PoseLandmark.LEFT_EAR, PoseLandmark.MOUTH_LEFT,
    PoseLandmark.LEFT_SHOULDER, PoseLandmark.LEFT_ELBOW,
    PoseLandmark.LEFT_WRIST, PoseLandmark.LEFT_PINKY, PoseLandmark.LEFT_INDEX,
    PoseLandmark.LEFT_THUMB, PoseLandmark.LEFT_HIP, PoseLandmark.LEFT_KNEE,
    PoseLandmark.LEFT_ANKLE, PoseLandmark.LEFT_HEEL,
    PoseLandmark.LEFT_FOOT_INDEX
])

_POSE_LANDMARKS_RIGHT = frozenset([
    PoseLandmark.RIGHT_EYE_INNER, PoseLandmark.RIGHT_EYE,
    PoseLandmark.RIGHT_EYE_OUTER, PoseLandmark.RIGHT_EAR,
    PoseLandmark.MOUTH_RIGHT, PoseLandmark.RIGHT_SHOULDER,
    PoseLandmark.RIGHT_ELBOW, PoseLandmark.RIGHT_WRIST,
    PoseLandmark.RIGHT_PINKY, PoseLandmark.RIGHT_INDEX,
    PoseLandmark.RIGHT_THUMB, PoseLandmark.RIGHT_HIP, PoseLandmark.RIGHT_KNEE,
    PoseLandmark.RIGHT_ANKLE, PoseLandmark.RIGHT_HEEL,
    PoseLandmark.RIGHT_FOOT_INDEX
])


def get_default_hand_landmarks_style() -> Mapping[int, DrawingSpec]:
  """Returns the default hand landmarks drawing style.

  Returns:
      A mapping from each hand landmark to its default drawing spec.
  """
  hand_landmark_style = {}
  for k, v in _HAND_LANDMARK_STYLE.items():
    for landmark in k:
      hand_landmark_style[landmark] = v
  return hand_landmark_style


def get_default_hand_connections_style(
) -> Mapping[Tuple[int, int], DrawingSpec]:
  """Returns the default hand connections drawing style.

  Returns:
      A mapping from each hand connection to its default drawing spec.
  """
  hand_connection_style = {}
  for k, v in _HAND_CONNECTION_STYLE.items():
    for connection in k:
      hand_connection_style[connection] = v
  return hand_connection_style


def get_default_face_mesh_contours_style(
) -> Mapping[Tuple[int, int], DrawingSpec]:
  """Returns the default face mesh contours drawing style.

  Returns:
      A mapping from each face mesh contours connection to its default drawing
      spec.
  """
  face_mesh_contours_connection_style = {}
  for k, v in _FACEMESH_CONTOURS_CONNECTION_STYLE.items():
    for connection in k:
      face_mesh_contours_connection_style[connection] = v
  return face_mesh_contours_connection_style


def get_default_face_mesh_tesselation_style() -> DrawingSpec:
  """Returns the default face mesh tesselation drawing style.

  Returns:
      A DrawingSpec.
  """
  return DrawingSpec(color=_GRAY, thickness=_THICKNESS_TESSELATION)


def get_default_face_mesh_iris_connections_style(
) -> Mapping[Tuple[int, int], DrawingSpec]:
  """Returns the default face mesh iris connections drawing style.

  Returns:
       A mapping from each iris connection to its default drawing spec.
  """
  face_mesh_iris_connections_style = {}
  left_spec = DrawingSpec(color=_GREEN, thickness=_THICKNESS_CONTOURS)
  for connection in face_mesh_connections.FACEMESH_LEFT_IRIS:
    face_mesh_iris_connections_style[connection] = left_spec
  right_spec = DrawingSpec(color=_RED, thickness=_THICKNESS_CONTOURS)
  for connection in face_mesh_connections.FACEMESH_RIGHT_IRIS:
    face_mesh_iris_connections_style[connection] = right_spec
  return face_mesh_iris_connections_style


def get_default_pose_landmarks_style() -> Mapping[int, DrawingSpec]:
  """Returns the default pose landmarks drawing style.

  Returns:
      A mapping from each pose landmark to its default drawing spec.
  """
  pose_landmark_style = {}
  left_spec = DrawingSpec(
      color=(0, 138, 255), thickness=_THICKNESS_POSE_LANDMARKS)
  right_spec = DrawingSpec(
      color=(231, 217, 0), thickness=_THICKNESS_POSE_LANDMARKS)
  for landmark in _POSE_LANDMARKS_LEFT:
    pose_landmark_style[landmark] = left_spec
  for landmark in _POSE_LANDMARKS_RIGHT:
    pose_landmark_style[landmark] = right_spec
  pose_landmark_style[PoseLandmark.NOSE] = DrawingSpec(
      color=_WHITE, thickness=_THICKNESS_POSE_LANDMARKS)
  return pose_landmark_style
