# Copyright 2020 The MediaPipe Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""MediaPipe solution drawing utils."""

import dataclasses
import math
from typing import Mapping, Optional, Union

import cv2
import matplotlib.pyplot as plt
import numpy as np

from mediapipe.tasks.python.components.containers import detections as detections_module
from mediapipe.tasks.python.components.containers import landmark as landmark_module
from mediapipe.tasks.python.vision import face_landmarker

_PRESENCE_THRESHOLD = 0.5
_VISIBILITY_THRESHOLD = 0.5
_BGR_CHANNELS = 3
_BGRA_CHANNELS = 4
_CONNECTION = face_landmarker.FaceLandmarksConnections.Connection

WHITE_COLOR = (224, 224, 224)
BLACK_COLOR = (0, 0, 0)
RED_COLOR = (0, 0, 255)
GREEN_COLOR = (0, 128, 0)
BLUE_COLOR = (255, 0, 0)


@dataclasses.dataclass
class DrawingSpec:
  # Color for drawing the annotation. Default to the white color.
  color: Union[tuple[int, int, int], tuple[int, int, int, int]] = WHITE_COLOR
  # Thickness for drawing the annotation. Default to 2 pixels.
  thickness: int = 2
  # Circle radius. Default to 2 pixels.
  circle_radius: int = 2


def _normalized_to_pixel_coordinates(
    normalized_x: float,
    normalized_y: float,
    image_width: int,
    image_height: int,
) -> Union[None, tuple[int, int]]:
  """Converts normalized value pair to pixel coordinates."""

  # Checks if the float value is between 0 and 1.
  def is_valid_normalized_value(value: float) -> bool:
    return (value > 0 or math.isclose(0, value)) and (value < 1 or
                                                      math.isclose(1, value))

  if not (is_valid_normalized_value(normalized_x) and
          is_valid_normalized_value(normalized_y)):
    # TODO: Draw coordinates even if it's outside of the image bounds.
    return None
  x_px = min(math.floor(normalized_x * image_width), image_width - 1)
  y_px = min(math.floor(normalized_y * image_height), image_height - 1)
  return x_px, y_px


def _validate_image_channels(image: np.ndarray) -> tuple[int, int, int]:
  """Validates that `image` is a 3- or 4-channel 2D image and returns shape."""
  if image.ndim != 3 or image.shape[2] not in (_BGR_CHANNELS, _BGRA_CHANNELS):
    raise ValueError('Input image must contain three or four channel bgr data.')
  return image.shape[0], image.shape[1], image.shape[2]


def _color_for_channels(
    color: tuple[int, ...], channels: int
) -> tuple[int, ...]:
  """Appends opaque alpha (255) when drawing 3-tuple colors onto 4-channel images."""
  if channels == _BGRA_CHANNELS and len(color) == _BGR_CHANNELS:
    return (*color, 255)
  return color


def draw_detection(
    image: np.ndarray,
    detection: detections_module.Detection,
    keypoint_drawing_spec: DrawingSpec = DrawingSpec(color=RED_COLOR),
    bbox_drawing_spec: DrawingSpec = DrawingSpec(),
):
  """Draws the detction bounding box and keypoints on the image.

  Args:
    image: A three or four channel BGR/BGRA image represented as numpy ndarray.
    detection: A detection object with a bounding box.
    keypoint_drawing_spec: A DrawingSpec object that specifies the keypoints'
      drawing settings such as color, line thickness, and circle radius.
    bbox_drawing_spec: A DrawingSpec object that specifies the bounding box's
      drawing settings such as color and line thickness.

  Raises:
    ValueError: If the input image is not three or four channel data.
  """
  image_rows, image_cols, image_channels = _validate_image_channels(image)

  # Draws keypoints.
  if detection.keypoints:
    keypoint_color = _color_for_channels(
        keypoint_drawing_spec.color, image_channels
    )
    for keypoint in detection.keypoints:
      keypoint_px = _normalized_to_pixel_coordinates(
          keypoint.x, keypoint.y, image_cols, image_rows  # pyrefly: ignore[bad-argument-type]
      )
      if keypoint_px:
        cv2.circle(
            image,
            keypoint_px,
            keypoint_drawing_spec.circle_radius,
            keypoint_color,
            keypoint_drawing_spec.thickness,
        )

  # Draws bounding box.
  rect_start_point = (
      detection.bounding_box.origin_x,
      detection.bounding_box.origin_y,
  )
  rect_end_point = (
      detection.bounding_box.origin_x + detection.bounding_box.width,
      detection.bounding_box.origin_y + detection.bounding_box.height,
  )
  cv2.rectangle(
      image,
      rect_start_point,
      rect_end_point,
      _color_for_channels(bbox_drawing_spec.color, image_channels),
      bbox_drawing_spec.thickness,
  )


def draw_landmarks(
    image: np.ndarray,
    landmark_list: list[landmark_module.NormalizedLandmark],
    connections: Optional[list[_CONNECTION]] = None,
    landmark_drawing_spec: Optional[
        Union[DrawingSpec, Mapping[int, DrawingSpec]]
    ] = DrawingSpec(color=RED_COLOR),
    connection_drawing_spec: Union[
        DrawingSpec, Mapping[tuple[int, int], DrawingSpec]
    ] = DrawingSpec(),
    is_drawing_landmarks: bool = True,
):
  """Draws the landmarks and the connections on the image.

  Args:
    image: A three or four channel BGR/BGRA image represented as numpy ndarray.
    landmark_list: A list of normalized landmarks to be annotated on the image.
    connections: A list of landmark index tuples that specifies how landmarks to
      be connected in the drawing.
    landmark_drawing_spec: Either a DrawingSpec object or a mapping from hand
      landmarks to the DrawingSpecs that specifies the landmarks' drawing
      settings such as color, line thickness, and circle radius. If this
      argument is explicitly set to None, no landmarks will be drawn.
    connection_drawing_spec: Either a DrawingSpec object or a mapping from hand
      connections to the DrawingSpecs that specifies the connections' drawing
      settings such as color and line thickness. If this argument is explicitly
      set to None, no landmark connections will be drawn.
    is_drawing_landmarks: Whether to draw landmarks. If set false, skip drawing
      landmarks, only contours will be drawed.

  Raises:
    ValueError: If one of the followings:
      a) If the input image is not three or four channel data.
      b) If any connetions contain invalid landmark index.
  """
  if not landmark_list:
    return
  image_rows, image_cols, image_channels = _validate_image_channels(image)
  idx_to_coordinates = {}
  for idx, landmark in enumerate(landmark_list):
    if (
        landmark.visibility is not None
        and landmark.visibility < _VISIBILITY_THRESHOLD
    ) or (
        landmark.presence is not None
        and landmark.presence < _PRESENCE_THRESHOLD
    ):
      continue
    landmark_px = _normalized_to_pixel_coordinates(landmark.x, landmark.y,  # pyrefly: ignore[bad-argument-type]
                                                   image_cols, image_rows)
    if landmark_px:
      idx_to_coordinates[idx] = landmark_px
  if connections:
    num_landmarks = len(landmark_list)
    # Draws the connections if the start and end landmarks are both visible.
    for connection in connections:
      start_idx = connection.start
      end_idx = connection.end
      if not (0 <= start_idx < num_landmarks and 0 <= end_idx < num_landmarks):
        raise ValueError(f'Landmark index is out of range. Invalid connection '
                         f'from landmark #{start_idx} to landmark #{end_idx}.')
      if start_idx in idx_to_coordinates and end_idx in idx_to_coordinates:
        drawing_spec = (
            connection_drawing_spec[(start_idx, end_idx)]
            if isinstance(connection_drawing_spec, Mapping)
            else connection_drawing_spec
        )
        cv2.line(
            image,
            idx_to_coordinates[start_idx],
            idx_to_coordinates[end_idx],
            _color_for_channels(drawing_spec.color, image_channels),
            drawing_spec.thickness,
        )
  # Draws landmark points after finishing the connection lines, which is
  # aesthetically better.
  if is_drawing_landmarks and landmark_drawing_spec:
    border_color = _color_for_channels(WHITE_COLOR, image_channels)
    for idx, landmark_px in idx_to_coordinates.items():
      drawing_spec = landmark_drawing_spec[idx] if isinstance(
          landmark_drawing_spec, Mapping) else landmark_drawing_spec
      # White circle border
      circle_border_radius = max(drawing_spec.circle_radius + 1,
                                 int(drawing_spec.circle_radius * 1.2))
      cv2.circle(image, landmark_px, circle_border_radius, border_color,
                 drawing_spec.thickness)
      # Fill color into the circle
      cv2.circle(
          image,
          landmark_px,
          drawing_spec.circle_radius,
          _color_for_channels(drawing_spec.color, image_channels),
          drawing_spec.thickness,
      )


def draw_axis(
    image: np.ndarray,
    rotation: np.ndarray,
    translation: np.ndarray,
    focal_length: tuple[float, float] = (1.0, 1.0),
    principal_point: tuple[float, float] = (0.0, 0.0),
    axis_length: float = 0.1,
    axis_drawing_spec: DrawingSpec = DrawingSpec(),
):
  """Draws the 3D axis on the image.

  Args:
    image: A three or four channel BGR/BGRA image represented as numpy ndarray.
    rotation: Rotation matrix from object to camera coordinate frame.
    translation: Translation vector from object to camera coordinate frame.
    focal_length: camera focal length along x and y directions.
    principal_point: camera principal point in x and y.
    axis_length: length of the axis in the drawing.
    axis_drawing_spec: A DrawingSpec object that specifies the xyz axis drawing
      settings such as line thickness.

  Raises:
    ValueError: If one of the followings:
      a) If the input image is not three or four channel data.
  """
  image_rows, image_cols, image_channels = _validate_image_channels(image)
  # Create axis points in camera coordinate frame.
  axis_world = np.float32([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])  # pyrefly: ignore[bad-argument-type]
  axis_cam = np.matmul(rotation, axis_length * axis_world.T).T + translation
  x = axis_cam[..., 0]
  y = axis_cam[..., 1]
  z = axis_cam[..., 2]
  # Project 3D points to NDC space.
  fx, fy = focal_length
  px, py = principal_point
  x_ndc = np.clip(-fx * x / (z + 1e-5) + px, -1., 1.)
  y_ndc = np.clip(-fy * y / (z + 1e-5) + py, -1., 1.)
  # Convert from NDC space to image space.
  x_im = np.int32((1 + x_ndc) * 0.5 * image_cols)
  y_im = np.int32((1 - y_ndc) * 0.5 * image_rows)
  # Draw xyz axis on the image.
  origin = (x_im[0], y_im[0])  # pyrefly: ignore[bad-index]
  x_axis = (x_im[1], y_im[1])  # pyrefly: ignore[bad-index]
  y_axis = (x_im[2], y_im[2])  # pyrefly: ignore[bad-index]
  z_axis = (x_im[3], y_im[3])  # pyrefly: ignore[bad-index]
  cv2.arrowedLine(
      image,
      origin,
      x_axis,
      _color_for_channels(RED_COLOR, image_channels),
      axis_drawing_spec.thickness,
  )
  cv2.arrowedLine(
      image,
      origin,
      y_axis,
      _color_for_channels(GREEN_COLOR, image_channels),
      axis_drawing_spec.thickness,
  )
  cv2.arrowedLine(
      image,
      origin,
      z_axis,
      _color_for_channels(BLUE_COLOR, image_channels),
      axis_drawing_spec.thickness,
  )


def _normalize_color(color):
  return tuple(v / 255. for v in color)


def _bgr_to_rgb_color(
    color: Union[tuple[int, int, int], tuple[int, int, int, int]],
) -> tuple[int, ...]:
  """Swaps BGR channels to RGB while preserving an optional 4th alpha channel."""
  return (*color[2::-1], *color[3:])


def plot_landmarks(
    landmark_list: list[landmark_module.NormalizedLandmark],
    connections: Optional[list[tuple[int, int]]] = None,
    landmark_drawing_spec: DrawingSpec = DrawingSpec(
        color=RED_COLOR, thickness=5
    ),
    connection_drawing_spec: DrawingSpec = DrawingSpec(
        color=BLACK_COLOR, thickness=5
    ),
    elevation: int = 10,
    azimuth: int = 10,
):
  """Plot the landmarks and the connections in matplotlib 3d.

  Args:
    landmark_list: A list of normalized landmarks to be plotted.
    connections: A list of landmark index tuples that specifies how landmarks to
      be connected.
    landmark_drawing_spec: A DrawingSpec object that specifies the landmarks'
      drawing settings such as color and line thickness.
    connection_drawing_spec: A DrawingSpec object that specifies the
      connections' drawing settings such as color and line thickness.
    elevation: The elevation from which to view the plot.
    azimuth: the azimuth angle to rotate the plot.

  Raises:
    ValueError: If any connection contains an invalid landmark index.
  """
  if not landmark_list:
    return
  plt.figure(figsize=(10, 10))
  ax = plt.axes(projection='3d')
  ax.view_init(elev=elevation, azim=azimuth)
  plotted_landmarks = {}
  for idx, landmark in enumerate(landmark_list):
    if (
        landmark.visibility is not None
        and landmark.visibility < _VISIBILITY_THRESHOLD
    ) or (
        landmark.presence is not None
        and landmark.presence < _PRESENCE_THRESHOLD
    ):
      continue
    ax.scatter3D(
        xs=[-landmark.z or 0],  # pyrefly: ignore[unsupported-operation]
        ys=[landmark.x or 0],
        zs=[-landmark.y or 0],  # pyrefly: ignore[unsupported-operation]
        color=_normalize_color(_bgr_to_rgb_color(landmark_drawing_spec.color)),
        linewidth=landmark_drawing_spec.thickness,
    )
    plotted_landmarks[idx] = (-landmark.z, landmark.x, -landmark.y)  # pyrefly: ignore[unsupported-operation]
  if connections:
    num_landmarks = len(landmark_list)
    # Draws the connections if the start and end landmarks are both visible.
    for connection in connections:
      start_idx = connection[0]
      end_idx = connection[1]
      if not (0 <= start_idx < num_landmarks and 0 <= end_idx < num_landmarks):
        raise ValueError(f'Landmark index is out of range. Invalid connection '
                         f'from landmark #{start_idx} to landmark #{end_idx}.')
      if start_idx in plotted_landmarks and end_idx in plotted_landmarks:
        landmark_pair = [
            plotted_landmarks[start_idx], plotted_landmarks[end_idx]
        ]
        ax.plot3D(
            xs=[landmark_pair[0][0], landmark_pair[1][0]],
            ys=[landmark_pair[0][1], landmark_pair[1][1]],
            zs=[landmark_pair[0][2], landmark_pair[1][2]],
            color=_normalize_color(
                _bgr_to_rgb_color(connection_drawing_spec.color)
            ),
            linewidth=connection_drawing_spec.thickness)
  plt.show()
