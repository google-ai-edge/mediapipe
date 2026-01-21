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
"""Rect data class."""

import dataclasses
from typing import Any, Optional

from mediapipe.tasks.python.components.containers import rect_c as rect_c_module
from mediapipe.tasks.python.core.optional_dependencies import doc_controls


@dataclasses.dataclass
class Rect:
  """A rectangle, used as part of detection results.

  Absolute coordinates. The origin is on the top-left corner of the image.

  Attributes:
    left: The X coordinate of the left side of the rectangle.
    top: The Y coordinate of the top of the rectangle.
    bottom: The Y coordinate of the bottom of the rectangle.
    right: The X coordinate of the right side of the rectangle.
  """

  left: int
  top: int
  bottom: int
  right: int

  @doc_controls.do_not_generate_docs
  def to_ctypes(self) -> rect_c_module.RectC:
    """Generates a C API RectC object."""
    return rect_c_module.RectC(
        left=self.left, top=self.top, bottom=self.bottom, right=self.right
    )


@dataclasses.dataclass
class RectF:
  """A rectangle, used as part as input region-of-interest.

  The coordinates are normalized wrt the image dimensions, i.e. generally in
  [0,1] but they may exceed these bounds if describing a region overlapping the
  image. The origin is on the top-left corner of the image.

  Attributes:
    left: The X coordinate of the left side of the rectangle.
    top: The Y coordinate of the top of the rectangle.
    bottom: The Y coordinate of the bottom of the rectangle.
    right: The X coordinate of the right side of the rectangle.
  """

  left: float
  top: float
  bottom: float
  right: float

  @doc_controls.do_not_generate_docs
  def to_ctypes(self) -> rect_c_module.RectFC:
    """Generates a C API RectFC object."""
    return rect_c_module.RectFC(
        left=self.left, top=self.top, bottom=self.bottom, right=self.right
    )


@dataclasses.dataclass
class NormalizedRect:
  """A rectangle with rotation in normalized coordinates.

  Location of the center of the rectangle in image coordinates. The (0.0, 0.0)
  point is at the (top, left) corner.

  The values of box center location and size are within [0, 1].

  Attributes:
    x_center: The normalized X coordinate of the rectangle, in image
      coordinates.
    y_center: The normalized Y coordinate of the rectangle, in image
      coordinates.
    width: The width of the rectangle.
    height: The height of the rectangle.
    rotation: Rotation angle is clockwise in radians.
    rect_id: Optional unique id to help associate different rectangles to each
      other.
  """

  x_center: float
  y_center: float
  width: float
  height: float
  rotation: Optional[float] = 0.0
  rect_id: Optional[int] = None

  def __eq__(self, other: Any) -> bool:
    """Checks if this object is equal to the given object.

    Args:
      other: The object to be compared with.

    Returns:
      True if the objects are equal.
    """
    if not isinstance(other, NormalizedRect):
      return False
    return (
        self.x_center == other.x_center
        and self.y_center == other.y_center
        and self.width == other.width
        and self.height == other.height
        and self.rotation == other.rotation
        and self.rect_id == other.rect_id
    )
