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

from mediapipe.framework.formats import rect_pb2
from mediapipe.tasks.python.core.optional_dependencies import doc_controls

_NormalizedRectProto = rect_pb2.NormalizedRect


@dataclasses.dataclass
class Rect:
  """A rectangle, used as part of detection results or as input region-of-interest.

  The coordinates are normalized wrt the image dimensions, i.e. generally in
  [0,1] but they may exceed these bounds if describing a region overlapping the
  image. The origin is on the top-left corner of the image.

  Attributes:
    left: The X coordinate of the left side of the rectangle.
    top: The Y coordinate of the top of the rectangle.
    right: The X coordinate of the right side of the rectangle.
    bottom: The Y coordinate of the bottom of the rectangle.
  """

  left: float
  top: float
  right: float
  bottom: float


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

  @doc_controls.do_not_generate_docs
  def to_pb2(self) -> _NormalizedRectProto:
    """Generates a NormalizedRect protobuf object."""
    return _NormalizedRectProto(
        x_center=self.x_center,
        y_center=self.y_center,
        width=self.width,
        height=self.height,
        rotation=self.rotation,
        rect_id=self.rect_id)

  @classmethod
  @doc_controls.do_not_generate_docs
  def create_from_pb2(cls, pb2_obj: _NormalizedRectProto) -> 'NormalizedRect':
    """Creates a `NormalizedRect` object from the given protobuf object."""
    return NormalizedRect(
        x_center=pb2_obj.x_center,
        y_center=pb2_obj.y_center,
        width=pb2_obj.width,
        height=pb2_obj.height,
        rotation=pb2_obj.rotation,
        rect_id=pb2_obj.rect_id)

  def __eq__(self, other: Any) -> bool:
    """Checks if this object is equal to the given object.

    Args:
      other: The object to be compared with.

    Returns:
      True if the objects are equal.
    """
    if not isinstance(other, NormalizedRect):
      return False

    return self.to_pb2().__eq__(other.to_pb2())
