# Copyright 2022 The MediaPipe Authors. All Rights Reserved.
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

_RectProto = rect_pb2.Rect
_NormalizedRectProto = rect_pb2.NormalizedRect


@dataclasses.dataclass
class Rect:
  """A rectangle with rotation in image coordinates.

  Attributes:
    x_center : The X coordinate of the top-left corner, in pixels.
    y_center : The Y coordinate of the top-left corner, in pixels.
    width: The width of the rectangle, in pixels.
    height: The height of the rectangle, in pixels.
    rotation: Rotation angle is clockwise in radians.
    rect_id:  Optional unique id to help associate different rectangles to each
      other.
  """

  x_center: int
  y_center: int
  width: int
  height: int
  rotation: Optional[float] = 0.0
  rect_id: Optional[int] = None

  @doc_controls.do_not_generate_docs
  def to_pb2(self) -> _RectProto:
    """Generates a Rect protobuf object."""
    return _RectProto(
        x_center=self.x_center,
        y_center=self.y_center,
        width=self.width,
        height=self.height,
    )

  @classmethod
  @doc_controls.do_not_generate_docs
  def create_from_pb2(cls, pb2_obj: _RectProto) -> 'Rect':
    """Creates a `Rect` object from the given protobuf object."""
    return Rect(
        x_center=pb2_obj.x_center,
        y_center=pb2_obj.y_center,
        width=pb2_obj.width,
        height=pb2_obj.height)

  def __eq__(self, other: Any) -> bool:
    """Checks if this object is equal to the given object.

    Args:
      other: The object to be compared with.

    Returns:
      True if the objects are equal.
    """
    if not isinstance(other, Rect):
      return False

    return self.to_pb2().__eq__(other.to_pb2())


@dataclasses.dataclass
class NormalizedRect:
  """A rectangle with rotation in normalized coordinates. The values of box
    center location and size are within [0, 1].

  Attributes:
    x_center : The X normalized coordinate of the top-left corner.
    y_center : The Y normalized coordinate of the top-left corner.
    width: The width of the rectangle.
    height: The height of the rectangle.
    rotation: Rotation angle is clockwise in radians.
    rect_id:  Optional unique id to help associate different rectangles to each
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
    )

  @classmethod
  @doc_controls.do_not_generate_docs
  def create_from_pb2(cls, pb2_obj: _NormalizedRectProto) -> 'NormalizedRect':
    """Creates a `NormalizedRect` object from the given protobuf object."""
    return NormalizedRect(
        x_center=pb2_obj.x_center,
        y_center=pb2_obj.y_center,
        width=pb2_obj.width,
        height=pb2_obj.height)

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
