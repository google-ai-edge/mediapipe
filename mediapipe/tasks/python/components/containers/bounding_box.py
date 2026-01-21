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
"""Bounding box data class."""

import dataclasses
from typing import Any

from mediapipe.tasks.python.components.containers import rect_c as rect_c_lib
from mediapipe.tasks.python.core.optional_dependencies import doc_controls


@dataclasses.dataclass
class BoundingBox:
  """An integer bounding box, axis aligned.

  Attributes:
    origin_x: The X coordinate of the top-left corner, in pixels.
    origin_y: The Y coordinate of the top-left corner, in pixels.
    width: The width of the bounding box, in pixels.
    height: The height of the bounding box, in pixels.
  """

  origin_x: int
  origin_y: int
  width: int
  height: int

  @classmethod
  @doc_controls.do_not_generate_docs
  def from_ctypes(cls, c_obj: rect_c_lib.RectC) -> 'BoundingBox':
    """Creates a `BoundingBox` object from a ctypes RectC struct."""
    return BoundingBox(
        origin_x=c_obj.left,
        origin_y=c_obj.top,
        width=c_obj.right - c_obj.left,
        height=c_obj.bottom - c_obj.top,
    )

  def __eq__(self, other: Any) -> bool:
    """Checks if this object is equal to the given object.

    Args:
      other: The object to be compared with.

    Returns:
      True if the objects are equal.
    """
    if not isinstance(other, BoundingBox):
      return False
    return (
        self.origin_x == other.origin_x
        and self.origin_y == other.origin_y
        and self.width == other.width
        and self.height == other.height
    )
