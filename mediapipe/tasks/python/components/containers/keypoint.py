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
"""Keypoint data class."""

import dataclasses
from typing import Any, Optional

from mediapipe.tasks.python.components.containers import keypoint_c as keypoint_c_lib
from mediapipe.tasks.python.core.optional_dependencies import doc_controls


@dataclasses.dataclass
class NormalizedKeypoint:
  """A normalized keypoint.

  Normalized keypoint represents a point in 2D space with x, y coordinates.
  x and y are normalized to [0.0, 1.0] by the image width and height
  respectively.

  Attributes:
    x: The x coordinates of the normalized keypoint.
    y: The y coordinates of the normalized keypoint.
    label: The optional label of the keypoint.
    score: The score of the keypoint.
  """

  x: Optional[float] = None
  y: Optional[float] = None
  label: Optional[str] = None
  score: Optional[float] = None

  @classmethod
  @doc_controls.do_not_generate_docs
  def from_ctypes(
      cls, c_obj: keypoint_c_lib.NormalizedKeypointC
  ) -> 'NormalizedKeypoint':
    """Creates a `NormalizedKeypoint` object from a `NormalizedKeypointC."""
    return NormalizedKeypoint(
        x=c_obj.x,
        y=c_obj.y,
        label=c_obj.label.decode('utf-8') if c_obj.label else None,
        score=c_obj.score,
    )

  def __eq__(self, other: Any) -> bool:
    """Checks if this object is equal to the given object.

    Args:
      other: The object to be compared with.

    Returns:
      True if the objects are equal.
    """
    if not isinstance(other, NormalizedKeypoint):
      return False
    return (
        self.x == other.x
        and self.y == other.y
        and self.label == other.label
        and self.score == other.score
    )
