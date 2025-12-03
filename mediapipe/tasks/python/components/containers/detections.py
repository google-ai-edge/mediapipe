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
"""Detections data class."""

import dataclasses
from typing import Any, List, Optional

from mediapipe.tasks.python.components.containers import bounding_box as bounding_box_lib
from mediapipe.tasks.python.components.containers import category as category_lib
from mediapipe.tasks.python.components.containers import category_c as category_c_lib
from mediapipe.tasks.python.components.containers import detections_c as detections_c_lib
from mediapipe.tasks.python.components.containers import keypoint as keypoint_lib
from mediapipe.tasks.python.core.optional_dependencies import doc_controls


@dataclasses.dataclass
class Detection:
  """Represents one detected object in the object detector's results.

  Attributes:
    bounding_box: A BoundingBox object.
    categories: A list of Category objects.
    keypoints: A list of NormalizedKeypoint objects.
  """

  bounding_box: bounding_box_lib.BoundingBox
  categories: List[category_lib.Category]
  keypoints: Optional[List[keypoint_lib.NormalizedKeypoint]] = None

  def __eq__(self, other: Any) -> bool:
    """Checks if this object is equal to the given object.

    Args:
      other: The object to be compared with.

    Returns:
      True if the objects are equal.
    """
    if not isinstance(other, Detection):
      return False
    return (
        self.bounding_box == other.bounding_box
        and self.categories == other.categories
        and self.keypoints == other.keypoints
    )

  @classmethod
  @doc_controls.do_not_generate_docs
  def from_ctypes(cls, c_obj: detections_c_lib.DetectionC) -> 'Detection':
    """Creates a `Detection` object from the given `DetectionC` object."""
    c_categories = category_c_lib.CategoriesC(
        categories=c_obj.categories, categories_count=c_obj.categories_count
    )

    py_categories = category_lib.create_list_of_categories_from_ctypes(
        c_categories
    )
    py_bounding_box = bounding_box_lib.BoundingBox.from_ctypes(
        c_obj.bounding_box
    )

    if c_obj.keypoints and c_obj.keypoints_count > 0:
      py_keypoints = [
          keypoint_lib.NormalizedKeypoint.from_ctypes(c_obj.keypoints[i])
          for i in range(c_obj.keypoints_count)
      ]
    else:
      py_keypoints = None

    return Detection(
        bounding_box=py_bounding_box,
        categories=py_categories,
        keypoints=py_keypoints,
    )


@dataclasses.dataclass
class DetectionResult:
  """Represents the list of detected objects.

  Attributes:
    detections: A list of `Detection` objects.
  """

  detections: List[Detection]

  def __eq__(self, other: Any) -> bool:
    """Checks if this object is equal to the given object.

    Args:
      other: The object to be compared with.

    Returns:
      True if the objects are equal.
    """
    if not isinstance(other, DetectionResult):
      return False
    return self.detections == other.detections

  @classmethod
  @doc_controls.do_not_generate_docs
  def from_ctypes(
      cls, c_obj: detections_c_lib.DetectionResultC
  ) -> 'DetectionResult':
    """Creates a `DetectionResult` object from a `DetectionResultC` object."""
    return DetectionResult(
        detections=[
            Detection.from_ctypes(c_obj.detections[i])
            for i in range(c_obj.detections_count)
        ]
    )
