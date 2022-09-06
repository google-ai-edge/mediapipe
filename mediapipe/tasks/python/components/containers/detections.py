# Copyright 2022 The TensorFlow Authors. All Rights Reserved.
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
from typing import Any, List

from mediapipe.framework.formats import detection_pb2
from mediapipe.framework.formats import location_data_pb2
from mediapipe.tasks.python.components.containers import bounding_box as bounding_box_module
from mediapipe.tasks.python.components.containers import category as category_module
from mediapipe.tasks.python.core.optional_dependencies import doc_controls

_DetectionListProto = detection_pb2.DetectionList
_DetectionProto = detection_pb2.Detection
_LocationDataProto = location_data_pb2.LocationData


@dataclasses.dataclass
class Detection:
  """Represents one detected object in the object detector's results.

  Attributes:
    bounding_box: A BoundingBox object.
    categories: A list of Category objects.
  """

  bounding_box: bounding_box_module.BoundingBox
  categories: List[category_module.Category]

  @doc_controls.do_not_generate_docs
  def to_pb2(self) -> _DetectionProto:
    """Generates a Detection protobuf object."""
    labels = []
    label_ids = []
    scores = []
    display_names = []
    for category in self.categories:
      scores.append(category.score)
      if category.index:
        label_ids.append(category.index)
      if category.category_name:
        labels.append(category.category_name)
      if category.display_name:
        display_names.append(category.display_name)
    return _DetectionProto(
        label=labels,
        label_id=label_ids,
        score=scores,
        display_name=display_names,
        location_data=_LocationDataProto(
            format=_LocationDataProto.Format.BOUNDING_BOX,
            bounding_box=self.bounding_box.to_pb2()))

  @classmethod
  @doc_controls.do_not_generate_docs
  def create_from_pb2(cls, pb2_obj: _DetectionProto) -> 'Detection':
    """Creates a `Detection` object from the given protobuf object."""
    categories = []
    for idx, score in enumerate(pb2_obj.score):
      categories.append(
          category_module.Category(
              score=score,
              index=pb2_obj.label_id[idx]
              if idx < len(pb2_obj.label_id) else None,
              category_name=pb2_obj.label[idx]
              if idx < len(pb2_obj.label) else None,
              display_name=pb2_obj.display_name[idx]
              if idx < len(pb2_obj.display_name) else None))

    return Detection(
        bounding_box=bounding_box_module.BoundingBox.create_from_pb2(
            pb2_obj.location_data.bounding_box),
        categories=categories)

  def __eq__(self, other: Any) -> bool:
    """Checks if this object is equal to the given object.

    Args:
      other: The object to be compared with.

    Returns:
      True if the objects are equal.
    """
    if not isinstance(other, Detection):
      return False

    return self.to_pb2().__eq__(other.to_pb2())


@dataclasses.dataclass
class DetectionResult:
  """Represents the list of detected objects.

  Attributes:
    detections: A list of `Detection` objects.
  """

  detections: List[Detection]

  @doc_controls.do_not_generate_docs
  def to_pb2(self) -> _DetectionListProto:
    """Generates a DetectionList protobuf object."""
    return _DetectionListProto(
        detection=[detection.to_pb2() for detection in self.detections])

  @classmethod
  @doc_controls.do_not_generate_docs
  def create_from_pb2(cls, pb2_obj: _DetectionListProto) -> 'DetectionResult':
    """Creates a `DetectionResult` object from the given protobuf object."""
    return DetectionResult(detections=[
        Detection.create_from_pb2(detection) for detection in pb2_obj.detection
    ])

  def __eq__(self, other: Any) -> bool:
    """Checks if this object is equal to the given object.

    Args:
      other: The object to be compared with.

    Returns:
      True if the objects are equal.
    """
    if not isinstance(other, DetectionResult):
      return False

    return self.to_pb2().__eq__(other.to_pb2())
