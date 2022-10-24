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
"""Gesture data class."""

import dataclasses
from typing import Any, List

from mediapipe.tasks.python.components.containers import classification
from mediapipe.tasks.python.components.containers import landmark
from mediapipe.tasks.python.core.optional_dependencies import doc_controls


@dataclasses.dataclass
class GestureRecognitionResult:
  """ The gesture recognition result from GestureRecognizer, where each vector
  element represents a single hand detected in the image.

  Attributes:
    gestures:  Recognized hand gestures with sorted order such that the
      winning label is the first item in the list.
    handedness: Classification of handedness.
    hand_landmarks: Detected hand landmarks in normalized image coordinates.
    hand_world_landmarks: Detected hand landmarks in world coordinates.
  """

  gestures: List[classification.ClassificationList]
  handedness: List[classification.ClassificationList]
  hand_landmarks: List[landmark.NormalizedLandmarkList]
  hand_world_landmarks: List[landmark.LandmarkList]

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
