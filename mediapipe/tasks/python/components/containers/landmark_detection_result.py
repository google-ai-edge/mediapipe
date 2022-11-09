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
"""Landmarks Detection Result data class."""

import dataclasses
from typing import Optional, List

from mediapipe.framework.formats import classification_pb2
from mediapipe.framework.formats import landmark_pb2
from mediapipe.tasks.cc.components.containers.proto import landmarks_detection_result_pb2
from mediapipe.tasks.python.components.containers import category as category_module
from mediapipe.tasks.python.components.containers import landmark as landmark_module
from mediapipe.tasks.python.components.containers import rect as rect_module
from mediapipe.tasks.python.core.optional_dependencies import doc_controls

_LandmarksDetectionResultProto = landmarks_detection_result_pb2.LandmarksDetectionResult
_ClassificationProto = classification_pb2.Classification
_ClassificationListProto = classification_pb2.ClassificationList
_LandmarkListProto = landmark_pb2.LandmarkList
_NormalizedLandmarkListProto = landmark_pb2.NormalizedLandmarkList
_NormalizedRect = rect_module.NormalizedRect
_Category = category_module.Category
_NormalizedLandmark = landmark_module.NormalizedLandmark
_Landmark = landmark_module.Landmark


@dataclasses.dataclass
class LandmarksDetectionResult:
  """Represents the landmarks detection result.

  Attributes: landmarks : A list of `NormalizedLandmark` objects. categories : A
  list of `Category` objects. world_landmarks : A list of `Landmark` objects.
  rect : A `NormalizedRect` object.
  """

  landmarks: Optional[List[_NormalizedLandmark]]
  categories: Optional[List[_Category]]
  world_landmarks: Optional[List[_Landmark]]
  rect: _NormalizedRect

  @doc_controls.do_not_generate_docs
  def to_pb2(self) -> _LandmarksDetectionResultProto:
    """Generates a LandmarksDetectionResult protobuf object."""

    landmarks = _NormalizedLandmarkListProto()
    classifications = _ClassificationListProto()
    world_landmarks = _LandmarkListProto()

    for landmark in self.landmarks:
      landmarks.landmark.append(landmark.to_pb2())

    for category in self.categories:
      classifications.classification.append(
          _ClassificationProto(
              index=category.index,
              score=category.score,
              label=category.category_name,
              display_name=category.display_name))

    return _LandmarksDetectionResultProto(
        landmarks=landmarks,
        classifications=classifications,
        world_landmarks=world_landmarks,
        rect=self.rect.to_pb2())

  @classmethod
  @doc_controls.do_not_generate_docs
  def create_from_pb2(
      cls,
      pb2_obj: _LandmarksDetectionResultProto) -> 'LandmarksDetectionResult':
    """Creates a `LandmarksDetectionResult` object from the given protobuf object."""
    categories = []
    landmarks = []
    world_landmarks = []

    for classification in pb2_obj.classifications.classification:
      categories.append(
          category_module.Category(
              score=classification.score,
              index=classification.index,
              category_name=classification.label,
              display_name=classification.display_name))

    for landmark in pb2_obj.landmarks.landmark:
      landmarks.append(_NormalizedLandmark.create_from_pb2(landmark))

    for landmark in pb2_obj.world_landmarks.landmark:
      world_landmarks.append(_Landmark.create_from_pb2(landmark))
    return LandmarksDetectionResult(
        landmarks=landmarks,
        categories=categories,
        world_landmarks=world_landmarks,
        rect=_NormalizedRect.create_from_pb2(pb2_obj.rect))
