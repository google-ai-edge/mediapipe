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
"""Landmarks Detection Result data class."""

import dataclasses
from typing import Optional, List

from mediapipe.tasks.python.components.containers import category as category_module
from mediapipe.tasks.python.components.containers import landmark as landmark_module
from mediapipe.tasks.python.components.containers import rect as rect_module

_Category = category_module.Category
_NormalizedLandmark = landmark_module.NormalizedLandmark
_Landmark = landmark_module.Landmark


@dataclasses.dataclass
class LandmarksDetectionResult:
  """Represents the landmarks detection result.

  Attributes:
    landmarks: A list of `NormalizedLandmark` objects.
    categories: A list of `Category` objects.
    world_landmarks: A list of `Landmark` objects.
    rect: A `NormalizedRect` object.
  """

  landmarks: Optional[List[_NormalizedLandmark]]
  categories: Optional[List[_Category]]
  world_landmarks: Optional[List[_Landmark]]
  rect: rect_module.NormalizedRect
