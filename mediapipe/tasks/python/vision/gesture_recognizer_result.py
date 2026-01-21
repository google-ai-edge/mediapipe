# Copyright 2025 The MediaPipe Authors.
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
"""Gesture Recognizer Result data class."""

import dataclasses

from mediapipe.tasks.python.components.containers import category as category_module
from mediapipe.tasks.python.components.containers import landmark as landmark_module
from mediapipe.tasks.python.core.optional_dependencies import doc_controls
from mediapipe.tasks.python.vision import gesture_recognizer_result_c

_GESTURE_DEFAULT_INDEX = -1


@dataclasses.dataclass
class GestureRecognizerResult:
  """The gesture recognition result from GestureRecognizer.

  Each vector element represents a single hand detected in the image.

  Attributes:
    gestures: Recognized hand gestures of detected hands. Note that the index of
      the gesture is always -1, because the raw indices from multiple gesture
      classifiers cannot consolidate to a meaningful index.
    handedness: Classification of handedness.
    hand_landmarks: Detected hand landmarks in normalized image coordinates.
    hand_world_landmarks: Detected hand landmarks in world coordinates.
  """

  gestures: list[list[category_module.Category]]
  handedness: list[list[category_module.Category]]
  hand_landmarks: list[list[landmark_module.NormalizedLandmark]]
  hand_world_landmarks: list[list[landmark_module.Landmark]]

  @classmethod
  @doc_controls.do_not_generate_docs
  def from_ctypes(
      cls, c_result: gesture_recognizer_result_c.GestureRecognizerResultC
  ) -> "GestureRecognizerResult":
    """Creates a `GestureRecognizerResult` object from the given ctypes struct."""
    gestures = []
    for i in range(c_result.gestures_count):
      gesture_categories_c = c_result.gestures[i]
      gesture_categories = (
          category_module.create_list_of_categories_from_ctypes(
              gesture_categories_c
          )
      )
      for category in gesture_categories:
        category.index = _GESTURE_DEFAULT_INDEX
      gestures.append(gesture_categories)

    handedness = [
        category_module.create_list_of_categories_from_ctypes(
            c_result.handedness[i]
        )
        for i in range(c_result.handedness_count)
    ]

    hand_landmarks = [
        [
            landmark_module.NormalizedLandmark.from_ctypes(
                c_result.hand_landmarks[i].landmarks[j]
            )
            for j in range(c_result.hand_landmarks[i].landmarks_count)
        ]
        for i in range(c_result.hand_landmarks_count)
    ]

    hand_world_landmarks = [
        [
            landmark_module.Landmark.from_ctypes(
                c_result.hand_world_landmarks[i].landmarks[j]
            )
            for j in range(c_result.hand_world_landmarks[i].landmarks_count)
        ]
        for i in range(c_result.hand_world_landmarks_count)
    ]

    return cls(
        gestures=gestures,
        handedness=handedness,
        hand_landmarks=hand_landmarks,
        hand_world_landmarks=hand_world_landmarks,
    )
