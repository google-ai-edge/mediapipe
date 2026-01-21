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
"""Gesture Recognizer Result C API types."""

import ctypes

from mediapipe.tasks.python.components.containers import category_c
from mediapipe.tasks.python.components.containers import landmark_c


class GestureRecognizerResultC(ctypes.Structure):
  """The gesture recognition result from GestureRecognizer in C API."""

  _fields_ = [
      ("gestures", ctypes.POINTER(category_c.CategoriesC)),
      ("gestures_count", ctypes.c_uint32),
      ("handedness", ctypes.POINTER(category_c.CategoriesC)),
      ("handedness_count", ctypes.c_uint32),
      (
          "hand_landmarks",
          ctypes.POINTER(landmark_c.NormalizedLandmarksC),
      ),
      ("hand_landmarks_count", ctypes.c_uint32),
      (
          "hand_world_landmarks",
          ctypes.POINTER(landmark_c.LandmarksC),
      ),
      ("hand_world_landmarks_count", ctypes.c_uint32),
  ]
