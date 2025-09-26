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
"""MediaPipe Detection Result C API types."""

import ctypes

from mediapipe.tasks.python.components.containers import category_c
from mediapipe.tasks.python.components.containers import keypoint_c
from mediapipe.tasks.python.components.containers import rect_c


class DetectionC(ctypes.Structure):
  """CTypes for a single detection."""

  _fields_ = [
      ('categories', ctypes.POINTER(category_c.CategoryC)),
      ('categories_count', ctypes.c_uint32),
      ('bounding_box', rect_c.RectC),
      ('keypoints', ctypes.POINTER(keypoint_c.NormalizedKeypointC)),
      ('keypoints_count', ctypes.c_uint32),
  ]


class DetectionResultC(ctypes.Structure):
  """CTypes for the detection result."""

  _fields_ = [
      ('detections', ctypes.POINTER(DetectionC)),
      ('detections_count', ctypes.c_uint32),
  ]
