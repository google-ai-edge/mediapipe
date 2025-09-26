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

from mediapipe.tasks.python.components.containers import bounding_box
from mediapipe.tasks.python.components.containers import category as category_lib
from mediapipe.tasks.python.components.containers import category_c
from mediapipe.tasks.python.components.containers import detections as detections_lib
from mediapipe.tasks.python.components.containers import keypoint
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

  def to_python_detection_result(self) -> detections_lib.DetectionResult:
    """Converts a ctypes DetectionResultC to a Python DetectionResult object."""
    py_detections = []
    for i in range(self.detections_count):
      c_detection = self.detections[i]
      py_categories = []
      for j in range(c_detection.categories_count):
        c_category = c_detection.categories[j]
        py_categories.append(category_lib.Category.from_ctypes(c_category))

      py_keypoints = []
      for j in range(c_detection.keypoints_count):
        c_keypoint = c_detection.keypoints[j]
        py_keypoints.append(keypoint.NormalizedKeypoint.from_ctypes(c_keypoint))

      py_bounding_box = bounding_box.BoundingBox.from_ctypes(
          c_detection.bounding_box
      )
      py_detections.append(
          detections_lib.Detection(
              bounding_box=py_bounding_box,
              categories=py_categories,
              keypoints=py_keypoints if py_keypoints else None,
          )
      )

    return detections_lib.DetectionResult(detections=py_detections)
