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
"""MediaPipe ImageProcessingOptions C API types."""

import ctypes

from mediapipe.tasks.python.components.containers import rect_c


class ImageProcessingOptionsC(ctypes.Structure):
  """CTypes for ImageProcessingOptions."""

  _fields_ = [
      ('has_region_of_interest', ctypes.c_bool),
      ('region_of_interest', rect_c.RectFC),
      ('rotation_degrees', ctypes.c_int),
  ]
