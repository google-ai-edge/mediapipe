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

"""C types for ClassificationResult."""

import ctypes

from mediapipe.tasks.python.components.containers import category_c


class ClassificationsC(ctypes.Structure):
  _fields_ = [
      ("categories", ctypes.POINTER(category_c.CategoryC)),
      ("categories_count", ctypes.c_uint32),
      ("head_index", ctypes.c_int),
      ("head_name", ctypes.c_char_p),
  ]


class ClassificationResultC(ctypes.Structure):
  _fields_ = [
      ("classifications", ctypes.POINTER(ClassificationsC)),
      ("classifications_count", ctypes.c_uint32),
      ("timestamp_ms", ctypes.c_int64),
      ("has_timestamp_ms", ctypes.c_bool),
  ]
