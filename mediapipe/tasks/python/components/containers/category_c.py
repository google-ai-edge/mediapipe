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

"""C types for Category."""

import ctypes

from mediapipe.tasks.python.components.containers import category as category_lib


class CategoryC(ctypes.Structure):
  _fields_ = [
      ("index", ctypes.c_int),
      ("score", ctypes.c_float),
      ("category_name", ctypes.c_char_p),
      ("display_name", ctypes.c_char_p),
  ]

  def to_python_category(self) -> category_lib.Category:
    """Converts a ctypes CategoryC to a Python Category object."""
    return category_lib.Category(
        index=self.index,
        score=self.score,
        category_name=(
            self.category_name.decode("utf-8") if self.category_name else None
        ),
        display_name=(
            self.display_name.decode("utf-8") if self.display_name else None
        ),
    )
