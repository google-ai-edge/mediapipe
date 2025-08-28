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
from typing import List

from mediapipe.tasks.python.components.containers import category as category_module
from mediapipe.tasks.python.components.containers import category_c as category_c_module
from mediapipe.tasks.python.components.containers import classification_result as classification_result_module


class ClassificationsC(ctypes.Structure):
  _fields_ = [
      ("categories", ctypes.POINTER(category_c_module.CategoryC)),
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


def convert_to_python_classification_result(
    src: ClassificationResultC,
) -> classification_result_module.ClassificationResult:
  """Converts a ctypes ClassificationResultC struct to a Python ClassificationResult object."""
  python_result = classification_result_module.ClassificationResult(
      classifications=[]
  )

  if src.classifications and src.classifications_count > 0:
    for i in range(src.classifications_count):
      c_classifications = src.classifications[i]
      python_categories: List[category_module.Category] = []

      if (
          c_classifications.categories
          and c_classifications.categories_count > 0
      ):
        for j in range(c_classifications.categories_count):
          c_category = c_classifications.categories[j]
          python_categories.append(
              category_module.Category(
                  index=c_category.index,
                  score=c_category.score,
                  category_name=c_category.category_name.decode("utf-8")
                  if c_category.category_name
                  else None,
                  display_name=c_category.display_name.decode("utf-8")
                  if c_category.display_name
                  else None,
              )
          )
      python_result.classifications.append(
          classification_result_module.Classifications(
              categories=python_categories,
              head_index=c_classifications.head_index,
              head_name=c_classifications.head_name.decode("utf-8")
              if c_classifications.head_name
              else None,
          )
      )

  python_result.timestamp_ms = (
      src.timestamp_ms if src.has_timestamp_ms else None
  )

  return python_result
