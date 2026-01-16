# Copyright 2022 The TensorFlow Authors.
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
"""Classifications data class."""

import dataclasses
from typing import List, Optional

from mediapipe.tasks.python.components.containers import category as category_lib
from mediapipe.tasks.python.components.containers import classification_result_c
from mediapipe.tasks.python.core.optional_dependencies import doc_controls


@dataclasses.dataclass
class Classifications:
  """Represents the classification results for a given classifier head.

  Attributes:
    categories: The array of predicted categories, usually sorted by descending
      scores (e.g. from high to low probability).
    head_index: The index of the classifier head these categories refer to. This
      is useful for multi-head models.
    head_name: The name of the classifier head, which is the corresponding
      tensor metadata name.
  """

  categories: List[category_lib.Category]
  head_index: int
  head_name: Optional[str] = None

  @classmethod
  @doc_controls.do_not_generate_docs
  def from_ctypes(
      cls, c_struct: classification_result_c.ClassificationsC
  ) -> 'Classifications':
    """Creates a `Classifications` object from the given ctypes struct."""
    if c_struct.categories and c_struct.categories_count > 0:
      categories = [
          category_lib.Category.from_ctypes(c_struct.categories[i])
          for i in range(c_struct.categories_count)
      ]
    else:
      categories = []

    return Classifications(
        categories=categories,
        head_index=c_struct.head_index,
        head_name=(
            c_struct.head_name.decode('utf-8') if c_struct.head_name else None
        ),
    )


@dataclasses.dataclass
class ClassificationResult:
  """Contains the classification results of a model.

  Attributes:
    classifications: A list of `Classifications` objects, each for a head of the
      model.
    timestamp_ms: The optional timestamp (in milliseconds) of the start of the
      chunk of data corresponding to these results. This is only used for
      classification on time series (e.g. audio classification). In these use
      cases, the amount of data to process might exceed the maximum size that
      the model can process: to solve this, the input data is split into
      multiple chunks starting at different timestamps.
  """

  classifications: List[Classifications]
  timestamp_ms: Optional[int] = None

  @classmethod
  @doc_controls.do_not_generate_docs
  def from_ctypes(
      cls, c_struct: classification_result_c.ClassificationResultC
  ) -> 'ClassificationResult':
    """Creates a `ClassificationResult` object from a ctypes struct."""
    if c_struct.classifications and c_struct.classifications_count > 0:
      classifications = [
          Classifications.from_ctypes(c_struct.classifications[i])
          for i in range(c_struct.classifications_count)
      ]
    else:
      classifications = []

    timestamp_ms = c_struct.timestamp_ms if c_struct.has_timestamp_ms else None
    return ClassificationResult(
        classifications=classifications, timestamp_ms=timestamp_ms
    )
