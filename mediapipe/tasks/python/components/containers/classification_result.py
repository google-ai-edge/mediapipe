# Copyright 2022 The TensorFlow Authors. All Rights Reserved.
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

from mediapipe.tasks.cc.components.containers.proto import classifications_pb2
from mediapipe.tasks.python.components.containers import category as category_module
from mediapipe.tasks.python.core.optional_dependencies import doc_controls

_ClassificationsProto = classifications_pb2.Classifications
_ClassificationResultProto = classifications_pb2.ClassificationResult


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

  categories: List[category_module.Category]
  head_index: int
  head_name: Optional[str] = None

  @classmethod
  @doc_controls.do_not_generate_docs
  def create_from_pb2(cls, pb2_obj: _ClassificationsProto) -> 'Classifications':
    """Creates a `Classifications` object from the given protobuf object."""
    categories = []
    for entry in pb2_obj.classification_list.classification:
      categories.append(
          category_module.Category(
              index=entry.index,
              score=entry.score,
              display_name=entry.display_name,
              category_name=entry.label))

    return Classifications(
        categories=categories,
        head_index=pb2_obj.head_index,
        head_name=pb2_obj.head_name)


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
  def create_from_pb2(
      cls, pb2_obj: _ClassificationResultProto) -> 'ClassificationResult':
    """Creates a `ClassificationResult` object from the given protobuf object.
    """
    return ClassificationResult(
        classifications=[
            Classifications.create_from_pb2(classification)
            for classification in pb2_obj.classifications
        ],
        timestamp_ms=pb2_obj.timestamp_ms)
