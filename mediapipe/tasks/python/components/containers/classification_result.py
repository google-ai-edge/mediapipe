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

from mediapipe.framework.formats import classification_pb2
from mediapipe.tasks.cc.components.containers.proto import classifications_pb2
from mediapipe.tasks.python.components.containers import category as category_module
from mediapipe.tasks.python.core.optional_dependencies import doc_controls

_ClassificationProto = classification_pb2.Classification
_ClassificationListProto = classification_pb2.ClassificationList
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

  @doc_controls.do_not_generate_docs
  def to_pb2(self) -> _ClassificationsProto:
    """Generates a Classifications protobuf object."""
    classification_list_proto = _ClassificationListProto()
    for category in self.categories:
      classification_proto = category.to_pb2()
      classification_list_proto.classification.append(classification_proto)
    return _ClassificationsProto(
        classification_list=classification_list_proto,
        head_index=self.head_index,
        head_name=self.head_name)

  @classmethod
  @doc_controls.do_not_generate_docs
  def create_from_pb2(cls, pb2_obj: _ClassificationsProto) -> 'Classifications':
    """Creates a `Classifications` object from the given protobuf object."""
    categories = []
    for classification in pb2_obj.classification_list.classification:
      categories.append(
          category_module.Category.create_from_pb2(classification))
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

  @doc_controls.do_not_generate_docs
  def to_pb2(self) -> _ClassificationResultProto:
    """Generates a ClassificationResult protobuf object."""
    return _ClassificationResultProto(
        classifications=[
            classification.to_pb2() for classification in self.classifications
        ],
        timestamp_ms=self.timestamp_ms)

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
