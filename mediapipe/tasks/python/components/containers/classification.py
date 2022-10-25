# Copyright 2022 The MediaPipe Authors. All Rights Reserved.
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
"""Classification data class."""

import dataclasses
from typing import Any, List, Optional

from mediapipe.framework.formats import classification_pb2
from mediapipe.tasks.python.core.optional_dependencies import doc_controls

_ClassificationProto = classification_pb2.Classification
_ClassificationListProto = classification_pb2.ClassificationList


@dataclasses.dataclass
class Classification:
  """A classification.

  Attributes:
    index: The index of the class in the corresponding label map.
    score: The probability score for this class.
    label_name: Label or name of the class.
    display_name: Optional human-readable string for display purposes.
  """

  index: Optional[int] = None
  score: Optional[float] = None
  label: Optional[str] = None
  display_name: Optional[str] = None

  @doc_controls.do_not_generate_docs
  def to_pb2(self) -> _ClassificationProto:
    """Generates a Classification protobuf object."""
    return _ClassificationProto(
        index=self.index,
        score=self.score,
        label=self.label,
        display_name=self.display_name)

  @classmethod
  @doc_controls.do_not_generate_docs
  def create_from_pb2(cls, pb2_obj: _ClassificationProto) -> 'Classification':
    """Creates a `Classification` object from the given protobuf object."""
    return Classification(
        index=pb2_obj.index,
        score=pb2_obj.score,
        label=pb2_obj.label,
        display_name=pb2_obj.display_name)

  def __eq__(self, other: Any) -> bool:
    """Checks if this object is equal to the given object.

    Args:
      other: The object to be compared with.

    Returns:
      True if the objects are equal.
    """
    if not isinstance(other, Classification):
      return False

    return self.to_pb2().__eq__(other.to_pb2())


@dataclasses.dataclass
class ClassificationList:
  """Represents the classifications for a given classifier.
  Attributes:
    classification : A list of `Classification` objects.
    tensor_index: Optional index of the tensor that produced these
      classifications.
    tensor_name:  Optional name of the tensor that produced these
      classifications tensor metadata name.
  """

  classifications: List[Classification]
  tensor_index: Optional[int] = None
  tensor_name: Optional[str] = None

  @doc_controls.do_not_generate_docs
  def to_pb2(self) -> _ClassificationListProto:
    """Generates a ClassificationList protobuf object."""
    return _ClassificationListProto(
      classification=[
          classification.to_pb2()
          for classification in self.classifications
      ],
      tensor_index=self.tensor_index,
      tensor_name=self.tensor_name)

  @classmethod
  @doc_controls.do_not_generate_docs
  def create_from_pb2(
      cls,
      pb2_obj: _ClassificationListProto
  ) -> 'ClassificationList':
    """Creates a `ClassificationList` object from the given protobuf object."""
    return ClassificationList(
      classifications=[
        Classification.create_from_pb2(classification)
        for classification in pb2_obj.classification
      ],
      tensor_index=pb2_obj.tensor_index,
      tensor_name=pb2_obj.tensor_name)

  def __eq__(self, other: Any) -> bool:
    """Checks if this object is equal to the given object.
    Args:
      other: The object to be compared with.
    Returns:
      True if the objects are equal.
    """
    if not isinstance(other, ClassificationList):
      return False

    return self.to_pb2().__eq__(other.to_pb2())
