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
"""Category data class."""

import dataclasses
from typing import Any

from mediapipe.tasks.cc.components.containers import category_pb2
from mediapipe.tasks.python.core.optional_dependencies import doc_controls

_CategoryProto = category_pb2.Category


@dataclasses.dataclass
class Category:
  """A classification category.

  Category is a util class, contains a label, its display name, a float
  value as score, and the index of the label in the corresponding label file.
  Typically it's used as the result of classification tasks.

  Attributes:
    index: The index of the label in the corresponding label file.
    score: The probability score of this label category.
    display_name: The display name of the label, which may be translated for
      different locales. For example, a label, "apple", may be translated into
      Spanish for display purpose, so that the `display_name` is "manzana".
    category_name: The label of this category object.
  """

  index: int
  score: float
  display_name: str
  category_name: str

  @doc_controls.do_not_generate_docs
  def to_pb2(self) -> _CategoryProto:
    """Generates a Category protobuf object."""
    return _CategoryProto(
        index=self.index,
        score=self.score,
        display_name=self.display_name,
        category_name=self.category_name)

  @classmethod
  @doc_controls.do_not_generate_docs
  def create_from_pb2(cls, pb2_obj: _CategoryProto) -> 'Category':
    """Creates a `Category` object from the given protobuf object."""
    return Category(
        index=pb2_obj.index,
        score=pb2_obj.score,
        display_name=pb2_obj.display_name,
        category_name=pb2_obj.category_name)

  def __eq__(self, other: Any) -> bool:
    """Checks if this object is equal to the given object.

    Args:
      other: The object to be compared with.

    Returns:
      True if the objects are equal.
    """
    if not isinstance(other, Category):
      return False

    return self.to_pb2().__eq__(other.to_pb2())
