# Copyright 2022 The MediaPipe Authors.
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
from typing import Any, Optional

from mediapipe.tasks.python.components.containers import category_c as category_c_module
from mediapipe.tasks.python.core.optional_dependencies import doc_controls


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

  index: Optional[int] = None
  score: Optional[float] = None
  display_name: Optional[str] = None
  category_name: Optional[str] = None

  @classmethod
  @doc_controls.do_not_generate_docs
  def from_ctypes(cls, c_obj: category_c_module.CategoryC) -> "Category":
    """Creates a `Category` object from the given `CategoryC` object.

    This function converts the `CategoryC` index of -1 to a Python value of None
    to retain the same semantic meaning. All other values are converted as-is.

    Args:
      c_obj: The `CategoryC` object to be converted.

    Returns:
      A `Category` object.
    """
    return Category(
        index=c_obj.index if c_obj.index != -1 else None,
        score=c_obj.score,
        category_name=(
            c_obj.category_name.decode("utf-8") if c_obj.category_name else None
        ),
        display_name=(
            c_obj.display_name.decode("utf-8") if c_obj.display_name else None
        ),
    )

  def __eq__(self, other: Any) -> bool:
    """Checks if this object is equal to the given object.

    Args:
      other: The object to be compared with.

    Returns:
      True if the objects are equal.
    """
    if not isinstance(other, Category):
      return False
    return (
        self.index == other.index
        and self.score == other.score
        and self.display_name == other.display_name
        and self.category_name == other.category_name
    )


@doc_controls.do_not_generate_docs
def create_list_of_categories_from_ctypes(
    c_obj: category_c_module.CategoriesC,
) -> list[Category]:
  """Creates a list of `Category` objects from a `CategoriesC` object."""
  return [
      Category.from_ctypes(c_obj.categories[i])
      for i in range(c_obj.categories_count)
  ]
