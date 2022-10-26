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
"""Classifier options data class."""

import dataclasses
from typing import Any, List, Optional

from mediapipe.tasks.cc.components.processors.proto import classifier_options_pb2
from mediapipe.tasks.python.core.optional_dependencies import doc_controls

_ClassifierOptionsProto = classifier_options_pb2.ClassifierOptions


@dataclasses.dataclass
class ClassifierOptions:
  """Options for classification processor.

  Attributes:
    display_names_locale: The locale to use for display names specified through
      the TFLite Model Metadata.
    max_results: The maximum number of top-scored classification results to
      return.
    score_threshold: Overrides the ones provided in the model metadata. Results
      below this value are rejected.
    category_allowlist: Allowlist of category names. If non-empty, detection
      results whose category name is not in this set will be filtered out.
      Duplicate or unknown category names are ignored. Mutually exclusive with
      `category_denylist`.
    category_denylist: Denylist of category names. If non-empty, detection
      results whose category name is in this set will be filtered out. Duplicate
      or unknown category names are ignored. Mutually exclusive with
      `category_allowlist`.
  """

  display_names_locale: Optional[str] = None
  max_results: Optional[int] = None
  score_threshold: Optional[float] = None
  category_allowlist: Optional[List[str]] = None
  category_denylist: Optional[List[str]] = None

  @doc_controls.do_not_generate_docs
  def to_pb2(self) -> _ClassifierOptionsProto:
    """Generates a ClassifierOptions protobuf object."""
    return _ClassifierOptionsProto(
        score_threshold=self.score_threshold,
        category_allowlist=self.category_allowlist,
        category_denylist=self.category_denylist,
        display_names_locale=self.display_names_locale,
        max_results=self.max_results)

  @classmethod
  @doc_controls.do_not_generate_docs
  def create_from_pb2(cls,
                      pb2_obj: _ClassifierOptionsProto) -> 'ClassifierOptions':
    """Creates a `ClassifierOptions` object from the given protobuf object."""
    return ClassifierOptions(
        score_threshold=pb2_obj.score_threshold,
        category_allowlist=[str(name) for name in pb2_obj.category_allowlist],
        category_denylist=[str(name) for name in pb2_obj.category_denylist],
        display_names_locale=pb2_obj.display_names_locale,
        max_results=pb2_obj.max_results)

  def __eq__(self, other: Any) -> bool:
    """Checks if this object is equal to the given object.

    Args:
      other: The object to be compared with.

    Returns:
      True if the objects are equal.
    """
    if not isinstance(other, ClassifierOptions):
      return False

    return self.to_pb2().__eq__(other.to_pb2())
