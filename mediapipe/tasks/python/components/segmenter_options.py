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
"""Segmenter options data class."""

import dataclasses
import enum
from typing import Any, Optional

from mediapipe.tasks.cc.components import segmenter_options_pb2
from mediapipe.tasks.python.core.optional_dependencies import doc_controls

_SegmenterOptionsProto = segmenter_options_pb2.SegmenterOptions


class OutputType(enum.Enum):
  UNSPECIFIED = 0
  CATEGORY_MASK = 1
  CONFIDENCE_MASK = 2


class Activation(enum.Enum):
  NONE = 0
  SIGMOID = 1
  SOFTMAX = 2


@dataclasses.dataclass
class SegmenterOptions:
  """Options for segmentation processor.
  Attributes:
    output_type: The output mask type allows specifying the type of
      post-processing to perform on the raw model results.
    activation: Activation function to apply to input tensor.
  """

  output_type: Optional[OutputType] = OutputType.CATEGORY_MASK
  activation: Optional[Activation] = Activation.NONE

  @doc_controls.do_not_generate_docs
  def to_pb2(self) -> _SegmenterOptionsProto:
    """Generates a protobuf object to pass to the C++ layer."""
    return _SegmenterOptionsProto(
        output_type=self.output_type.value,
        activation=self.activation.value
    )

  @classmethod
  @doc_controls.do_not_generate_docs
  def create_from_pb2(
      cls, pb2_obj: _SegmenterOptionsProto) -> "SegmenterOptions":
    """Creates a `SegmenterOptions` object from the given protobuf object."""
    return SegmenterOptions(
        output_type=OutputType(pb2_obj.output_type),
        activation=Activation(pb2_obj.output_type)
    )

  def __eq__(self, other: Any) -> bool:
    """Checks if this object is equal to the given object.
    Args:
      other: The object to be compared with.
    Returns:
      True if the objects are equal.
    """
    if not isinstance(other, SegmenterOptions):
      return False

    return self.to_pb2().__eq__(other.to_pb2())
