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
"""Embedder options data class."""

import dataclasses
from typing import Any, Optional

from mediapipe.tasks.cc.components.processors.proto import embedder_options_pb2
from mediapipe.tasks.python.core.optional_dependencies import doc_controls

_EmbedderOptionsProto = embedder_options_pb2.EmbedderOptions


@dataclasses.dataclass
class EmbedderOptions:
  """Shared options used by all embedding extraction tasks.

  Attributes:
    l2_normalize: Whether to normalize the returned feature vector with L2 norm.
      Use this option only if the model does not already contain a native
      L2_NORMALIZATION TF Lite Op. In most cases, this is already the case and
      L2 norm is thus achieved through TF Lite inference.
    quantize: Whether the returned embedding should be quantized to bytes via
      scalar quantization. Embeddings are implicitly assumed to be unit-norm and
      therefore any dimension is guaranteed to have a value in [-1.0, 1.0]. Use
      the l2_normalize option if this is not the case.
  """

  l2_normalize: Optional[bool] = None
  quantize: Optional[bool] = None

  @doc_controls.do_not_generate_docs
  def to_pb2(self) -> _EmbedderOptionsProto:
    """Generates a EmbedderOptions protobuf object."""
    return _EmbedderOptionsProto(
        l2_normalize=self.l2_normalize, quantize=self.quantize)

  @classmethod
  @doc_controls.do_not_generate_docs
  def create_from_pb2(cls, pb2_obj: _EmbedderOptionsProto) -> 'EmbedderOptions':
    """Creates a `EmbedderOptions` object from the given protobuf object."""
    return EmbedderOptions(
        l2_normalize=pb2_obj.l2_normalize, quantize=pb2_obj.quantize)

  def __eq__(self, other: Any) -> bool:
    """Checks if this object is equal to the given object.

    Args:
      other: The object to be compared with.

    Returns:
      True if the objects are equal.
    """
    if not isinstance(other, EmbedderOptions):
      return False

    return self.to_pb2().__eq__(other.to_pb2())
