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
"""Embeddings data class."""

import dataclasses
from typing import Any, Optional, List

import numpy as np
from mediapipe.tasks.cc.components.containers.proto import embeddings_pb2
from mediapipe.tasks.python.core.optional_dependencies import doc_controls

_FloatEmbeddingProto = embeddings_pb2.FloatEmbedding
_QuantizedEmbeddingProto = embeddings_pb2.QuantizedEmbedding
_EmbeddingProto = embeddings_pb2.Embedding
_EmbeddingResultProto = embeddings_pb2.EmbeddingResult


@dataclasses.dataclass
class FloatEmbedding:
  """Defines a dense floating-point embedding.

  Attributes:
    values: A NumPy array indicating the raw output of the embedding layer.
  """

  values: np.ndarray

  @doc_controls.do_not_generate_docs
  def to_pb2(self) -> _FloatEmbeddingProto:
    """Generates a FloatEmbedding protobuf object."""
    return _FloatEmbeddingProto(values=self.values)

  @classmethod
  @doc_controls.do_not_generate_docs
  def create_from_pb2(
      cls, pb2_obj: _FloatEmbeddingProto) -> 'FloatEmbedding':
    """Creates a `FloatEmbedding` object from the given protobuf object."""
    return FloatEmbedding(values=np.array(pb2_obj.value_float, dtype=float))

  def __eq__(self, other: Any) -> bool:
    """Checks if this object is equal to the given object.
    Args:
      other: The object to be compared with.
    Returns:
      True if the objects are equal.
    """
    if not isinstance(other, FloatEmbedding):
      return False

    return self.to_pb2().__eq__(other.to_pb2())


@dataclasses.dataclass
class QuantizedEmbedding:
  """Defines a dense scalar-quantized embedding.

  Attributes:
    values: A NumPy array indicating the raw output of the embedding layer.
  """

  values: np.ndarray

  @doc_controls.do_not_generate_docs
  def to_pb2(self) -> _QuantizedEmbeddingProto:
    """Generates a QuantizedEmbedding protobuf object."""
    return _QuantizedEmbeddingProto(values=self.values)

  @classmethod
  @doc_controls.do_not_generate_docs
  def create_from_pb2(
      cls, pb2_obj: _QuantizedEmbeddingProto) -> 'QuantizedEmbedding':
    """Creates a `QuantizedEmbedding` object from the given protobuf object."""
    return QuantizedEmbedding(
        values=np.array(bytearray(pb2_obj.value_string), dtype=np.uint8))

  def __eq__(self, other: Any) -> bool:
    """Checks if this object is equal to the given object.
    Args:
      other: The object to be compared with.
    Returns:
      True if the objects are equal.
    """
    if not isinstance(other, QuantizedEmbedding):
      return False

    return self.to_pb2().__eq__(other.to_pb2())


@dataclasses.dataclass
class Embedding:
  """Embedding result for a given embedder head.

  Attributes:
    embedding: The actual embedding, either floating-point or scalar-quantized.
    head_index: The index of the embedder head that produced this embedding.
      This is useful for multi-head models.
    head_name: The name of the embedder head, which is the corresponding tensor
      metadata name (if any). This is useful for multi-head models.
  """

  embedding: np.ndarray
  head_index: Optional[int] = None
  head_name: Optional[str] = None

  @doc_controls.do_not_generate_docs
  def to_pb2(self) -> _EmbeddingProto:
    """Generates a Embedding protobuf object."""

    if self.embedding.dtype == float:
      return _EmbeddingProto(float_embedding=self.embedding,
                             head_index=self.head_index,
                             head_name=self.head_name)

    elif self.embedding.dtype == np.uint8:
      return _EmbeddingProto(quantized_embedding=bytes(self.embedding),
                             head_index=self.head_index,
                             head_name=self.head_name)

    else:
      raise ValueError("Invalid dtype. Only float and np.uint8 are supported.")

  @classmethod
  @doc_controls.do_not_generate_docs
  def create_from_pb2(
      cls, pb2_obj: _EmbeddingProto) -> 'Embedding':
    """Creates a `Embedding` object from the given protobuf object."""

    quantized_embedding = np.array(
        bytearray(pb2_obj.quantized_embedding.values))
    float_embedding = np.array(pb2_obj.float_embedding.values, dtype=float)

    if len(quantized_embedding) == 0:
      return Embedding(embedding=float_embedding,
                       head_index=pb2_obj.head_index,
                       head_name=pb2_obj.head_name)
    else:
      return Embedding(embedding=quantized_embedding,
                       head_index=pb2_obj.head_index,
                       head_name=pb2_obj.head_name)

  def __eq__(self, other: Any) -> bool:
    """Checks if this object is equal to the given object.
    Args:
      other: The object to be compared with.
    Returns:
      True if the objects are equal.
    """
    if not isinstance(other, Embedding):
      return False

    return self.to_pb2().__eq__(other.to_pb2())


@dataclasses.dataclass
class EmbeddingResult:
  """Embedding results for a given embedder model.
  Attributes:
    embeddings: A list of `Embedding` objects.
    timestamp_ms: The optional timestamp (in milliseconds) of the start of the
      chunk of data corresponding to these results. This is only used for
      embedding extraction on time series (e.g. audio embedding). In these use
      cases, the amount of data to process might exceed the maximum size that
      the model can process: to solve this, the input data is split into
      multiple chunks starting at different timestamps.
  """

  embeddings: List[Embedding]
  timestamp_ms: Optional[int] = None

  @doc_controls.do_not_generate_docs
  def to_pb2(self) -> _EmbeddingResultProto:
    """Generates a EmbeddingResult protobuf object."""
    return _EmbeddingResultProto(
        embeddings=[
            embedding.to_pb2() for embedding in self.embeddings
        ])

  @classmethod
  @doc_controls.do_not_generate_docs
  def create_from_pb2(
      cls, pb2_obj: _EmbeddingResultProto) -> 'EmbeddingResult':
    """Creates a `EmbeddingResult` object from the given protobuf object."""
    return EmbeddingResult(
        embeddings=[
            Embedding.create_from_pb2(embedding)
            for embedding in pb2_obj.embeddings
        ])

  def __eq__(self, other: Any) -> bool:
    """Checks if this object is equal to the given object.
    Args:
      other: The object to be compared with.
    Returns:
      True if the objects are equal.
    """
    if not isinstance(other, EmbeddingResult):
      return False

    return self.to_pb2().__eq__(other.to_pb2())
