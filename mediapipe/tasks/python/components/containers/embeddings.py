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
_EmbeddingEntryProto = embeddings_pb2.EmbeddingEntry
_EmbeddingsProto = embeddings_pb2.Embeddings
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
class EmbeddingEntry:
  """Floating-point or scalar-quantized embedding with an optional timestamp.

  Attributes:
    embedding: The actual embedding, either floating-point or scalar-quantized.
    timestamp_ms: The optional timestamp (in milliseconds) associated to the
      embedding entry. This is useful for time series use cases, e.g. audio
      embedding.
  """

  embedding: np.ndarray
  timestamp_ms: Optional[int] = None

  @doc_controls.do_not_generate_docs
  def to_pb2(self) -> _EmbeddingEntryProto:
    """Generates a EmbeddingEntry protobuf object."""

    if self.embedding.dtype == float:
      return _EmbeddingEntryProto(float_embedding=self.embedding)

    elif self.embedding.dtype == np.uint8:
      return _EmbeddingEntryProto(quantized_embedding=bytes(self.embedding))

    else:
      raise ValueError("Invalid dtype. Only float and np.uint8 are supported.")

  @classmethod
  @doc_controls.do_not_generate_docs
  def create_from_pb2(
      cls, pb2_obj: _EmbeddingEntryProto) -> 'EmbeddingEntry':
    """Creates a `EmbeddingEntry` object from the given protobuf object."""

    if pb2_obj.float_embedding:
      return EmbeddingEntry(
          embedding=np.array(pb2_obj.float_embedding.values, dtype=float))

    elif pb2_obj.quantized_embedding:
      return EmbeddingEntry(
          embedding=np.array(bytearray(pb2_obj.quantized_embedding.values),
                             dtype=np.uint8))

    else:
      raise ValueError("Either float_embedding or quantized_embedding must "
                       "exist.")

  def __eq__(self, other: Any) -> bool:
    """Checks if this object is equal to the given object.
    Args:
      other: The object to be compared with.
    Returns:
      True if the objects are equal.
    """
    if not isinstance(other, EmbeddingEntry):
      return False

    return self.to_pb2().__eq__(other.to_pb2())


@dataclasses.dataclass
class Embeddings:
  """Embeddings for a given embedder head.
  Attributes:
    entries: A list of `ClassificationEntry` objects.
    head_index: The index of the embedder head that produced this embedding.
      This is useful for multi-head models.
    head_name: The name of the embedder head, which is the corresponding tensor
      metadata name (if any). This is useful for multi-head models.
  """

  entries: List[EmbeddingEntry]
  head_index: int
  head_name: str

  @doc_controls.do_not_generate_docs
  def to_pb2(self) -> _EmbeddingsProto:
    """Generates a Embeddings protobuf object."""
    return _EmbeddingsProto(
        entries=[entry.to_pb2() for entry in self.entries],
        head_index=self.head_index,
        head_name=self.head_name)

  @classmethod
  @doc_controls.do_not_generate_docs
  def create_from_pb2(cls, pb2_obj: _EmbeddingsProto) -> 'Embeddings':
    """Creates a `Embeddings` object from the given protobuf object."""
    return Embeddings(
        entries=[
            EmbeddingEntry.create_from_pb2(entry)
            for entry in pb2_obj.entries
        ],
        head_index=pb2_obj.head_index,
        head_name=pb2_obj.head_name)

  def __eq__(self, other: Any) -> bool:
    """Checks if this object is equal to the given object.
    Args:
      other: The object to be compared with.
    Returns:
      True if the objects are equal.
    """
    if not isinstance(other, Embeddings):
      return False

    return self.to_pb2().__eq__(other.to_pb2())


@dataclasses.dataclass
class EmbeddingResult:
  """Contains one set of results per embedder head.
  Attributes:
    embeddings: A list of `Embeddings` objects.
  """

  embeddings: List[Embeddings]

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
            Embeddings.create_from_pb2(embedding)
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
