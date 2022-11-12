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
from typing import Optional, List

import numpy as np
from mediapipe.tasks.cc.components.containers.proto import embeddings_pb2
from mediapipe.tasks.python.core.optional_dependencies import doc_controls

_FloatEmbeddingProto = embeddings_pb2.FloatEmbedding
_QuantizedEmbeddingProto = embeddings_pb2.QuantizedEmbedding
_EmbeddingProto = embeddings_pb2.Embedding
_EmbeddingResultProto = embeddings_pb2.EmbeddingResult


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

  @classmethod
  @doc_controls.do_not_generate_docs
  def create_from_pb2(cls, pb2_obj: _EmbeddingProto) -> 'Embedding':
    """Creates a `Embedding` object from the given protobuf object."""

    quantized_embedding = np.array(
        bytearray(pb2_obj.quantized_embedding.values))
    float_embedding = np.array(pb2_obj.float_embedding.values, dtype=float)

    if not pb2_obj.quantized_embedding.values:
      return Embedding(
          embedding=float_embedding,
          head_index=pb2_obj.head_index,
          head_name=pb2_obj.head_name)
    else:
      return Embedding(
          embedding=quantized_embedding,
          head_index=pb2_obj.head_index,
          head_name=pb2_obj.head_name)


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

  @classmethod
  @doc_controls.do_not_generate_docs
  def create_from_pb2(cls, pb2_obj: _EmbeddingResultProto) -> 'EmbeddingResult':
    """Creates a `EmbeddingResult` object from the given protobuf object."""
    return EmbeddingResult(embeddings=[
        Embedding.create_from_pb2(embedding) for embedding in pb2_obj.embeddings
    ])
