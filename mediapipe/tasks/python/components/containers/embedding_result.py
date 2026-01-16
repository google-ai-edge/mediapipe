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
"""Embeddings data class."""

import dataclasses
from typing import List, Optional

import numpy as np

from mediapipe.tasks.python.components.containers import embedding_result_c
from mediapipe.tasks.python.core.optional_dependencies import doc_controls


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
  def from_ctypes(
      cls, c_result: embedding_result_c.EmbeddingResultC
  ) -> 'EmbeddingResult':
    """Converts a C EmbeddingResult to a Python EmbeddingResult."""
    python_embeddings = []
    for i in range(c_result.embeddings_count):
      c_embedding = c_result.embeddings[i]
      embedding = None
      if c_embedding.float_embedding:
        embedding_array = list(
            c_embedding.float_embedding[: c_embedding.values_count]
        )
        embedding = np.array(embedding_array, dtype=float)
      if c_embedding.quantized_embedding:
        embedding_array = list(
            c_embedding.quantized_embedding[: c_embedding.values_count]
        )
        embedding = np.array(embedding_array, dtype=np.uint8)

      python_embedding = Embedding(
          embedding=embedding,
          head_index=c_embedding.head_index,
          head_name=(
              c_embedding.head_name.decode('utf-8')
              if c_embedding.head_name
              else None
          ),
      )
      python_embeddings.append(python_embedding)
    timestamp_ms = c_result.timestamp_ms if c_result.has_timestamp_ms else None
    return EmbeddingResult(
        embeddings=python_embeddings, timestamp_ms=timestamp_ms
    )
