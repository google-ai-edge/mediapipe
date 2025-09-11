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
"""Embeddings C API types."""

import ctypes

import numpy as np

from mediapipe.tasks.python.components.containers import embedding_result as embedding_result_module


class EmbeddingC(ctypes.Structure):
  """C struct for an embedding."""

  _fields_ = [
      ('float_embedding', ctypes.POINTER(ctypes.c_float)),
      ('quantized_embedding', ctypes.c_char_p),
      ('values_count', ctypes.c_uint32),
      ('head_index', ctypes.c_int),
      ('head_name', ctypes.c_char_p),
  ]


class EmbeddingResultC(ctypes.Structure):
  """C struct for embedding result."""

  _fields_ = [
      ('embeddings', ctypes.POINTER(EmbeddingC)),
      ('embeddings_count', ctypes.c_uint32),
      ('has_timestamp_ms', ctypes.c_bool),
      ('timestamp_ms', ctypes.c_int64),
  ]


def convert_to_python_embedding_result(
    c_result: EmbeddingResultC,
) -> embedding_result_module.EmbeddingResult:
  """Converts a C EmbeddingResult to a Python EmbeddingResult."""
  python_embeddings = []
  for i in range(c_result.embeddings_count):
    c_embedding = c_result.embeddings[i]
    emedding = None
    if c_embedding.float_embedding:
      emedding_array = list(
          c_embedding.float_embedding[: c_embedding.values_count]
      )
      emedding = np.array(emedding_array, dtype=float)
    if c_embedding.quantized_embedding:
      emedding_array = bytearray(
          c_embedding.quantized_embedding[: c_embedding.values_count]
      )
      emedding = np.array(emedding_array)

    python_embedding = embedding_result_module.Embedding(
        embedding=emedding,
        head_index=c_embedding.head_index,
        head_name=(
            c_embedding.head_name.decode('utf-8')
            if c_embedding.head_name
            else None
        ),
    )
    python_embeddings.append(python_embedding)
  timestamp_ms = c_result.timestamp_ms if c_result.has_timestamp_ms else None
  return embedding_result_module.EmbeddingResult(
      embeddings=python_embeddings, timestamp_ms=timestamp_ms
  )
