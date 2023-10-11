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
"""Cosine similarity utilities."""

import numpy as np

from mediapipe.tasks.python.components.containers import embedding_result

_Embedding = embedding_result.Embedding


def _compute_cosine_similarity(u: np.ndarray, v: np.ndarray):
  """Computes cosine similarity between two embeddings."""

  if len(u) <= 0:
    raise ValueError("Cannot compute cosing similarity on empty embeddings.")

  norm_u = np.linalg.norm(u)
  norm_v = np.linalg.norm(v)

  if norm_u <= 0 or norm_v <= 0:
    raise ValueError(
        "Cannot compute cosine similarity on embedding with 0 norm.")

  return u.dot(v) / (norm_u * norm_v)


def cosine_similarity(u: _Embedding, v: _Embedding) -> float:
  """Utility function to compute cosine similarity between two embedding.

  May return an InvalidArgumentError if e.g. the feature vectors are of
  different types (quantized vs. float), have different sizes, or have an
  L2-norm of 0.

  Args:
    u: An embedding.
    v: An embedding.

  Returns:
    Cosine similarity value.
  """
  if len(u.embedding) != len(v.embedding):
    raise ValueError(f"Cannot compute cosine similarity between embeddings "
                     f"of different sizes "
                     f"({len(u.embedding)} vs. {len(v.embedding)}).")

  if u.embedding.dtype == float and v.embedding.dtype == float:
    return _compute_cosine_similarity(u.embedding, v.embedding)

  if u.embedding.dtype == np.uint8 and v.embedding.dtype == np.uint8:
    return _compute_cosine_similarity(
        u.embedding.view("int8").astype("float"),
        v.embedding.view("int8").astype("float"),
    )

  raise ValueError("Cannot compute cosine similarity between quantized and "
                   "float embeddings.")
