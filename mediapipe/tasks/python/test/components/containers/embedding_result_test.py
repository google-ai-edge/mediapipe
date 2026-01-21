# Copyright 2025 The MediaPipe Authors.
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
"""Tests for EmbeddingResult conversion between Python and C."""

import ctypes

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np

from mediapipe.tasks.python.components.containers import embedding_result as embedding_result_module
from mediapipe.tasks.python.components.containers import embedding_result_c as embedding_result_c_module

_EmbeddingC = embedding_result_c_module.EmbeddingC
_EmbeddingResultC = embedding_result_c_module.EmbeddingResultC
_EmbeddingResult = embedding_result_module.EmbeddingResult


def _create_c_embedding_result(
    values,
    is_quantized: bool,
    timestamp_ms: int | None = None,
) -> _EmbeddingResultC:
  """Creates a mock C EmbeddingResultC struct."""
  c_embedding = _EmbeddingC(
      values_count=len(values),
      head_index=0,
      head_name=b"feature",
  )
  if is_quantized:
    c_quantized_array = (ctypes.c_uint8 * len(values))(*values)
    c_embedding.quantized_embedding = c_quantized_array
    c_embedding.float_embedding = None
  else:
    c_float_array = (ctypes.c_float * len(values))(*values)
    c_embedding.float_embedding = c_float_array
    c_embedding.quantized_embedding = None

  c_embeddings = (_EmbeddingC * 1)(c_embedding)
  return _EmbeddingResultC(
      embeddings=c_embeddings,
      embeddings_count=1,
      has_timestamp_ms=timestamp_ms is not None,
      timestamp_ms=timestamp_ms if timestamp_ms is not None else 0,
  )


class EmbeddingResultTest(parameterized.TestCase):

  def _assert_embedding_result(
      self,
      py_result: _EmbeddingResult,
      expected_values: list[float] | list[int],
      is_quantized: bool,
      expected_timestamp: int | None = None,
  ):
    """Asserts that the Python EmbeddingResult matches the expected values."""
    self.assertLen(py_result.embeddings, 1)
    embedding = py_result.embeddings[0]
    if is_quantized:
      np.testing.assert_array_equal(embedding.embedding, expected_values)
    else:
      np.testing.assert_array_almost_equal(embedding.embedding, expected_values)
    self.assertEqual(embedding.head_index, 0)
    self.assertEqual(embedding.head_name, "feature")
    self.assertEqual(py_result.timestamp_ms, expected_timestamp)

  def test_from_ctypes_float_embedding(self):
    float_values = [0.1, 0.2, 0.3]
    original_result = _create_c_embedding_result(
        float_values, is_quantized=False
    )
    converted_result = _EmbeddingResult.from_ctypes(original_result)
    self._assert_embedding_result(
        converted_result, float_values, is_quantized=False
    )

  def test_from_ctypes_quantized_embedding(self):
    quantized_values = [100, 200, 255]
    original_result = _create_c_embedding_result(
        quantized_values, is_quantized=True
    )
    converted_result = _EmbeddingResult.from_ctypes(original_result)
    self._assert_embedding_result(
        converted_result, quantized_values, is_quantized=True
    )

  def test_from_ctypes_quantized_embedding_contains_a_zero_value(self):
    """Tests conversion logic for strings with zero values."""
    quantized_values = [0, 100, 200]
    original_result = _create_c_embedding_result(
        quantized_values, is_quantized=True
    )
    converted_result = _EmbeddingResult.from_ctypes(original_result)
    self._assert_embedding_result(
        converted_result, quantized_values, is_quantized=True
    )

  def test_from_ctypes_with_timestamp(self):
    float_values = [0.1, 0.2]
    original_result = _create_c_embedding_result(
        float_values, is_quantized=False, timestamp_ms=12345
    )
    converted_result = _EmbeddingResult.from_ctypes(original_result)
    self._assert_embedding_result(
        converted_result,
        float_values,
        is_quantized=False,
        expected_timestamp=12345,
    )


if __name__ == "__main__":
  absltest.main()
