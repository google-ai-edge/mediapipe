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
"""Tests for Matrix conversion between Python and C."""

import ctypes
import unittest

import numpy as np

from mediapipe.tasks.python.components.containers import matrix_c as matrix_c_lib


class MatrixTest(unittest.TestCase):

  def test_to_numpy_conversion(self):
    c_data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    c_data_array = (ctypes.c_float * len(c_data))(*c_data)
    matrix_c = matrix_c_lib.MatrixC(rows=2, cols=3, data=c_data_array)

    np_array = matrix_c.to_numpy()

    expected_np_array = np.array([[1.0, 3.0, 5.0], [2.0, 4.0, 6.0]])
    np.testing.assert_array_almost_equal(np_array, expected_np_array)

  def test_to_numpy_empty(self):
    matrix_c = matrix_c_lib.MatrixC(rows=0, cols=0, data=None)
    np_array = matrix_c.to_numpy()
    self.assertEqual(np_array.shape, (0,))

  def test_to_numpy_no_data(self):
    matrix_c = matrix_c_lib.MatrixC(rows=2, cols=3, data=None)
    np_array = matrix_c.to_numpy()
    self.assertEqual(np_array.shape, (0,))

  def test_to_numpy_data_is_copied(self):
    # Test that the C data is copied, since the C data is not owned by the
    # MatrixC instance but rather by the MediaPipe C API, which frees the data
    # once the result is processed.
    original_c_data = [1.0, 2.0, 3.0, 4.0]
    c_data_array = (ctypes.c_float * len(original_c_data))(*original_c_data)
    matrix_c = matrix_c_lib.MatrixC(rows=2, cols=2, data=c_data_array)

    # Convert to numpy.
    np_array = matrix_c.to_numpy()

    # Modify the original C data.
    c_data_array[0] = 0.0

    # The numpy array should still be the same.
    expected_np_array = np.array([[1.0, 3.0], [2.0, 4.0]])
    np.testing.assert_array_almost_equal(np_array, expected_np_array)


if __name__ == '__main__':
  unittest.main()
