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

"""C types for Matrix."""

import ctypes

import numpy as np

from mediapipe.tasks.python.core.optional_dependencies import doc_controls


class MatrixC(ctypes.Structure):
  """The ctypes struct for Matrix."""

  _fields_ = [
      ('rows', ctypes.c_uint32),
      ('cols', ctypes.c_uint32),
      ('data', ctypes.POINTER(ctypes.c_float)),
  ]

  @doc_controls.do_not_generate_docs
  def to_numpy(self) -> np.ndarray:
    """Copies the MatrixC struct to a numpy array.

    The data is reshaped from column-major order to (cols, rows) and then
    transposed to a row-major numpy array. The underlying data is copied to
    ensure the data is not freed when the underlying C Matrix is closed.

    Returns:
      A numpy array representing the MatrixC struct.
    """
    if not self.data or self.rows == 0 or self.cols == 0:
      return np.empty((0,))

    np_array = np.ctypeslib.as_array(
        self.data, shape=(self.cols, self.rows)
    ).copy()
    return np_array.T
