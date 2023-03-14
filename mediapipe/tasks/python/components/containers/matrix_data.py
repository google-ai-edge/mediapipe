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
"""Matrix data data class."""

import dataclasses
import enum
from typing import Any, Optional

import numpy as np
from mediapipe.framework.formats import matrix_data_pb2
from mediapipe.tasks.python.core.optional_dependencies import doc_controls

_MatrixDataProto = matrix_data_pb2.MatrixData


@dataclasses.dataclass
class MatrixData:
  """This stores the Matrix data.

  Here the data is stored in column-major order by default.

  Attributes:
    rows: The number of rows in the matrix.
    cols: The number of columns in the matrix.
    data: The data stored in the matrix as a NumPy array.
    layout: The order in which the data are stored. Defaults to COLUMN_MAJOR.
  """

  class Layout(enum.Enum):
    COLUMN_MAJOR = 0
    ROW_MAJOR = 1

  rows: int = None
  cols: int = None
  data: np.ndarray = None
  layout: Optional[Layout] = Layout.COLUMN_MAJOR

  @doc_controls.do_not_generate_docs
  def to_pb2(self) -> _MatrixDataProto:
    """Generates a MatrixData protobuf object."""
    return _MatrixDataProto(
        rows=self.rows,
        cols=self.cols,
        data=self.data.tolist(),
        layout=self.layout)

  @classmethod
  @doc_controls.do_not_generate_docs
  def create_from_pb2(cls, pb2_obj: _MatrixDataProto) -> 'MatrixData':
    """Creates a `MatrixData` object from the given protobuf object."""
    return MatrixData(
        rows=pb2_obj.rows,
        cols=pb2_obj.cols,
        data=np.array(pb2_obj.data),
        layout=pb2_obj.layout)

  def __eq__(self, other: Any) -> bool:
    """Checks if this object is equal to the given object.

    Args:
      other: The object to be compared with.

    Returns:
      True if the objects are equal.
    """
    if not isinstance(other, MatrixData):
      return False

    return self.to_pb2().__eq__(other.to_pb2())
