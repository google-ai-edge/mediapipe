# Copyright 2024 The MediaPipe Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""ModelWriter for writing a set of weights as binary files."""

import contextlib
import os
from typing import Dict, Tuple

from jax import numpy as jnp
import numpy as np

from mediapipe.tasks.python.genai.converter import converter_base
from mediapipe.tasks.python.genai.converter import quantization_util


@contextlib.contextmanager
def filemanager(filename: str, mode: str):
  try:
    with open(filename, mode) as f:
      yield f
  finally:
    pass


def removeprefix(s, prefix):
  """Removes the prefix from a string."""
  if s.startswith(prefix):
    return s[len(prefix) :]
  return s


class WeightBinsWriter(converter_base.ModelWriterBase):
  """A ModelWriter for writing a set of weights as binary files."""

  def get_weight_info(self, var_name: str, weight: np.ndarray) -> str:
    """Gets the string that describes the weights."""
    dtype_str = str(weight.dtype)
    shape_str = '_'.join(map(str, weight.shape))
    return f'mdl_vars.{var_name}.{dtype_str}.{shape_str}\n'

  def write_variables(self, variables: Dict[str, Tuple[np.ndarray, bool]]):
    """Writes variable to the binary files. One for each layer.

    Args:
      variables: A dictionary that maps from the target variable names to the
        quantized tensor values along with a boolean that indicates whether to
        pack the values (only applicable for the 4-bit quantized tensors).
    """
    weights_info = []
    for var_name, value in variables.items():
      output = value[0]
      if value[1]:
        # Squeeze the tensor to make sure it is a 1D array for packing.
        output = np.expand_dims(np.ravel(output), axis=-1)
        # Extra pack needed for 4 bit. We always pack the weights along the
        # first dimension since the tensor has already been squeezed.
        output = quantization_util.pack_4bit(output, 0, jnp.int8)
      if 'combined_qkv' in var_name:
        var_name = removeprefix(var_name, 'mld_vars.')
        var_name_q = var_name.replace('combined_qkv', 'q')
        var_name_k = var_name.replace('combined_qkv', 'k')
        var_name_v = var_name.replace('combined_qkv', 'v')
        if output.shape[0] == 3:
          weight_q, weight_k, weight_v = output
          assert weight_q.shape == weight_k.shape == weight_v.shape
        else:  # LoRA right weight is shared across q, k, v
          weight_q = weight_k = weight_v = output
          weights_info.append(self.get_weight_info(var_name_q, weight_q))
        path_q = os.path.join(self._output_dir, var_name_q)
        with filemanager(path_q, 'wb') as f:
          f.write(weight_q.tobytes())
          weights_info.append(self.get_weight_info(var_name_k, weight_k))
        path_k = os.path.join(self._output_dir, var_name_k)
        with filemanager(path_k, 'wb') as f:
          f.write(weight_k.tobytes())
        path_v = os.path.join(self._output_dir, var_name_v)
        with filemanager(path_v, 'wb') as f:
          f.write(weight_v.tobytes())
          weights_info.append(self.get_weight_info(var_name_v, weight_v))
      else:
        if 'key' in var_name:
          var_name = var_name.replace('key', 'k')
        if 'query' in var_name:
          var_name = var_name.replace('query', 'q')
        if 'value' in var_name:
          var_name = var_name.replace('value', 'v')
        path = os.path.join(
            self._output_dir, removeprefix(var_name, 'mdl_vars.')
        )
        with filemanager(path, 'wb') as f:
          f.write(output.tobytes())
          weights_info.append(self.get_weight_info(var_name, output))

      # Sort weights_info
      weights_info.sort()
      with filemanager(
          os.path.join(self._output_dir, 'layer_info.txt'), 'w'
      ) as finfo:
        for line in weights_info:
          finfo.write(line + '\n')
