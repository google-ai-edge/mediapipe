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
# ==============================================================================
"""Helper methods for writing metadata into TFLite models."""

from typing import List

from mediapipe.tasks.metadata import schema_py_generated as _schema_fb


def get_input_tensor_names(model_buffer: bytearray) -> List[str]:
  """Gets a list of the input tensor names."""
  subgraph = get_subgraph(model_buffer)
  tensor_names = []
  for i in range(subgraph.InputsLength()):
    index = subgraph.Inputs(i)
    tensor_names.append(subgraph.Tensors(index).Name().decode("utf-8"))
  return tensor_names


def get_output_tensor_names(model_buffer: bytearray) -> List[str]:
  """Gets a list of the output tensor names."""
  subgraph = get_subgraph(model_buffer)
  tensor_names = []
  for i in range(subgraph.OutputsLength()):
    index = subgraph.Outputs(i)
    tensor_names.append(subgraph.Tensors(index).Name().decode("utf-8"))
  return tensor_names


def get_input_tensor_types(
    model_buffer: bytearray) -> List[_schema_fb.TensorType]:
  """Gets a list of the input tensor types."""
  subgraph = get_subgraph(model_buffer)
  tensor_types = []
  for i in range(subgraph.InputsLength()):
    index = subgraph.Inputs(i)
    tensor_types.append(subgraph.Tensors(index).Type())
  return tensor_types


def get_output_tensor_types(
    model_buffer: bytearray) -> List[_schema_fb.TensorType]:
  """Gets a list of the output tensor types."""
  subgraph = get_subgraph(model_buffer)
  tensor_types = []
  for i in range(subgraph.OutputsLength()):
    index = subgraph.Outputs(i)
    tensor_types.append(subgraph.Tensors(index).Type())
  return tensor_types


def get_output_tensor_indices(model_buffer: bytearray) -> List[int]:
  """Gets a list of the output tensor indices."""
  subgraph = get_subgraph(model_buffer)
  return subgraph.OutputsAsNumpy()


def get_subgraph(model_buffer: bytearray) -> _schema_fb.SubGraph:
  """Gets the subgraph of the model.

  TFLite does not support multi-subgraph. A model should have exactly one
  subgraph.

  Args:
    model_buffer: valid buffer of the model file.

  Returns:
    The subgraph of the model.

  Raises:
    ValueError: if the model has more than one subgraph or has no subgraph.
  """

  model = _schema_fb.Model.GetRootAsModel(model_buffer, 0)

  # Use the first subgraph as default. TFLite Interpreter doesn't support
  # multiple subgraphs yet, but models with mini-benchmark may have multiple
  # subgraphs for acceleration evaluation purpose.
  return model.Subgraphs(0)
