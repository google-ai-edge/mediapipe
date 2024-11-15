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

"""Defines a couple base classes for the conversion/quantization process."""

from typing import Iterator
import os
from typing import Dict, List, Optional, Tuple
import numpy as np


class QuantizationAction:
  """Container of the tensor values and its corresponding quantization settings.

  The container is responsible for hosting all of the information that is
  required to execute the weight-only quantization.

  Attributes:
    tensor_name: A string that represents the input tensor name.
    tensor_value: A numpy array that contains the unquantized tensor values.
    target_name: A string that represents the updated tensor name.
    quantize_axis: A list of integers representing the dimensions to be
      quantized along. For example, if an input tensor has shape [128, 256] and
      the quantize_axis==[0], it means the quantization happens along the 0-th
      dimension, resulting in [256] scaling factors.
    quantize_bits: An integer that specifies the target quantization bits. It
      currently only supports either 8 or 4 bits.
    pack_dim: An integer specifying which dimension to pack the quantized bits.
      This is only applicable when the quantize_bits == 4.
  """

  def __init__(
      self,
      tensor_name: str,
      tensor_value: Optional[np.ndarray] = None,
      target_name: Optional[str] = None,
      quantize_axis: Optional[List[int]] = None,
      quantize_bits: Optional[int] = None,
      pack_dim: Optional[int] = 0,
  ):
    """Initializes the model attributes."""
    self.tensor_name = tensor_name
    self.tensor_value = tensor_value
    self.target_name = target_name
    self.quantize_axis = quantize_axis
    self.quantize_bits = quantize_bits
    self.pack_dim = pack_dim

  def __str__(self) -> str:
    output_string = "QuantizationAction(\n"
    output_string += f"  tensor_name: {self.tensor_name}\n"
    output_string += f"  target_name: {self.target_name}\n"
    output_string += f"  quantize_axis: {self.quantize_axis}\n"
    output_string += f"  quantize_bits: {self.quantize_bits}\n"
    output_string += f"  pack_dim: {self.pack_dim}\n"
    if self.tensor_value is not None:
      output_string += f"  tensor_value: {self.tensor_value.shape}\n"
    output_string += ")\n"
    return output_string


class CkptLoaderBase:
  """Base class for loading the checkpoint.

  This class is responsible for loading the checkpoint files into the layer
  weight tensors (as numpy arrays) + quantization setting information (8/4
  bits). The returned data should be a list of QuantizationAction that describes
  how to quantize each layer weights.
  """

  def __init__(
      self,
      ckpt_path: str,
      is_symmetric: bool,
      attention_quant_bits: int,
      feedforward_quant_bits: int,
      embedding_quant_bits: int,
  ):
    """Initializes the loader.

    Args:
      ckpt_path: The filepath to the checkpoint.
      is_symmetric: Whether to apply symmetric or asymmetric quantization.
      attention_quant_bits: An integer that specify the target quantization bits
        (support 8 or 4) for the attention layers.
      feedforward_quant_bits: An integer that specify the target quantization
        bits (support 8 or 4) for the feedforward layers in each Transformer
        blocks.
      embedding_quant_bits: An integer that specify the target quantization bits
        (support 8 or 4) for the embedding (and the final projection) layers.
    """
    self._ckpt_path = ckpt_path
    self._is_symmetric = is_symmetric
    self._attention_quant_bits = attention_quant_bits
    self._feedforward_quant_bits = feedforward_quant_bits
    self._embedding_quant_bits = embedding_quant_bits

  def load_to_actions(
      self,
  ) -> Iterator[Optional[List[QuantizationAction]]]:
    """Loads the checkpoint and returns the quantization actions."""
    raise NotImplementedError("The load_to_actions method is not implemented.")


class LayerActionMapperBase:
  """Base class for mapping the layer weights to quantization actions.

  This class is responsible for mapping from each layer to its corresponding
  quantization information (e.g. target quantization bits / updated tensor
  name...).
  """

  def __init__(
      self,
      is_symmetric: bool,
      attention_quant_bits: int,
      feedforward_quant_bits: int,
      embedding_quant_bits: int,
      backend: str,
  ):
    self._is_symmetric = is_symmetric
    self._attention_quant_bits = attention_quant_bits
    self._feedforward_quant_bits = feedforward_quant_bits
    self._embedding_quant_bits = embedding_quant_bits
    self._backend = backend

  def map_to_actions(
      self, layer_name: str
  ) -> Optional[List[QuantizationAction]]:
    """Maps the layer weights to quantization actions.

    Args:
      layer_name: A string representing the name of the layer weight. Note that
        it is expected the layer information is contained in the name which is
        enough to determine the target quantization information. Any child class
        is expected to implement this function.
    """
    raise NotImplementedError("The map_to_actions method is not implemented.")


class ModelWriterBase:
  """Base class for writing the quantized model.

  This class is responsible for taking a dictionary of the quantized
  tensors/names and writing them into the format that can be loaded by the
  on-device inference engine.
  """

  def __init__(self, output_dir: str, backend: str):
    """Initializes the class.

    Args:
      output_dir: A string that represents the output directory to write the
        resulting file(s).
      backend: A string that represents the target backend to run the output
        file(s).
    """
    self._output_dir = output_dir
    if not os.path.exists(self._output_dir):
      os.mkdir(self._output_dir)
    self._backend = backend

  def write_variables(
      self,
      variables: Dict[str, Tuple[np.ndarray, bool]],
      use_fake_values: bool = False,
  ):
    raise NotImplementedError("The write_variables method is not implemented.")
