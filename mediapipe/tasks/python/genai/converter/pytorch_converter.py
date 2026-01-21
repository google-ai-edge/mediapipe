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

"""CkptLoader implementation for loading the Pytorch file."""

from typing import Iterator
import enum
import os
from typing import List, Optional

import numpy as np

from mediapipe.tasks.python.genai.converter import converter_base
from mediapipe.tasks.python.genai.converter import external_dependencies

torch = external_dependencies.torch


class _PytorchReader:
  """Pytorch reader."""

  def __init__(self, model_path: str):
    if not os.path.exists(model_path):
      raise ValueError(f"{model_path} does not exists.")
    self._model = torch.load(model_path, map_location=torch.device("cpu"))

  def read_tensor_as_numpy(self, tensor_name) -> np.ndarray:
    tensor = (
        self._model[tensor_name]
        .to(torch.float32)
        .t()
        .contiguous()
        .detach()
        .cpu()
        .numpy()
    )
    return tensor

  def get_tensor_names(self) -> List[str]:
    names = list(self._model.keys())
    return names


class LayerType(enum.Enum):
  """Enum for layer type."""

  NONE = 0
  ATTENTION = 1  # Layer is part of the attention module.
  FEEDFORWARD = 2  # Layer is part of the feedforward module in the Transformer.
  EMBEDDING = 3  # Layer is the embedding lookup or final projection layer.
  LAYER_NORM = (
      4  # Layer is layer normalization before and after attention layer.
  )

  @classmethod
  def get_layer_type(cls, layer_name: str):
    """Gets the layer type of the given layer name."""
    ffn_layers = [
        "mlp",
    ]
    attn_layers = [
        "self_attention",
    ]
    emb_layers = [
        "word_embeddings",
        "lm_head",
    ]
    layer_norms = [
        "input_layernorm",
        "post_attention_layernorm",
        "ln_f",
    ]
    if any(sub_name in layer_name for sub_name in attn_layers):
      return LayerType.ATTENTION
    if any(sub_name in layer_name for sub_name in ffn_layers):
      return LayerType.FEEDFORWARD
    if any(sub_name in layer_name for sub_name in emb_layers):
      return LayerType.EMBEDDING
    if any(sub_name in layer_name for sub_name in layer_norms):
      return LayerType.LAYER_NORM
    else:
      return LayerType.NONE


class FalconMapper(converter_base.LayerActionMapperBase):
  """LayerActionMapper for handling the Falcon-rw-1b model."""

  def __init__(
      self,
      is_symmetric: bool,
      attention_quant_bits: int,
      feedforward_quant_bits: int,
      embedding_quant_bits: int,
      backend: str,
      reader: _PytorchReader,
  ):
    super().__init__(
        is_symmetric=is_symmetric,
        attention_quant_bits=attention_quant_bits,
        feedforward_quant_bits=feedforward_quant_bits,
        embedding_quant_bits=embedding_quant_bits,
        backend=backend,
    )
    self._reader = reader

  def map_to_actions(
      self, layer_name: str
  ) -> Optional[List[converter_base.QuantizationAction]]:
    """Map the given layer name to actions."""
    actions = []
    tensor_value = self._reader.read_tensor_as_numpy(layer_name)
    if "query_key_value" in layer_name:
      qkv_tensors = self._decompose_falcon_qkv(tensor_value)
      for tensor, name in zip(qkv_tensors, ["q", "k", "v"]):
        decomposed_name = layer_name.replace("query_key_value", name)
        action = self._map_to_action_helper(tensor, decomposed_name)
        actions.append(action)
    else:
      actions.append(self._map_to_action_helper(tensor_value, layer_name))
    return actions

  def _map_to_action_helper(
      self, tensor_value: np.ndarray, layer_name: str
  ) -> converter_base.QuantizationAction:
    quantize_axis = None
    quantize_bits = None
    layer_type = LayerType.get_layer_type(layer_name)

    if layer_type != LayerType.LAYER_NORM and layer_name.endswith(".weight"):
      layer_type = LayerType.get_layer_type(layer_name)
      quantize_axis = [0]
      if layer_type == LayerType.FEEDFORWARD:
        quantize_bits = self._feedforward_quant_bits
      elif layer_type == LayerType.ATTENTION:
        quantize_bits = self._attention_quant_bits
        if self._backend == "cpu" and ".dense." in layer_name:
          tensor_value = np.transpose(tensor_value)
          quantize_axis = [1]
      elif layer_type == LayerType.EMBEDDING:
        quantize_bits = self._embedding_quant_bits
        if self._backend == "cpu" and "word_embeddings" in layer_name:
          tensor_value = np.transpose(tensor_value)
          quantize_axis = [1]
    target_name = self.update_target_name(layer_name)

    return converter_base.QuantizationAction(
        tensor_name=layer_name,
        tensor_value=tensor_value,
        target_name=target_name,
        quantize_axis=quantize_axis,
        quantize_bits=quantize_bits,
        pack_dim=0,
    )

  def _decompose_falcon_qkv(self, tensor_value: np.ndarray) -> List[np.ndarray]:
    """Decomposes combined qkv tensor used in falcon model into separate q, k and v tensors."""
    chunk_size = 64
    hidden_size = 2048

    tensor_value = tensor_value.transpose()

    q_tensor = np.zeros(
        (hidden_size,)
        + ((hidden_size,) if len(tensor_value.shape) == 2 else ()),
        dtype=tensor_value.dtype,
    )
    k_tensor = np.zeros_like(q_tensor, dtype=tensor_value.dtype)
    v_tensor = np.zeros_like(k_tensor, dtype=tensor_value.dtype)

    j = 0
    for i in range(0 * chunk_size, hidden_size * 3, chunk_size * 3):
      q_tensor[j : j + chunk_size] = tensor_value[i : i + chunk_size]
      j += chunk_size

    j = 0
    for i in range(1 * chunk_size, hidden_size * 3, chunk_size * 3):
      k_tensor[j : j + chunk_size] = tensor_value[i : i + chunk_size]
      j += chunk_size

    j = 0
    for i in range(2 * chunk_size, hidden_size * 3, chunk_size * 3):
      v_tensor[j : j + chunk_size] = tensor_value[i : i + chunk_size]
      j += chunk_size

    return [
        np.ascontiguousarray(q_tensor.transpose()),
        np.ascontiguousarray(k_tensor.transpose()),
        np.ascontiguousarray(v_tensor.transpose()),
    ]

  def update_target_name(self, target_name: str) -> str:
    """Updates the target name to match the tensor name convention."""
    layer_type = LayerType.get_layer_type(target_name)

    target_name = target_name.replace(
        "transformer.h.", "params.lm.transformer.x_layers_"
    )

    if layer_type == LayerType.FEEDFORWARD:
      target_name = target_name.replace(".weight", ".linear.w")
      target_name = target_name.replace(".bias", ".bias.b")
      target_name = target_name.replace(
          "mlp.dense_h_to_4h", "ff_layer.ffn_layer1"
      )
      target_name = target_name.replace(
          "mlp.dense_4h_to_h", "ff_layer.ffn_layer2"
      )
    elif layer_type == LayerType.ATTENTION:
      target_name = target_name.replace("dense", "post")
      target_name = target_name.replace(".weight", ".linear.w")
      target_name = target_name.replace(".bias", ".bias.b")
    elif layer_type == LayerType.EMBEDDING:
      target_name = target_name.replace(
          "transformer.word_embeddings", "params.lm.token_embedding"
      )
      target_name = target_name.replace(
          "lm_head", "params.lm.softmax.logits_ffn"
      )
      target_name = target_name.replace(".weight", ".w")
    elif layer_type == LayerType.LAYER_NORM:
      target_name = target_name.replace("input_layernorm", "pre_layer_norm")
      target_name = target_name.replace(
          "pre_layer_norm.weight", "pre_layer_norm.scale"
      )
      if self._backend == "cpu":
        target_name = target_name.replace(
            "post_attention_layernorm", "ff_layer.pre_layer_norm"
        )
        target_name = target_name.replace(
            "ff_layer.pre_layer_norm.weight", "ff_layer.pre_layer_norm.scale"
        )
      else:
        target_name = target_name.replace(
            "post_attention_layernorm", "post_layer_norm"
        )
        target_name = target_name.replace(
            "post_layer_norm.weight", "post_layer_norm.scale"
        )
      target_name = target_name.replace(
          "transformer.ln_f.weight", "params.lm.final_ln.scale"
      )
      target_name = target_name.replace(
          "transformer.ln_f.bias", "params.lm.final_ln.bias"
      )

    return target_name


class PytorchCkptLoader(converter_base.CkptLoaderBase):
  """CkptLoader implementation for loading the Pytorch model."""

  def __init__(
      self,
      ckpt_path: str,
      is_symmetric: bool,
      attention_quant_bits: int,
      feedforward_quant_bits: int,
      embedding_quant_bits: int,
      special_model: str,
      backend: str,
  ):
    """Initializes the loader.

    Args:
      ckpt_path: The filepath to the safetensors file.
      is_symmetric: Whether to apply symmetric or asymmetric quantization.
      attention_quant_bits: An integer that specify the target quantization bits
        (support 8 or 4) for the attention layers.
      feedforward_quant_bits: An integer that specify the target quantization
        bits (support 8 or 4) for the feedforward layers in each Transformer
        blocks.
      embedding_quant_bits: An integer that specify the target quantization bits
        (support 8 or 4) for the embedding (and the final projection) layers.
      special_model: A string that indicates which input model is and whether
        any special treatment is needed.
      backend: A string indicating the backend used when converting this model.
        Valid options are "cpu" and "gpu".
    """
    super().__init__(
        ckpt_path,
        is_symmetric,
        attention_quant_bits,
        feedforward_quant_bits,
        embedding_quant_bits,
    )

    self._special_model = special_model
    self._reader = _PytorchReader(ckpt_path)
    if special_model in ["FALCON_RW_1B"]:
      self.mapper = FalconMapper(
          is_symmetric,
          attention_quant_bits,
          feedforward_quant_bits,
          embedding_quant_bits,
          backend,
          self._reader,
      )
    else:
      raise ValueError(f"Unknown special model: {special_model}")

  def load_to_actions(
      self,
  ) -> Iterator[List[converter_base.QuantizationAction]]:
    tensor_names = self._reader.get_tensor_names()
    for tensor_name in tensor_names:
      tensor_actions = self.mapper.map_to_actions(tensor_name)
      if tensor_actions is None:
        continue
      yield tensor_actions
