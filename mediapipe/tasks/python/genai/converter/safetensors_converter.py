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

"""CkptLoader implementation for loading the Safetensors."""

import array
from typing import Iterator
import enum
import glob
import json
import os
from typing import List, Optional

import numpy as np
import torch

from mediapipe.tasks.python.genai.converter import converter_base


DTYPE_MAP = {
    "F16": torch.float16,
    "BF16": torch.bfloat16,
    "F32": torch.float32,
}


class _SafetensorsShardReader:
  """Reads a single safetensors shard."""

  _HEAD_BYTES = 8

  def __init__(self, shard_path: str):
    self._shard_path = shard_path
    if not os.path.exists(self._shard_path):
      raise ValueError(f"{self._shard_path} does not exists.")
    with open(self._shard_path, "rb") as f:
      head_bytes = f.read(self._HEAD_BYTES)
      metadata_bytes_num = np.frombuffer(head_bytes, dtype=np.uint64)[0]
      metadata_bytes = f.read(metadata_bytes_num)
      self.layers_info = json.loads(metadata_bytes)
      self.metadata_bytes_num = metadata_bytes_num

  def read_tensor_as_numpy(self, tensor_name) -> np.ndarray:
    """Reads a tensor from the model file as a numpy array with np.float32 type."""
    tensor_info = self.layers_info[tensor_name]
    with open(self._shard_path, "rb") as f:
      shape = tensor_info["shape"]
      dtype = tensor_info["dtype"]
      if dtype not in DTYPE_MAP:
        raise ValueError(f"{dtype} is not supported.")
      data_offsets = tensor_info["data_offsets"]
      f.seek(int(self._HEAD_BYTES + self.metadata_bytes_num + data_offsets[0]))
      tensor_bytes = f.read(data_offsets[1] - data_offsets[0])
      raw_tensor = torch.frombuffer(
          array.array("b", tensor_bytes), dtype=DTYPE_MAP[dtype]
      ).reshape(shape)
      return raw_tensor.float().t().contiguous().numpy()

  def get_tensor_names(self) -> List[str]:
    names = list(self.layers_info.keys())
    if "__metadata__" in names:
      names.remove("__metadata__")
    return names


class _SafetensorsReader:
  """Reads all the safetensors shards."""

  def __init__(self, ckpt_path: str):
    shards = []
    if os.path.isdir(ckpt_path):
      # Read all safetensors files within checkpoint
      for shard_path in glob.glob(os.path.join(ckpt_path, "*.safetensors")):
        shards.append(_SafetensorsShardReader(shard_path))
    else:
      # Assume the ckpt_path is a file or a file pattern to match.
      for shard_path in glob.glob(ckpt_path):
        shards.append(_SafetensorsShardReader(shard_path))
    assert shards is not None

    self._ckpt_path = ckpt_path
    self._tensors_map = {}
    for shard in shards:
      tensor_names = shard.get_tensor_names()
      for tensor_name in tensor_names:
        if tensor_name in self._tensors_map:
          raise ValueError(f"Duplicate tensor name: {tensor_name}")
        self._tensors_map[tensor_name] = shard

  def get_tensor_names(self) -> List[str]:
    return list(self._tensors_map.keys())

  def read_tensor_as_numpy(self, tensor_name: str) -> np.ndarray:
    return self._tensors_map[tensor_name].read_tensor_as_numpy(tensor_name)


class LayerType(enum.Enum):
  """Enum for layer type."""

  NONE = 0
  ATTENTION = 1  # Layer is part of the attention module.
  FEEDFORWARD = 2  # Layer is part of the feedforward module in the Transformer.
  EMBEDDING = 3  # Layer is the embedding lookup or final projection layer.
  LAYER_NORM = (
      4  # Layer is layer normalization before and after attention layer.
  )
  LORA = 5  # Layer is LoRA weights augmented on the base model layers.

  @classmethod
  def get_layer_type(cls, layer_name: str):
    """Gets the layer type of the given layer name."""
    ffn_layers = [
        "mlp",
    ]
    attn_layers = [
        "self_attn",
    ]
    emb_layers = [
        "embed_tokens",
        "lm_head",
    ]
    layer_norms = [
        "input_layernorm",
        "post_attention_layernorm",
        "final_layernorm",
        "model.norm.weight",
        "pre_feedforward_layernorm",
        "post_feedforward_layernorm",
    ]
    lora_layers = ["lora"]
    if any(sub_name in layer_name for sub_name in lora_layers):
      return LayerType.LORA
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


class StablelmMapper(converter_base.LayerActionMapperBase):
  """LayerActionMapper for handling the StableLM model."""

  def __init__(
      self,
      is_symmetric: bool,
      attention_quant_bits: int,
      feedforward_quant_bits: int,
      embedding_quant_bits: int,
      backend: str,
      reader: _SafetensorsReader,
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
    tensor_value = self._reader.read_tensor_as_numpy(layer_name)
    quantize_axis = None
    quantize_bits = None
    layer_type = LayerType.get_layer_type(layer_name)

    if layer_type != LayerType.LAYER_NORM and layer_name.endswith(".weight"):
      quantize_axis = [0]
      if layer_type == LayerType.FEEDFORWARD:
        quantize_bits = self._feedforward_quant_bits
      elif layer_type == LayerType.ATTENTION:
        quantize_bits = self._attention_quant_bits
        if self._backend == "cpu" and ".o_proj." in layer_name:
          tensor_value = np.transpose(tensor_value)
          quantize_axis = [1]
      elif layer_type == LayerType.EMBEDDING:
        quantize_bits = self._embedding_quant_bits
        if self._backend == "cpu" and ".embed_tokens." in layer_name:
          tensor_value = np.transpose(tensor_value)
          quantize_axis = [1]
    target_name = self.update_target_name(layer_name)

    actions = [
        converter_base.QuantizationAction(
            tensor_name=layer_name,
            tensor_value=tensor_value,
            target_name=target_name,
            quantize_axis=quantize_axis,
            quantize_bits=quantize_bits,
            pack_dim=0,
        )
    ]
    return actions

  def update_target_name(self, target_name: str) -> str:
    """Updates the target name to match the tensor name convention."""
    target_name = target_name.replace(
        "model.layers.", "params.lm.transformer.x_layers_"
    )
    target_name = target_name.replace("mlp.up_proj", "ff_layer.ffn_layer1")
    target_name = target_name.replace("mlp.down_proj", "ff_layer.ffn_layer2")
    target_name = target_name.replace(
        "mlp.gate_proj", "ff_layer.ffn_layer1_gate"
    )
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
    target_name = target_name.replace("self_attn.q_proj", "self_attention.q")
    target_name = target_name.replace("self_attn.k_proj", "self_attention.k")
    target_name = target_name.replace("self_attn.v_proj", "self_attention.v")
    target_name = target_name.replace("self_attn.o_proj", "self_attention.post")
    target_name = target_name.replace(
        "model.embed_tokens", "params.lm.token_embedding"
    )
    target_name = target_name.replace("model.norm", "params.lm.final_ln")
    target_name = target_name.replace("final_ln.weight", "final_ln.scale")
    target_name = target_name.replace("lm_head", "params.lm.softmax.logits_ffn")
    target_name = target_name.replace(".weight", ".w")

    return target_name


class PhiMapper(converter_base.LayerActionMapperBase):
  """LayerActionMapper for handling the Phi model."""

  def __init__(
      self,
      is_symmetric: bool,
      attention_quant_bits: int,
      feedforward_quant_bits: int,
      embedding_quant_bits: int,
      backend: str,
      reader: _SafetensorsReader,
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
    tensor_value = self._reader.read_tensor_as_numpy(layer_name)
    quantize_axis = None
    quantize_bits = None
    layer_type = LayerType.get_layer_type(layer_name)

    if (
        layer_type != LayerType.LAYER_NORM
        and layer_name.endswith(".weight")
        and layer_type != LayerType.LORA
    ):
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
        if self._backend == "cpu" and ".embed_tokens." in layer_name:
          tensor_value = np.transpose(tensor_value)
          quantize_axis = [1]
    target_name = self.update_target_name(layer_name)

    actions = [
        converter_base.QuantizationAction(
            tensor_name=layer_name,
            tensor_value=tensor_value,
            target_name=target_name,
            quantize_axis=quantize_axis,
            quantize_bits=quantize_bits,
            pack_dim=0,
        )
    ]
    return actions

  def update_target_name(self, target_name: str) -> str:
    """Updates the target name to match the tensor name convention."""
    target_name = target_name.replace("base_model.model.", "")
    target_name = target_name.replace(
        "model.layers.", "params.lm.transformer.x_layers_"
    )

    layer_type = LayerType.get_layer_type(target_name)
    if layer_type == LayerType.FEEDFORWARD:
      target_name = target_name.replace(".weight", ".linear.w")
      target_name = target_name.replace(".bias", ".bias.b")
      target_name = target_name.replace("mlp.fc1", "ff_layer.ffn_layer1")
      target_name = target_name.replace("mlp.fc2", "ff_layer.ffn_layer2")

    elif layer_type == LayerType.ATTENTION:
      target_name = target_name.replace(".weight", ".linear.w")
      target_name = target_name.replace(".bias", ".bias.b")
      target_name = target_name.replace("self_attn.q_proj", "self_attention.q")
      target_name = target_name.replace("self_attn.k_proj", "self_attention.k")
      target_name = target_name.replace("self_attn.v_proj", "self_attention.v")
      target_name = target_name.replace(
          "self_attn.dense", "self_attention.post"
      )
    elif layer_type == LayerType.EMBEDDING:
      target_name = target_name.replace(
          "model.embed_tokens", "params.lm.token_embedding"
      )
      target_name = target_name.replace(
          "lm_head", "params.lm.softmax.logits_ffn"
      )
      target_name = target_name.replace(
          "logits_ffn.weight", "logits_ffn.linear.w"
      )
      target_name = target_name.replace("logits_ffn.bias", "logits_ffn.bias.b")
    elif layer_type == LayerType.LAYER_NORM:
      target_name = target_name.replace("input_layernorm", "pre_layer_norm")
      target_name = target_name.replace(
          "pre_layer_norm.weight", "pre_layer_norm.scale"
      )
      target_name = target_name.replace(
          "model.final_layernorm", "params.lm.final_ln"
      )
      target_name = target_name.replace("final_ln.weight", "final_ln.scale")
    target_name = target_name.replace(".weight", ".w")

    # For LoRA weights
    if "post" in target_name:
      target_name = target_name.replace("lora_A.linear.w", "w_prime_right")
      target_name = target_name.replace("lora_B.linear.w", "w_prime_left")
    else:
      target_name = target_name.replace("lora_A.linear.w", "w_prime_left")
      target_name = target_name.replace("lora_B.linear.w", "w_prime_right")

    return target_name


class GemmaMapper(converter_base.LayerActionMapperBase):
  """LayerActionMapper for handling the Gemma model."""

  def __init__(
      self,
      is_symmetric: bool,
      attention_quant_bits: int,
      feedforward_quant_bits: int,
      embedding_quant_bits: int,
      backend: str,
      reader: _SafetensorsReader,
      is_v2: bool,
  ):
    super().__init__(
        is_symmetric=is_symmetric,
        attention_quant_bits=attention_quant_bits,
        feedforward_quant_bits=feedforward_quant_bits,
        embedding_quant_bits=embedding_quant_bits,
        backend=backend,
    )
    self._reader = reader
    self._is_v2 = is_v2

  def map_to_actions(
      self, layer_name: str
  ) -> Optional[List[converter_base.QuantizationAction]]:
    """Map the given layer name to actions."""
    tensor_value = self._reader.read_tensor_as_numpy(layer_name)
    quantize_axis = None
    quantize_bits = None
    layer_type = LayerType.get_layer_type(layer_name)

    if (
        layer_type != LayerType.LAYER_NORM
        and layer_name.endswith(".weight")
        and layer_type != LayerType.LORA
    ):
      quantize_axis = [0]
      if layer_type == LayerType.FEEDFORWARD:
        quantize_bits = self._feedforward_quant_bits
      elif layer_type == LayerType.ATTENTION:
        quantize_bits = self._attention_quant_bits
        if "o_proj" in layer_name:
          tensor_value = np.transpose(tensor_value)
          quantize_axis = [1]
      elif layer_type == LayerType.EMBEDDING:
        quantize_bits = self._embedding_quant_bits

    target_name = self.update_target_name(layer_name)

    actions = [
        converter_base.QuantizationAction(
            tensor_name=layer_name,
            tensor_value=tensor_value,
            target_name=target_name,
            quantize_axis=quantize_axis,
            quantize_bits=quantize_bits,
            pack_dim=0,
        )
    ]
    return actions

  def update_target_name(self, target_name: str) -> str:
    """Updates the target name to match the tensor name convention."""
    target_name = target_name.replace("base_model.model.", "")
    target_name = target_name.replace(
        "model.layers.", "params.lm.transformer.x_layers_"
    )
    target_name = target_name.replace("mlp.up_proj", "ff_layer.ffn_layer1")
    target_name = target_name.replace("mlp.down_proj", "ff_layer.ffn_layer2")
    target_name = target_name.replace(
        "mlp.gate_proj", "ff_layer.ffn_layer1_gate"
    )
    target_name = target_name.replace("input_layernorm", "pre_layer_norm")
    target_name = target_name.replace(
        "pre_layer_norm.weight", "pre_layer_norm.scale"
    )

    # Gemma and Gemma2 differ slightly in their use of the
    # "post_attention_layernorm" tensor name.
    if self._is_v2:
      target_name = target_name.replace(
          "post_attention_layernorm", "post_layer_norm"
      )
    else:
      target_name = target_name.replace(
          "post_attention_layernorm", "ff_layer.pre_layer_norm"
      )

    target_name = target_name.replace(
        "pre_feedforward_layernorm", "ff_layer.pre_layer_norm"
    )
    target_name = target_name.replace(
        "post_feedforward_layernorm", "ff_layer.post_layer_norm"
    )
    target_name = target_name.replace(
        "ff_layer.pre_layer_norm.weight", "ff_layer.pre_layer_norm.scale"
    )
    target_name = target_name.replace(
        "ff_layer.post_layer_norm.weight", "ff_layer.post_layer_norm.scale"
    )
    target_name = target_name.replace(
        "post_layer_norm.weight", "post_layer_norm.scale"
    )
    target_name = target_name.replace("self_attn.q_proj", "self_attention.q")
    target_name = target_name.replace("self_attn.k_proj", "self_attention.k")
    target_name = target_name.replace("self_attn.v_proj", "self_attention.v")
    target_name = target_name.replace("self_attn.o_proj", "self_attention.post")
    target_name = target_name.replace(
        "model.embed_tokens", "params.lm.softmax.logits_ffn"
    )
    target_name = target_name.replace("model.norm", "params.lm.final_ln")
    target_name = target_name.replace("final_ln.weight", "final_ln.scale")
    target_name = target_name.replace(".weight", ".w")

    # For LoRA weights
    if "post" in target_name:
      target_name = target_name.replace("lora_A.w", "w_prime_right")
      target_name = target_name.replace("lora_B.w", "w_prime_left")
    else:
      target_name = target_name.replace("lora_A.w", "w_prime_left")
      target_name = target_name.replace("lora_B.w", "w_prime_right")

    return target_name


class SafetensorsCkptLoader(converter_base.CkptLoaderBase):
  """CkptLoader implementation for loading the Safetensors."""

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
    self._reader = _SafetensorsReader(ckpt_path)
    if special_model in ["STABLELM_4E1T_3B"]:
      self.mapper = StablelmMapper(
          is_symmetric,
          attention_quant_bits,
          feedforward_quant_bits,
          embedding_quant_bits,
          backend,
          self._reader,
      )
    elif special_model in ["PHI_2"]:
      self.mapper = PhiMapper(
          is_symmetric,
          attention_quant_bits,
          feedforward_quant_bits,
          embedding_quant_bits,
          backend,
          self._reader,
      )
    elif special_model in ["GEMMA_2B", "GEMMA_7B", "GEMMA2_2B"]:
      self.mapper = GemmaMapper(
          is_symmetric,
          attention_quant_bits,
          feedforward_quant_bits,
          embedding_quant_bits,
          backend,
          self._reader,
          True if special_model in ["GEMMA2_2B"] else False,
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
