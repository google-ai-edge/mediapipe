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

"""Functions to perform the checkpoint conversion."""

import contextlib
import ctypes
import os
from typing import Any, List, Optional

from absl import logging
import numpy as np

from mediapipe.tasks.python.core import mediapipe_c_bindings
from mediapipe.tasks.python.core import mediapipe_c_utils
from mediapipe.tasks.python.core import serial_dispatcher
from mediapipe.tasks.python.genai.converter import converter_base
from mediapipe.tasks.python.genai.converter import converter_factory
from mediapipe.tasks.python.genai.converter import external_dependencies
from mediapipe.tasks.python.genai.converter import quantization_util

jnp = external_dependencies.jnp

_CTYPES_SIGNATURES = (
    mediapipe_c_utils.CStatusFunction(
        func_name='MpLlmConverterGenerateCpuTfLite',
        core_argtypes=(
            ctypes.c_char_p,  # model_type
            ctypes.c_char_p,  # weight_path
            ctypes.c_char_p,  # vocab_model_file
            ctypes.c_bool,  # is_quantized
            ctypes.c_char_p,  # output_tflite_file
        ),
    ),
    mediapipe_c_utils.CStatusFunction(
        func_name='MpLlmConverterGenerateGpuTfLite',
        core_argtypes=(
            ctypes.c_char_p,  # model_type
            ctypes.c_char_p,  # weight_path
            ctypes.c_char_p,  # vocab_model_file
            ctypes.c_bool,  # is_quantized
            ctypes.c_bool,  # obfuscate
            ctypes.c_char_p,  # output_tflite_file
            ctypes.c_int,  # lora_rank
            ctypes.c_char_p,  # lora_weight_path
            ctypes.c_char_p,  # lora_output_tflite_file
            ctypes.c_char_p,  # lora_main_model_type
            ctypes.c_char_p,  # image_encoder_file
            ctypes.c_char_p,  # image_adapter_file
            ctypes.c_char_p,  # submodel_type
            ctypes.c_bool,  # use_dynamic_ple
            ctypes.c_bool,  # apply_srq
        ),
    ),
    mediapipe_c_utils.CStatusFunction(
        func_name='MpLlmConverterConvertHfTokenizer',
        core_argtypes=(
            ctypes.c_char_p,  # vocab_model_file
            ctypes.c_char_p,  # output_vocab_file
        ),
    ),
)


def _safe_encode_str(value: str | None) -> bytes:
  """Encodes a string to bytes, returning an empty byte string if None."""
  return value.encode('utf-8') if value is not None else b''


class ConversionConfig(object):
  """Config for checkpoint conversion.

  Attributes:
    input_ckpt: Directory or path for the input checkpoint.
    ckpt_format: Checkpoint format, e.g. 'safetensors', 'pytorch'.
    model_type: Name of the model, e.g. GEMMA_2B.
    backend: Target backend to run the model. Can be either "cpu" or "gpu".
    output_dir: Where the output file(s) to be stored.
    is_quantized: Whether the checkpoint is already quantized. If the checkpoint
      is already quantized, the converter will not quantize it again and it will
      ignore the quantization parameters.
    is_symmetric: Whether to quantize symmetrically.
    attention_quant_bits: Target quantization bits for the attention layers.
    feedforward_quant_bits: Target quantization bits for the feedforward layers.
    embedding_quant_bits: Target quantization bits for the embedding layers.
    combine_file_only: Whether to combine the weight files only (assuming the
      weight files are already existed).
    vocab_model_file: The file path to the 1) SentencePiece vocab model; 2)
      Hugging Face BPE tokenizer files; 1) is applicable for the Gemma model and
      2) is applicable for other models. When 2) is used, the provided path is
      expected to point to a directory that contains both tokenizer.json and
      tokenizer_config.json files.
    obfuscate: Whether to obfuscate the model.
    output_tflite_file: (optional) the output tflite filename. If not provided,
      the output will be `model.tflite` stored in the output_dir.
    fp16_scale: A scalar value between [0, 1]. Some models can run into
      activation overflow issue when running in 16-bit floating point mode. To
      solve this, we need to scale down the weights of certain layers. See
      go/llm-on-device-fp16 for more detailed explanation.
    lora_ckpt: The directory or path for the lora checkpoint. Required in order
      to convert the lora weights.
    lora_rank: An integer representing the rank of LoRA. Required in order to
      convert the lora weights.If not provided, then the converter assumes there
      is no LoRA weights. Note that only the GPU backend supports LoRA.
    lora_alpha: A float representing the scale of LoRA weights.
    lora_output_tflite_file: A string indicating the name of the generated
      tflite file for the LoRA weight. Only applicable when the lora_rank is not
      zero.
    lora_main_model_type: If the LoRA will be applied to a submodel packaged
      with a main model, what the main model type is.
    image_encoder_file: A string with the name of the image encoder tflite file.
    image_adapter_file: A string with the name of the image adapter tflite file.
    submodel_type: Name of submodel, e.g. GEMMA_2B.
    use_fake_weights: Whether to use fake weights. If set to True, the weights
      will be filled with zeros.
    use_dynamic_ple: Whether any PLE embeddings should be loaded dynamically.
      Default is true, which will cause embeddings to only be loaded into VRAM
      on demand.
    use_mse_quant: Whether to use MSE quantization for recomputing scales.
  """

  def __init__(
      self,
      input_ckpt: str,
      ckpt_format: str,
      model_type: str,
      backend: str,
      output_dir: str,
      is_quantized: bool = False,
      is_symmetric: bool = True,
      attention_quant_bits: int = 8,
      feedforward_quant_bits: int = 8,
      embedding_quant_bits: int = 8,
      combine_file_only: bool = False,
      vocab_model_file: str = '',
      obfuscate: bool = False,
      output_tflite_file: Optional[str] = None,
      fp16_scale: Optional[float] = None,
      lora_ckpt: Optional[str] = None,
      lora_rank: Optional[int] = None,
      lora_alpha: Optional[float] = None,
      lora_output_tflite_file: Optional[str] = None,
      lora_main_model_type: Optional[str] = None,
      image_encoder_file: Optional[str] = None,
      image_adapter_file: Optional[str] = None,
      submodel_type: Optional[str] = None,
      use_fake_weights: bool = False,
      use_dynamic_ple: bool = True,
      use_mse_quant: bool = False,
  ):
    self.input_ckpt = input_ckpt
    self.ckpt_format = ckpt_format
    self.model_type = model_type
    self.backend = backend
    if os.path.isfile(output_dir):
      raise ValueError('Output directory mush not point to an existing file.')
    if not os.path.isdir(output_dir):
      logging.info('Creating output directory: %s', output_dir)
      os.makedirs(output_dir, exist_ok=True)
    self.output_dir = output_dir
    self.is_quantized = is_quantized
    self.is_symmetric = is_symmetric
    self.attention_quant_bits = attention_quant_bits
    self.feedforward_quant_bits = feedforward_quant_bits
    self.embedding_quant_bits = embedding_quant_bits
    self.combine_file_only = combine_file_only
    self.vocab_model_file = vocab_model_file
    self.obfuscate = obfuscate
    self.image_encoder_file = image_encoder_file
    self.image_adapter_file = image_adapter_file
    self.submodel_type = submodel_type
    self.use_fake_weights = use_fake_weights
    self.use_dynamic_ple = use_dynamic_ple
    self.use_mse_quant = use_mse_quant
    if output_tflite_file:
      parent_dir = os.path.dirname(output_tflite_file)
      if not os.path.isdir(parent_dir):
        logging.info('Creating tflite parent directory: %s', parent_dir)
        os.makedirs(parent_dir, exist_ok=True)
      self.output_tflite_file = output_tflite_file
    else:
      self.output_tflite_file = os.path.join(output_dir, 'model.tflite')

    self.fp16_scale = None
    self.lora_ckpt = lora_ckpt
    self.lora_rank = lora_rank
    self.lora_alpha = lora_alpha
    self.lora_output_tflite_file = lora_output_tflite_file
    self.lora_main_model_type = lora_main_model_type
    if (self.lora_ckpt is None) ^ (self.lora_rank is None):
      raise ValueError(
          'lora_ckpt and lora_rank must be either both provided or both not'
          ' provided.'
      )
    if self.lora_rank is not None:
      if backend == 'cpu':
        raise ValueError('LoRA is not supported for CPU backend.')
      lora_applicable_models = [
          'GEMMA_2B',
          'GEMMA2_2B',
          'PHI_2',
          'GEMMA3_1B',
          'GEMMA3_4B',
          'GEMMA3_12B',
          'GEMMA3_27B',
          'GEMMA3_300M',
      ]
      if model_type not in lora_applicable_models:
        raise ValueError(
            'LoRA is only applicable for the model_type:'
            f' {", ".join(lora_applicable_models)}, but get model_type:'
            f' {model_type}.'
        )


@contextlib.contextmanager
def filemanager(filename: str, mode: str):
  try:
    with open(filename, mode) as f:
      yield f
  finally:
    pass


class _LlmConverter:
  """Bundles all conversion logic for LLM models."""

  def __init__(self, lib: Any):
    self._lib = serial_dispatcher.SerialDispatcher(lib, _CTYPES_SIGNATURES)

  def quantize_by_actions(
      self,
      actions: List[converter_base.QuantizationAction],
      backend: str,
      is_symmetric: bool,
      use_mse_quant: bool = False,
  ):
    """Quantizes the weights by actions.

    Args:
      actions: A list of QuantizationAction that contains the information and
        tensor values to be quantized.
      backend: Target backend to run the model. Can be either "cpu" or "gpu".
      is_symmetric: Whether to quantize symmetrically.
      use_mse_quant: Whether to use MSE quantization for recomputing scales.

    Returns:
      A dictionary that maps from the updated tensor names to the quantized
      tensor values + a boolean that indicates whether the tensor values need to
      be packed (only applicable for the 4-bit quantized weights).
    """
    output_tensors = {}
    qvalue_suffix = '_quantized_value'
    scale_suffix = '_quantized_scale'
    zp_suffix = '_quantized_zp'
    for action in actions:
      if action.tensor_value is None:
        continue
      # The dtype needs to be compared in string as it is a custom numpy dtype.
      # Explicitly cast the bfloat16 and float16 dtype to float32 to make sure
      # its value is converted and serialized correctly.
      if (
          str(action.tensor_value.dtype) == 'bfloat16'
          or action.tensor_value.dtype == np.float16
      ):
        action.tensor_value = action.tensor_value.astype(np.float32)
      if (
          (not action.is_quantized)
          and action.tensor_value.dtype != np.float32
          and action.tensor_value.dtype != np.int8
      ):
        raise ValueError(
            'All tensors should be casted to either float32 or int8, but got: '
            '%s'
            % action.tensor_value.dtype
        )
      if action.is_quantized:
        pack = action.tensor_value.dtype == jnp.int4
        if qvalue_suffix in action.target_name:
          target_name = action.target_name[: -len(qvalue_suffix)]
          # Stores the quantized value in int8 for 4-bit quantization.
          if pack:
            action.tensor_value = action.tensor_value.astype(jnp.int8)
          output_tensors[target_name] = (action.tensor_value, pack)
        elif (
            scale_suffix in action.target_name
            or zp_suffix in action.target_name
        ):
          output_tensors[action.target_name] = (
              action.tensor_value,
              False,
          )
        else:
          output_tensors[action.target_name] = (action.tensor_value, False)
      if action.quantize_axis:
        pack = action.quantize_bits == 4
        if action.tensor_value.dtype == np.int8:
          if backend == 'cpu' and pack:
            raise ValueError(
                'Converting pre-quantized checkpoint into 4-bit is not'
                ' supported for CPU backend.'
            )
          output_tensors[action.target_name] = (action.tensor_value, pack)
        else:
          if is_symmetric:
            target_var, scale = quantization_util.quantize_tensor(
                var=action.tensor_value,
                axis=action.quantize_axis,
                sym=is_symmetric,
                number_bits=action.quantize_bits,
                use_mse_quant=use_mse_quant,
            )
            output_tensors[action.target_name] = (target_var, pack)
            output_tensors[action.target_name + scale_suffix] = (
                scale,
                False,
            )
            zp = None
          else:
            target_var, scale, zp = quantization_util.quantize_tensor(
                var=action.tensor_value,
                axis=action.quantize_axis,
                sym=is_symmetric,
                number_bits=action.quantize_bits,
                use_mse_quant=use_mse_quant,
            )
          if backend == 'cpu' and pack:
            target_var, scale, zp = quantization_util.update_to_uint4(
                target_var, scale, zp
            )
          output_tensors[action.target_name] = (target_var, pack)
          output_tensors[action.target_name + scale_suffix] = (scale, False)
          if zp is not None:
            output_tensors[action.target_name + zp_suffix] = (zp, False)
      else:
        output_tensors[action.target_name] = (action.tensor_value, False)
    return output_tensors

  def combined_weight_bins_to_tflite(
      self,
      model_type: str,
      backend: str,
      weight_path: str,
      output_tflite_file: str,
      obfuscate: bool,
      vocab_model_file: str,
      lora_rank: Optional[int] = None,
      lora_weight_path: Optional[str] = None,
      lora_output_tflite_file: Optional[str] = None,
      lora_main_model_type: Optional[str] = None,
      image_encoder_file: Optional[str] = None,
      image_adapter_file: Optional[str] = None,
      submodel_type: Optional[str] = None,
      use_dynamic_ple: Optional[bool] = None,
      apply_srq: Optional[bool] = None,
  ):
    """Combines weight files to tflite file."""
    if backend == 'cpu':
      if lora_rank is not None:
        logging.fatal('LoRA is not supported for CPU backend.')
      self._lib.MpLlmConverterGenerateCpuTfLite(
          model_type.encode('utf-8'),
          weight_path.encode('utf-8'),
          vocab_model_file.encode('utf-8'),
          True,
          output_tflite_file.encode('utf-8'),
      )
    elif backend == 'gpu':
      self._lib.MpLlmConverterGenerateGpuTfLite(
          model_type.encode('utf-8'),
          weight_path.encode('utf-8'),
          vocab_model_file.encode('utf-8'),
          True,
          obfuscate,
          output_tflite_file.encode('utf-8'),
          0 if lora_rank is None else lora_rank,
          _safe_encode_str(lora_weight_path),
          _safe_encode_str(lora_output_tflite_file),
          _safe_encode_str(lora_main_model_type),
          _safe_encode_str(image_encoder_file),
          _safe_encode_str(image_adapter_file),
          _safe_encode_str(submodel_type),
          True if use_dynamic_ple is None else use_dynamic_ple,
          False if apply_srq is None else apply_srq,
      )
    else:
      raise ValueError(f'Unsupported backend: {backend}')

  def convert_bpe_vocab(self, vocab_model_file: str, output_dir: str) -> str:
    """Converts the BPE vocab model file to SPM format.

    Args:
      vocab_model_file: The input BPE vocab model file path.
      output_dir: The output directory to store the SPM vocab model file.

    Returns:
      The path to the output SPM vocab model file.
    """
    if not os.path.isdir(vocab_model_file):
      raise ValueError(
          'The input BPE vocab model file path is expected to be a directory'
          ' that contains both tokenizer.json and tokenizer_config.json files.'
      )
    output_vocab_file = os.path.join(output_dir, 'spm.model')
    self._lib.MpLlmConverterConvertHfTokenizer(
        vocab_model_file.encode('utf-8'),
        output_vocab_file.encode('utf-8'),
    )
    return output_vocab_file

  def sort_layer_info(self, layer_info_file: str) -> None:
    """Loads and sorts the layer info file."""
    layer_info = []
    with filemanager(layer_info_file, 'r') as finfo:
      for line in finfo:
        line = line.strip()
        if line:
          layer_info.append(line)
    layer_info = list(set(layer_info))
    layer_info.sort()
    with filemanager(layer_info_file, 'w') as finfo:
      for line in layer_info:
        finfo.write(line + '\n')
        finfo.write('\n')

  def maybe_quantize_and_write_tensors_to_bins(
      self,
      ckpt_loader: converter_base.CkptLoaderBase,
      config: ConversionConfig,
  ) -> None:
    """Quantizes the weight tensors according to the loader and writes them to bins."""
    actions = ckpt_loader.load_to_actions()

    for action in actions:
      # Quantize the weight
      quantized_tensors = self.quantize_by_actions(
          action, config.backend, config.is_symmetric, config.use_mse_quant
      )
      del action
      # Write the tensors into file(s).
      writer = converter_factory.create_writer(
          writer_type='weight_bins',
          output_dir=config.output_dir,
          backend=config.backend,
      )
      writer.write_variables(quantized_tensors, config.use_fake_weights)
      del quantized_tensors
      del writer

  def convert_checkpoint(self, config: ConversionConfig) -> None:
    """Converts the checkpoint to tflite file."""
    logging.info('input folder: %s', config.input_ckpt)

    if os.path.isdir(config.vocab_model_file):
      vocab_model_path = self.convert_bpe_vocab(
          config.vocab_model_file, config.output_dir
      )
    else:
      vocab_model_path = config.vocab_model_file

    if not config.combine_file_only:
      # Load the layer weights and prepare the quantization configurations.
      loader = converter_factory.create_ckpt_loader(
          config.ckpt_format,
          ckpt_path=config.input_ckpt,
          is_quantized=config.is_quantized,
          is_symmetric=config.is_symmetric,
          backend=config.backend,
          attention_quant_bits=config.attention_quant_bits,
          feedforward_quant_bits=config.feedforward_quant_bits,
          embedding_quant_bits=config.embedding_quant_bits,
          special_model=config.model_type,
          fp16_scale=config.fp16_scale,
      )
      self.maybe_quantize_and_write_tensors_to_bins(loader, config)

      if config.lora_ckpt is not None and config.lora_ckpt != config.input_ckpt:
        # If lora ckpt and the input ckpt is the same. The lora conversion is
        # handled in the previous loader.
        lora_loader = converter_factory.create_ckpt_loader(
            config.ckpt_format,
            ckpt_path=config.lora_ckpt,
            is_quantized=config.is_quantized,
            is_symmetric=config.is_symmetric,
            backend=config.backend,
            attention_quant_bits=config.attention_quant_bits,
            feedforward_quant_bits=config.feedforward_quant_bits,
            embedding_quant_bits=config.embedding_quant_bits,
            special_model=config.model_type,
        )
        self.maybe_quantize_and_write_tensors_to_bins(lora_loader, config)

      self.sort_layer_info(os.path.join(config.output_dir, 'layer_info.txt'))

    self.combined_weight_bins_to_tflite(
        config.model_type,
        config.backend,
        weight_path=config.output_dir,
        output_tflite_file=config.output_tflite_file,
        obfuscate=config.obfuscate,
        vocab_model_file=vocab_model_path,
        lora_rank=config.lora_rank,
        lora_weight_path=config.output_dir,
        lora_output_tflite_file=config.lora_output_tflite_file,
        lora_main_model_type=config.lora_main_model_type,
        image_encoder_file=config.image_encoder_file,
        image_adapter_file=config.image_adapter_file,
        submodel_type=config.submodel_type,
        use_dynamic_ple=config.use_dynamic_ple,
        # Fow now, any pre-quantized model is assumed to require SRQ support.
        apply_srq=config.is_quantized,
    )

  def __enter__(self) -> '_LlmConverter':
    """Returns `self` upon entering the runtime context."""
    return self

  def __exit__(self, *args) -> None:
    """Shuts down the LlmConverter on exit of the context manager.

    Args:
      *args: Unused.
    Raises:
      RuntimeError: If the LLM converter failed to close.
    """
    del args  # Unused.
    self._lib.close()


def convert_checkpoint(config: ConversionConfig) -> None:
  """Converts the checkpoint to tflite file."""
  lib = mediapipe_c_bindings.load_raw_library(_CTYPES_SIGNATURES)
  with _LlmConverter(lib) as converter:
    converter.convert_checkpoint(config)


def quantize_by_actions(
    actions: List[converter_base.QuantizationAction],
    backend: str,
    is_symmetric: bool,
    use_mse_quant: bool = False,
):
  """Quantizes the weights by actions.

  Args:
    actions: A list of QuantizationAction that contains the information and
      tensor values to be quantized.
    backend: Target backend to run the model. Can be either "cpu" or "gpu".
    is_symmetric: Whether to quantize symmetrically.
    use_mse_quant: Whether to use MSE quantization for recomputing scales.

  Returns:
    A dictionary that maps from the updated tensor names to the quantized
    tensor values + a boolean that indicates whether the tensor values need to
    be packed (only applicable for the 4-bit quantized weights).
  """
  lib = mediapipe_c_bindings.load_raw_library(_CTYPES_SIGNATURES)
  with _LlmConverter(lib) as converter:
    return converter.quantize_by_actions(
        actions, backend, is_symmetric, use_mse_quant
    )
