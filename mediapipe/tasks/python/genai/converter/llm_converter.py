"""Functions to perform the checkpoint conversion."""

import os
from typing import List, Optional

from absl import logging

from mediapipe.tasks.python.genai.converter import converter_base
from mediapipe.tasks.python.genai.converter import converter_factory
from mediapipe.tasks.python.genai.converter import model_ckpt_util
from mediapipe.tasks.python.genai.converter import quantization_util


class ConversionConfig(object):
  """Config for checkpoint conversion.

  Attributes:
    input_ckpt: Directory or path for the input checkpoint.
    ckpt_format: Checkpoint format, e.g. 'safetensors', 'pytorch'.
    model_type: Name of the model, e.g. GEMMA_2B.
    backend: Target backend to run the model. Can be either "cpu" or "gpu".
    output_dir: Where the output file(s) to be stored.
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
    output_tflite_file: (optional) the output tflite filename. If not provided,
      the output will be `model.tflite` stored in the output_dir.
  """

  def __init__(
      self,
      input_ckpt: str,
      ckpt_format: str,
      model_type: str,
      backend: str,
      output_dir: str,
      is_symmetric: bool = True,
      attention_quant_bits: int = 8,
      feedforward_quant_bits: int = 8,
      embedding_quant_bits: int = 8,
      combine_file_only: bool = False,
      vocab_model_file: str = '',
      output_tflite_file: Optional[str] = None,
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
    self.is_symmetric = is_symmetric
    self.attention_quant_bits = attention_quant_bits
    self.feedforward_quant_bits = feedforward_quant_bits
    self.embedding_quant_bits = embedding_quant_bits
    self.combine_file_only = combine_file_only
    self.vocab_model_file = vocab_model_file
    if output_tflite_file:
      parent_dir = os.path.dirname(output_tflite_file)
      if not os.path.isdir(parent_dir):
        logging.info('Creating tflite parent directory: %s', parent_dir)
        os.makedirs(parent_dir, exist_ok=True)
      self.output_tflite_file = output_tflite_file
    else:
      self.output_tflite_file = os.path.join(output_dir, 'model.tflite')


def quantize_by_actions(
    actions: List[converter_base.QuantizationAction],
    backend: str,
    is_symmetric: bool,
):
  """Quantizes the weights by actions.

  Args:
    actions: A list of QuantizationAction that contains the information and
      tensor values to be quantized.
    backend: Target backend to run the model. Can be either "cpu" or "gpu".
    is_symmetric: Whether to quantize symmetrically.

  Returns:
    A dictionary that maps from the updated tensor names to the quantized
    tensor values + a boolean that indicates whether the tensor values need to
    be packed (only applicable for the 4-bit quantized weights).
  """
  output_tensors = {}
  for action in actions:
    if action.quantize_axis:
      pack = action.quantize_bits == 4
      if is_symmetric:
        target_var, scale = quantization_util.quantize_tensor(
            var=action.tensor_value,
            axis=action.quantize_axis,
            sym=is_symmetric,
            number_bits=action.quantize_bits,
        )
        output_tensors[action.target_name] = (target_var, pack)
        output_tensors[action.target_name + '_quantized_scale'] = (scale, False)
        zp = None
      else:
        target_var, scale, zp = quantization_util.quantize_tensor(
            var=action.tensor_value,
            axis=action.quantize_axis,
            sym=is_symmetric,
            number_bits=action.quantize_bits,
        )
      if backend == 'cpu' and pack:
        target_var, scale, zp = quantization_util.update_to_uint4(
            target_var, scale, zp
        )
      output_tensors[action.target_name] = (target_var, pack)
      output_tensors[action.target_name + '_quantized_scale'] = (scale, False)
      if zp is not None:
        output_tensors[action.target_name + '_quantized_zp'] = (zp, False)
    else:
      output_tensors[action.target_name] = (action.tensor_value, False)
  return output_tensors


def combined_weight_bins_to_tflite(
    model_type: str,
    backend: str,
    weight_path: str,
    output_tflite_file: str,
    vocab_model_file: str,
):
  """Combines weight files to tflite file."""
  # TODO: Figure out whether to clean up the weight files after this.
  if backend == 'cpu':
    model_ckpt_util.GenerateCpuTfLite(
        model_type,
        weight_path,
        vocab_model_file,
        True,
        output_tflite_file,
    )
  elif backend == 'gpu':
    model_ckpt_util.GenerateGpuTfLite(
        model_type,
        weight_path,
        vocab_model_file,
        True,
        output_tflite_file,
    )
  else:
    raise ValueError('Unsupported backend: %s' % backend)


def convert_bpe_vocab(vocab_model_file: str, output_dir: str) -> str:
  if not os.path.isdir(vocab_model_file):
    raise ValueError(
        'The input BPE vocab model file path is expected to be a directory that'
        ' conatins both tokenizer.json and tokenizer_config.json files.'
    )
  output_vocab_file = os.path.join(output_dir, 'spm.model')
  model_ckpt_util.ConvertHfTokenizer(vocab_model_file, output_vocab_file)
  return output_vocab_file


def convert_checkpoint(config: ConversionConfig) -> None:
  """Converts the checkpoint to tflite file."""
  logging.info('input folder: %s', config.input_ckpt)

  if config.model_type == 'GEMMA_2B' or config.model_type == 'GEMINI_XXS':
    vocab_model_path = config.vocab_model_file
  else:
    vocab_model_path = convert_bpe_vocab(
        config.vocab_model_file, config.output_dir
    )

  if not config.combine_file_only:
    # Load the layer weights and prepare the quantization configurations.
    loader = converter_factory.create_ckpt_loader(
        config.ckpt_format,
        ckpt_path=config.input_ckpt,
        is_symmetric=config.is_symmetric,
        backend=config.backend,
        attention_quant_bits=config.attention_quant_bits,
        feedforward_quant_bits=config.feedforward_quant_bits,
        embedding_quant_bits=config.embedding_quant_bits,
        special_model=config.model_type,
    )
    actions = loader.load_to_actions()

    # Quantize the weights.
    quantized_tensors = quantize_by_actions(
        actions, config.backend, config.is_symmetric
    )

    # Write the quantized tensors into file(s).
    writer = converter_factory.create_writer(
        writer_type='weight_bins',
        output_dir=config.output_dir,
        backend=config.backend,
    )
    writer.write_variables(quantized_tensors)

  combined_weight_bins_to_tflite(
      config.model_type,
      config.backend,
      weight_path=config.output_dir,
      output_tflite_file=config.output_tflite_file,
      vocab_model_file=vocab_model_path,
  )
