r"""Script to convert and quantize the model chekpoints.

This script is used to convert the large language model checkpoints into the
format that can be loaded by our on-device inference engine.

Example command to run:

bazel run mediapipe/model_maker/python/llm:example_main -- \
    --input_ckpt=$HOME/Downloads/g_mini_2b/ \
    --ckpt_format=safetensors \
    --backend=xnnpack \
    --model_type=G_MINI_2B \
    --output_dir=$HOME/Downloads/g_mini_2b/odml_weights/ \
    --output_tflite_file=$HOME/Downloads/g_mini_2b/odml_weights/model.tflite \
    --logtostderr
"""

from typing import Sequence

from absl import app
from absl import flags

from mediapipe.model_maker.python.llm import llm_converter


_INPUT_CKPT = flags.DEFINE_string(
    'input_ckpt', None, 'Directory or path for the input checkpoint.'
)

_CKPT_FORMAT = flags.DEFINE_string('ckpt_format', None, 'Checkpoint format.')

_IS_SYMMETRIC = flags.DEFINE_bool(
    'is_symmetric', True, 'Whether to quantize symmetrically.'
)

_BACKEND = flags.DEFINE_string(
    'backend',
    'xnnpack',
    'Target backend to run the model. Can be either xnnpack (CPU) or ml_drift'
    ' (GPU).',
)

_ATTENTION_QUAT_BITS = flags.DEFINE_integer(
    'attention_quant_bits',
    8,
    'Target quantization bits for the attention layers.',
)

_FEEDFORWARD_QUAT_BITS = flags.DEFINE_integer(
    'feedforward_quant_bits',
    8,
    'Target quantization bits for the feedforward layers.',
)

_EMBEDDING_QUAT_BITS = flags.DEFINE_integer(
    'embedding_quant_bits',
    8,
    'Target quantization bits for the embedding layers.',
)

_MODEL_TYPE = flags.DEFINE_string(
    'model_type',
    None,
    'Name of the model, e.g. G_MINI_2B.',
)

_OUTPUT_DIR = flags.DEFINE_string('output_dir', None, 'Output directory.')

_COMBINE_FILE_ONLY = flags.DEFINE_bool(
    'combine_file_only',
    False,
    'Whether to combine the weight files only (assuming the weight files are'
    ' already existed).',
)

_VOCAB_MODEL_FILE = flags.DEFINE_string(
    'vocab_model_file', '', 'Vocab model file name.'
)

_OUTPUT_TFLITE_FILE = flags.DEFINE_string(
    'output_tflite_file', '', 'Output tflite file name.'
)


def main(argv: Sequence[str]) -> None:
  del argv

  config = llm_converter.ConversionConfig(
      input_ckpt=_INPUT_CKPT.value,
      ckpt_format=_CKPT_FORMAT.value,
      model_type=_MODEL_TYPE.value,
      backend=_BACKEND.value,
      output_dir=_OUTPUT_DIR.value,
      is_symmetric=_IS_SYMMETRIC.value,
      attention_quant_bits=_ATTENTION_QUAT_BITS.value,
      feedforward_quant_bits=_FEEDFORWARD_QUAT_BITS.value,
      embedding_quant_bits=_EMBEDDING_QUAT_BITS.value,
      combine_file_only=_COMBINE_FILE_ONLY.value,
      vocab_model_file=_VOCAB_MODEL_FILE.value,
      output_tflite_file=_OUTPUT_TFLITE_FILE.value,
  )
  llm_converter.convert_checkpoint(config)


if __name__ == '__main__':
  app.run(main)
