# Copyright 2025 The MediaPipe Authors.
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

from unittest import mock

from absl.testing import parameterized
import numpy as np
import tensorflow as tf

import unittest
from mediapipe.tasks.python.genai.converter import converter_base
from mediapipe.tasks.python.genai.converter import llm_converter

_LLM_CONVERTER_FUNCTIONS = (
    'MpLlmConverterGenerateCpuTfLite',
    'MpLlmConverterGenerateGpuTfLite',
    'MpLlmConverterConvertHfTokenizer',
)


class LlmConverterTest(googletest.TestCase, parameterized.TestCase):

  def get_fake_action(self, input_dtype):
    if input_dtype == 'bfloat16':
      # Create a TensorFlow bfloat16 tensor
      bfloat16_tensor = tf.constant([1.0, -1.0, 2.0, -2.0], dtype=tf.bfloat16)
      # Convert the TensorFlow tensor to a NumPy array
      tensor_value = bfloat16_tensor.numpy()
    else:
      tensor_value = np.array(
          [1.0, -1.0, 2.0, -2.0], dtype=np.dtype(input_dtype)
      )
    return converter_base.QuantizationAction(
        tensor_name='mdl_vars.params.lm.softmax.logits_ffn.w',
        target_name='params.lm.softmax.logits_ffn.w',
        quantize_axis=[0],
        quantize_bits=8,
        pack_dim=0,
        tensor_value=tensor_value,
    )

  @parameterized.parameters(
      {'input_dtype': 'float32'},
      {'input_dtype': 'float16'},
      {'input_dtype': 'bfloat16'},
      {'input_dtype': 'int8'},
  )
  def test_quantize_by_actions(self, input_dtype):
    out = llm_converter.quantize_by_actions(
        [self.get_fake_action(input_dtype)], backend='gpu', is_symmetric=True
    )

    if input_dtype == 'int8':
      # The values are pre-quantized and should be the same.
      np.testing.assert_allclose(
          out['params.lm.softmax.logits_ffn.w'][0],
          np.array([1, -1, 2, -2], dtype=np.int8),
      )
    else:
      np.testing.assert_allclose(
          out['params.lm.softmax.logits_ffn.w'][0],
          np.array([64, -64, 127, -127], dtype=np.int8),
      )
      np.testing.assert_allclose(
          out['params.lm.softmax.logits_ffn.w_quantized_scale'][0],
          np.array(0.015748, dtype=np.float32),
          rtol=1e-03,
      )

  def test_generate_cpu_tflite_receives_correct_args(self):
    c_lib = mock.MagicMock(spec_set=_LLM_CONVERTER_FUNCTIONS)
    c_lib.MpLlmConverterGenerateCpuTfLite.return_value = 0

    config = llm_converter.ConversionConfig(
        model_type='GEMMA_2B',
        backend='cpu',
        input_ckpt='/tmp/ckpt',
        ckpt_format='fake',
        output_dir='/tmp/',
        vocab_model_file='/tmp/vocab.model',
        output_tflite_file='/tmp/output.tflite',
    )
    llm_converter_lib = llm_converter._LlmConverter(c_lib)

    llm_converter_lib.combined_weight_bins_to_tflite(
        model_type=config.model_type,
        backend=config.backend,
        weight_path=config.output_dir,
        output_tflite_file=config.output_tflite_file,
        obfuscate=config.obfuscate,
        vocab_model_file=config.vocab_model_file,
    )

    c_lib.MpLlmConverterGenerateCpuTfLite.assert_called_once_with(
        b'GEMMA_2B',
        b'/tmp/',
        b'/tmp/vocab.model',
        True,
        b'/tmp/output.tflite',
        mock.ANY,  # error_message
    )

  def test_generate_cpu_tflite_propagates_failure(self):
    c_lib = mock.MagicMock(spec_set=_LLM_CONVERTER_FUNCTIONS)
    c_lib.MpLlmConverterGenerateCpuTfLite.return_value = 13  # Simulate failure

    config = llm_converter.ConversionConfig(
        model_type='GEMMA_2B',
        backend='cpu',
        input_ckpt='/tmp/ckpt',
        ckpt_format='fake',
        output_dir='/tmp/',
        vocab_model_file='/tmp/vocab.model',
        output_tflite_file='/tmp/output.tflite',
    )
    llm_converter_lib = llm_converter._LlmConverter(c_lib)

    with self.assertRaises(RuntimeError):
      llm_converter_lib.combined_weight_bins_to_tflite(
          model_type=config.model_type,
          backend=config.backend,
          weight_path=config.output_dir,
          output_tflite_file=config.output_tflite_file,
          obfuscate=config.obfuscate,
          vocab_model_file=config.vocab_model_file,
      )

  def test_generate_gpu_tflite_receives_correct_args(self):
    c_lib = mock.MagicMock(spec_set=_LLM_CONVERTER_FUNCTIONS)
    c_lib.MpLlmConverterGenerateGpuTfLite.return_value = 0

    config = llm_converter.ConversionConfig(
        model_type='GEMMA_2B',
        backend='gpu',
        input_ckpt='/tmp/ckpt',
        ckpt_format='fake',
        output_dir='/tmp/',
        vocab_model_file='/tmp/vocab.model',
        output_tflite_file='/tmp/output.tflite',
        use_dynamic_ple=True,
    )
    llm_converter_lib = llm_converter._LlmConverter(c_lib)

    llm_converter_lib.combined_weight_bins_to_tflite(
        model_type=config.model_type,
        backend=config.backend,
        weight_path=config.output_dir,
        output_tflite_file=config.output_tflite_file,
        obfuscate=config.obfuscate,
        vocab_model_file=config.vocab_model_file,
        lora_rank=4,
        lora_weight_path='/tmp/lora_weights',
        lora_output_tflite_file='/tmp/lora_output.tflite',
        lora_main_model_type='GEMMA_2B',
        image_encoder_file='/tmp/image_encoder.tflite',
        image_adapter_file='/tmp/image_adapter.tflite',
        submodel_type='GEMMA_2B',
        use_dynamic_ple=config.use_dynamic_ple,
        apply_srq=False,
    )

    c_lib.MpLlmConverterGenerateGpuTfLite.assert_called_once_with(
        b'GEMMA_2B',
        b'/tmp/',
        b'/tmp/vocab.model',
        True,
        False,
        b'/tmp/output.tflite',
        4,
        b'/tmp/lora_weights',
        b'/tmp/lora_output.tflite',
        b'GEMMA_2B',
        b'/tmp/image_encoder.tflite',
        b'/tmp/image_adapter.tflite',
        b'GEMMA_2B',
        True,
        False,
        mock.ANY,  # error_message
    )

  def test_generate_gpu_tflite_propagates_failure(self):
    c_lib = mock.MagicMock(spec_set=_LLM_CONVERTER_FUNCTIONS)
    c_lib.MpLlmConverterGenerateGpuTfLite.return_value = 13  # Simulate failure

    config = llm_converter.ConversionConfig(
        model_type='GEMMA_2B',
        backend='gpu',
        input_ckpt='/tmp/ckpt',
        ckpt_format='fake',
        output_dir='/tmp/',
        vocab_model_file='/tmp/vocab.model',
        output_tflite_file='/tmp/output.tflite',
        use_dynamic_ple=True,
    )
    llm_converter_lib = llm_converter._LlmConverter(c_lib)

    with self.assertRaises(RuntimeError):
      llm_converter_lib.combined_weight_bins_to_tflite(
          model_type=config.model_type,
          backend=config.backend,
          weight_path=config.output_dir,
          output_tflite_file=config.output_tflite_file,
          obfuscate=config.obfuscate,
          vocab_model_file=config.vocab_model_file,
          lora_rank=4,
          lora_weight_path='/tmp/lora_weights',
          lora_output_tflite_file='/tmp/lora_output.tflite',
          lora_main_model_type='GEMMA_2B',
          image_encoder_file='/tmp/image_encoder.tflite',
          image_adapter_file='/tmp/image_adapter.tflite',
          submodel_type='GEMMA_2B',
          use_dynamic_ple=config.use_dynamic_ple,
          apply_srq=False,
      )

  @mock.patch('os.path.isdir', return_value=True)
  @mock.patch('os.path.join', return_value='/tmp/spm.model')
  def test_convert_hf_tokenizer_receives_correct_args(
      self, mock_join, mock_isdir
  ):
    del mock_join, mock_isdir

    c_lib = mock.MagicMock(spec_set=_LLM_CONVERTER_FUNCTIONS)
    c_lib.MpLlmConverterConvertHfTokenizer.return_value = 0

    llm_converter_lib = llm_converter._LlmConverter(c_lib)
    output_file = llm_converter_lib.convert_bpe_vocab(
        vocab_model_file='/tmp/hf_tokenizer', output_dir='/tmp'
    )

    c_lib.MpLlmConverterConvertHfTokenizer.assert_called_once_with(
        b'/tmp/hf_tokenizer', b'/tmp/spm.model', mock.ANY
    )
    self.assertEqual(output_file, '/tmp/spm.model')

  @mock.patch('os.path.isdir', return_value=True)
  @mock.patch('os.path.join', return_value='/tmp/spm.model')
  def test_convert_hf_tokenizer_propagates_failure(self, mock_join, mock_isdir):
    del mock_join, mock_isdir

    c_lib = mock.MagicMock(spec_set=_LLM_CONVERTER_FUNCTIONS)
    c_lib.MpLlmConverterConvertHfTokenizer.return_value = 13  # Simulate failure

    llm_converter_lib = llm_converter._LlmConverter(c_lib)
    with self.assertRaises(RuntimeError):
      llm_converter_lib.convert_bpe_vocab(
          vocab_model_file='/tmp/hf_tokenizer', output_dir='/tmp'
      )


if __name__ == '__main__':
  googletest.main()
