"""Tests for llm_converter."""

from absl.testing import parameterized
import numpy as np
import tensorflow as tf

import unittest
from mediapipe.tasks.python.genai.converter import converter_base
from mediapipe.tasks.python.genai.converter import llm_converter


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


if __name__ == '__main__':
  googletest.main()
