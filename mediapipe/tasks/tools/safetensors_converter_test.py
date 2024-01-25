"""Unit tests for safetensors_converter."""

import os

from absl.testing import absltest
from absl.testing import parameterized

from mediapipe.tasks.python.test import test_utils
from mediapipe.tasks.tools import safetensors_converter

_TEST_DATA_DIR = 'mediapipe/tasks/testdata/text'
_SAFETENSORS_FILE = test_utils.get_test_data_path(
    os.path.join(_TEST_DATA_DIR, 'loader_testdata.safetensors')
)


class SafetensorsConverterTest(parameterized.TestCase):
  VARIABLE_NAMES = [
      'model.embed_tokens.weight',
      'model.layers.0.input_layernorm.bias',
      'model.layers.0.input_layernorm.weight',
      'model.layers.0.mlp.down_proj.weight',
      'model.layers.0.mlp.gate_proj.weight',
      'model.layers.0.mlp.up_proj.weight',
      'model.layers.0.post_attention_layernorm.bias',
      'model.layers.0.post_attention_layernorm.weight',
      'model.layers.0.self_attn.k_proj.weight',
      'model.layers.0.self_attn.o_proj.weight',
      'model.layers.0.self_attn.q_proj.weight',
      'model.layers.0.self_attn.v_proj.weight',
      'model.norm.bias',
      'model.norm.weight',
      'lm_head.weight',
  ]

  def test_init(self):
    loader = safetensors_converter.SafetensorsCkptLoader(
        ckpt_path=_SAFETENSORS_FILE,
        is_symmetric=True,
        attention_quant_bits=8,
        feedforward_quant_bits=8,
        embedding_quant_bits=8,
        special_model='STABLELM_4E1T_3B',
    )
    self.assertEqual(loader._ckpt_path, _SAFETENSORS_FILE)
    self.assertEqual(loader._is_symmetric, True)
    self.assertEqual(loader._attention_quant_bits, 8)
    self.assertEqual(loader._feedforward_quant_bits, 8)

  @parameterized.product(
      quant_bits=(4, 8),
  )
  def test_load_to_actions(self, quant_bits):
    loader = safetensors_converter.SafetensorsCkptLoader(
        ckpt_path=_SAFETENSORS_FILE,
        is_symmetric=True,
        attention_quant_bits=8,
        feedforward_quant_bits=quant_bits,
        embedding_quant_bits=8,
        special_model='STABLELM_4E1T_3B',
    )
    actions = loader.load_to_actions()
    self.assertLen(actions, 15)


if __name__ == '__main__':
  absltest.main()
