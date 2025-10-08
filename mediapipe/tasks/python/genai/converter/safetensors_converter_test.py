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

"""Unit tests for safetensors_converter."""

import os
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np

from mediapipe.tasks.python.genai.converter import safetensors_converter
from mediapipe.tasks.python.test import test_utils

_TEST_DATA_DIR = 'mediapipe/tasks/testdata/text'
_SAFETENSORS_FILE = test_utils.get_test_data_path(
    os.path.join(_TEST_DATA_DIR, 'stablelm_3b_4e1t_test_weight.safetensors')
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
        backend='gpu',
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
        backend='gpu',
    )
    actions = loader.load_to_actions()
    self.assertLen(list(actions), 15)

  @mock.patch.object(safetensors_converter, '_SafetensorsReader')
  def testGemma3NConversion(self, MockReader):
    """Tests the conversion of a Gemma 3N model."""
    mock_reader_instance = MockReader.return_value
    gemma_3n_variable_names = [
        # Standard language model layers with the 'language_model.' prefix
        'language_model.model.embed_tokens.weight',
        'language_model.model.layers.0.input_layernorm.weight',
        'language_model.model.layers.0.mlp.down_proj.weight',
        'language_model.model.layers.0.self_attn.o_proj.weight',
        'language_model.model.norm.weight',
        # Vision tower layers that should be skipped
        'vision_tower.vision_tower.encoder.layers.0.blocks.0.attn.qkv.weight',
        'multi_modal_projector.linear_1.weight',
    ]
    mock_reader_instance.get_tensor_names.return_value = gemma_3n_variable_names
    mock_reader_instance.read_tensor_as_numpy.return_value = np.zeros(
        (1, 1), dtype=np.float32
    )

    loader = safetensors_converter.SafetensorsCkptLoader(
        ckpt_path='/fake/path',
        is_symmetric=True,
        attention_quant_bits=8,
        feedforward_quant_bits=8,
        embedding_quant_bits=8,
        special_model='GEMMA3N_4B',
        backend='gpu',
    )
    actions_list = list(loader.load_to_actions())

    # Check that the vision layers were skipped, and only 5 actions were created
    self.assertLen(actions_list, 5)

    # Check that the 'language_model.' prefix was correctly removed
    target_names = [actions[0].target_name for actions in actions_list]
    self.assertIn(
        'params.lm.softmax.logits_ffn.w', target_names
    )
    self.assertIn(
        'params.lm.transformer.x_layers_0.pre_layer_norm.scale', target_names
    )
    self.assertIn(
        'params.lm.transformer.x_layers_0.ff_layer.ffn_layer2.w', target_names
    )
    self.assertIn(
        'params.lm.transformer.x_layers_0.self_attention.post.w', target_names
    )
    self.assertIn('params.lm.final_ln.scale', target_names)


if __name__ == '__main__':
  absltest.main()
