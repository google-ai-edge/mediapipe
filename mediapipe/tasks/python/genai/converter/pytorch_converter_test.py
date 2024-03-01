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

"""Unit tests for pytorch_converter."""

import os

from absl.testing import absltest
from absl.testing import parameterized

from mediapipe.tasks.python.genai.converter import pytorch_converter
from mediapipe.tasks.python.test import test_utils

_TEST_DATA_DIR = 'mediapipe/tasks/testdata/text'
_PYTORCH_FILE = test_utils.get_test_data_path(
    os.path.join(_TEST_DATA_DIR, 'falcon_rw_1b_test_weight.pt')
)


class PytorchConverterTest(parameterized.TestCase):
  VARIABLE_NAMES = [
      'transformer.word_embeddings.weight',
      'transformer.h.0.input_layernorm.weight',
      'transformer.h.0.input_layernorm.bias',
      'transformer.h.0.self_attention.query_key_value.weight',
      'transformer.h.0.self_attention.query_key_value.bias',
      'transformer.h.0.self_attention.dense.weight',
      'transformer.h.0.self_attention.dense.bias',
      'transformer.h.0.post_attention_layernorm.weight',
      'transformer.h.0.post_attention_layernorm.bias',
      'transformer.h.0.mlp.dense_h_to_4h.weight',
      'transformer.h.0.mlp.dense_h_to_4h.bias',
      'transformer.h.0.mlp.dense_4h_to_h.weight',
      'transformer.h.0.mlp.dense_4h_to_h.bias',
      'transformer.ln_f.weight',
      'transformer.ln_f.bias',
      'lm_head.weight',
  ]

  def test_init(self):
    loader = pytorch_converter.PytorchCkptLoader(
        ckpt_path=_PYTORCH_FILE,
        is_symmetric=True,
        attention_quant_bits=8,
        feedforward_quant_bits=8,
        embedding_quant_bits=8,
        special_model='FALCON_RW_1B',
        backend='cpu',
    )
    self.assertEqual(loader._ckpt_path, _PYTORCH_FILE)
    self.assertEqual(loader._is_symmetric, True)
    self.assertEqual(loader._attention_quant_bits, 8)
    self.assertEqual(loader._feedforward_quant_bits, 8)

  @parameterized.product(
      quant_bits=(4, 8),
  )
  def test_load_to_actions(self, quant_bits):
    loader = pytorch_converter.PytorchCkptLoader(
        ckpt_path=_PYTORCH_FILE,
        is_symmetric=True,
        attention_quant_bits=8,
        feedforward_quant_bits=quant_bits,
        embedding_quant_bits=8,
        special_model='FALCON_RW_1B',
        backend='cpu',
    )
    actions = loader.load_to_actions()
    # There are 16 layers in the model, but qkv weight and bias would be
    # decomposed to q, k, v tensors, so there would be 20 quantization actions.
    self.assertEqual(sum(len(action) for action in actions), 20)


if __name__ == '__main__':
  absltest.main()
