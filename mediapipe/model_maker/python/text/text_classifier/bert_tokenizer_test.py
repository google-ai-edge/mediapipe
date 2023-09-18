# Copyright 2022 The MediaPipe Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import tempfile
from unittest import mock as unittest_mock

from absl.testing import parameterized
import tensorflow as tf
import tensorflow_hub

from mediapipe.model_maker.python.text.text_classifier import bert_tokenizer
from mediapipe.model_maker.python.text.text_classifier import model_spec


class BertTokenizerTest(tf.test.TestCase, parameterized.TestCase):

  def setUp(self):
    super().setUp()
    # Mock tempfile.gettempdir() to be unique for each test to avoid race
    # condition when downloading model since these tests may run in parallel.
    mock_gettempdir = unittest_mock.patch.object(
        tempfile,
        'gettempdir',
        return_value=self.create_tempdir(),
        autospec=True,
    )
    self.mock_gettempdir = mock_gettempdir.start()
    self.addCleanup(mock_gettempdir.stop)
    ms = model_spec.SupportedModels.MOBILEBERT_CLASSIFIER.value()
    self._vocab_file = os.path.join(
        tensorflow_hub.resolve(ms.get_path()), 'assets', 'vocab.txt'
    )

  @parameterized.named_parameters(
      dict(
          testcase_name='fulltokenizer',
          tokenizer_class=bert_tokenizer.BertFullTokenizer,
      ),
      dict(
          testcase_name='fasttokenizer',
          tokenizer_class=bert_tokenizer.BertFastTokenizer,
      ),
  )
  def test_bert_full_tokenizer(self, tokenizer_class):
    tokenizer = tokenizer_class(self._vocab_file, True, 16)
    text_input = tf.constant(['this is an éxamplé input ¿foo'.encode('utf-8')])
    result = tokenizer.process(text_input)
    self.assertAllEqual(
        result['input_word_ids'],
        [
            101,
            2023,
            2003,
            2019,
            2742,
            7953,
            1094,
            29379,
            102,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
        ],
    )
    self.assertAllEqual(
        result['input_mask'], [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0]
    )
    self.assertAllEqual(
        result['input_type_ids'],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    )


if __name__ == '__main__':
  tf.test.main()
