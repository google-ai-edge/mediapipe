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

import tensorflow as tf
import tensorflow_hub

from mediapipe.model_maker.python.core.utils import hub_loader
from mediapipe.model_maker.python.text.text_classifier import bert_tokenizer
from mediapipe.model_maker.python.text.text_classifier import model_spec
from mediapipe.model_maker.python.text.text_classifier import model_with_tokenizer


class BertTokenizerTest(tf.test.TestCase):
  _SEQ_LEN = 128

  def setUp(self):
    super().setUp()
    # Mock tempfile.gettempdir() to be unique for each test to avoid race
    # condition when downloading model since these tests may run in parallel.
    mock_gettempdir = unittest_mock.patch.object(
        tempfile,
        "gettempdir",
        return_value=self.create_tempdir(),
        autospec=True,
    )
    self.mock_gettempdir = mock_gettempdir.start()
    self.addCleanup(mock_gettempdir.stop)
    self._ms = model_spec.SupportedModels.MOBILEBERT_CLASSIFIER.value()
    self._tokenizer = self._create_tokenizer()
    self._model = self._create_model()

  def _create_tokenizer(self):
    vocab_file = os.path.join(
        tensorflow_hub.resolve(self._ms.get_path()), "assets", "vocab.txt"
    )
    return bert_tokenizer.BertFastTokenizer(vocab_file, True, self._SEQ_LEN)

  def _create_model(self):
    encoder_inputs = dict(
        input_word_ids=tf.keras.layers.Input(
            shape=(self._SEQ_LEN,),
            dtype=tf.int32,
            name="input_word_ids",
        ),
        input_mask=tf.keras.layers.Input(
            shape=(self._SEQ_LEN,),
            dtype=tf.int32,
            name="input_mask",
        ),
        input_type_ids=tf.keras.layers.Input(
            shape=(self._SEQ_LEN,),
            dtype=tf.int32,
            name="input_type_ids",
        ),
    )
    renamed_inputs = dict(
        input_ids=encoder_inputs["input_word_ids"],
        input_mask=encoder_inputs["input_mask"],
        segment_ids=encoder_inputs["input_type_ids"],
    )
    encoder = hub_loader.HubKerasLayerV1V2(
        self._ms.get_path(),
        signature="tokens",
        output_key="pooled_output",
        trainable=True,
    )
    pooled_output = encoder(renamed_inputs)

    output = tf.keras.layers.Dropout(rate=0.1)(pooled_output)
    initializer = tf.keras.initializers.TruncatedNormal(stddev=0.02)
    output = tf.keras.layers.Dense(
        2,
        kernel_initializer=initializer,
        name="output",
        activation="softmax",
        dtype=tf.float32,
    )(output)
    return tf.keras.Model(inputs=encoder_inputs, outputs=output)

  def test_model_with_tokenizer(self):
    model = model_with_tokenizer.ModelWithTokenizer(
        self._tokenizer, self._model
    )
    output = model(tf.constant(["Example input".encode("utf-8")]))
    self.assertAllEqual(output.shape, (2,))
    self.assertEqual(tf.reduce_sum(output), 1)


if __name__ == "__main__":
  tf.test.main()
