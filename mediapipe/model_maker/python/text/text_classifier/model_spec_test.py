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
"""Tests for model_spec."""

import os
import tempfile
from unittest import mock as unittest_mock

import tensorflow as tf

from mediapipe.model_maker.python.text.text_classifier import hyperparameters as hp
from mediapipe.model_maker.python.text.text_classifier import model_options as classifier_model_options
from mediapipe.model_maker.python.text.text_classifier import model_spec as ms


class ModelSpecTest(tf.test.TestCase):

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

  def test_predefined_bert_spec(self):
    model_spec_obj = ms.SupportedModels.MOBILEBERT_CLASSIFIER.value()
    self.assertIsInstance(model_spec_obj, ms.BertClassifierSpec)
    self.assertEqual(model_spec_obj.name, 'MobileBERT')
    self.assertTrue(model_spec_obj.files)
    self.assertTrue(model_spec_obj.do_lower_case)
    self.assertEqual(
        model_spec_obj.tflite_input_name,
        {
            'ids': 'serving_default_input_word_ids:0',
            'mask': 'serving_default_input_mask:0',
            'segment_ids': 'serving_default_input_type_ids:0',
        },
    )
    self.assertEqual(
        model_spec_obj.model_options,
        classifier_model_options.BertModelOptions(
            seq_len=128, do_fine_tuning=True, dropout_rate=0.1))
    self.assertEqual(
        model_spec_obj.hparams,
        hp.BertHParams(
            epochs=3,
            batch_size=48,
            learning_rate=3e-5,
            distribution_strategy='off',
        ),
    )

  def test_predefined_average_word_embedding_spec(self):
    model_spec_obj = (
        ms.SupportedModels.AVERAGE_WORD_EMBEDDING_CLASSIFIER.value())
    self.assertIsInstance(model_spec_obj, ms.AverageWordEmbeddingClassifierSpec)
    self.assertEqual(model_spec_obj.name, 'AverageWordEmbedding')
    self.assertEqual(
        model_spec_obj.model_options,
        classifier_model_options.AverageWordEmbeddingModelOptions(
            seq_len=256,
            wordvec_dim=16,
            do_lower_case=True,
            vocab_size=10000,
            dropout_rate=0.2))
    self.assertEqual(
        model_spec_obj.hparams,
        hp.AverageWordEmbeddingHParams(
            epochs=10,
            batch_size=32,
            learning_rate=0,
            steps_per_epoch=None,
            shuffle=False,
            distribution_strategy='off',
            num_gpus=0,
            tpu='',
        ),
    )

  def test_custom_bert_spec(self):
    custom_bert_classifier_options = (
        classifier_model_options.BertModelOptions(
            seq_len=512, do_fine_tuning=False, dropout_rate=0.3))
    model_spec_obj = (
        ms.SupportedModels.MOBILEBERT_CLASSIFIER.value(
            model_options=custom_bert_classifier_options))
    self.assertEqual(model_spec_obj.model_options,
                     custom_bert_classifier_options)

  def test_custom_average_word_embedding_spec(self):
    custom_hparams = hp.AverageWordEmbeddingHParams(
        learning_rate=0.4,
        batch_size=64,
        epochs=10,
        steps_per_epoch=10,
        shuffle=True,
        export_dir='foo/bar',
        distribution_strategy='mirrored',
        num_gpus=3,
        tpu='tpu/address',
    )
    custom_average_word_embedding_model_options = (
        classifier_model_options.AverageWordEmbeddingModelOptions(
            seq_len=512,
            wordvec_dim=32,
            do_lower_case=False,
            vocab_size=5000,
            dropout_rate=0.5))
    model_spec_obj = (
        ms.SupportedModels.AVERAGE_WORD_EMBEDDING_CLASSIFIER.value(
            model_options=custom_average_word_embedding_model_options,
            hparams=custom_hparams))
    self.assertEqual(model_spec_obj.model_options,
                     custom_average_word_embedding_model_options)
    self.assertEqual(model_spec_obj.hparams, custom_hparams)


if __name__ == '__main__':
  # Load compressed models from tensorflow_hub
  os.environ['TFHUB_MODEL_LOAD_FORMAT'] = 'COMPRESSED'
  tf.test.main()
