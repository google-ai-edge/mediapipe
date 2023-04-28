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

import csv
import filecmp
import os
import tempfile
import unittest
from unittest import mock as unittest_mock

import tensorflow as tf

from mediapipe.model_maker.python.text import text_classifier
from mediapipe.tasks.python.test import test_utils


@unittest.skip('b/275624089')
class TextClassifierTest(tf.test.TestCase):

  _AVERAGE_WORD_EMBEDDING_JSON_FILE = (
      test_utils.get_test_data_path('average_word_embedding_metadata.json'))
  _BERT_CLASSIFIER_JSON_FILE = test_utils.get_test_data_path(
      'bert_metadata.json'
  )

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

  def _get_data(self):
    labels_and_text = (('pos', 'super good'), (('neg', 'really bad')))
    csv_file = os.path.join(self.create_tempdir(), 'data.csv')
    if os.path.exists(csv_file):
      return csv_file
    fieldnames = ['text', 'label']
    with open(csv_file, 'w') as f:
      writer = csv.DictWriter(f, fieldnames=fieldnames)
      writer.writeheader()
      for label, text in labels_and_text:
        writer.writerow({'text': text, 'label': label})
    csv_params = text_classifier.CSVParams(
        text_column='text', label_column='label')
    all_data = text_classifier.Dataset.from_csv(
        filename=csv_file, csv_params=csv_params)
    return all_data.split(0.5)

  def test_create_and_train_average_word_embedding_model(self):
    train_data, validation_data = self._get_data()
    options = (
        text_classifier.TextClassifierOptions(
            supported_model=(text_classifier.SupportedModels
                             .AVERAGE_WORD_EMBEDDING_CLASSIFIER),
            hparams=text_classifier.HParams(
                epochs=1, batch_size=1, learning_rate=0)))
    average_word_embedding_classifier = (
        text_classifier.TextClassifier.create(train_data, validation_data,
                                              options))

    _, accuracy = average_word_embedding_classifier.evaluate(validation_data)
    self.assertGreaterEqual(accuracy, 0.0)

    # Test export_model
    average_word_embedding_classifier.export_model()
    output_metadata_file = os.path.join(options.hparams.export_dir,
                                        'metadata.json')
    output_tflite_file = os.path.join(options.hparams.export_dir,
                                      'model.tflite')

    self.assertTrue(os.path.exists(output_tflite_file))
    self.assertGreater(os.path.getsize(output_tflite_file), 0)

    self.assertTrue(os.path.exists(output_metadata_file))
    self.assertGreater(os.path.getsize(output_metadata_file), 0)
    filecmp.clear_cache()
    self.assertTrue(
        filecmp.cmp(
            output_metadata_file,
            self._AVERAGE_WORD_EMBEDDING_JSON_FILE,
            shallow=False))

  def test_create_and_train_bert(self):
    train_data, validation_data = self._get_data()
    options = text_classifier.TextClassifierOptions(
        supported_model=text_classifier.SupportedModels.MOBILEBERT_CLASSIFIER,
        model_options=text_classifier.BertModelOptions(
            do_fine_tuning=False, seq_len=2),
        hparams=text_classifier.HParams(
            epochs=1,
            batch_size=1,
            learning_rate=3e-5,
            distribution_strategy='off'))
    bert_classifier = text_classifier.TextClassifier.create(
        train_data, validation_data, options)

    _, accuracy = bert_classifier.evaluate(validation_data)
    self.assertGreaterEqual(accuracy, 0.0)

    # Test export_model
    bert_classifier.export_model()
    output_metadata_file = os.path.join(
        options.hparams.export_dir, 'metadata.json'
    )
    output_tflite_file = os.path.join(
        options.hparams.export_dir, 'model.tflite'
    )

    self.assertTrue(os.path.exists(output_tflite_file))
    self.assertGreater(os.path.getsize(output_tflite_file), 0)

    self.assertTrue(os.path.exists(output_metadata_file))
    self.assertGreater(os.path.getsize(output_metadata_file), 0)
    filecmp.clear_cache()
    self.assertTrue(
        filecmp.cmp(
            output_metadata_file, self._BERT_CLASSIFIER_JSON_FILE, shallow=False
        )
    )

  def test_label_mismatch(self):
    options = (
        text_classifier.TextClassifierOptions(
            supported_model=(
                text_classifier.SupportedModels.MOBILEBERT_CLASSIFIER)))
    train_tf_dataset = tf.data.Dataset.from_tensor_slices([[0]])
    train_data = text_classifier.Dataset(train_tf_dataset, 1, ['foo'])
    validation_tf_dataset = tf.data.Dataset.from_tensor_slices([[0]])
    validation_data = text_classifier.Dataset(validation_tf_dataset, 1, ['bar'])
    with self.assertRaisesRegex(
        ValueError,
        'Training data label names .* not equal to validation data label names'
    ):
      text_classifier.TextClassifier.create(train_data, validation_data,
                                            options)

  def test_options_mismatch(self):
    train_data, validation_data = self._get_data()

    avg_options = (
        text_classifier.TextClassifierOptions(
            supported_model=(
                text_classifier.SupportedModels.MOBILEBERT_CLASSIFIER),
            model_options=text_classifier.AverageWordEmbeddingModelOptions()))
    with self.assertRaisesRegex(
        ValueError, 'Expected AVERAGE_WORD_EMBEDDING_CLASSIFIER, got'
        ' SupportedModels.MOBILEBERT_CLASSIFIER'):
      text_classifier.TextClassifier.create(train_data, validation_data,
                                            avg_options)

    bert_options = (
        text_classifier.TextClassifierOptions(
            supported_model=(text_classifier.SupportedModels
                             .AVERAGE_WORD_EMBEDDING_CLASSIFIER),
            model_options=text_classifier.BertModelOptions()))
    with self.assertRaisesRegex(
        ValueError, 'Expected MOBILEBERT_CLASSIFIER, got'
        ' SupportedModels.AVERAGE_WORD_EMBEDDING_CLASSIFIER'):
      text_classifier.TextClassifier.create(train_data, validation_data,
                                            bert_options)


if __name__ == '__main__':
  # Load compressed models from tensorflow_hub
  os.environ['TFHUB_MODEL_LOAD_FORMAT'] = 'COMPRESSED'
  tf.test.main()
