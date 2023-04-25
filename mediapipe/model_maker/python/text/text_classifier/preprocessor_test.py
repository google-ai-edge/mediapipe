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
import os
import tempfile
from unittest import mock as unittest_mock

import numpy as np
import numpy.testing as npt
import tensorflow as tf

from mediapipe.model_maker.python.text.text_classifier import dataset as text_classifier_ds
from mediapipe.model_maker.python.text.text_classifier import model_spec
from mediapipe.model_maker.python.text.text_classifier import preprocessor


class PreprocessorTest(tf.test.TestCase):
  CSV_PARAMS_ = text_classifier_ds.CSVParameters(
      text_column='text', label_column='label')

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

  def _get_csv_file(self):
    labels_and_text = (('pos', 'super super super super good'),
                       (('neg', 'really bad')))
    csv_file = os.path.join(self.get_temp_dir(), 'data.csv')
    if os.path.exists(csv_file):
      return csv_file
    fieldnames = ['text', 'label']
    with open(csv_file, 'w') as f:
      writer = csv.DictWriter(f, fieldnames=fieldnames)
      writer.writeheader()
      for label, text in labels_and_text:
        writer.writerow({'text': text, 'label': label})
    return csv_file

  def test_average_word_embedding_preprocessor(self):
    csv_file = self._get_csv_file()
    dataset = text_classifier_ds.Dataset.from_csv(
        filename=csv_file, csv_params=self.CSV_PARAMS_)
    average_word_embedding_preprocessor = (
        preprocessor.AverageWordEmbeddingClassifierPreprocessor(
            seq_len=5,
            do_lower_case=True,
            texts=['super super super super good', 'really bad'],
            vocab_size=7))
    preprocessed_dataset = (
        average_word_embedding_preprocessor.preprocess(dataset))
    labels = []
    features_list = []
    for features, label in preprocessed_dataset.gen_tf_dataset():
      self.assertEqual(label.shape, [1])
      labels.append(label.numpy()[0])
      self.assertEqual(features.shape, [1, 5])
      features_list.append(features.numpy()[0])
    self.assertEqual(labels, [1, 0])
    npt.assert_array_equal(
        np.stack(features_list), np.array([[1, 3, 3, 3, 3], [1, 5, 6, 0, 0]]))

  def test_bert_preprocessor(self):
    csv_file = self._get_csv_file()
    dataset = text_classifier_ds.Dataset.from_csv(
        filename=csv_file, csv_params=self.CSV_PARAMS_)
    bert_spec = model_spec.SupportedModels.MOBILEBERT_CLASSIFIER.value()
    bert_preprocessor = preprocessor.BertClassifierPreprocessor(
        seq_len=5,
        do_lower_case=bert_spec.do_lower_case,
        uri=bert_spec.downloaded_files.get_path(),
    )
    preprocessed_dataset = bert_preprocessor.preprocess(dataset)
    labels = []
    input_masks = []
    for features, label in preprocessed_dataset.gen_tf_dataset():
      self.assertEqual(label.shape, [1])
      labels.append(label.numpy()[0])
      self.assertSameElements(
          features.keys(), ['input_word_ids', 'input_mask', 'input_type_ids'])
      for feature in features.values():
        self.assertEqual(feature.shape, [1, 5])
      input_masks.append(features['input_mask'].numpy()[0])
      npt.assert_array_equal(features['input_type_ids'].numpy()[0],
                             [0, 0, 0, 0, 0])
    npt.assert_array_equal(
        np.stack(input_masks), np.array([[1, 1, 1, 1, 1], [1, 1, 1, 1, 0]]))
    self.assertEqual(labels, [1, 0])


if __name__ == '__main__':
  # Load compressed models from tensorflow_hub
  os.environ['TFHUB_MODEL_LOAD_FORMAT'] = 'COMPRESSED'
  tf.test.main()
