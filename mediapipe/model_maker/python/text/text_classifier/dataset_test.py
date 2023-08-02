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

import tensorflow as tf

from mediapipe.model_maker.python.text.text_classifier import dataset


class DatasetTest(tf.test.TestCase):

  def _get_csv_file(self):
    labels_and_text = (('neutral', 'indifferent'), ('pos', 'extremely great'),
                       ('neg', 'totally awful'), ('pos', 'super good'),
                       ('neg', 'really bad'))
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

  def test_from_csv(self):
    csv_file = self._get_csv_file()
    csv_params = dataset.CSVParameters(text_column='text', label_column='label')
    data = dataset.Dataset.from_csv(filename=csv_file, csv_params=csv_params)
    self.assertLen(data, 5)
    self.assertEqual(data.num_classes, 3)
    self.assertEqual(data.label_names, ['neg', 'neutral', 'pos'])
    data_values = set([(text.numpy()[0], label.numpy()[0])
                       for text, label in data.gen_tf_dataset()])
    expected_data_values = set([(b'indifferent', 1), (b'extremely great', 2),
                                (b'totally awful', 0), (b'super good', 2),
                                (b'really bad', 0)])
    self.assertEqual(data_values, expected_data_values)

  def test_split(self):
    ds = tf.data.Dataset.from_tensor_slices(['good', 'bad', 'neutral', 'odd'])
    data = dataset.Dataset(ds, ['pos', 'neg'], size=4)
    train_data, test_data = data.split(0.5)
    expected_train_data = [b'good', b'bad']
    expected_test_data = [b'neutral', b'odd']

    self.assertLen(train_data, 2)
    train_data_values = [elem.numpy() for elem in train_data._dataset]
    self.assertEqual(train_data_values, expected_train_data)
    self.assertEqual(train_data.num_classes, 2)
    self.assertEqual(train_data.label_names, ['pos', 'neg'])

    self.assertLen(test_data, 2)
    test_data_values = [elem.numpy() for elem in test_data._dataset]
    self.assertEqual(test_data_values, expected_test_data)
    self.assertEqual(test_data.num_classes, 2)
    self.assertEqual(test_data.label_names, ['pos', 'neg'])


if __name__ == '__main__':
  tf.test.main()
