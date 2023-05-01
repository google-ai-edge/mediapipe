# Copyright 2023 The MediaPipe Authors.
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

import tensorflow as tf

from mediapipe.model_maker.python.vision.face_stylizer import dataset
from mediapipe.tasks.python.test import test_utils


class DatasetTest(tf.test.TestCase):

  def setUp(self):
    super().setUp()
    # TODO: Replace the stylize image dataset with licensed images.
    self._test_data_dirname = 'testdata'

  def test_from_folder(self):
    input_data_dir = test_utils.get_test_data_path(self._test_data_dirname)
    data = dataset.Dataset.from_folder(dirname=input_data_dir)
    self.assertEqual(data.num_classes, 2)
    self.assertEqual(data.label_names, ['cartoon', 'sketch'])
    self.assertLen(data, 2)

  def test_from_folder_raise_value_error_for_invalid_path(self):
    with self.assertRaisesRegex(ValueError, 'Invalid input data directory'):
      dataset.Dataset.from_folder(dirname='invalid')

  def test_from_folder_raise_value_error_for_valid_no_data_path(self):
    input_data_dir = test_utils.get_test_data_path('face_stylizer')
    with self.assertRaisesRegex(
        ValueError, 'No images found under given directory'
    ):
      dataset.Dataset.from_folder(dirname=input_data_dir)


if __name__ == '__main__':
  tf.test.main()
