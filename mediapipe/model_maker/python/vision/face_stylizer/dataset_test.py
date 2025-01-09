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

import numpy as np
import tensorflow as tf

from mediapipe.model_maker.python.vision.core import image_utils
from mediapipe.model_maker.python.vision.face_stylizer import dataset
from mediapipe.tasks.python.test import test_utils


class DatasetTest(tf.test.TestCase):

  def setUp(self):
    super().setUp()

  def test_from_image(self):
    test_image_file = 'input/style/cartoon/cartoon.jpg'
    input_image_path = test_utils.get_test_data_path(test_image_file)
    data = dataset.Dataset.from_image(filename=input_image_path)
    self.assertEqual(data.num_classes, 1)
    self.assertEqual(data.label_names, ['cartoon'])
    self.assertLen(data, 1)

  def test_from_image_raise_value_error_for_invalid_path(self):
    with self.assertRaisesRegex(ValueError, 'Unsupported image formats: .zip'):
      dataset.Dataset.from_image(filename='input/style/cartoon/cartoon.zip')

  def test_from_image_raise_value_error_for_invalid_image(self):
    with self.assertRaisesRegex(ValueError, 'Invalid image'):
      test_image_file = 'input/style/sketch/boy-6030802_1280.jpg'
      input_image_path = test_utils.get_test_data_path(test_image_file)
      dataset.Dataset.from_image(filename=input_image_path)


if __name__ == '__main__':
  tf.test.main()
