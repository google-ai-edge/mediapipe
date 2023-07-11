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
import random
import numpy as np
import tensorflow as tf

from mediapipe.model_maker.python.vision.core import image_utils
from mediapipe.model_maker.python.vision.core import test_utils
from mediapipe.model_maker.python.vision.image_classifier import dataset


class DatasetTest(tf.test.TestCase):

  def setUp(self):
    super().setUp()
    self.image_path = os.path.join(self.get_temp_dir(), 'random_image_dir')
    if os.path.exists(self.image_path):
      return
    os.mkdir(self.image_path)
    for class_name in ('daisy', 'tulips'):
      class_subdir = os.path.join(self.image_path, class_name)
      os.mkdir(class_subdir)
      test_utils.write_filled_jpeg_file(
          os.path.join(class_subdir, '0.jpeg'),
          [random.uniform(0, 255) for _ in range(3)],
          224,
      )

  def test_split(self):
    ds = tf.data.Dataset.from_tensor_slices([[0, 1], [1, 1], [0, 0], [1, 0]])
    data = dataset.Dataset(dataset=ds, label_names=['pos', 'neg'], size=4)
    train_data, test_data = data.split(fraction=0.5)

    self.assertLen(train_data, 2)
    for i, elem in enumerate(train_data._dataset):
      self.assertTrue((elem.numpy() == np.array([i, 1])).all())
    self.assertEqual(train_data.num_classes, 2)
    self.assertEqual(train_data.label_names, ['pos', 'neg'])

    self.assertLen(test_data, 2)
    for i, elem in enumerate(test_data._dataset):
      self.assertTrue((elem.numpy() == np.array([i, 0])).all())
    self.assertEqual(test_data.num_classes, 2)
    self.assertEqual(test_data.label_names, ['pos', 'neg'])

  def test_from_folder(self):
    data = dataset.Dataset.from_folder(dirname=self.image_path)

    self.assertLen(data, 2)
    self.assertEqual(data.num_classes, 2)
    self.assertEqual(data.label_names, ['daisy', 'tulips'])
    for image, label in data.gen_tf_dataset():
      self.assertTrue(label.numpy() == 1 or label.numpy() == 0)
      if label.numpy() == 0:
        raw_image_tensor = image_utils.load_image(
            os.path.join(self.image_path, 'daisy', '0.jpeg')
        )
      else:
        raw_image_tensor = image_utils.load_image(
            os.path.join(self.image_path, 'tulips', '0.jpeg')
        )
      self.assertTrue((image.numpy() == raw_image_tensor.numpy()).all())

  def test_from_tfds(self):
    # TODO: Remove this once tfds download error is fixed.
    self.skipTest('Temporarily skip the unittest due to tfds download error.')
    train_data, validation_data, test_data = (
        dataset.Dataset.from_tfds('beans'))
    self.assertIsInstance(train_data.gen_tf_dataset(), tf.data.Dataset)
    self.assertLen(train_data, 1034)
    self.assertEqual(train_data.num_classes, 3)
    self.assertEqual(train_data.label_names,
                     ['angular_leaf_spot', 'bean_rust', 'healthy'])

    self.assertIsInstance(validation_data.gen_tf_dataset(), tf.data.Dataset)
    self.assertLen(validation_data, 133)
    self.assertEqual(validation_data.num_classes, 3)
    self.assertEqual(validation_data.label_names,
                     ['angular_leaf_spot', 'bean_rust', 'healthy'])

    self.assertIsInstance(test_data.gen_tf_dataset(), tf.data.Dataset)
    self.assertLen(test_data, 128)
    self.assertEqual(test_data.num_classes, 3)
    self.assertEqual(test_data.label_names,
                     ['angular_leaf_spot', 'bean_rust', 'healthy'])


if __name__ == '__main__':
  tf.test.main()
