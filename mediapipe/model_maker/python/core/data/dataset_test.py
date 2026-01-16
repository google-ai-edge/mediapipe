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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from mediapipe.model_maker.python.core.data import dataset as ds
from mediapipe.model_maker.python.core.utils import test_util


class DatasetTest(tf.test.TestCase):

  def test_split(self):
    dataset = tf.data.Dataset.from_tensor_slices([[0, 1], [1, 1], [0, 0],
                                                  [1, 0]])
    data = ds.Dataset(dataset, 4)
    train_data, test_data = data.split(0.5)

    self.assertLen(train_data, 2)
    self.assertIsInstance(train_data, ds.Dataset)
    self.assertIsInstance(test_data, ds.Dataset)
    for i, elem in enumerate(train_data.gen_tf_dataset()):
      self.assertTrue((elem.numpy() == np.array([i, 1])).all())

    self.assertLen(test_data, 2)
    for i, elem in enumerate(test_data.gen_tf_dataset()):
      self.assertTrue((elem.numpy() == np.array([i, 0])).all())

  def test_len(self):
    size = 4
    dataset = tf.data.Dataset.from_tensor_slices([[0, 1], [1, 1], [0, 0],
                                                  [1, 0]])
    data = ds.Dataset(dataset, size)
    self.assertLen(data, size)

  def test_gen_tf_dataset(self):
    input_dim = 8
    data = test_util.create_dataset(
        data_size=2, input_shape=[input_dim], num_classes=2)

    dataset = data.gen_tf_dataset()
    self.assertLen(dataset, 2)
    for (feature, label) in dataset:
      self.assertTrue((tf.shape(feature).numpy() == np.array([1, 8])).all())
      self.assertTrue((tf.shape(label).numpy() == np.array([1])).all())

    dataset2 = data.gen_tf_dataset(batch_size=2)
    self.assertLen(dataset2, 1)
    for (feature, label) in dataset2:
      self.assertTrue((tf.shape(feature).numpy() == np.array([2, 8])).all())
      self.assertTrue((tf.shape(label).numpy() == np.array([2])).all())

    dataset3 = data.gen_tf_dataset(batch_size=2, is_training=True, shuffle=True)
    self.assertEqual(dataset3.cardinality(), 1)
    for (feature, label) in dataset3.take(10):
      self.assertTrue((tf.shape(feature).numpy() == np.array([2, 8])).all())
      self.assertTrue((tf.shape(label).numpy() == np.array([2])).all())


if __name__ == '__main__':
  tf.test.main()
