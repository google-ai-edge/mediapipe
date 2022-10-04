# Copyright 2022 The MediaPipe Authors. All Rights Reserved.
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

# Dependency imports

import tensorflow as tf

from mediapipe.model_maker.python.core.data import classification_dataset


class ClassificationDataLoaderTest(tf.test.TestCase):

  def test_split(self):

    class MagicClassificationDataLoader(
        classification_dataset.ClassificationDataset):

      def __init__(self, dataset, size, index_to_label, value):
        super(MagicClassificationDataLoader,
              self).__init__(dataset, size, index_to_label)
        self.value = value

      def split(self, fraction):
        return self._split(fraction, self.index_to_label, self.value)

    # Some dummy inputs.
    magic_value = 42
    num_classes = 2
    index_to_label = (False, True)

    # Create data loader from sample data.
    ds = tf.data.Dataset.from_tensor_slices([[0, 1], [1, 1], [0, 0], [1, 0]])
    data = MagicClassificationDataLoader(ds, len(ds), index_to_label,
                                         magic_value)

    # Train/Test data split.
    fraction = .25
    train_data, test_data = data.split(fraction)

    # `split` should return instances of child DataLoader.
    self.assertIsInstance(train_data, MagicClassificationDataLoader)
    self.assertIsInstance(test_data, MagicClassificationDataLoader)

    # Make sure number of entries are right.
    self.assertEqual(len(train_data.gen_tf_dataset()), len(train_data))
    self.assertLen(train_data, fraction * len(ds))
    self.assertLen(test_data, len(ds) - len(train_data))

    # Make sure attributes propagated correctly.
    self.assertEqual(train_data.num_classes, num_classes)
    self.assertEqual(test_data.index_to_label, index_to_label)
    self.assertEqual(train_data.value, magic_value)
    self.assertEqual(test_data.value, magic_value)


if __name__ == '__main__':
  tf.test.main()
