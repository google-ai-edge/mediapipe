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

from typing import Any, Tuple, TypeVar

# Dependency imports

import tensorflow as tf

from mediapipe.model_maker.python.core.data import classification_dataset

_DatasetT = TypeVar(
    '_DatasetT', bound='ClassificationDatasetTest.MagicClassificationDataset')


class ClassificationDatasetTest(tf.test.TestCase):

  def test_split(self):

    class MagicClassificationDataset(
        classification_dataset.ClassificationDataset):
      """A mock classification dataset class for testing purpose.

      Attributes:
        value: A value variable stored by the mock dataset class for testing.
      """

      def __init__(self, dataset: tf.data.Dataset, size: int,
                   index_by_label: Any, value: Any):
        super().__init__(
            dataset=dataset, size=size, index_by_label=index_by_label)
        self.value = value

      def split(self, fraction: float) -> Tuple[_DatasetT, _DatasetT]:
        return self._split(fraction, self.index_by_label, self.value)

    # Some dummy inputs.
    magic_value = 42
    num_classes = 2
    index_by_label = (False, True)

    # Create data loader from sample data.
    ds = tf.data.Dataset.from_tensor_slices([[0, 1], [1, 1], [0, 0], [1, 0]])
    data = MagicClassificationDataset(
        dataset=ds,
        size=len(ds),
        index_by_label=index_by_label,
        value=magic_value)

    # Train/Test data split.
    fraction = .25
    train_data, test_data = data.split(fraction=fraction)

    # `split` should return instances of child DataLoader.
    self.assertIsInstance(train_data, MagicClassificationDataset)
    self.assertIsInstance(test_data, MagicClassificationDataset)

    # Make sure number of entries are right.
    self.assertEqual(len(train_data.gen_tf_dataset()), len(train_data))
    self.assertLen(train_data, fraction * len(ds))
    self.assertLen(test_data, len(ds) - len(train_data))

    # Make sure attributes propagated correctly.
    self.assertEqual(train_data.num_classes, num_classes)
    self.assertEqual(test_data.index_by_label, index_by_label)
    self.assertEqual(train_data.value, magic_value)
    self.assertEqual(test_data.value, magic_value)


if __name__ == '__main__':
  tf.test.main()
