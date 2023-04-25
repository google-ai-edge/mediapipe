# Copyright 2022 The MediaPipe Authors.
#
# Licensed under the Apache License, Version 2.0 (the 'License');
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an 'AS IS' BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os

# Dependency imports

import tensorflow as tf

from mediapipe.model_maker.python.core.tasks import classifier
from mediapipe.model_maker.python.core.utils import test_util


class MockClassifier(classifier.Classifier):
  """A mock class with implementation of abstract methods for testing."""

  def train(self, train_data, validation_data=None, **kwargs):
    pass

  def evaluate(self, data, **kwargs):
    pass


class ClassifierTest(tf.test.TestCase):

  def setUp(self):
    super(ClassifierTest, self).setUp()
    label_names = ['cat', 'dog']
    self.model = MockClassifier(
        model_spec=None, label_names=label_names, shuffle=False)
    self.model.model = test_util.build_model(input_shape=[4], num_classes=2)

  def _check_nonempty_file(self, filepath):
    self.assertTrue(os.path.isfile(filepath))
    self.assertGreater(os.path.getsize(filepath), 0)

  def test_export_labels(self):
    export_path = os.path.join(self.get_temp_dir(), 'export/')
    self.model.export_labels(export_dir=export_path)
    self._check_nonempty_file(os.path.join(export_path, 'labels.txt'))


if __name__ == '__main__':
  tf.test.main()
