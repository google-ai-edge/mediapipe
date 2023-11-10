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


from absl.testing import parameterized
import numpy as np
import tensorflow as tf

from mediapipe.model_maker.python.core.utils import metrics


class SparseMetricTest(tf.test.TestCase, parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.y_true = np.array([0, 0, 1, 1, 0, 1])
    self.y_pred = np.array([
        [0.9, 0.1],  # 0, 0 y
        [0.8, 0.2],  # 0, 0 y
        [0.7, 0.3],  # 0, 1 n
        [0.6, 0.4],  # 0, 1 n
        [0.3, 0.7],  # 1, 0 y
        [0.3, 0.7],  # 1, 1 y
    ])

  def _assert_metric_equals(self, metric, value):
    metric.update_state(self.y_true, self.y_pred)
    self.assertEqual(metric.result(), value)

  def test_sparse_recall(self):
    metric = metrics.SparseRecall()
    self._assert_metric_equals(metric, 1 / 3)

  def test_sparse_precision(self):
    metric = metrics.SparsePrecision()
    self._assert_metric_equals(metric, 1 / 2)

  def test_binary_sparse_recall_at_precision(self):
    metric = metrics.BinarySparseRecallAtPrecision(1.0)
    self._assert_metric_equals(metric, 0.0)  # impossible to achieve precision=1
    metric = metrics.BinarySparseRecallAtPrecision(0.4)
    self._assert_metric_equals(metric, 1.0)

  def test_binary_sparse_precision_at_recall(self):
    metric = metrics.BinarySparsePrecisionAtRecall(1.0)
    self._assert_metric_equals(metric, 3 / 4)
    metric = metrics.BinarySparsePrecisionAtRecall(0.7)
    self._assert_metric_equals(metric, 3 / 4)

  def test_binary_sparse_precision_at_recall_class_id_error(self):
    # class_id=1 case should not error
    _ = metrics.BinarySparsePrecisionAtRecall(1.0, class_id=1)
    # class_id=2 case should error
    with self.assertRaisesRegex(
        ValueError,
        'Custom BinarySparseMetric for class:PrecisionAtRecall is only'
        ' supported for class_id=1, got class_id=2 instead',
    ):
      _ = metrics.BinarySparsePrecisionAtRecall(1.0, class_id=2)

  def test_binary_auc(self):
    metric = metrics.BinaryAUC(num_thresholds=1000)
    self._assert_metric_equals(metric, 0.7222222)


if __name__ == '__main__':
  tf.test.main()
