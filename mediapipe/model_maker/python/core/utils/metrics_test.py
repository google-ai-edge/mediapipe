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

import functools

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
        [0.3, 0.7],  # 1, 0 n
        [0.3, 0.7],  # 1, 1 y
    ])

  def _assert_metric_equals(self, metric, value, sparse=True):
    if not sparse:
      y_true = tf.one_hot(self.y_true, 2)
      metric.update_state(y_true, self.y_pred)
    else:
      metric.update_state(self.y_true, self.y_pred)
    self.assertEqual(metric.result(), value)

  def test_sparse_recall(self):
    metric = metrics.BinarySparseRecall()
    self._assert_metric_equals(metric, 1 / 3)

  def test_sparse_precision(self):
    metric = metrics.BinarySparsePrecision()
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
    metric = metrics.BinaryAUC(num_thresholds=1000, class_id=1)
    self._assert_metric_equals(metric, 0.7222222, sparse=False)

  def test_binary_sparse_auc(self):
    metric = metrics.BinarySparseAUC(num_thresholds=1000)
    self._assert_metric_equals(metric, 0.7222222)


class OneVsAllSparseMetricTest(tf.test.TestCase, parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.y_true = np.array([[0], [0], [2], [1], [1], [0], [2], [1]])
    self.y_pred = np.array([
        [0.9, 0.1, 0.0],  # Correctly classified as 0
        [0.8, 0.1, 0.1],  # Correctly classified as 0
        [0.1, 0.7, 0.2],  # 2 misclassified as 1
        [0.7, 0.1, 0.2],  # 1 misclassified as 0
        [0.6, 0.4, 0.0],  # 1 misclassified as 0
        [0.3, 0.6, 0.1],  # 0 misclassified as 1
        [0.1, 0.1, 0.8],  # Correctly classified as 2
        [0.0, 0.7, 0.3],  # Correctly classified as 1
    ])
    self.sample_weight = np.ones_like(self.y_true)

  def _assert_metric_equals(self, metric, expected_result):
    metric.update_state(self.y_true, self.y_pred, self.sample_weight)
    self.assertEqual(metric.result(), expected_result)

  @parameterized.named_parameters(
      dict(
          testcase_name='one_vs_all_precision',
          metric_cls=metrics.MultiClassSparsePrecision,
          class0_expected_result=1 / 2,  # TP: 2, FP: 4
          class1_expected_result=1 / 3,  # TP: 1, FP: 2
          class2_expected_result=1.0,  # TP: 1, FP: 0
      ),
      dict(
          testcase_name='one_vs_all_recall',
          metric_cls=metrics.MultiClassSparseRecall,
          class0_expected_result=2 / 3,  # TP: 2, FN: 1
          class1_expected_result=1 / 3,  # TP: 1, FN: 2
          class2_expected_result=1 / 2,  # TP: 1, FN: 1
      ),
  )
  def test_metric(
      self,
      metric_cls,
      class0_expected_result,
      class1_expected_result,
      class2_expected_result,
  ):
    self._assert_metric_equals(metric_cls(class_id=0), class0_expected_result)
    self._assert_metric_equals(metric_cls(class_id=1), class1_expected_result)
    self._assert_metric_equals(metric_cls(class_id=2), class2_expected_result)


class MaskedMetricTest(tf.test.TestCase, parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.y_true = np.array([
        [0, 1],
        [0, 0],
        [0, 0],
        [1, 1],
    ])
    self.y_pred = np.array([
        [0.8, 0.8],  # 1, 1
        [0.1, 0.3],  # 0, 0
        [0.1, 0.8],  # 0, 1
        [0.3, 0.7],  # 0, 1
    ])
    self.mask = np.array([
        [1, 1],
        [1, 0],
        [0, 1],
        [1, 0],
    ])
    self.mask_all_ones = np.array([
        [1, 1],
        [1, 1],
        [1, 1],
        [1, 1],
    ])

  def _assert_masked_metric_equals(self, metric, value):
    # Sanity check that the metric without mask is the same as all ones mask.
    metric.update_state(self.y_true, self.y_pred)
    result1 = metric.result()
    metric.reset_state()
    metric.update_state(self.y_true, self.y_pred, self.mask_all_ones)
    result2 = metric.result()
    self.assertEqual(result1, result2)
    metric.reset_state()
    # Check that the metric with mask matches value.
    metric.update_state(self.y_true, self.y_pred, self.mask)
    self.assertEqual(metric.result(), value)

  @parameterized.named_parameters(
      dict(
          testcase_name='masked_binary_precision',
          metric_cls=metrics.MaskedBinaryPrecision,
          class0_value=0.0,
          class1_value=1 / 2,
      ),
      dict(
          testcase_name='masked_binary_recall',
          metric_cls=metrics.MaskedBinaryRecall,
          class0_value=0.0,
          class1_value=1.0,
      ),
      dict(
          testcase_name='masked_binary_recall_at_precision',
          metric_cls=functools.partial(
              metrics.MaskedBinaryRecallAtPrecision, 1 / 2
          ),
          class0_value=1.0,
          class1_value=1.0,
      ),
      dict(
          testcase_name='masked_binary_precision_at_recall',
          metric_cls=functools.partial(
              metrics.MaskedBinaryPrecisionAtRecall, 1 / 2
          ),
          class0_value=1 / 2,
          class1_value=1 / 2,
      ),
  )
  def test_metric(self, metric_cls, class0_value, class1_value):
    metric = metric_cls(class_id=0)
    self._assert_masked_metric_equals(metric, class0_value)
    metric = metric_cls(class_id=1)
    self._assert_masked_metric_equals(metric, class1_value)


if __name__ == '__main__':
  tf.test.main()
