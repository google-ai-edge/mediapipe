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
"""Metrics utility library."""

import tensorflow as tf


def _get_binary_sparse_metric(metric: tf.metrics.Metric):
  """Helper method to create a BinarySparse version of a tf.keras.Metric.

  BinarySparse is an implementation where the update_state(y_true, y_pred) takes
  in shapes y_true=(batch_size, 1) y_pred=(batch_size, 2). Note that this only
  supports the binary classification case, and that class_id=0 is the negative
  class and class_id=1 is the positive class.

  Currently supported tf.metric.Metric classes
    1. BinarySparseRecallAtPrecision
    2. BinarySparsePrecisionAtRecall

  Args:
    metric: A tf.metric.Metric class for which we want to generate a
      BinarySparse version of this metric.

  Returns:
    A class for the BinarySparse version of the specified tf.metrics.Metric
  """

  class BinarySparseMetric(metric):
    """A BinarySparse wrapper class for a tf.keras.Metric.

    This class has the same parameters and functions as the underlying
    metric class. For example, the parameters for BinarySparseRecallAtPrecision
    is the same as tf.keras.metrics.RecallAtPrecision. The only new constraint
    is that class_id must be set to 1 (or not specified) for the Binary metric.
    """

    def __init__(self, *args, **kwargs):
      if 'class_id' in kwargs and kwargs['class_id'] != 1:
        raise ValueError(
            f'Custom BinarySparseMetric for class:{metric.__name__} is '
            'only supported for class_id=1, got class_id='
            f'{kwargs["class_id"]} instead'
        )
      else:
        kwargs['class_id'] = 1
      super().__init__(*args, **kwargs)

    def update_state(self, y_true, y_pred, sample_weight=None):
      y_true = tf.cast(tf.reshape(y_true, [-1]), tf.int32)
      y_true_one_hot = tf.one_hot(y_true, 2)
      return super().update_state(
          y_true_one_hot, y_pred, sample_weight=sample_weight
      )

  return BinarySparseMetric


def _get_sparse_metric(metric: tf.metrics.Metric):
  """Helper method to create a Sparse version of a tf.keras.Metric.

  Sparse is an implementation where the update_state(y_true, y_pred) takes in
  shapes y_true=(batch_size, 1) and y_pred=(batch_size, num_classes).

  Currently supported tf.metrics.Metric classes:
    1. tf.metrics.Recall
    2. tf.metrics.Precision

  Args:
    metric: A tf.metric.Metric class for which we want to generate a Sparse
      version of this metric.

  Returns:
    A class for the Sparse version of the specified tf.keras.Metric.
  """

  class SparseMetric(metric):
    """A Sparse wrapper class for a tf.keras.Metric."""

    def update_state(self, y_true, y_pred, sample_weight=None):
      y_pred = tf.math.argmax(y_pred, axis=-1)
      return super().update_state(y_true, y_pred, sample_weight=sample_weight)

  return SparseMetric


class BinaryAUC(tf.keras.metrics.AUC):
  """A Binary AUC metric for binary classification tasks.

  For update state, the shapes of y_true and y_pred are expected to be:
    - y_true: [batch_size x 1] array of 0 for negatives and 1 for positives
    - y_pred: [batch_size x 2] array of probabilities where y_pred[:,0] are the
      probabilities of the 0th(negative) class and y_pred[:,1] are the
      probabilities of the 1st(positive) class

  See https://www.tensorflow.org/api_docs/python/tf/keras/metrics/AUC for
    details.
  """

  def update_state(self, y_true, y_pred, sample_weight=None):
    super().update_state(y_true, y_pred[:, 1], sample_weight)


SparseRecall = _get_sparse_metric(tf.metrics.Recall)
SparsePrecision = _get_sparse_metric(tf.metrics.Precision)
BinarySparseRecallAtPrecision = _get_binary_sparse_metric(
    tf.metrics.RecallAtPrecision
)
BinarySparsePrecisionAtRecall = _get_binary_sparse_metric(
    tf.metrics.PrecisionAtRecall
)
