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

from typing import Sequence
import math
from typing import Optional
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
    3. BinarySparsePrecision
    4. BinarySparseRecall

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
      super().update_state(y_true_one_hot, y_pred, sample_weight=sample_weight)

  return BinarySparseMetric


def _get_multiclass_sparse_metric(metric: tf.metrics.Metric):
  """Creates a sparse version of a tf.keras.Metric for multi-class tasks.

  This class evaluates metrics for a specified class_id in a one-vs-all binary
  fashion where all other classes are treated as a combined negative class, and
  class_id is the positive class. The model's predicted class is always the
  argmax of the scores.

  Currently supported tf.metrics.Metric classes:
    1. tf.metrics.Recall
    2. tf.metrics.Precision

  For update state, the shapes of y_true, y_pred, and sample_weight are expected
  to be:
    - y_true: [batch_size x 1] array of indices, indicating the true labels.
    - y_pred: [batch_size x num_classes] array of probabilities where
    y_pred[:,i] is the probability of the i-th class.
    - sample_weight: [batch_size x 1] array of sample weights.

  Args:
    metric: A tf.metric.Metric class for which we want to generate a multiclass
      sparse version of this metric.

  Returns:
    A class for the multiclass sparse version of the specified tf.keras.Metric.
  """

  class MultiClassSparseMetric(metric):
    """A multiclass sparse wrapper class for a tf.keras.Metric."""

    def __init__(self, *args, **kwargs):
      if 'class_id' not in kwargs:
        raise ValueError(
            f'Custom MultiClassSparseMetric for class:{metric.__name__} must'
            ' have class_id specified upon initialization.'
        )
      super().__init__(*args, **kwargs)
      self._class_id = kwargs['class_id']

    def update_state(self, y_true, y_pred, sample_weight=None):
      num_classes = y_pred.shape[-1]
      y_true = tf.reshape(y_true, [-1])
      y_true = tf.where(y_true != self._class_id, -1, y_true)
      y_true = tf.one_hot(y_true, depth=num_classes)

      y_pred = tf.argmax(y_pred, axis=-1)
      y_pred = tf.cast(y_pred, tf.int32)
      y_pred = tf.where(y_pred != self._class_id, -1, y_pred)
      y_pred = tf.one_hot(y_pred, depth=num_classes)
      super().update_state(
          y_true,
          y_pred,
          tf.reshape(sample_weight, [-1])
          if sample_weight is not None
          else None,
      )

  return MultiClassSparseMetric


class BinaryAUC(tf.keras.metrics.AUC):
  """A Binary AUC metric for multi-label tasks.

  class_id is the index of the class/label that we want to compute Binary AUC
  for.

  For update state, the shapes of y_true, y_pred, and sample_weight are expected
  to be:
    - y_true: [batch_size x num_classes] array of one-hot encoded labels (note,
    these could be in a multi-label setting where the sum of y_true can be > 1)
    - y_pred: [batch_size x num_classes] array of probabilities where
    y_pred[:,i] is the probability of the i-th class.
    - sample_weight: [batch_size x num_classes] array of sample weights.
  """

  def __init__(self, *args, class_id: int = 1, **kwargs):
    super().__init__(*args, **kwargs)
    self._class_id = class_id

  def update_state(self, y_true, y_pred, sample_weight=None):
    super().update_state(
        y_true[:, self._class_id],
        y_pred[:, self._class_id],
        sample_weight[:, self._class_id] if sample_weight is not None else None,
    )


class BinarySparseAUC(tf.keras.metrics.AUC):
  """A Binary Sparse AUC metric for binary classification tasks.

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


def _get_masked_binary_metric(metric: tf.metrics.Metric):
  """Helper method to create a Masked version of a tf.keras.Metric."""

  class MaskedBinaryMetric(metric):
    """A Masked Binary metric wrapper class for a tf.keras.Metric.

    This class assumes that the underlying metric is used in a binary fashion
    with `class_id` specified in **kwargs.

    The sample_weight in `update_state` is used as a mask over the metric
    calculations. sample_weight should have shape [batch_size x num_classes]
    when specified, and we only care about sample_weight[:, class_id].
    """

    def __init__(self, *args, **kwargs):
      assert 'class_id' in kwargs
      if 'class_id' not in kwargs:
        raise ValueError(
            f'Custom MaskedBinaryMetric for class:{metric.__name__} must have '
            'class_id specified upon initialization.'
        )
      self._class_id = kwargs['class_id']
      super().__init__(*args, **kwargs)

    def update_state(self, y_true, y_pred, sample_weight=None):
      return super().update_state(
          y_true,
          y_pred,
          sample_weight[:, self._class_id]
          if sample_weight is not None
          else None,
      )

  return MaskedBinaryMetric


class WeightedSumMetric(tf.keras.metrics.Metric):
  """A weighted sum metric for combining multiclass sparse metrics.

  addend_metrics: A list of metrics to be combined.
  weights: A list of weights for each metric. If None, then the weights of the
  metrics will be equal.
  name: The name of the weighted sum metric.
  """

  def __init__(
      self,
      addend_metrics: Sequence[tf.keras.metrics.Metric],
      weights: Optional[Sequence[float]] = None,
      name: str = 'weighted_sum',
      **kwargs,
  ):
    super().__init__(name=name, **kwargs)
    self._metrics = addend_metrics
    self._weights = weights

    if self._weights is None:
      self._weights = [1.0 / len(self._metrics)] * len(self._metrics)
    else:
      if len(self._weights) != len(self._metrics):
        raise ValueError('Number of weights must match the number of metrics.')

    if any(w < 0 for w in self._weights) or not math.isclose(
        sum(self._weights), 1.0
    ):
      raise ValueError('Weights must be non-negative and sum to 1.')

  def update_state(self, y_true, y_pred, sample_weight=None):
    for metric in self._metrics:
      metric.update_state(y_true, y_pred, sample_weight)

  def result(self):
    weighted_sum = 0.0
    for metric, weight in zip(self._metrics, self._weights):
      weighted_sum += weight * metric.result()
    return weighted_sum

  def reset_states(self):
    for metric in self._metrics:
      metric.reset_states()


BinarySparseRecall = _get_binary_sparse_metric(tf.metrics.Recall)
BinarySparsePrecision = _get_binary_sparse_metric(tf.metrics.Precision)
BinarySparseRecallAtPrecision = _get_binary_sparse_metric(
    tf.metrics.RecallAtPrecision
)
BinarySparsePrecisionAtRecall = _get_binary_sparse_metric(
    tf.metrics.PrecisionAtRecall
)
MultiClassSparseRecall = _get_multiclass_sparse_metric(tf.metrics.Recall)
MultiClassSparsePrecision = _get_multiclass_sparse_metric(tf.metrics.Precision)
MaskedBinaryPrecision = _get_masked_binary_metric(tf.metrics.Precision)
MaskedBinaryRecall = _get_masked_binary_metric(tf.metrics.Recall)
MaskedBinaryRecallAtPrecision = _get_masked_binary_metric(
    tf.metrics.RecallAtPrecision
)
MaskedBinaryPrecisionAtRecall = _get_masked_binary_metric(
    tf.metrics.PrecisionAtRecall
)
