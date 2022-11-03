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
"""Loss function utility library."""

from typing import Optional, Sequence

import tensorflow as tf


class FocalLoss(tf.keras.losses.Loss):
  """Implementation of focal loss (https://arxiv.org/pdf/1708.02002.pdf).

  This class computes the focal loss between labels and prediction. Focal loss
  is a weighted loss function that modulates the standard cross-entropy loss
  based on how well the neural network performs on a specific example of a
  class. The labels should be provided in a `one_hot` vector representation.
  There should be `#classes` floating point values per prediction.
  The loss is reduced across all samples using 'sum_over_batch_size' reduction
  (see https://www.tensorflow.org/api_docs/python/tf/keras/losses/Reduction).

  Example usage:
  >>> y_true = [[0, 1, 0], [0, 0, 1]]
  >>> y_pred = [[0.05, 0.95, 0], [0.1, 0.8, 0.1]]
  >>> gamma = 2
  >>> focal_loss = FocalLoss(gamma)
  >>> focal_loss(y_true, y_pred).numpy()
  0.9326

  >>> # Calling with 'sample_weight'.
  >>> focal_loss(y_true, y_pred, sample_weight=tf.constant([0.3, 0.7])).numpy()
  0.6528

  Usage with the `compile()` API:
  ```python
  model.compile(optimizer='sgd', loss=FocalLoss(gamma))
  ```

  """

  def __init__(self, gamma, class_weight: Optional[Sequence[float]] = None):
    """Constructor.

    Args:
      gamma: Focal loss gamma, as described in class docs.
      class_weight: A weight to apply to the loss, one for each class. The
        weight is applied for each input where the ground truth label matches.
    """
    super().__init__()
    # Used for clipping min/max values of probability values in y_pred to avoid
    # NaNs and Infs in computation.
    self._epsilon = 1e-7
    # This is a tunable "focusing parameter"; should be >= 0.
    # When gamma = 0, the loss returned is the standard categorical
    # cross-entropy loss.
    self._gamma = gamma
    self._class_weight = class_weight
    # tf.keras.losses.Loss class implementation requires a Reduction specified
    # in self.reduction. To use this reduction, we should use tensorflow's
    # compute_weighted_loss function however it is only compatible with v1 of
    # Tensorflow: https://www.tensorflow.org/api_docs/python/tf/compat/v1/losses/compute_weighted_loss?hl=en.  pylint: disable=line-too-long
    # So even though it is specified here, we don't use self.reduction in the
    # loss function call.
    self.reduction = tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE

  def __call__(self,
               y_true: tf.Tensor,
               y_pred: tf.Tensor,
               sample_weight: Optional[tf.Tensor] = None) -> tf.Tensor:
    if self._class_weight:
      class_weight = tf.convert_to_tensor(self._class_weight, dtype=tf.float32)
      label = tf.argmax(y_true, axis=1)
      loss_weight = tf.gather(class_weight, label)
    else:
      loss_weight = tf.ones(tf.shape(y_true)[0])
    y_true = tf.cast(y_true, y_pred.dtype)
    y_pred = tf.clip_by_value(y_pred, self._epsilon, 1 - self._epsilon)
    batch_size = tf.cast(tf.shape(y_pred)[0], y_pred.dtype)
    if sample_weight is None:
      sample_weight = tf.constant(1.0)
    weight_shape = sample_weight.shape
    weight_rank = weight_shape.ndims
    y_pred_rank = y_pred.shape.ndims
    if y_pred_rank - weight_rank == 1:
      sample_weight = tf.expand_dims(sample_weight, [-1])
    elif weight_rank != 0:
      raise ValueError(f'Unexpected sample_weights, should be either a scalar'
                       f'or a vector of batch_size:{batch_size.numpy()}')
    ce = -tf.math.log(y_pred)
    modulating_factor = tf.math.pow(1 - y_pred, self._gamma)
    losses = y_true * modulating_factor * ce * sample_weight
    losses = losses * loss_weight[:, tf.newaxis]
    # By default, this function uses "sum_over_batch_size" reduction for the
    # loss per batch.
    return tf.reduce_sum(losses) / batch_size
