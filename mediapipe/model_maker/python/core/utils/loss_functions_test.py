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

import math
from typing import Optional

from absl.testing import parameterized
import tensorflow as tf

from mediapipe.model_maker.python.core.utils import loss_functions


class LossFunctionsTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.named_parameters(
      dict(testcase_name='no_sample_weight', sample_weight=None),
      dict(
          testcase_name='with_sample_weight',
          sample_weight=tf.constant([0.2, 0.2, 0.3, 0.1, 0.2])))
  def test_focal_loss_gamma_0_is_cross_entropy(
      self, sample_weight: Optional[tf.Tensor]):
    y_true = tf.constant([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0, 0], [0, 1,
                                                                       0]])
    y_pred = tf.constant([[0.7, 0.1, 0.2], [0.6, 0.3, 0.1], [0.1, 0.5, 0.4],
                          [0.8, 0.1, 0.1], [0.4, 0.5, 0.1]])

    tf_cce = tf.keras.losses.CategoricalCrossentropy(
        from_logits=False,
        reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE)
    focal_loss = loss_functions.FocalLoss(gamma=0)
    self.assertAllClose(
        tf_cce(y_true, y_pred, sample_weight=sample_weight),
        focal_loss(y_true, y_pred, sample_weight=sample_weight), 1e-4)

  def test_focal_loss_with_sample_weight(self):
    y_true = tf.constant([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0, 0], [0, 1,
                                                                       0]])
    y_pred = tf.constant([[0.7, 0.1, 0.2], [0.6, 0.3, 0.1], [0.1, 0.5, 0.4],
                          [0.8, 0.1, 0.1], [0.4, 0.5, 0.1]])

    focal_loss = loss_functions.FocalLoss(gamma=0)

    sample_weight = tf.constant([0.2, 0.2, 0.3, 0.1, 0.2])

    self.assertGreater(
        focal_loss(y_true=y_true, y_pred=y_pred),
        focal_loss(y_true=y_true, y_pred=y_pred, sample_weight=sample_weight))

  @parameterized.named_parameters(
      dict(testcase_name='gt_0.1', y_pred=tf.constant([0.1, 0.9])),
      dict(testcase_name='gt_0.3', y_pred=tf.constant([0.3, 0.7])),
      dict(testcase_name='gt_0.5', y_pred=tf.constant([0.5, 0.5])),
      dict(testcase_name='gt_0.7', y_pred=tf.constant([0.7, 0.3])),
      dict(testcase_name='gt_0.9', y_pred=tf.constant([0.9, 0.1])),
  )
  def test_focal_loss_decreases_with_increasing_gamma(self, y_pred: tf.Tensor):
    y_true = tf.constant([[1, 0]])

    focal_loss_gamma_0 = loss_functions.FocalLoss(gamma=0)
    loss_gamma_0 = focal_loss_gamma_0(y_true, y_pred)
    focal_loss_gamma_0p5 = loss_functions.FocalLoss(gamma=0.5)
    loss_gamma_0p5 = focal_loss_gamma_0p5(y_true, y_pred)
    focal_loss_gamma_1 = loss_functions.FocalLoss(gamma=1)
    loss_gamma_1 = focal_loss_gamma_1(y_true, y_pred)
    focal_loss_gamma_2 = loss_functions.FocalLoss(gamma=2)
    loss_gamma_2 = focal_loss_gamma_2(y_true, y_pred)
    focal_loss_gamma_5 = loss_functions.FocalLoss(gamma=5)
    loss_gamma_5 = focal_loss_gamma_5(y_true, y_pred)

    self.assertGreater(loss_gamma_0, loss_gamma_0p5)
    self.assertGreater(loss_gamma_0p5, loss_gamma_1)
    self.assertGreater(loss_gamma_1, loss_gamma_2)
    self.assertGreater(loss_gamma_2, loss_gamma_5)

  @parameterized.named_parameters(
      dict(testcase_name='index_0', true_class=0),
      dict(testcase_name='index_1', true_class=1),
      dict(testcase_name='index_2', true_class=2),
  )
  def test_focal_loss_class_weight_is_applied(self, true_class: int):
    class_weight = [1.0, 3.0, 10.0]
    y_pred = tf.constant([[1.0, 1.0, 1.0]]) / 3.0
    y_true = tf.one_hot(true_class, depth=3)[tf.newaxis, :]
    expected_loss = -math.log(1.0 / 3.0) * class_weight[true_class]

    loss_fn = loss_functions.FocalLoss(gamma=0, class_weight=class_weight)
    loss = loss_fn(y_true, y_pred)
    self.assertNear(loss, expected_loss, 1e-4)


if __name__ == '__main__':
  tf.test.main()
