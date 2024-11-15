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

import math
import tempfile
from typing import Dict, Optional, Sequence
from unittest import mock as unittest_mock

from absl.testing import parameterized
import tensorflow as tf

from mediapipe.model_maker.python.core.utils import loss_functions


class MaskedBinaryCrossentropyTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.named_parameters(
      dict(
          testcase_name='mask_all_ones',
          mask=tf.constant(
              [[1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1]]
          ),
      ),
      dict(
          testcase_name='mask_some_zeros',
          mask=tf.constant(
              [[1, 1, 1], [0, 1, 1], [0, 0, 1], [1, 0, 0], [1, 0, 1]]
          ),
      ),
  )
  def test_masked_binary_crossentropy(self, mask):
    y_true = tf.constant(
        [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0, 0], [0, 1, 0]]
    )
    y_pred = tf.constant([
        [0.7, 0.1, 0.2],
        [0.6, 0.3, 0.1],
        [0.1, 0.5, 0.4],
        [0.8, 0.1, 0.1],
        [0.4, 0.5, 0.1],
    ])

    loss = loss_functions.MaskedBinaryCrossentropy(
        reduction=tf.keras.losses.Reduction.SUM
    )
    loss_value = loss(y_true, y_pred, sample_weight=mask)

    y_true_masked = tf.boolean_mask(y_true, mask)[:, tf.newaxis]
    y_pred_masked = tf.boolean_mask(y_pred, mask)[:, tf.newaxis]
    print(y_true_masked)
    print(y_pred_masked)

    gt_loss = tf.keras.losses.BinaryCrossentropy(
        reduction=tf.keras.losses.Reduction.SUM
    )
    gt_loss_value = gt_loss(y_true_masked, y_pred_masked)

    self.assertAlmostEqual(loss_value, gt_loss_value, delta=1e-4)


class FocalLossTest(tf.test.TestCase, parameterized.TestCase):

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


class SparseFocalLossTest(tf.test.TestCase):

  def test_sparse_focal_loss_matches_focal_loss(self):
    num_classes = 2
    y_pred = tf.constant([[0.8, 0.2], [0.3, 0.7]])
    y_true = tf.constant([1, 0])
    y_true_one_hot = tf.one_hot(y_true, num_classes)
    for gamma in [0.0, 0.5, 1.0]:
      expected_loss_fn = loss_functions.FocalLoss(gamma=gamma)
      loss_fn = loss_functions.SparseFocalLoss(
          gamma=gamma, num_classes=num_classes
      )
      expected_loss = expected_loss_fn(y_true_one_hot, y_pred)
      loss = loss_fn(y_true, y_pred)
      self.assertNear(loss, expected_loss, 1e-4)


class MockPerceptualLoss(loss_functions.PerceptualLoss):
  """A mock class with implementation of abstract methods for testing."""

  def __init__(
      self,
      use_mock_loss_op: bool = False,
      feature_weight: Optional[Sequence[float]] = None,
      loss_weight: Optional[loss_functions.PerceptualLossWeight] = None,
  ):
    super().__init__(feature_weight=feature_weight, loss_weight=loss_weight)
    if use_mock_loss_op:
      self._loss_op = lambda x, y: tf.math.reduce_mean(x - y)

  def _compute_features(self, img: tf.Tensor) -> Sequence[tf.Tensor]:
    return [tf.random.normal(shape=(1, 8, 8, 3))] * 5


class PerceptualLossTest(tf.test.TestCase, parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self._img1 = tf.fill(dims=(8, 8), value=0.2)
    self._img2 = tf.fill(dims=(8, 8), value=0.8)

  def test_invalid_feature_weight_raise_value_error(self):
    with self.assertRaisesRegex(
        ValueError,
        'Input feature weight length 2 is smaller than feature length 5',
    ):
      MockPerceptualLoss(feature_weight=[1.0, 2.0])(
          img1=self._img1, img2=self._img2
      )

  @parameterized.named_parameters(
      dict(
          testcase_name='default_loss_weight_and_loss_op',
          use_mock_loss_op=False,
          feature_weight=None,
          loss_weight=None,
          loss_values={
              'style_loss': 0.032839,
              'content_loss': 5.639870,
          },
      ),
      dict(
          testcase_name='style_loss_weight_is_0_default_loss_op',
          use_mock_loss_op=False,
          feature_weight=None,
          loss_weight=loss_functions.PerceptualLossWeight(style=0),
          loss_values={
              'style_loss': 0,
              'content_loss': 5.639870,
          },
      ),
      dict(
          testcase_name='content_loss_weight_is_0_default_loss_op',
          use_mock_loss_op=False,
          feature_weight=None,
          loss_weight=loss_functions.PerceptualLossWeight(content=0),
          loss_values={
              'style_loss': 0.032839,
              'content_loss': 0,
          },
      ),
      dict(
          testcase_name='customized_loss_weight_default_loss_op',
          use_mock_loss_op=False,
          feature_weight=None,
          loss_weight=loss_functions.PerceptualLossWeight(
              style=1.0, content=2.0
          ),
          loss_values={'style_loss': 0.032839, 'content_loss': 11.279739},
      ),
      dict(
          testcase_name=(
              'customized_feature_weight_and_loss_weight_default_loss_op'
          ),
          use_mock_loss_op=False,
          feature_weight=[1.0, 2.0, 3.0, 4.0, 5.0],
          loss_weight=loss_functions.PerceptualLossWeight(
              style=1.0, content=2.0
          ),
          loss_values={'style_loss': 0.164193, 'content_loss': 33.839218},
      ),
      dict(
          testcase_name='no_loss_change_if_extra_feature_weight_provided',
          use_mock_loss_op=False,
          feature_weight=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
          loss_weight=loss_functions.PerceptualLossWeight(
              style=1.0, content=2.0
          ),
          loss_values={
              'style_loss': 0.164193,
              'content_loss': 33.839218,
          },
      ),
      dict(
          testcase_name='customized_loss_weight_custom_loss_op',
          use_mock_loss_op=True,
          feature_weight=None,
          loss_weight=loss_functions.PerceptualLossWeight(
              style=1.0, content=2.0
          ),
          loss_values={'style_loss': 0.000395, 'content_loss': -1.533469},
      ),
  )
  def test_weighted_perceptul_loss(
      self,
      use_mock_loss_op: bool,
      feature_weight: Sequence[float],
      loss_weight: loss_functions.PerceptualLossWeight,
      loss_values: Dict[str, float],
  ):
    perceptual_loss = MockPerceptualLoss(
        use_mock_loss_op=use_mock_loss_op,
        feature_weight=feature_weight,
        loss_weight=loss_weight,
    )
    loss = perceptual_loss(img1=self._img1, img2=self._img2)
    self.assertEqual(list(loss.keys()), ['style_loss', 'content_loss'])
    self.assertNear(loss['style_loss'], loss_values['style_loss'], 1e-4)
    self.assertNear(loss['content_loss'], loss_values['content_loss'], 1e-4)


class VGGPerceptualLossTest(tf.test.TestCase, parameterized.TestCase):

  def setUp(self):
    super().setUp()
    # Mock tempfile.gettempdir() to be unique for each test to avoid race
    # condition when downloading model since these tests may run in parallel.
    mock_gettempdir = unittest_mock.patch.object(
        tempfile,
        'gettempdir',
        return_value=self.create_tempdir(),
        autospec=True,
    )
    self.mock_gettempdir = mock_gettempdir.start()
    self.addCleanup(mock_gettempdir.stop)
    self._img1 = tf.fill(dims=(1, 256, 256, 3), value=0.1)
    self._img2 = tf.fill(dims=(1, 256, 256, 3), value=0.9)

  @parameterized.named_parameters(
      dict(
          testcase_name='default_loss_weight',
          loss_weight=None,
          loss_values={
              'style_loss': 5.8363257e-06,
              'content_loss': 1.7016045,
          },
      ),
      dict(
          testcase_name='customized_loss_weight',
          loss_weight=loss_functions.PerceptualLossWeight(
              style=10.0, content=20.0
          ),
          loss_values={
              'style_loss': 5.8363257e-05,
              'content_loss': 34.03208,
          },
      ),
  )
  def test_vgg_perceptual_loss(self, loss_weight, loss_values):
    vgg_loss = loss_functions.VGGPerceptualLoss(loss_weight=loss_weight)
    loss = vgg_loss(img1=self._img1, img2=self._img2)
    self.assertEqual(list(loss.keys()), ['style_loss', 'content_loss'])
    self.assertNear(
        loss['style_loss'],
        loss_values['style_loss'],
        loss_values['style_loss'] / 1e5,
    )
    self.assertNear(
        loss['content_loss'],
        loss_values['content_loss'],
        loss_values['content_loss'] / 1e5,
    )


class ImagePerceptualQualityLossTest(tf.test.TestCase, parameterized.TestCase):

  def setUp(self):
    super().setUp()
    # Mock tempfile.gettempdir() to be unique for each test to avoid race
    # condition when downloading model since these tests may run in parallel.
    mock_gettempdir = unittest_mock.patch.object(
        tempfile,
        'gettempdir',
        return_value=self.create_tempdir(),
        autospec=True,
    )
    self.mock_gettempdir = mock_gettempdir.start()
    self.addCleanup(mock_gettempdir.stop)
    self._img1 = tf.fill(dims=(1, 256, 256, 3), value=0.1)
    self._img2 = tf.fill(dims=(1, 256, 256, 3), value=0.9)

  @parameterized.named_parameters(
      dict(
          testcase_name='default_loss_weight',
          loss_weight=None,
          loss_value=2.501612,
      ),
      dict(
          testcase_name='customized_loss_weight_zero_l1',
          loss_weight=loss_functions.PerceptualLossWeight(
              l1=0.0, style=10.0, content=20.0
          ),
          loss_value=34.032139,
      ),
      dict(
          testcase_name='customized_loss_weight_nonzero_l1',
          loss_weight=loss_functions.PerceptualLossWeight(
              l1=10.0, style=10.0, content=20.0
          ),
          loss_value=42.032139,
      ),
  )
  def test_image_perceptual_quality_loss(self, loss_weight, loss_value):
    image_quality_loss = loss_functions.ImagePerceptualQualityLoss(
        loss_weight=loss_weight
    )
    loss = image_quality_loss(img1=self._img1, img2=self._img2)
    self.assertNear(loss, loss_value, 1e-4)


if __name__ == '__main__':
  tf.test.main()
