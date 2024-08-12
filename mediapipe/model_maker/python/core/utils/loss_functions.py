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
"""Loss function utility library."""

import abc
from typing import Mapping, Sequence
import dataclasses
from typing import Any, Optional

import numpy as np
import tensorflow as tf

from mediapipe.model_maker.python.core.utils import file_util
from mediapipe.model_maker.python.core.utils import model_util
from official.modeling import tf_utils


_VGG_IMAGENET_PERCEPTUAL_MODEL_URL = 'https://storage.googleapis.com/mediapipe-assets/vgg_feature_extractor.tar.gz'


class MaskedBinaryCrossentropy(tf.keras.losses.BinaryCrossentropy):
  """Modified BinaryCrossentropy that uses sample_weights as a mask.

  This loss is similar to BinaryCrossentropy, but it allows for the
  specification of sample_weights as a mask for each item of y_pred.
  Typical usage of BinaryCrossentropy expects sample_weights to specify a single
  scalar per example which does not provide granularity to mask each element of
  y_pred.

  This loss differs from BinaryCrossentropy in that it does not AVG loss per
  example and SUM over batch. Instead, it SUMs loss per example and batch.
  """

  def __init__(
      self, *args, class_weights: Optional[Sequence[float]] = None, **kwargs
  ):
    """Initializes MaskedBinaryCrossentropy and sets class_weights.

    Args:
      *args: Args to pass to the base class.
      class_weights: Optional class weights. If provided, the loss will be
        weighted by the class weights. Expected to be in shape [num_classes].
      **kwargs: Kwargs to pass to the base class.
    """
    super().__init__(*args, **kwargs)
    self._class_weights = (
        tf.expand_dims(tf.convert_to_tensor(class_weights), axis=0)
        if class_weights is not None
        else None
    )

  def __call__(self, y_true, y_pred, sample_weight=None):
    """Override the __call__ method to apply the sample_weight as a mask.

    Args:
      y_true: The ground truth values. Expected to be in shape [batch_size,
        num_classes]
      y_pred: The predicted values. Expected to be in shape [batch_size,
        num_classes]
      sample_weight: Optional weighting of each example. Defaults to 1.0.
        Expected to be in shape [batch_size, num_classes]

    Returns:
      The loss.
    """
    if self._class_weights is not None:
      if sample_weight is not None:
        sample_weight = sample_weight * self._class_weights
      else:
        sample_weight = tf.repeat(
            self._class_weights, repeats=y_true.shape[0], axis=0
        )

    return super().__call__(  # pytype: disable=attribute-error
        tf.reshape(y_true, [-1, 1]),
        tf.reshape(y_pred, [-1, 1]),
        tf.reshape(sample_weight, [-1, 1])
        if sample_weight is not None
        else None,
    )


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
    """Initializes FocalLoss.

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


class SparseFocalLoss(FocalLoss):
  """Sparse implementation of Focal Loss.

  This is the same as FocalLoss, except the labels are expected to be class ids
  instead of 1-hot encoded vectors. See FocalLoss class documentation defined
  in this same file for more details.

  Example usage:
  >>> y_true = [1, 2]
  >>> y_pred = [[0.05, 0.95, 0], [0.1, 0.8, 0.1]]
  >>> gamma = 2
  >>> focal_loss = SparseFocalLoss(gamma, 3)
  >>> focal_loss(y_true, y_pred).numpy()
  0.9326

  >>> # Calling with 'sample_weight'.
  >>> focal_loss(y_true, y_pred, sample_weight=tf.constant([0.3, 0.7])).numpy()
  0.6528
  """

  def __init__(
      self, gamma, num_classes, class_weight: Optional[Sequence[float]] = None
  ):
    """Initializes SparseFocalLoss.

    Args:
      gamma: Focal loss gamma, as described in class docs.
      num_classes: Number of classes.
      class_weight: A weight to apply to the loss, one for each class. The
        weight is applied for each input where the ground truth label matches.
    """
    super().__init__(gamma, class_weight=class_weight)
    self._num_classes = num_classes

  def __call__(
      self,
      y_true: tf.Tensor,
      y_pred: tf.Tensor,
      sample_weight: Optional[tf.Tensor] = None,
  ) -> tf.Tensor:
    y_true = tf.cast(tf.reshape(y_true, [-1]), tf.int32)
    y_true_one_hot = tf.one_hot(y_true, self._num_classes)
    return super().__call__(y_true_one_hot, y_pred, sample_weight=sample_weight)


@dataclasses.dataclass
class PerceptualLossWeight:
  """The weight for each perceptual loss.

  Attributes:
    l1: weight for L1 loss.
    content: weight for content loss.
    style: weight for style loss.
  """

  l1: float = 1.0
  content: float = 1.0
  style: float = 1.0


class ImagePerceptualQualityLoss(tf.keras.losses.Loss):
  """Image perceptual quality loss.

  It obtains a weighted loss between the VGGPerceptualLoss and L1 loss.
  """

  def __init__(
      self,
      loss_weight: Optional[PerceptualLossWeight] = None,
      reduction: tf.keras.losses.Reduction = tf.keras.losses.Reduction.NONE,
  ):
    """Initializes ImagePerceptualQualityLoss."""
    self._loss_weight = loss_weight
    self._losses = {}
    self._vgg_loss = VGGPerceptualLoss(self._loss_weight)
    self._reduction = reduction

  def _l1_loss(
      self,
      reduction: tf.keras.losses.Reduction = tf.keras.losses.Reduction.NONE,
  ) -> Any:
    """Calculates L1 loss."""
    return tf.keras.losses.MeanAbsoluteError(reduction)

  def __call__(
      self,
      img1: tf.Tensor,
      img2: tf.Tensor,
  ) -> tf.Tensor:
    """Computes image perceptual quality loss."""
    loss_value = []
    if self._loss_weight is None:
      self._loss_weight = PerceptualLossWeight()

    if self._loss_weight.content > 0 or self._loss_weight.style > 0:
      vgg_loss = self._vgg_loss(img1, img2)
      vgg_loss_value = tf.math.add_n(vgg_loss.values())
      loss_value.append(vgg_loss_value)
    if self._loss_weight.l1 > 0:
      l1_loss = self._l1_loss(reduction=self._reduction)(img1, img2)
      l1_loss_value = tf_utils.safe_mean(l1_loss * self._loss_weight.l1)
      loss_value.append(l1_loss_value)
    total_loss = tf.math.add_n(loss_value)
    return total_loss


class PerceptualLoss(tf.keras.Model, metaclass=abc.ABCMeta):
  """Base class for perceptual loss model."""

  def __init__(
      self,
      feature_weight: Optional[Sequence[float]] = None,
      loss_weight: Optional[PerceptualLossWeight] = None,
  ):
    """Instantiates perceptual loss.

    Args:
      feature_weight: The weight coefficients of multiple model extracted
        features used for calculating the perceptual loss.
      loss_weight: The weight coefficients between `style_loss` and
        `content_loss`.
    """
    super().__init__()
    self._loss_op = lambda x, y: tf.math.reduce_mean(tf.abs(x - y))
    self._loss_style = tf.constant(0.0)
    self._loss_content = tf.constant(0.0)
    self._feature_weight = feature_weight
    self._loss_weight = loss_weight

  def __call__(
      self,
      img1: tf.Tensor,
      img2: tf.Tensor,
  ) -> Mapping[str, tf.Tensor]:
    """Computes perceptual loss between two images.

    Args:
      img1: First batch of images. The pixel values should be normalized to [-1,
        1].
      img2: Second batch of images. The pixel values should be normalized to
        [-1, 1].

    Returns:
      A mapping between loss name and loss tensors.
    """
    x_features = self._compute_features(img1)
    y_features = self._compute_features(img2)

    if self._loss_weight is None:
      self._loss_weight = PerceptualLossWeight()

    # If the _feature_weight is not initialized, then initialize it as a list of
    # all the element equals to 1.0.
    if self._feature_weight is None:
      self._feature_weight = [1.0] * len(x_features)

    # If the length of _feature_weight smallert than the length of the feature,
    # raise a ValueError. Otherwise, only use the first len(x_features) weight
    # for computing the loss.
    if len(self._feature_weight) < len(x_features):
      raise ValueError(
          f'Input feature weight length {len(self._feature_weight)} is smaller'
          f' than feature length {len(x_features)}'
      )

    if self._loss_weight.style > 0.0:
      self._loss_style = tf_utils.safe_mean(
          self._loss_weight.style
          * self._get_style_loss(x_feats=x_features, y_feats=y_features)
      )
    if self._loss_weight.content > 0.0:
      self._loss_content = tf_utils.safe_mean(
          self._loss_weight.content
          * self._get_content_loss(x_feats=x_features, y_feats=y_features)
      )

    return {'style_loss': self._loss_style, 'content_loss': self._loss_content}

  @abc.abstractmethod
  def _compute_features(self, img: tf.Tensor) -> Sequence[tf.Tensor]:
    """Computes features from the given image tensor.

    Args:
      img: Image tensor.

    Returns:
      A list of multi-scale feature maps.
    """

  def _get_content_loss(
      self, x_feats: Sequence[tf.Tensor], y_feats: Sequence[tf.Tensor]
  ) -> tf.Tensor:
    """Gets weighted multi-scale content loss.

    Args:
      x_feats: Reconstructed face image.
      y_feats: Target face image.

    Returns:
      A scalar tensor for the content loss.
    """
    content_losses = []
    for coef, x_feat, y_feat in zip(self._feature_weight, x_feats, y_feats):
      content_loss = self._loss_op(x_feat, y_feat) * coef
      content_losses.append(content_loss)
    return tf.math.reduce_sum(content_losses)

  def _get_style_loss(
      self, x_feats: Sequence[tf.Tensor], y_feats: Sequence[tf.Tensor]
  ) -> tf.Tensor:
    """Gets weighted multi-scale style loss.

    Args:
      x_feats: Reconstructed face image.
      y_feats: Target face image.

    Returns:
      A scalar tensor for the style loss.
    """
    style_losses = []
    i = 0
    for coef, x_feat, y_feat in zip(self._feature_weight, x_feats, y_feats):
      x_feat_g = _compute_gram_matrix(x_feat)
      y_feat_g = _compute_gram_matrix(y_feat)
      style_loss = self._loss_op(x_feat_g, y_feat_g) * coef
      style_losses.append(style_loss)
      i = i + 1

    return tf.math.reduce_sum(style_loss)


class VGGPerceptualLoss(PerceptualLoss):
  """Perceptual loss based on VGG19 pretrained on the ImageNet dataset.

  Reference:
  - [Perceptual Losses for Real-Time Style Transfer and Super-Resolution](
      https://arxiv.org/abs/1603.08155) (ECCV 2016)

  Perceptual loss measures high-level perceptual and semantic differences
  between images.
  """

  def __init__(
      self,
      loss_weight: Optional[PerceptualLossWeight] = None,
  ):
    """Initializes image quality loss essentials.

    Args:
      loss_weight: Loss weight coefficients.
    """
    super().__init__(
        feature_weight=np.array([0.1, 0.1, 1.0, 1.0, 1.0]),
        loss_weight=loss_weight,
    )

    rgb_mean = tf.constant([0.485, 0.456, 0.406])
    rgb_std = tf.constant([0.229, 0.224, 0.225])

    self._rgb_mean = tf.reshape(rgb_mean, (1, 1, 1, 3))
    self._rgb_std = tf.reshape(rgb_std, (1, 1, 1, 3))

    model_path = file_util.DownloadedFiles(
        'vgg_feature_extractor',
        _VGG_IMAGENET_PERCEPTUAL_MODEL_URL,
        is_folder=True,
    )
    self._vgg19 = model_util.load_keras_model(model_path.get_path())

  def _compute_features(self, img: tf.Tensor) -> Sequence[tf.Tensor]:
    """Computes VGG19 features."""
    img = (img + 1) / 2.0
    norm_img = (img - self._rgb_mean) / self._rgb_std
    # no grad, as it only serves as a frozen feature extractor.
    return self._vgg19(norm_img)


def _compute_gram_matrix(feature: tf.Tensor) -> tf.Tensor:
  """Computes gram matrix for the feature map.

  Args:
    feature: [B, H, W, C] feature map.

  Returns:
    [B, C, C] gram matrix.
  """
  h, w, c = feature.shape[1:].as_list()
  feat_reshaped = tf.reshape(feature, shape=(-1, h * w, c))
  feat_gram = tf.matmul(
      tf.transpose(feat_reshaped, perm=[0, 2, 1]), feat_reshaped
  )
  return feat_gram / (c * h * w)
