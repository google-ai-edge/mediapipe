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
# ==============================================================================
"""ImageNet preprocessing."""

import tensorflow as tf

IMAGE_SIZE = 224
CROP_PADDING = 32


class Preprocessor(object):
  """Preprocessor for image classification."""

  def __init__(
      self,
      input_shape,
      num_classes,
      mean_rgb,
      stddev_rgb,
      use_augmentation=False,
      one_hot=True,
  ):
    self.input_shape = input_shape
    self.num_classes = num_classes
    self.mean_rgb = mean_rgb
    self.stddev_rgb = stddev_rgb
    self.use_augmentation = use_augmentation
    self.one_hot = one_hot

  def __call__(self, image, label, is_training=True):
    if self.use_augmentation:
      return self._preprocess_with_augmentation(image, label, is_training)
    return self._preprocess_without_augmentation(image, label)

  def _preprocess_with_augmentation(self, image, label, is_training):
    """Image preprocessing method with data augmentation."""
    image_size = self.input_shape[0]
    if is_training:
      image = preprocess_for_train(image, image_size)
    else:
      image = preprocess_for_eval(image, image_size)

    image -= tf.constant(self.mean_rgb, shape=[1, 1, 3], dtype=image.dtype)
    image /= tf.constant(self.stddev_rgb, shape=[1, 1, 3], dtype=image.dtype)
    if self.one_hot:
      label = tf.one_hot(label, depth=self.num_classes)
    return image, label

  # TODO: Changes to preprocess to support batch input.
  def _preprocess_without_augmentation(self, image, label):
    """Image preprocessing method without data augmentation."""
    image = tf.cast(image, tf.float32)

    image -= tf.constant(self.mean_rgb, shape=[1, 1, 3], dtype=image.dtype)
    image /= tf.constant(self.stddev_rgb, shape=[1, 1, 3], dtype=image.dtype)

    image = tf.compat.v1.image.resize(image, self.input_shape)
    if self.one_hot:
      label = tf.one_hot(label, depth=self.num_classes)
    return image, label


def _distorted_bounding_box_crop(image,
                                 bbox,
                                 min_object_covered=0.1,
                                 aspect_ratio_range=(0.75, 1.33),
                                 area_range=(0.05, 1.0),
                                 max_attempts=100):
  """Generates cropped_image using one of the bboxes randomly distorted.

  See `tf.image.sample_distorted_bounding_box` for more documentation.

  Args:
    image: 4-D Tensor of shape [batch, height, width, channels] or 3-D Tensor of
      shape [height, width, channels].
    bbox: `Tensor` of bounding boxes arranged `[1, num_boxes, coords]` where
      each coordinate is [0, 1) and the coordinates are arranged as `[ymin,
      xmin, ymax, xmax]`. If num_boxes is 0 then use the whole image.
    min_object_covered: An optional `float`. Defaults to `0.1`. The cropped area
      of the image must contain at least this fraction of any bounding box
      supplied.
    aspect_ratio_range: An optional list of `float`s. The cropped area of the
      image must have an aspect ratio = width / height within this range.
    area_range: An optional list of `float`s. The cropped area of the image must
      contain a fraction of the supplied image within in this range.
    max_attempts: An optional `int`. Number of attempts at generating a cropped
      region of the image of the specified constraints. After `max_attempts`
      failures, return the entire image.

  Returns:
    A cropped image `Tensor`
  """
  with tf.name_scope('distorted_bounding_box_crop'):
    shape = tf.shape(image)
    sample_distorted_bounding_box = tf.image.sample_distorted_bounding_box(
        shape,
        bounding_boxes=bbox,
        min_object_covered=min_object_covered,
        aspect_ratio_range=aspect_ratio_range,
        area_range=area_range,
        max_attempts=max_attempts,
        use_image_if_no_bounding_boxes=True)
    bbox_begin, bbox_size, _ = sample_distorted_bounding_box

    # Crop the image to the specified bounding box.
    offset_y, offset_x, _ = tf.unstack(bbox_begin)
    target_height, target_width, _ = tf.unstack(bbox_size)
    image = tf.image.crop_to_bounding_box(image, offset_y, offset_x,
                                          target_height, target_width)

    return image


def _at_least_x_are_equal(a, b, x):
  """At least `x` of `a` and `b` `Tensors` are equal."""
  match = tf.equal(a, b)
  match = tf.cast(match, tf.int32)
  return tf.greater_equal(tf.reduce_sum(match), x)


def _resize_image(image, image_size, method=None):
  if method is not None:
    tf.compat.v1.logging.info('Use customized resize method {}'.format(method))
    return tf.compat.v1.image.resize([image], [image_size, image_size],
                                     method)[0]
  tf.compat.v1.logging.info('Use default resize_bicubic.')
  return tf.compat.v1.image.resize_bicubic([image], [image_size, image_size])[0]


def _decode_and_random_crop(original_image, image_size, resize_method=None):
  """Makes a random crop of image_size."""
  bbox = tf.constant([0.0, 0.0, 1.0, 1.0], dtype=tf.float32, shape=[1, 1, 4])
  image = _distorted_bounding_box_crop(
      original_image,
      bbox,
      min_object_covered=0.1,
      aspect_ratio_range=(3. / 4, 4. / 3.),
      area_range=(0.08, 1.0),
      max_attempts=10)
  original_shape = tf.shape(original_image)
  bad = _at_least_x_are_equal(original_shape, tf.shape(image), 3)

  image = tf.cond(bad,
                  lambda: _decode_and_center_crop(original_image, image_size),
                  lambda: _resize_image(image, image_size, resize_method))

  return image


def _decode_and_center_crop(image, image_size, resize_method=None):
  """Crops to center of image with padding then scales image_size."""
  shape = tf.shape(image)
  image_height = shape[0]
  image_width = shape[1]

  padded_center_crop_size = tf.cast(
      ((image_size / (image_size + CROP_PADDING)) *
       tf.cast(tf.minimum(image_height, image_width), tf.float32)), tf.int32)

  offset_height = ((image_height - padded_center_crop_size) + 1) // 2
  offset_width = ((image_width - padded_center_crop_size) + 1) // 2
  image = tf.image.crop_to_bounding_box(image, offset_height, offset_width,
                                        padded_center_crop_size,
                                        padded_center_crop_size)
  image = _resize_image(image, image_size, resize_method)
  return image


def _flip(image):
  """Random horizontal image flip."""
  image = tf.image.random_flip_left_right(image)
  return image


def preprocess_for_train(
    image: tf.Tensor,
    image_size: int = IMAGE_SIZE,
    resize_method: str = tf.image.ResizeMethod.BILINEAR) -> tf.Tensor:
  """Preprocesses the given image for evaluation.

  Args:
    image: 4-D Tensor of shape [batch, height, width, channels] or 3-D Tensor of
      shape [height, width, channels].
    image_size: image size.
    resize_method: resize method. If none, use bicubic.

  Returns:
    A preprocessed image `Tensor`.
  """
  image = _decode_and_random_crop(image, image_size, resize_method)
  image = _flip(image)
  image = tf.reshape(image, [image_size, image_size, 3])

  image = tf.image.convert_image_dtype(image, dtype=tf.float32)

  return image


def preprocess_for_eval(
    image: tf.Tensor,
    image_size: int = IMAGE_SIZE,
    resize_method: str = tf.image.ResizeMethod.BILINEAR) -> tf.Tensor:
  """Preprocesses the given image for evaluation.

  Args:
    image: 4-D Tensor of shape [batch, height, width, channels] or 3-D Tensor of
      shape [height, width, channels].
    image_size: image size.
    resize_method: if None, use bicubic.

  Returns:
    A preprocessed image `Tensor`.
  """
  image = _decode_and_center_crop(image, image_size, resize_method)
  image = tf.reshape(image, [image_size, image_size, 3])
  image = tf.image.convert_image_dtype(image, dtype=tf.float32)
  return image
