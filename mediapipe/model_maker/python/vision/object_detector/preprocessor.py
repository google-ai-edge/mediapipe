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
"""Preprocessor for object detector."""
from typing import Any, Mapping, Tuple

import tensorflow as tf

from mediapipe.model_maker.python.vision.object_detector import model_spec as ms
from official.vision.dataloaders import utils
from official.vision.ops import anchor
from official.vision.ops import box_ops
from official.vision.ops import preprocess_ops


# TODO Combine preprocessing logic with image_preprocessor.
class Preprocessor(object):
  """Preprocessor for object detector."""

  def __init__(self, model_spec: ms.ModelSpec):
    """Initialize a Preprocessor."""
    self._mean_norm = model_spec.mean_norm
    self._stddev_norm = model_spec.stddev_norm
    self._output_size = model_spec.input_image_shape[:2]
    self._min_level = model_spec.min_level
    self._max_level = model_spec.max_level
    self._num_scales = 3
    self._aspect_ratios = [0.5, 1, 2]
    self._anchor_size = 3
    self._dtype = tf.float32
    self._match_threshold = 0.5
    self._unmatched_threshold = 0.5
    self._aug_scale_min = 0.5
    self._aug_scale_max = 2.0
    self._max_num_instances = 100

    self._padded_size = preprocess_ops.compute_padded_size(
        self._output_size, 2**self._max_level
    )

    input_anchor = anchor.build_anchor_generator(
        min_level=self._min_level,
        max_level=self._max_level,
        num_scales=self._num_scales,
        aspect_ratios=self._aspect_ratios,
        anchor_size=self._anchor_size,
    )
    self._anchor_boxes = input_anchor(image_size=self._output_size)
    self._anchor_labeler = anchor.AnchorLabeler(
        self._match_threshold, self._unmatched_threshold
    )

  @property
  def anchor_boxes(self):
    return self._anchor_boxes

  def __call__(
      self, data: Mapping[str, Any], is_training: bool = True
  ) -> Tuple[tf.Tensor, Mapping[str, Any]]:
    """Run the preprocessor on an example.

    The data dict should contain the following keys always:
      - image
      - groundtruth_classes
      - groundtruth_boxes
      - groundtruth_is_crowd
    Additional keys needed when is_training is set to True:
      - groundtruth_area
      - source_id
      - height
      - width

    Args:
      data: A dict of object detector inputs.
      is_training: Whether or not the data is used for training.

    Returns:
      A tuple of (image, labels) where image is a Tensor and labels is a dict.
    """
    classes = data['groundtruth_classes']
    boxes = data['groundtruth_boxes']

    # Get original image.
    image = data['image']
    image_shape = tf.shape(input=image)[0:2]

    # Normalize image with mean and std pixel values.
    image = preprocess_ops.normalize_image(
        image, self._mean_norm, self._stddev_norm
    )

    # Flip image randomly during training.
    if is_training:
      image, boxes, _ = preprocess_ops.random_horizontal_flip(image, boxes)

    # Convert boxes from normalized coordinates to pixel coordinates.
    boxes = box_ops.denormalize_boxes(boxes, image_shape)

    # Resize and crop image.
    image, image_info = preprocess_ops.resize_and_crop_image(
        image,
        self._output_size,
        padded_size=self._padded_size,
        aug_scale_min=(self._aug_scale_min if is_training else 1.0),
        aug_scale_max=(self._aug_scale_max if is_training else 1.0),
    )

    # Resize and crop boxes.
    image_scale = image_info[2, :]
    offset = image_info[3, :]
    boxes = preprocess_ops.resize_and_crop_boxes(
        boxes, image_scale, image_info[1, :], offset
    )
    # Filter out ground-truth boxes that are all zeros.
    indices = box_ops.get_non_empty_box_indices(boxes)
    boxes = tf.gather(boxes, indices)
    classes = tf.gather(classes, indices)

    # Assign anchors.
    (cls_targets, box_targets, _, cls_weights, box_weights) = (
        self._anchor_labeler.label_anchors(
            self.anchor_boxes, boxes, tf.expand_dims(classes, axis=1)
        )
    )

    # Cast input image to desired data type.
    image = tf.cast(image, dtype=self._dtype)

    # Pack labels for model_fn outputs.
    labels = {
        'cls_targets': cls_targets,
        'box_targets': box_targets,
        'anchor_boxes': self.anchor_boxes,
        'cls_weights': cls_weights,
        'box_weights': box_weights,
        'image_info': image_info,
    }
    if not is_training:
      groundtruths = {
          'source_id': data['source_id'],
          'height': data['height'],
          'width': data['width'],
          'num_detections': tf.shape(data['groundtruth_classes']),
          'image_info': image_info,
          'boxes': box_ops.denormalize_boxes(
              data['groundtruth_boxes'], image_shape
          ),
          'classes': data['groundtruth_classes'],
          'areas': data['groundtruth_area'],
          'is_crowds': tf.cast(data['groundtruth_is_crowd'], tf.int32),
      }
      groundtruths['source_id'] = utils.process_source_id(
          groundtruths['source_id']
      )
      groundtruths = utils.pad_groundtruths_to_fixed_size(
          groundtruths, self._max_num_instances
      )
      labels.update({'groundtruths': groundtruths})
    return image, labels
