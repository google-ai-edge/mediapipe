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
"""Face stylizer dataset library."""

from typing import Sequence
import logging
import os

import tensorflow as tf

from mediapipe.model_maker.python.core.data import classification_dataset
from mediapipe.model_maker.python.vision.face_stylizer import constants
from mediapipe.python._framework_bindings import image as image_module
from mediapipe.tasks.python.core import base_options as base_options_module
from mediapipe.tasks.python.vision import face_aligner


def _preprocess_face_dataset(
    all_image_paths: Sequence[str],
) -> Sequence[tf.Tensor]:
  """Preprocess face image dataset by aligning the face."""
  path = constants.FACE_ALIGNER_TASK_FILES.get_path()
  base_options = base_options_module.BaseOptions(model_asset_path=path)
  options = face_aligner.FaceAlignerOptions(base_options=base_options)
  aligner = face_aligner.FaceAligner.create_from_options(options)

  preprocessed_images = []
  for path in all_image_paths:
    tf.compat.v1.logging.info('Preprocess image %s', path)
    image = image_module.Image.create_from_file(path)
    aligned_image = aligner.align(image)
    if aligned_image is None:
      raise ValueError(
          'ERROR: Invalid image. No face is detected and aligned. Please make'
          ' sure the image has a single face that is facing straightforward and'
          ' not significantly rotated.'
      )
    aligned_image_tensor = tf.convert_to_tensor(aligned_image.numpy_view())
    preprocessed_images.append(aligned_image_tensor)

  return preprocessed_images


# TODO: Change to a unlabeled dataset if it makes sense.
class Dataset(classification_dataset.ClassificationDataset):
  """Dataset library for face stylizer fine tuning."""

  @classmethod
  def from_image(
      cls, filename: str
  ) -> classification_dataset.ClassificationDataset:
    """Creates a dataset from single image.

    Supported input image formats include 'jpg', 'jpeg', 'png'.

    Args:
      filename: Name of the image file.

    Returns:
      Dataset containing image and label and other related info.
    """
    file_path = os.path.abspath(filename)
    image_filename = os.path.basename(filename)
    image_name, ext_name = os.path.splitext(image_filename)

    if not ext_name.endswith(('.jpg', '.jpeg', '.png')):
      raise ValueError('Unsupported image formats: %s' % ext_name)

    image_data = _preprocess_face_dataset([file_path])
    label_names = [image_name]

    image_ds = tf.data.Dataset.from_tensor_slices(image_data)

    # Load label
    label_ds = tf.data.Dataset.from_tensor_slices(tf.cast([0], tf.int64))

    # Create a dataset of (image, label) pairs
    image_label_ds = tf.data.Dataset.zip((image_ds, label_ds))

    logging.info('Create dataset for style: %s.', image_name)

    return Dataset(
        dataset=image_label_ds,
        label_names=label_names,
        size=1,
    )
