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
"""Image classifier dataset library."""

import dataclasses
import functools
import os
import random
from typing import List, Optional

import tensorflow as tf

from mediapipe.model_maker.python.core.data import classification_dataset
from mediapipe.model_maker.python.vision.core import image_utils


class Dataset(classification_dataset.ClassificationDataset):
  """Dataset library for image classifier."""

  @classmethod
  def from_folder(
      cls,
      dirname: str,
      shuffle: bool = True) -> classification_dataset.ClassificationDataset:
    """Loads images and labels from the given directory.

    Assume the image data of the same label are in the same subdirectory.

    Args:
      dirname: Name of the directory containing the data files.
      shuffle: boolean, if true, random shuffle data.

    Returns:
      Dataset containing images and labels and other related info.
    Raises:
      ValueError: if the input data directory is empty.
    """
    data_root = os.path.abspath(dirname)

    # Assumes the image data of the same label are in the same subdirectory,
    # gets image path and label names.
    all_image_paths = list(tf.io.gfile.glob(data_root + r'/*/*'))
    all_image_size = len(all_image_paths)
    if all_image_size == 0:
      raise ValueError('Image size is zero')

    if shuffle:
      # Random shuffle data.
      random.shuffle(all_image_paths)

    label_names = sorted(
        name for name in os.listdir(data_root)
        if os.path.isdir(os.path.join(data_root, name)))
    all_label_size = len(label_names)
    index_by_label = dict(
        (name, index) for index, name in enumerate(label_names))
    all_image_labels = [
        index_by_label[os.path.basename(os.path.dirname(path))]
        for path in all_image_paths
    ]

    path_ds = tf.data.Dataset.from_tensor_slices(all_image_paths)

    image_ds = path_ds.map(
        image_utils.load_image, num_parallel_calls=tf.data.AUTOTUNE
    )

    # Load label
    label_ds = tf.data.Dataset.from_tensor_slices(
        tf.cast(all_image_labels, tf.int64))

    # Create a dataset if (image, label) pairs
    image_label_ds = tf.data.Dataset.zip((image_ds, label_ds))

    tf.compat.v1.logging.info(
        'Load image with size: %d, num_label: %d, labels: %s.', all_image_size,
        all_label_size, ', '.join(label_names))
    return Dataset(
        dataset=image_label_ds, label_names=label_names, size=all_image_size
    )
