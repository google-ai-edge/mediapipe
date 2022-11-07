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
"""Custom classifier."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from typing import Any, List

import tensorflow as tf

from mediapipe.model_maker.python.core.data import dataset
from mediapipe.model_maker.python.core.tasks import custom_model


class Classifier(custom_model.CustomModel):
  """An abstract base class that represents a TensorFlow classifier."""

  def __init__(self, model_spec: Any, label_names: List[str], shuffle: bool):
    """Initilizes a classifier with its specifications.

    Args:
        model_spec: Specification for the model.
        label_names: A list of label names for the classes.
        shuffle: Whether the dataset should be shuffled.
    """
    super(Classifier, self).__init__(model_spec, shuffle)
    self._label_names = label_names
    self._num_classes = len(label_names)

  def evaluate(self, data: dataset.Dataset, batch_size: int = 32) -> Any:
    """Evaluates the classifier with the provided evaluation dataset.

    Args:
        data: Evaluation dataset
        batch_size: Number of samples per evaluation step.

    Returns:
      The loss value and accuracy.
    """
    ds = data.gen_tf_dataset(
        batch_size, is_training=False, preprocess=self._preprocess)
    return self._model.evaluate(ds)

  def export_labels(self, export_dir: str, label_filename: str = 'labels.txt'):
    """Exports classification labels into a label file.

    Args:
      export_dir: The directory to save exported files.
      label_filename: File name to save labels model. The full export path is
        {export_dir}/{label_filename}.
    """
    if not tf.io.gfile.exists(export_dir):
      tf.io.gfile.makedirs(export_dir)

    label_filepath = os.path.join(export_dir, label_filename)
    tf.compat.v1.logging.info('Saving labels in %s', label_filepath)
    with tf.io.gfile.GFile(label_filepath, 'w') as f:
      f.write('\n'.join(self._label_names))
