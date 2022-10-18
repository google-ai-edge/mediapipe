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
"""Interface to define a custom model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import os
from typing import Any, Callable, Optional

# Dependency imports

import tensorflow as tf

from mediapipe.model_maker.python.core.data import dataset
from mediapipe.model_maker.python.core.utils import model_util
from mediapipe.model_maker.python.core.utils import quantization


class CustomModel(abc.ABC):
  """The abstract base class that represents a custom TensorFlow model."""

  def __init__(self, model_spec: Any, shuffle: bool):
    """Initializes a custom model with model specs and other parameters.

    Args:
      model_spec: Specification for the model.
      shuffle: Whether the training data need be shuffled.
    """
    self._model_spec = model_spec
    self._shuffle = shuffle
    self._preprocess = None
    self._model = None

  @abc.abstractmethod
  def evaluate(self, data: dataset.Dataset, **kwargs):
    """Evaluates the model with the provided data."""
    return

  def summary(self):
    """Prints a summary of the model."""
    self._model.summary()

  def export_tflite(
      self,
      export_dir: str,
      tflite_filename: str = 'model.tflite',
      quantization_config: Optional[quantization.QuantizationConfig] = None,
      preprocess: Optional[Callable[..., bool]] = None):
    """Converts the model to requested formats.

    Args:
      export_dir: The directory to save exported files.
      tflite_filename: File name to save tflite model. The full export path is
        {export_dir}/{tflite_filename}.
      quantization_config: The configuration for model quantization.
      preprocess: A callable to preprocess the representative dataset for
        quantization. The callable takes three arguments in order: feature,
        label, and is_training.
    """
    if not tf.io.gfile.exists(export_dir):
      tf.io.gfile.makedirs(export_dir)

    tflite_filepath = os.path.join(export_dir, tflite_filename)
    # TODO: Populate metadata to the exported TFLite model.
    model_util.export_tflite(
        self._model,
        tflite_filepath,
        quantization_config,
        preprocess=preprocess)
    tf.compat.v1.logging.info(
        'TensorFlow Lite model exported successfully: %s' % tflite_filepath)
