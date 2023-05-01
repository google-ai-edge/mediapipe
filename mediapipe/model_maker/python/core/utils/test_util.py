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
"""Test utilities for model maker."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from typing import Sequence
from typing import Dict, List, Union

# Dependency imports

import numpy as np
import tensorflow as tf

from mediapipe.model_maker.python.core.data import dataset as ds
from mediapipe.model_maker.python.core.utils import model_util


def create_dataset(data_size: int,
                   input_shape: List[int],
                   num_classes: int,
                   max_input_value: int = 1000) -> ds.Dataset:
  """Creates and returns a simple `Dataset` object for test."""
  features = tf.random.uniform(
      shape=[data_size] + input_shape,
      minval=0,
      maxval=max_input_value,
      dtype=tf.float32)

  labels = tf.random.uniform(
      shape=[data_size], minval=0, maxval=num_classes, dtype=tf.int32)

  tf_dataset = tf.data.Dataset.from_tensor_slices((features, labels))
  dataset = ds.Dataset(tf_dataset, data_size)
  return dataset


def create_random_sample(size: Union[int, List[int]],
                         low: float = 0,
                         high: float = 1) -> np.ndarray:
  """Creates and returns a random sample with floating point values.

  Args:
    size: Size of the output multi-dimensional array.
    low: Lower boundary of the output values.
    high: Higher boundary of the output values.

  Returns:
    1D array if the size is scalar. Otherwise, N-D array whose dimension equals
    input size.
  """
  np.random.seed(0)
  return np.random.uniform(low=low, high=high, size=size).astype(np.float32)


def build_model(input_shape: List[int], num_classes: int) -> tf.keras.Model:
  """Builds a simple Keras model for test."""
  inputs = tf.keras.layers.Input(shape=input_shape)
  if len(input_shape) == 3:  # Image inputs.
    outputs = tf.keras.layers.GlobalAveragePooling2D()(inputs)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(outputs)
  elif len(input_shape) == 1:  # Text inputs.
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(inputs)
  else:
    raise ValueError("Model inputs should be 2D tensor or 4D tensor.")

  model = tf.keras.Model(inputs=inputs, outputs=outputs)
  return model


def is_same_output(tflite_model: bytearray,
                   keras_model: tf.keras.Model,
                   input_tensors: Union[List[tf.Tensor], tf.Tensor],
                   atol: float = 1e-04) -> bool:
  """Returns if the output of TFLite model and keras model are identical."""
  # Gets output from lite model.
  lite_runner = model_util.get_lite_runner(tflite_model)
  lite_output = lite_runner.run(input_tensors)

  # Gets output from keras model.
  keras_output = keras_model.predict_on_batch(input_tensors)

  return np.allclose(lite_output, keras_output, atol=atol)


def run_tflite(
    tflite_filename: str,
    input_tensors: Union[List[tf.Tensor], Dict[str, tf.Tensor]],
) -> Union[Sequence[tf.Tensor], tf.Tensor]:
  """Runs TFLite model inference."""
  with tf.io.gfile.GFile(tflite_filename, "rb") as f:
    tflite_model = f.read()
  lite_runner = model_util.get_lite_runner(tflite_model)
  return lite_runner.run(input_tensors)


def test_tflite(keras_model: tf.keras.Model,
                tflite_model: bytearray,
                size: Union[int, List[int]],
                high: float = 1,
                atol: float = 1e-04) -> bool:
  """Verifies if the output of TFLite model and TF Keras model are identical.

  Args:
    keras_model: Input TensorFlow Keras model.
    tflite_model: Input TFLite model flatbuffer.
    size: Size of the input tesnor.
    high: Higher boundary of the values in input tensors.
    atol: Absolute tolerance of the difference between the outputs of Keras
      model and TFLite model.

  Returns:
    True if the output of TFLite model and TF Keras model are identical.
    Otherwise, False.
  """
  random_input = create_random_sample(size=size, high=high)
  random_input = tf.convert_to_tensor(random_input)

  return is_same_output(
      tflite_model=tflite_model,
      keras_model=keras_model,
      input_tensors=random_input,
      atol=atol)


def test_tflite_file(keras_model: tf.keras.Model,
                     tflite_file: bytearray,
                     size: Union[int, List[int]],
                     high: float = 1,
                     atol: float = 1e-04) -> bool:
  """Verifies if the output of TFLite model and TF Keras model are identical.

  Args:
    keras_model: Input TensorFlow Keras model.
    tflite_file: Input TFLite model file.
    size: Size of the input tesnor.
    high: Higher boundary of the values in input tensors.
    atol: Absolute tolerance of the difference between the outputs of Keras
      model and TFLite model.

  Returns:
    True if the output of TFLite model and TF Keras model are identical.
    Otherwise, False.
  """
  with tf.io.gfile.GFile(tflite_file, "rb") as f:
    tflite_model = f.read()
  return test_tflite(keras_model, tflite_model, size, high, atol)
