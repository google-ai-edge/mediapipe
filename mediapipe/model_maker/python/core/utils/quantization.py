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
"""Libraries for post-training quantization."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from typing import Any, Callable, List, Optional, Union

import tensorflow as tf

from mediapipe.model_maker.python.core.data import dataset as ds

DEFAULT_QUANTIZATION_STEPS = 500


def _get_representative_dataset_generator(dataset: tf.data.Dataset,
                                          num_steps: int) -> Callable[[], Any]:
  """Gets a representative dataset generator for post-training quantization.

  The generator is to provide a small dataset to calibrate or estimate the
  range, i.e, (min, max) of all floating-point arrays in the model for
  quantization. Usually, this is a small subset of a few hundred samples
  randomly chosen, in no particular order, from the training or evaluation
  dataset. See tf.lite.RepresentativeDataset for more details.

  Args:
    dataset: Input dataset for extracting representative sub dataset.
    num_steps: The number of quantization steps which also reflects the size of
      the representative dataset.

  Returns:
    A representative dataset generator.
  """

  def representative_dataset_gen():
    """Generates representative dataset for quantization."""
    for data, _ in dataset.take(num_steps):
      yield [data]

  return representative_dataset_gen


class QuantizationConfig(object):
  """Configuration for post-training quantization.

  Refer to
  https://www.tensorflow.org/lite/performance/post_training_quantization
  for different post-training quantization options.
  """

  def __init__(
      self,
      optimizations: Optional[Union[tf.lite.Optimize,
                                    List[tf.lite.Optimize]]] = None,
      representative_data: Optional[ds.Dataset] = None,
      quantization_steps: Optional[int] = None,
      inference_input_type: Optional[tf.dtypes.DType] = None,
      inference_output_type: Optional[tf.dtypes.DType] = None,
      supported_ops: Optional[Union[tf.lite.OpsSet,
                                    List[tf.lite.OpsSet]]] = None,
      supported_types: Optional[Union[tf.dtypes.DType,
                                      List[tf.dtypes.DType]]] = None,
      experimental_new_quantizer: bool = False,
  ):
    """Constructs QuantizationConfig.

    Args:
      optimizations: A list of optimizations to apply when converting the model.
        If not set, use `[Optimize.DEFAULT]` by default.
      representative_data: A representative ds.Dataset for post-training
        quantization.
      quantization_steps: Number of post-training quantization calibration steps
        to run (default to DEFAULT_QUANTIZATION_STEPS).
      inference_input_type: Target data type of real-number input arrays. Allows
        for a different type for input arrays. Defaults to None. If set, must be
        be `{tf.float32, tf.uint8, tf.int8}`.
      inference_output_type: Target data type of real-number output arrays.
        Allows for a different type for output arrays. Defaults to None. If set,
        must be `{tf.float32, tf.uint8, tf.int8}`.
      supported_ops: Set of OpsSet options supported by the device. Used to Set
        converter.target_spec.supported_ops.
      supported_types: List of types for constant values on the target device.
        Supported values are types exported by lite.constants. Frequently, an
        optimization choice is driven by the most compact (i.e. smallest) type
        in this list (default [constants.FLOAT]).
      experimental_new_quantizer: Whether to enable experimental new quantizer.

    Raises:
      ValueError: if inference_input_type or inference_output_type are set but
      not in {tf.float32, tf.uint8, tf.int8}.
    """
    if inference_input_type is not None and inference_input_type not in {
        tf.float32, tf.uint8, tf.int8
    }:
      raise ValueError('Unsupported inference_input_type %s' %
                       inference_input_type)
    if inference_output_type is not None and inference_output_type not in {
        tf.float32, tf.uint8, tf.int8
    }:
      raise ValueError('Unsupported inference_output_type %s' %
                       inference_output_type)

    if optimizations is None:
      optimizations = [tf.lite.Optimize.DEFAULT]
    if not isinstance(optimizations, list):
      optimizations = [optimizations]
    self.optimizations = optimizations

    self.representative_data = representative_data
    if self.representative_data is not None and quantization_steps is None:
      quantization_steps = DEFAULT_QUANTIZATION_STEPS
    self.quantization_steps = quantization_steps

    self.inference_input_type = inference_input_type
    self.inference_output_type = inference_output_type

    if supported_ops is not None and not isinstance(supported_ops, list):
      supported_ops = [supported_ops]
    self.supported_ops = supported_ops

    if supported_types is not None and not isinstance(supported_types, list):
      supported_types = [supported_types]
    self.supported_types = supported_types

    self.experimental_new_quantizer = experimental_new_quantizer

  @classmethod
  def for_dynamic(cls) -> 'QuantizationConfig':
    """Creates configuration for dynamic range quantization."""
    return QuantizationConfig()

  @classmethod
  def for_int8(
      cls,
      representative_data: ds.Dataset,
      quantization_steps: int = DEFAULT_QUANTIZATION_STEPS,
      inference_input_type: tf.dtypes.DType = tf.uint8,
      inference_output_type: tf.dtypes.DType = tf.uint8,
      supported_ops: tf.lite.OpsSet = tf.lite.OpsSet.TFLITE_BUILTINS_INT8
  ) -> 'QuantizationConfig':
    """Creates configuration for full integer quantization.

    Args:
      representative_data: Representative data used for post-training
        quantization.
      quantization_steps: Number of post-training quantization calibration steps
        to run.
      inference_input_type: Target data type of real-number input arrays.
      inference_output_type: Target data type of real-number output arrays.
      supported_ops: Set of `tf.lite.OpsSet` options, where each option
        represents a set of operators supported by the target device.

    Returns:
      QuantizationConfig.
    """
    return QuantizationConfig(
        representative_data=representative_data,
        quantization_steps=quantization_steps,
        inference_input_type=inference_input_type,
        inference_output_type=inference_output_type,
        supported_ops=supported_ops)

  @classmethod
  def for_float16(cls) -> 'QuantizationConfig':
    """Creates configuration for float16 quantization."""
    return QuantizationConfig(supported_types=[tf.float16])

  def set_converter_with_quantization(self, converter: tf.lite.TFLiteConverter,
                                      **kwargs: Any) -> tf.lite.TFLiteConverter:
    """Sets input TFLite converter with quantization configurations.

    Args:
      converter: input tf.lite.TFLiteConverter.
      **kwargs: arguments used by ds.Dataset.gen_tf_dataset.

    Returns:
      tf.lite.TFLiteConverter with quantization configurations.
    """
    converter.optimizations = self.optimizations

    if self.representative_data is not None:
      tf_ds = self.representative_data.gen_tf_dataset(
          batch_size=1, is_training=False, **kwargs)
      converter.representative_dataset = tf.lite.RepresentativeDataset(
          _get_representative_dataset_generator(tf_ds, self.quantization_steps))

    if self.inference_input_type:
      converter.inference_input_type = self.inference_input_type
    if self.inference_output_type:
      converter.inference_output_type = self.inference_output_type
    if self.supported_ops:
      converter.target_spec.supported_ops = self.supported_ops
    if self.supported_types:
      converter.target_spec.supported_types = self.supported_types

    if self.experimental_new_quantizer is not None:
      converter.experimental_new_quantizer = self.experimental_new_quantizer
    return converter
