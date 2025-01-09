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

from absl.testing import parameterized
import tensorflow as tf

from mediapipe.model_maker.python.core.utils import quantization
from mediapipe.model_maker.python.core.utils import test_util
from ai_edge_litert import interpreter as tfl_interpreter


class QuantizationTest(tf.test.TestCase, parameterized.TestCase):

  def test_create_dynamic_quantization_config(self):
    config = quantization.QuantizationConfig.for_dynamic()
    self.assertEqual(config.optimizations, [tf.lite.Optimize.DEFAULT])
    self.assertIsNone(config.representative_data)
    self.assertIsNone(config.inference_input_type)
    self.assertIsNone(config.inference_output_type)
    self.assertIsNone(config.supported_ops)
    self.assertIsNone(config.supported_types)
    self.assertFalse(config.experimental_new_quantizer)

  def test_create_int8_quantization_config(self):
    representative_data = test_util.create_dataset(
        data_size=10, input_shape=[4], num_classes=3)
    config = quantization.QuantizationConfig.for_int8(
        representative_data=representative_data)
    self.assertEqual(config.optimizations, [tf.lite.Optimize.DEFAULT])
    self.assertEqual(config.inference_input_type, tf.uint8)
    self.assertEqual(config.inference_output_type, tf.uint8)
    self.assertEqual(config.supported_ops,
                     [tf.lite.OpsSet.TFLITE_BUILTINS_INT8])
    self.assertFalse(config.experimental_new_quantizer)

  def test_set_converter_with_quantization_from_int8_config(self):
    representative_data = test_util.create_dataset(
        data_size=10, input_shape=[4], num_classes=3)
    config = quantization.QuantizationConfig.for_int8(
        representative_data=representative_data)
    model = test_util.build_model(input_shape=[4], num_classes=3)
    saved_model_dir = self.get_temp_dir()
    model.save(saved_model_dir)
    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
    converter = config.set_converter_with_quantization(converter=converter)
    self.assertEqual(config.optimizations, [tf.lite.Optimize.DEFAULT])
    self.assertEqual(config.inference_input_type, tf.uint8)
    self.assertEqual(config.inference_output_type, tf.uint8)
    self.assertEqual(config.supported_ops,
                     [tf.lite.OpsSet.TFLITE_BUILTINS_INT8])
    tflite_model = converter.convert()
    interpreter = tfl_interpreter.Interpreter(model_content=tflite_model)
    self.assertEqual(interpreter.get_input_details()[0]['dtype'], tf.uint8)
    self.assertEqual(interpreter.get_output_details()[0]['dtype'], tf.uint8)

  def test_create_float16_quantization_config(self):
    config = quantization.QuantizationConfig.for_float16()
    self.assertEqual(config.optimizations, [tf.lite.Optimize.DEFAULT])
    self.assertIsNone(config.representative_data)
    self.assertIsNone(config.inference_input_type)
    self.assertIsNone(config.inference_output_type)
    self.assertIsNone(config.supported_ops)
    self.assertEqual(config.supported_types, [tf.float16])
    self.assertFalse(config.experimental_new_quantizer)

  def test_set_converter_with_quantization_from_float16_config(self):
    config = quantization.QuantizationConfig.for_float16()
    model = test_util.build_model(input_shape=[4], num_classes=3)
    saved_model_dir = self.get_temp_dir()
    model.save(saved_model_dir)
    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
    converter = config.set_converter_with_quantization(converter=converter)
    self.assertEqual(config.supported_types, [tf.float16])
    tflite_model = converter.convert()
    interpreter = tfl_interpreter.Interpreter(model_content=tflite_model)
    # The input and output are expected to be set to float32 by default.
    self.assertEqual(interpreter.get_input_details()[0]['dtype'], tf.float32)
    self.assertEqual(interpreter.get_output_details()[0]['dtype'], tf.float32)

  @parameterized.named_parameters(
      dict(
          testcase_name='invalid_inference_input_type',
          inference_input_type=tf.uint8,
          inference_output_type=tf.int64),
      dict(
          testcase_name='invalid_inference_output_type',
          inference_input_type=tf.int64,
          inference_output_type=tf.float32))
  def test_create_quantization_config_failure(self, inference_input_type,
                                              inference_output_type):
    with self.assertRaises(ValueError):
      _ = quantization.QuantizationConfig(
          inference_input_type=inference_input_type,
          inference_output_type=inference_output_type)


if __name__ == '__main__':
  tf.test.main()
