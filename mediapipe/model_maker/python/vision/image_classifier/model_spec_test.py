# Copyright 2022 The MediaPipe Authors.
#
# Licensed under the Apache License, Version 2.0 (the 'License');
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an 'AS IS' BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os

from typing import Callable, List
from absl.testing import parameterized
import tensorflow as tf

from mediapipe.model_maker.python.vision.image_classifier import model_spec as ms


class ModelSpecTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.named_parameters(
      dict(
          testcase_name='mobilenet_v2_spec_test',
          model_spec=ms.mobilenet_v2_spec,
          expected_uri='https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/4',
          expected_name='mobilenet_v2',
          expected_input_image_shape=[224, 224]),
      dict(
          testcase_name='efficientnet_lite0_spec_test',
          model_spec=ms.efficientnet_lite0_spec,
          expected_uri='https://tfhub.dev/tensorflow/efficientnet/lite0/feature-vector/2',
          expected_name='efficientnet_lite0',
          expected_input_image_shape=[224, 224]),
      dict(
          testcase_name='efficientnet_lite2_spec_test',
          model_spec=ms.efficientnet_lite2_spec,
          expected_uri='https://tfhub.dev/tensorflow/efficientnet/lite2/feature-vector/2',
          expected_name='efficientnet_lite2',
          expected_input_image_shape=[260, 260]),
      dict(
          testcase_name='efficientnet_lite4_spec_test',
          model_spec=ms.efficientnet_lite4_spec,
          expected_uri='https://tfhub.dev/tensorflow/efficientnet/lite4/feature-vector/2',
          expected_name='efficientnet_lite4',
          expected_input_image_shape=[300, 300]),
  )
  def test_predefiend_spec(self, model_spec: Callable[..., ms.ModelSpec],
                           expected_uri: str, expected_name: str,
                           expected_input_image_shape: List[int]):
    model_spec_obj = model_spec()
    self.assertIsInstance(model_spec_obj, ms.ModelSpec)
    self.assertEqual(model_spec_obj.uri, expected_uri)
    self.assertEqual(model_spec_obj.name, expected_name)
    self.assertEqual(model_spec_obj.input_image_shape,
                     expected_input_image_shape)

  def test_create_spec(self):
    custom_model_spec = ms.ModelSpec(
        uri='https://custom_model',
        input_image_shape=[128, 128],
        name='custom_model')
    self.assertEqual(custom_model_spec.uri, 'https://custom_model')
    self.assertEqual(custom_model_spec.name, 'custom_model')
    self.assertEqual(custom_model_spec.input_image_shape, [128, 128])


if __name__ == '__main__':
  # Load compressed models from tensorflow_hub
  os.environ['TFHUB_MODEL_LOAD_FORMAT'] = 'COMPRESSED'
  tf.test.main()
