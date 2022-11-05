# Copyright 2022 The MediaPipe Authors. All Rights Reserved.
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

import filecmp
import os

from absl.testing import parameterized
import numpy as np
import tensorflow as tf

from mediapipe.model_maker.python.vision import image_classifier
from mediapipe.tasks.python.test import test_utils


def _fill_image(rgb, image_size):
  r, g, b = rgb
  return np.broadcast_to(
      np.array([[[r, g, b]]], dtype=np.uint8),
      shape=(image_size, image_size, 3))


class ImageClassifierTest(tf.test.TestCase, parameterized.TestCase):
  IMAGE_SIZE = 24
  IMAGES_PER_CLASS = 2
  CMY_NAMES_AND_RGB_VALUES = (('cyan', (0, 255, 255)),
                              ('magenta', (255, 0, 255)), ('yellow', (255, 255,
                                                                      0)))

  def _gen(self):
    for i, (_, rgb) in enumerate(self.CMY_NAMES_AND_RGB_VALUES):
      for _ in range(self.IMAGES_PER_CLASS):
        yield (_fill_image(rgb, self.IMAGE_SIZE), i)

  def _gen_cmy_data(self):
    ds = tf.data.Dataset.from_generator(
        self._gen, (tf.uint8, tf.int64), (tf.TensorShape(
            [self.IMAGE_SIZE, self.IMAGE_SIZE, 3]), tf.TensorShape([])))
    data = image_classifier.Dataset(ds, self.IMAGES_PER_CLASS * 3,
                                    ['cyan', 'magenta', 'yellow'])
    return data

  def setUp(self):
    super(ImageClassifierTest, self).setUp()
    all_data = self._gen_cmy_data()
    # Splits data, 90% data for training, 10% for testing
    self.train_data, self.test_data = all_data.split(0.9)

  @parameterized.named_parameters(
      dict(
          testcase_name='mobilenet_v2',
          model_spec=image_classifier.SupportedModels.MOBILENET_V2,
          hparams=image_classifier.HParams(
              train_epochs=1, batch_size=1, shuffle=True)),
      dict(
          testcase_name='efficientnet_lite0',
          model_spec=image_classifier.SupportedModels.EFFICIENTNET_LITE0,
          hparams=image_classifier.HParams(
              train_epochs=1, batch_size=1, shuffle=True)),
      dict(
          testcase_name='efficientnet_lite2',
          model_spec=image_classifier.SupportedModels.EFFICIENTNET_LITE2,
          hparams=image_classifier.HParams(
              train_epochs=1, batch_size=1, shuffle=True)),
      dict(
          testcase_name='efficientnet_lite4',
          model_spec=image_classifier.SupportedModels.EFFICIENTNET_LITE4,
          hparams=image_classifier.HParams(
              train_epochs=1, batch_size=1, shuffle=True)),
  )
  def test_create_and_train_model(self,
                                  model_spec: image_classifier.SupportedModels,
                                  hparams: image_classifier.HParams):
    model = image_classifier.ImageClassifier.create(
        model_spec=model_spec,
        train_data=self.train_data,
        hparams=hparams,
        validation_data=self.test_data)
    self._test_accuracy(model)

  def test_efficientnetlite0_model_train_and_export(self):
    hparams = image_classifier.HParams(
        train_epochs=1, batch_size=1, shuffle=True)
    model = image_classifier.ImageClassifier.create(
        model_spec=image_classifier.SupportedModels.EFFICIENTNET_LITE0,
        train_data=self.train_data,
        hparams=hparams,
        validation_data=self.test_data)
    self._test_accuracy(model)

    # Test export_model
    model.export_model()
    output_metadata_file = os.path.join(hparams.model_dir, 'metadata.json')
    output_tflite_file = os.path.join(hparams.model_dir, 'model.tflite')
    expected_metadata_file = test_utils.get_test_data_path('metadata.json')

    self.assertTrue(os.path.exists(output_tflite_file))
    self.assertGreater(os.path.getsize(output_tflite_file), 0)

    self.assertTrue(os.path.exists(output_metadata_file))
    self.assertGreater(os.path.getsize(output_metadata_file), 0)
    self.assertTrue(filecmp.cmp(output_metadata_file, expected_metadata_file))

  def _test_accuracy(self, model, threshold=0.0):
    _, accuracy = model.evaluate(self.test_data)
    self.assertGreaterEqual(accuracy, threshold)


if __name__ == '__main__':
  # Load compressed models from tensorflow_hub
  os.environ['TFHUB_MODEL_LOAD_FORMAT'] = 'COMPRESSED'
  tf.test.main()
