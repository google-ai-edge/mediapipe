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

import filecmp
import io
import os
import tempfile

from unittest import mock as unittest_mock
from absl.testing import parameterized
import mock
import numpy as np
import tensorflow as tf

from mediapipe.model_maker.python.vision import image_classifier
from mediapipe.model_maker.python.vision.image_classifier import hyperparameters
from mediapipe.model_maker.python.vision.image_classifier import model_options
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
    data = image_classifier.Dataset(
        ds, ['cyan', 'magenta', 'yellow'], self.IMAGES_PER_CLASS * 3
    )
    return data

  def setUp(self):
    super(ImageClassifierTest, self).setUp()
    all_data = self._gen_cmy_data()
    # Splits data, 90% data for training, 10% for testing
    self._train_data, self._test_data = all_data.split(0.9)

  @parameterized.named_parameters(
      dict(
          testcase_name='mobilenet_v2',
          options=image_classifier.ImageClassifierOptions(
              supported_model=image_classifier.SupportedModels.MOBILENET_V2,
              hparams=image_classifier.HParams(
                  epochs=1,
                  batch_size=1,
                  shuffle=True,
                  export_dir=tempfile.mkdtemp()))),
      dict(
          testcase_name='efficientnet_lite0',
          options=image_classifier.ImageClassifierOptions(
              supported_model=(
                  image_classifier.SupportedModels.EFFICIENTNET_LITE0),
              hparams=image_classifier.HParams(
                  epochs=1,
                  batch_size=1,
                  shuffle=True,
                  export_dir=tempfile.mkdtemp()))),
      dict(
          testcase_name='efficientnet_lite0_change_dropout_rate',
          options=image_classifier.ImageClassifierOptions(
              supported_model=(
                  image_classifier.SupportedModels.EFFICIENTNET_LITE0),
              model_options=image_classifier.ModelOptions(dropout_rate=0.1),
              hparams=image_classifier.HParams(
                  epochs=1,
                  batch_size=1,
                  shuffle=True,
                  export_dir=tempfile.mkdtemp()))),
      dict(
          testcase_name='efficientnet_lite2',
          options=image_classifier.ImageClassifierOptions(
              supported_model=(
                  image_classifier.SupportedModels.EFFICIENTNET_LITE2),
              hparams=image_classifier.HParams(
                  epochs=1,
                  batch_size=1,
                  shuffle=True,
                  export_dir=tempfile.mkdtemp()))),
      dict(
          testcase_name='efficientnet_lite4',
          options=image_classifier.ImageClassifierOptions(
              supported_model=(
                  image_classifier.SupportedModels.EFFICIENTNET_LITE4),
              hparams=image_classifier.HParams(
                  epochs=1,
                  batch_size=1,
                  shuffle=True,
                  export_dir=tempfile.mkdtemp()))),
  )
  def test_create_and_train_model(
      self, options: image_classifier.ImageClassifierOptions):
    model = image_classifier.ImageClassifier.create(
        train_data=self._train_data,
        validation_data=self._test_data,
        options=options)
    self._test_accuracy(model)

    # Test export_model
    model.export_model()
    output_metadata_file = os.path.join(options.hparams.export_dir,
                                        'metadata.json')
    output_tflite_file = os.path.join(options.hparams.export_dir,
                                      'model.tflite')
    expected_metadata_file = test_utils.get_test_data_path('metadata.json')

    self.assertTrue(os.path.exists(output_tflite_file))
    self.assertGreater(os.path.getsize(output_tflite_file), 0)

    self.assertTrue(os.path.exists(output_metadata_file))
    self.assertGreater(os.path.getsize(output_metadata_file), 0)
    filecmp.clear_cache()
    self.assertTrue(
        filecmp.cmp(
            output_metadata_file, expected_metadata_file, shallow=False))

  def test_continual_training_by_loading_checkpoint(self):
    mock_stdout = io.StringIO()
    with mock.patch('sys.stdout', mock_stdout):
      options = image_classifier.ImageClassifierOptions(
          supported_model=image_classifier.SupportedModels.EFFICIENTNET_LITE0,
          hparams=image_classifier.HParams(
              epochs=5, batch_size=1, shuffle=True))
      model = image_classifier.ImageClassifier.create(
          train_data=self._train_data,
          validation_data=self._test_data,
          options=options)
      model = image_classifier.ImageClassifier.create(
          train_data=self._train_data,
          validation_data=self._test_data,
          options=options)
      self._test_accuracy(model)

    self.assertRegex(mock_stdout.getvalue(), 'Resuming from')

  def _test_accuracy(self, model, threshold=0.0):
    _, accuracy = model.evaluate(self._test_data)
    self.assertGreaterEqual(accuracy, threshold)

  @unittest_mock.patch.object(
      hyperparameters,
      'HParams',
      autospec=True,
      return_value=hyperparameters.HParams(epochs=1))
  @unittest_mock.patch.object(
      model_options,
      'ImageClassifierModelOptions',
      autospec=True,
      return_value=model_options.ImageClassifierModelOptions())
  def test_create_hparams_and_model_options_if_none_in_image_classifier_options(
      self, mock_hparams, mock_model_options):
    options = image_classifier.ImageClassifierOptions(
        supported_model=(image_classifier.SupportedModels.EFFICIENTNET_LITE0))
    image_classifier.ImageClassifier.create(
        train_data=self._train_data,
        validation_data=self._test_data,
        options=options)
    mock_hparams.assert_called_once()
    mock_model_options.assert_called_once()


if __name__ == '__main__':
  # Load compressed models from tensorflow_hub
  os.environ['TFHUB_MODEL_LOAD_FORMAT'] = 'COMPRESSED'
  tf.test.main()
