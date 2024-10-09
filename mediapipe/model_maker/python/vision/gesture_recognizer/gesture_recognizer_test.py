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

import io
import os
import tempfile
from unittest import mock as unittest_mock
import zipfile

import tensorflow as tf

from mediapipe.model_maker.python.core.utils import test_util
from mediapipe.model_maker.python.vision import gesture_recognizer
from mediapipe.model_maker.python.vision.gesture_recognizer import hyperparameters
from mediapipe.model_maker.python.vision.gesture_recognizer import model_options
from mediapipe.tasks.python.test import test_utils

_TEST_DATA_DIR = 'mediapipe/model_maker/python/vision/gesture_recognizer/testdata'
tf.keras.backend.experimental.enable_tf_random_generator()


class GestureRecognizerTest(tf.test.TestCase):

  def _load_data(self):
    input_data_dir = test_utils.get_test_data_path(
        os.path.join(_TEST_DATA_DIR, 'raw_data'))

    data = gesture_recognizer.Dataset.from_folder(
        dirname=input_data_dir,
        hparams=gesture_recognizer.HandDataPreprocessingParams(shuffle=True))
    return data

  def setUp(self):
    super().setUp()
    tf.keras.utils.set_random_seed(87654321)
    # Mock tempfile.gettempdir() to be unique for each test to avoid race
    # condition when downloading model since these tests may run in parallel.
    mock_gettempdir = unittest_mock.patch.object(
        tempfile,
        'gettempdir',
        return_value=self.create_tempdir(),
        autospec=True,
    )
    self.mock_gettempdir = mock_gettempdir.start()
    self.addCleanup(mock_gettempdir.stop)
    # Load dataset used by tests
    all_data = self._load_data()
    # Splits data, 90% data for training, 10% for validation
    self._train_data, self._validation_data = all_data.split(0.9)

  def test_gesture_recognizer_model(self):
    mo = gesture_recognizer.ModelOptions()
    hparams = gesture_recognizer.HParams(
        export_dir=tempfile.mkdtemp(), epochs=2)
    gesture_recognizer_options = gesture_recognizer.GestureRecognizerOptions(
        model_options=mo, hparams=hparams)
    model = gesture_recognizer.GestureRecognizer.create(
        train_data=self._train_data,
        validation_data=self._validation_data,
        options=gesture_recognizer_options)

    self._test_accuracy(model)

  @unittest_mock.patch.object(
      tf.keras.layers, 'Dense', wraps=tf.keras.layers.Dense
  )
  def test_gesture_recognizer_model_layer_widths(self, mock_dense):
    layer_widths = [64, 32]
    mo = gesture_recognizer.ModelOptions(layer_widths=layer_widths)
    hparams = gesture_recognizer.HParams(
        export_dir=tempfile.mkdtemp(), epochs=2)
    gesture_recognizer_options = gesture_recognizer.GestureRecognizerOptions(
        model_options=mo, hparams=hparams)
    model = gesture_recognizer.GestureRecognizer.create(
        train_data=self._train_data,
        validation_data=self._validation_data,
        options=gesture_recognizer_options)
    expected_calls = [
        unittest_mock.call(w, name=f'custom_gesture_recognizer_{i}')
        for i, w in enumerate(layer_widths)
    ]
    expected_calls.append(
        unittest_mock.call(
            len(self._train_data.label_names),
            activation='softmax',
            name='custom_gesture_recognizer_out'))
    self.assertLen(mock_dense.call_args_list, len(expected_calls))
    mock_dense.assert_has_calls(expected_calls)
    self._test_accuracy(model)

  def test_export_gesture_recognizer_model(self):
    mo = gesture_recognizer.ModelOptions()
    hparams = gesture_recognizer.HParams(
        export_dir=tempfile.mkdtemp(), epochs=2)
    gesture_recognizer_options = gesture_recognizer.GestureRecognizerOptions(
        model_options=mo, hparams=hparams)
    model = gesture_recognizer.GestureRecognizer.create(
        train_data=self._train_data,
        validation_data=self._validation_data,
        options=gesture_recognizer_options)
    model.export_model()
    model_bundle_file = os.path.join(hparams.export_dir,
                                     'gesture_recognizer.task')
    with zipfile.ZipFile(model_bundle_file) as zf:
      self.assertEqual(
          set(zf.namelist()),
          set(['hand_landmarker.task', 'hand_gesture_recognizer.task']))
      zf.extractall(self.get_temp_dir())
    hand_gesture_recognizer_bundle_file = os.path.join(
        self.get_temp_dir(), 'hand_gesture_recognizer.task')
    with zipfile.ZipFile(hand_gesture_recognizer_bundle_file) as zf:
      self.assertEqual(
          set(zf.namelist()),
          set([
              'canned_gesture_classifier.tflite',
              'custom_gesture_classifier.tflite', 'gesture_embedder.tflite'
          ]))
      zf.extractall(self.get_temp_dir())
    gesture_classifier_tflite_file = os.path.join(
        self.get_temp_dir(), 'custom_gesture_classifier.tflite')
    test_util.test_tflite_file(
        keras_model=model._model,
        tflite_file=gesture_classifier_tflite_file,
        size=[1, model.embedding_size])

  def _test_accuracy(self, model, threshold=0.0):
    # Test on _train_data because of our limited dataset size
    _, accuracy = model.evaluate(self._train_data)
    tf.compat.v1.logging.info(f'train accuracy: {accuracy}')
    self.assertGreater(accuracy, threshold)

  @unittest_mock.patch.object(
      hyperparameters,
      'HParams',
      autospec=True,
      return_value=gesture_recognizer.HParams(epochs=1),
  )
  @unittest_mock.patch.object(
      model_options,
      'GestureRecognizerModelOptions',
      autospec=True,
      return_value=gesture_recognizer.ModelOptions(),
  )
  def test_create_hparams_and_model_options_if_none_in_gesture_recognizer_options(
      self, mock_hparams, mock_model_options):
    options = gesture_recognizer.GestureRecognizerOptions()
    gesture_recognizer.GestureRecognizer.create(
        train_data=self._train_data,
        validation_data=self._validation_data,
        options=options)
    mock_hparams.assert_called_once()
    mock_model_options.assert_called_once()

  def test_continual_training_by_loading_checkpoint(self):
    mo = gesture_recognizer.ModelOptions()
    hparams = gesture_recognizer.HParams(
        export_dir=tempfile.mkdtemp(), epochs=2)
    gesture_recognizer_options = gesture_recognizer.GestureRecognizerOptions(
        model_options=mo, hparams=hparams)
    mock_stdout = io.StringIO()
    with unittest_mock.patch('sys.stdout', mock_stdout):
      model = gesture_recognizer.GestureRecognizer.create(
          train_data=self._train_data,
          validation_data=self._validation_data,
          options=gesture_recognizer_options)
      model = gesture_recognizer.GestureRecognizer.create(
          train_data=self._train_data,
          validation_data=self._validation_data,
          options=gesture_recognizer_options)
      self._test_accuracy(model)

    self.assertRegex(mock_stdout.getvalue(), 'Resuming from')


if __name__ == '__main__':
  tf.test.main()
