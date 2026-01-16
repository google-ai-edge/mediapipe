# Copyright 2023 The MediaPipe Authors.
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
import tempfile
import unittest
from unittest import mock as unittest_mock

from absl.testing import parameterized
import tensorflow as tf

from mediapipe.model_maker.python.vision import object_detector
from mediapipe.tasks.python.test import test_utils as task_test_utils


class ObjectDetectorTest(tf.test.TestCase, parameterized.TestCase):

  def setUp(self):
    super().setUp()
    dataset_folder = task_test_utils.get_test_data_path('coco_data')
    cache_dir = self.create_tempdir()
    self.data = object_detector.Dataset.from_coco_folder(
        dataset_folder, cache_dir=cache_dir
    )
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

  @unittest.skip("Timeouts/Flaky")
  def test_object_detector(self):
    hparams = object_detector.HParams(
        epochs=1,
        batch_size=2,
        learning_rate=0.9,
        shuffle=False,
        export_dir=self.create_tempdir(),
    )
    options = object_detector.ObjectDetectorOptions(
        supported_model=object_detector.SupportedModels.MOBILENET_V2,
        hparams=hparams,
    )
    # Test `create``
    model = object_detector.ObjectDetector.create(
        train_data=self.data, validation_data=self.data, options=options
    )
    losses, coco_metrics = model.evaluate(self.data)
    self._assert_ap_greater(coco_metrics)
    self.assertFalse(model._is_qat)
    # Test float export_model
    model.export_model()
    output_metadata_file = os.path.join(
        options.hparams.export_dir, 'metadata.json'
    )
    output_tflite_file = os.path.join(
        options.hparams.export_dir, 'model.tflite'
    )
    self.assertTrue(os.path.exists(output_tflite_file))
    self.assertGreater(os.path.getsize(output_tflite_file), 0)
    self.assertTrue(os.path.exists(output_metadata_file))
    self.assertGreater(os.path.getsize(output_metadata_file), 0)

    # Test `quantization_aware_training`
    qat_hparams = object_detector.QATHParams(
        learning_rate=0.9,
        batch_size=2,
        epochs=1,
        decay_steps=6,
        decay_rate=0.96,
    )
    model.quantization_aware_training(self.data, self.data, qat_hparams)
    qat_losses, qat_coco_metrics = model.evaluate(self.data)
    self._assert_ap_greater(qat_coco_metrics)
    self.assertNotAllEqual(losses, qat_losses)
    self.assertTrue(model._is_qat)
    model.export_model('model_qat.tflite')
    output_metadata_file = os.path.join(
        options.hparams.export_dir, 'metadata.json'
    )
    output_tflite_file = os.path.join(
        options.hparams.export_dir, 'model_qat.tflite'
    )
    self.assertTrue(os.path.exists(output_tflite_file))
    self.assertGreater(os.path.getsize(output_tflite_file), 0)
    self.assertLess(os.path.getsize(output_tflite_file), 3500000)
    self.assertTrue(os.path.exists(output_metadata_file))
    self.assertGreater(os.path.getsize(output_metadata_file), 0)

    # Load float ckpt test
    model.restore_float_ckpt()
    losses_2, _ = model.evaluate(self.data)
    self.assertAllEqual(losses, losses_2)
    self.assertNotAllEqual(qat_losses, losses_2)
    self.assertFalse(model._is_qat)

  def _assert_ap_greater(self, coco_metrics, threshold=0.0):
    self.assertGreaterEqual(coco_metrics['AP'], threshold)


if __name__ == '__main__':
  tf.test.main()
