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
from unittest import mock as unittest_mock

import numpy as np
import tensorflow as tf

from mediapipe.model_maker.python.vision.object_detector import dataset as ds
from mediapipe.model_maker.python.vision.object_detector import model as model_lib
from mediapipe.model_maker.python.vision.object_detector import model_options as model_opt
from mediapipe.model_maker.python.vision.object_detector import model_spec as ms
from mediapipe.model_maker.python.vision.object_detector import preprocessor
from mediapipe.tasks.python.test import test_utils as task_test_utils


def _dicts_match(dict_1, dict_2):
  for key in dict_1:
    if key not in dict_2 or np.any(dict_1[key] != dict_2[key]):
      return False
  return True


def _outputs_match(output1, output2):
  return _dicts_match(
      output1['cls_outputs'], output2['cls_outputs']
  ) and _dicts_match(output1['box_outputs'], output2['box_outputs'])


class ObjectDetectorModelTest(tf.test.TestCase):

  def setUp(self):
    super().setUp()
    dataset_folder = task_test_utils.get_test_data_path('coco_data')
    cache_dir = self.create_tempdir()
    self.data = ds.Dataset.from_coco_folder(dataset_folder, cache_dir=cache_dir)
    self.model_spec = ms.SupportedModels.MOBILENET_V2.value()
    self.preprocessor = preprocessor.Preprocessor(self.model_spec)
    self.fake_inputs = np.random.uniform(
        low=0, high=1, size=(1, 256, 256, 3)
    ).astype(np.float32)
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

  def _create_model(self):
    model_options = model_opt.ObjectDetectorModelOptions()
    model = model_lib.ObjectDetectorModel(
        self.model_spec, model_options, self.data.num_classes
    )
    return model

  def _train_model(self, model):
    """Helper to run a simple training run on the model."""
    dataset = self.data.gen_tf_dataset(
        batch_size=2,
        is_training=True,
        shuffle=False,
        preprocess=self.preprocessor,
    )
    optimizer = tf.keras.optimizers.experimental.SGD(
        learning_rate=0.03, momentum=0.9
    )
    model.compile(optimizer=optimizer)
    model.fit(
        x=dataset, epochs=2, steps_per_epoch=None, validation_data=dataset
    )

  def test_model(self):
    model = self._create_model()
    outputs_before = model(self.fake_inputs, training=True)
    self._train_model(model)
    outputs_after = model(self.fake_inputs, training=True)
    self.assertFalse(_outputs_match(outputs_before, outputs_after))

  def test_model_convert_to_qat(self):
    model_options = model_opt.ObjectDetectorModelOptions()
    model = model_lib.ObjectDetectorModel(
        self.model_spec, model_options, self.data.num_classes
    )
    outputs_before = model(self.fake_inputs, training=True)
    model.convert_to_qat()
    outputs_after = model(self.fake_inputs, training=True)
    self.assertFalse(_outputs_match(outputs_before, outputs_after))
    outputs_before = outputs_after
    self._train_model(model)
    outputs_after = model(self.fake_inputs, training=True)
    self.assertFalse(_outputs_match(outputs_before, outputs_after))

  def test_model_save_and_load_checkpoint(self):
    model = self._create_model()
    checkpoint_path = os.path.join(self.create_tempdir(), 'ckpt')
    model.save_checkpoint(checkpoint_path)
    data_checkpoint_file = checkpoint_path + '.data-00000-of-00001'
    index_checkpoint_file = checkpoint_path + '.index'
    self.assertTrue(os.path.exists(data_checkpoint_file))
    self.assertTrue(os.path.exists(index_checkpoint_file))
    self.assertGreater(os.path.getsize(data_checkpoint_file), 0)
    self.assertGreater(os.path.getsize(index_checkpoint_file), 0)
    outputs_before = model(self.fake_inputs, training=True)

    # Check model output is different after training
    self._train_model(model)
    outputs_after = model(self.fake_inputs, training=True)
    self.assertFalse(_outputs_match(outputs_before, outputs_after))

    # Check model output is the same after loading previous checkpoint
    model.load_checkpoint(checkpoint_path, include_last_layer=True)
    outputs_after = model(self.fake_inputs, training=True)
    self.assertTrue(_outputs_match(outputs_before, outputs_after))

  def test_export_saved_model(self):
    export_dir = self.create_tempdir()
    export_path = os.path.join(export_dir, 'saved_model')
    model = self._create_model()
    model.export_saved_model(export_path)
    self.assertTrue(os.path.exists(export_path))
    self.assertGreater(os.path.getsize(export_path), 0)
    model.convert_to_qat()
    export_path = os.path.join(export_dir, 'saved_model_qat')
    model.export_saved_model(export_path)
    self.assertTrue(os.path.exists(export_path))
    self.assertGreater(os.path.getsize(export_path), 0)


if __name__ == '__main__':
  tf.test.main()
