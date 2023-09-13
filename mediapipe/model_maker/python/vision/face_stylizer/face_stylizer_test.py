# Copyright 2023 The MediaPipe Authors. All Rights Reserved.
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
import zipfile

import tensorflow as tf

from mediapipe.model_maker.python.core.utils import test_util as mm_test_util
from mediapipe.model_maker.python.vision import face_stylizer
from mediapipe.tasks.python.test import test_utils


class FaceStylizerTest(tf.test.TestCase):

  def _create_training_dataset(self):
    """Creates training dataset."""
    input_style_image_file = test_utils.get_test_data_path(
        'input/style/cartoon/cartoon.jpg'
    )

    data = face_stylizer.Dataset.from_image(filename=input_style_image_file)
    return data

  def _create_eval_dataset(self):
    """Create evaluation dataset."""
    input_test_image_file = test_utils.get_test_data_path(
        'input/raw/face/portrait.jpg'
    )

    data = face_stylizer.Dataset.from_image(filename=input_test_image_file)
    return data

  def _evaluate_saved_model(self, model: face_stylizer.FaceStylizer):
    """Evaluates the fine-tuned face stylizer model."""
    test_image = tf.ones(shape=(256, 256, 3), dtype=tf.float32)
    test_image_batch = test_image[tf.newaxis]
    in_latent = model._encoder(test_image_batch)
    output = model._decoder({'inputs': in_latent + model.w_avg})
    self.assertEqual(output['image'][-1].shape, (1, 256, 256, 3))

  def setUp(self):
    super().setUp()
    self._train_data = self._create_training_dataset()
    self._eval_data = self._create_eval_dataset()

  def test_finetuning_face_stylizer_with_single_input_style_image(self):
    with self.test_session(use_gpu=True):
      face_stylizer_options = face_stylizer.FaceStylizerOptions(
          model=face_stylizer.SupportedModels.BLAZE_FACE_STYLIZER_256,
          hparams=face_stylizer.HParams(epochs=1),
      )
      model = face_stylizer.FaceStylizer.create(
          train_data=self._train_data, options=face_stylizer_options
      )
      self._evaluate_saved_model(model)

  def test_evaluate_face_stylizer(self):
    with self.test_session(use_gpu=True):
      face_stylizer_options = face_stylizer.FaceStylizerOptions(
          model=face_stylizer.SupportedModels.BLAZE_FACE_STYLIZER_256,
          hparams=face_stylizer.HParams(epochs=1),
      )
      model = face_stylizer.FaceStylizer.create(
          train_data=self._train_data, options=face_stylizer_options
      )
      eval_output = model.stylize(self._eval_data)
      self.assertLen(eval_output, 1)
      eval_output_data = eval_output.gen_tf_dataset()
      iterator = iter(eval_output_data)
      self.assertEqual(iterator.get_next().shape, (1, 256, 256, 3))

  def test_export_face_stylizer_tflite_model(self):
    with self.test_session(use_gpu=True):
      model_enum = face_stylizer.SupportedModels.BLAZE_FACE_STYLIZER_256
      face_stylizer_options = face_stylizer.FaceStylizerOptions(
          model=model_enum,
          hparams=face_stylizer.HParams(
              epochs=0, export_dir=self.get_temp_dir()
          ),
      )
      model = face_stylizer.FaceStylizer.create(
          train_data=self._train_data, options=face_stylizer_options
      )
      model.export_model()
      model_bundle_file = os.path.join(
          self.get_temp_dir(), 'face_stylizer.task'
      )
      with zipfile.ZipFile(model_bundle_file) as zf:
        self.assertEqual(
            set(zf.namelist()),
            set([
                'face_detector.tflite',
                'face_landmarks_detector.tflite',
                'face_stylizer.tflite',
            ]),
        )
        zf.extractall(self.get_temp_dir())

      face_stylizer_tflite_file = os.path.join(
          self.get_temp_dir(), 'face_stylizer.tflite'
      )
      spec = face_stylizer.SupportedModels.get(model_enum)
      input_image_shape = spec.input_image_shape
      input_tensor_shape = [1] + list(input_image_shape) + [3]
      input_tensor = mm_test_util.create_random_sample(size=input_tensor_shape)
      output = mm_test_util.run_tflite(face_stylizer_tflite_file, input_tensor)
      self.assertTrue((output >= 0.0).all())
      self.assertTrue((output <= 1.0).all())


if __name__ == '__main__':
  tf.test.main()
