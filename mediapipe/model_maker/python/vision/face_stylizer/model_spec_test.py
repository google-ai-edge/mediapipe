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


import tensorflow as tf

from mediapipe.model_maker.python.vision.face_stylizer import model_spec as ms


class ModelSpecTest(tf.test.TestCase):

  def test_predefine_spec(self):
    blaze_face_stylizer_256_spec = ms.blaze_face_stylizer_256_spec()
    self.assertIsInstance(blaze_face_stylizer_256_spec, ms.ModelSpec)
    self.assertEqual(blaze_face_stylizer_256_spec.style_block_num, 12)
    self.assertAllEqual(
        blaze_face_stylizer_256_spec.input_image_shape, [256, 256]
    )
    self.assertEqual(
        blaze_face_stylizer_256_spec.name, 'blaze_face_stylizer_256'
    )

  def test_predefine_spec_enum(self):
    blaze_face_stylizer_256 = ms.SupportedModels.BLAZE_FACE_STYLIZER_256
    spec = ms.SupportedModels.get(blaze_face_stylizer_256)
    self.assertIsInstance(spec, ms.ModelSpec)
    self.assertEqual(spec.style_block_num, 12)
    self.assertAllEqual(spec.input_image_shape, [256, 256])
    self.assertEqual(spec.name, 'blaze_face_stylizer_256')


if __name__ == '__main__':
  tf.test.main()
