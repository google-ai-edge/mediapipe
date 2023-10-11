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
import tensorflow as tf

from mediapipe.model_maker.python.vision.core import image_utils
from mediapipe.model_maker.python.vision.core import test_utils


class ImageUtilsTest(tf.test.TestCase):

  def setUp(self):
    super().setUp()
    self.jpeg_img = os.path.join(self.get_temp_dir(), 'image.jpeg')
    if os.path.exists(self.jpeg_img):
      return
    test_utils.write_filled_jpeg_file(self.jpeg_img, [0, 125, 255], 224)

  def test_load_image(self):
    img_tensor = image_utils.load_image(self.jpeg_img)
    self.assertEqual(img_tensor.shape, (224, 224, 3))


if __name__ == '__main__':
  tf.test.main()
