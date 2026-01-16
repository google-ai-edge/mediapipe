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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from absl import flags
import tensorflow as tf

from mediapipe.model_maker.python.core.data import data_util

_WORKSPACE = "mediapipe"
_TEST_DATA_DIR = os.path.join(
    _WORKSPACE, 'mediapipe/model_maker/python/core/data/testdata')

FLAGS = flags.FLAGS


class DataUtilTest(tf.test.TestCase):

  def test_load_rgb_image(self):
    image_path = os.path.join(FLAGS.test_srcdir, _TEST_DATA_DIR, 'test.jpg')
    image_data = data_util.load_image(image_path)
    self.assertEqual(image_data.shape, (5184, 3456, 3))


if __name__ == '__main__':
  tf.test.main()
