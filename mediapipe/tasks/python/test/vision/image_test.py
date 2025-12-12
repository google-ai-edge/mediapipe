# Copyright 2025 The MediaPipe Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np

from mediapipe.tasks.python.test import test_utils
from mediapipe.tasks.python.vision.core import image

_TEST_DATA_DIR = 'mediapipe/tasks/testdata/vision'
_IMAGE_FILE = 'portrait.jpg'


class ImageTest(parameterized.TestCase):

  def test_create_from_numpy(self):
    np_array = np.zeros((3, 4, 3), dtype=np.uint8)
    img = image.Image(image.ImageFormat.SRGB, np_array)
    self.assertEqual(img.width, 4)
    self.assertEqual(img.height, 3)
    self.assertEqual(img.channels, 3)
    self.assertEqual(img.image_format, image.ImageFormat.SRGB)
    np.testing.assert_array_equal(img.numpy_view(), np_array)

  def test_create_from_file(self):
    test_image_path = test_utils.get_test_data_path(
        os.path.join(_TEST_DATA_DIR, _IMAGE_FILE)
    )
    img = image.Image.create_from_file(test_image_path)
    # portrait.jpg is 820x1024, 3 channels (SRGB)
    self.assertEqual(img.width, 820)
    self.assertEqual(img.height, 1024)
    self.assertEqual(img.channels, 3)
    self.assertEqual(img.image_format, image.ImageFormat.SRGB)

  def test_get_item_uint8(self):
    pixel_data = np.array(
        [[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]], dtype=np.uint8
    )
    img = image.Image(image.ImageFormat.SRGB, pixel_data)
    self.assertEqual(img[0, 1, 1], 5)  # row 0, col 1, channel 1
    self.assertEqual(img[1, 0, 2], 9)  # row 1, col 0, channel 2

  def test_get_item_uint8_grayscale(self):
    pixel_data = np.array([[[1], [2]], [[3], [4]]], dtype=np.uint8)
    img = image.Image(image.ImageFormat.GRAY8, pixel_data)
    self.assertEqual(img[0, 1], 2)  # row 0, col 1
    self.assertEqual(img[1, 0], 3)  # row 1, col 0

  def test_get_item_uint16(self):
    pixel_data = np.array(
        [
            [[100, 200, 300], [400, 500, 600]],
            [[700, 800, 900], [1000, 1100, 1200]],
        ],
        dtype=np.uint16,
    )
    img = image.Image(image.ImageFormat.SRGB48, pixel_data)
    self.assertEqual(img[0, 1, 1], 500)  # row 0, col 1, channel 1
    self.assertEqual(img[1, 0, 0], 700)  # row 1, col 0, channel 0

  def test_get_item_float32(self):
    pixel_data = np.array(
        [[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], dtype=np.float32
    )
    img = image.Image(image.ImageFormat.VEC32F2, pixel_data)
    self.assertAlmostEqual(img[0, 1, 1], 4.0)  # row 0, col 1, channel 1
    self.assertAlmostEqual(img[1, 0, 0], 5.0)  # row 1, col 0, channel 0

  def test_uses_gpu(self):
    pixel_data = np.array([[[1], [2]], [[3], [4]]], dtype=np.uint8)
    img = image.Image(image.ImageFormat.GRAY8, pixel_data)
    self.assertFalse(img.uses_gpu())

  def test_is_contiguous(self):
    pixel_data = np.array([[[1], [2]], [[3], [4]]], dtype=np.uint8)
    img = image.Image(image.ImageFormat.GRAY8, pixel_data)
    self.assertFalse(img.is_contiguous())

  def test_is_empty(self):
    pixel_data = np.array([[[1], [2]], [[3], [4]]], dtype=np.uint8)
    img = image.Image(image.ImageFormat.GRAY8, pixel_data)
    self.assertFalse(img.is_empty())

  def test_is_aligned(self):
    pixel_data = np.array([[[1], [2]], [[3], [4]]], dtype=np.uint8)
    img = image.Image(image.ImageFormat.GRAY8, pixel_data)
    self.assertTrue(img.is_aligned(1))
    self.assertFalse(img.is_aligned(32))

  def test_create_from_numpy_error(self):
    # SRGB expects 3 channels, but the array has 2.
    np_array = np.zeros((3, 4, 2), dtype=np.uint8)
    with self.assertRaisesRegex(ValueError, 'Pixel data size is too small'):
      image.Image(image.ImageFormat.SRGB, np_array)


if __name__ == '__main__':
  absltest.main()
