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

import gc
import os
import sys
import threading
from unittest import mock
import weakref

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

  def test_numpy_view_non_contiguous_float32(self):
    pixel_data = np.array([[[1.0], [2.0]], [[3.0], [4.0]]], dtype=np.float32)
    img = image.Image(image.ImageFormat.VEC32F1, pixel_data)
    self.assertFalse(img.is_contiguous())
    np.testing.assert_array_equal(img.numpy_view(), pixel_data)

  def test_numpy_view_non_contiguous_uint16(self):
    pixel_data = np.array([[[100, 200, 300], [400, 500, 600]]], dtype=np.uint16)
    img = image.Image(image.ImageFormat.SRGB48, pixel_data)
    self.assertFalse(img.is_contiguous())
    np.testing.assert_array_equal(img.numpy_view(), pixel_data)

  def test_numpy_view_is_unwritable(self):
    img = image.Image(image.ImageFormat.SRGB, np.zeros((3, 4, 3), np.uint8))
    self.assertFalse(img.numpy_view().flags.writeable)

  @parameterized.named_parameters(
      ('contiguous', (4, 16, 3), True),
      ('non_contiguous', (3, 5, 3), False),
  )
  def test_numpy_view_keeps_image_alive(self, shape, is_contiguous):
    pixel_data = np.arange(np.prod(shape), dtype=np.uint8).reshape(shape)
    img = image.Image(image.ImageFormat.SRGB, pixel_data)
    self.assertEqual(img.is_contiguous(), is_contiguous)
    img_ref = weakref.ref(img)
    sub_view = img.numpy_view()[1:, ::2]
    del img
    gc.collect()

    # The array points into the pixel data owned by the Image, so the Image
    # must not be freed while the array (or a view of it) is in use.
    self.assertIsNotNone(img_ref())
    # Allocate more images to overwrite the memory of a freed Image.
    unused_images = [
        image.Image(
            image.ImageFormat.SRGB, np.full(shape, 255, dtype=np.uint8)
        )
        for _ in range(50)
    ]
    np.testing.assert_array_equal(sub_view, pixel_data[1:, ::2])

    # The Image is released together with the last array that uses its data.
    del sub_view, unused_images
    gc.collect()
    self.assertIsNone(img_ref())

  @parameterized.named_parameters(
      (
          'uint8_flipped_columns',
          image.ImageFormat.SRGB,
          np.uint8,
          3,
          lambda a: a[:, ::-1],
      ),
      (
          'uint8_reversed_channels',
          image.ImageFormat.SRGB,
          np.uint8,
          3,
          lambda a: a[..., ::-1],
      ),
      (
          'uint8_strided_slice',
          image.ImageFormat.SRGB,
          np.uint8,
          3,
          lambda a: a[::2, 1::2],
      ),
      (
          'uint8_transposed',
          image.ImageFormat.SRGB,
          np.uint8,
          3,
          lambda a: a.transpose(1, 0, 2),
      ),
      (
          'uint16_flipped_rows',
          image.ImageFormat.SRGB48,
          np.uint16,
          3,
          lambda a: a[::-1],
      ),
      (
          'float32_flipped_columns',
          image.ImageFormat.VEC32F2,
          np.float32,
          2,
          lambda a: a[:, ::-1],
      ),
  )
  def test_create_from_non_contiguous_numpy(
      self, image_format, dtype, channels, make_view
  ):
    base = np.arange(4 * 8 * channels, dtype=dtype).reshape(4, 8, channels)
    pixel_data = make_view(base)
    self.assertFalse(pixel_data.flags.c_contiguous)
    expected = np.array(pixel_data)  # A C-contiguous copy.

    img = image.Image(image_format, pixel_data)

    self.assertEqual((img.height, img.width), expected.shape[:2])
    np.testing.assert_array_equal(img.numpy_view(), expected)
    self.assertEqual(img[1, 2, 0], expected[1, 2, 0])
    # The Image holds a copy, so it doesn't depend on the input afterwards.
    base[...] = 0
    np.testing.assert_array_equal(img.numpy_view(), expected)

  def test_create_from_non_contiguous_two_dimensional_numpy(self):
    pixel_data = np.arange(3 * 5, dtype=np.uint8).reshape(3, 5).T
    self.assertFalse(pixel_data.flags.c_contiguous)

    img = image.Image(image.ImageFormat.GRAY8, pixel_data)

    self.assertEqual((img.height, img.width), (5, 3))
    np.testing.assert_array_equal(img.numpy_view()[..., 0], pixel_data)

  def test_numpy_view_is_thread_safe(self):
    pixel_data = np.arange(5 * 7 * 3, dtype=np.uint8).reshape(5, 7, 3)
    num_threads = 4
    for _ in range(300):
      # The pixel data of a new non-contiguous Image is copied into a cache by
      # the first numpy_view() call.
      img = image.Image(image.ImageFormat.SRGB, pixel_data)
      self.assertFalse(img.is_contiguous())
      start = threading.Barrier(num_threads)
      views = []

      def get_view():
        start.wait()
        views.append(img.numpy_view())  # pylint: disable=cell-var-from-loop

      threads = [threading.Thread(target=get_view) for _ in range(num_threads)]
      for thread in threads:
        thread.start()
      for thread in threads:
        thread.join()

      self.assertLen(views, num_threads)
      for view in views:
        np.testing.assert_array_equal(view, pixel_data)

  def test_failed_creation_does_not_raise_when_garbage_collected(self):
    unraisable = []
    with mock.patch.object(sys, 'unraisablehook', unraisable.append):
      with self.assertRaisesRegex(ValueError, 'Unsupported number of dim'):
        image.Image(image.ImageFormat.SRGB, np.zeros(5, dtype=np.uint8))
      gc.collect()
    self.assertEmpty(unraisable)


if __name__ == '__main__':
  absltest.main()
