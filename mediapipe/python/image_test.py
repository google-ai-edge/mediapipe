# Copyright 2021 The MediaPipe Authors.
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

"""Tests for mediapipe.python._framework_bindings.image."""

import gc
import os
import random
import sys

from absl.testing import absltest
import cv2
import numpy as np
import PIL.Image

# resources dependency
from mediapipe.python._framework_bindings import image
from mediapipe.python._framework_bindings import image_frame

TEST_IMAGE_PATH = 'mediapipe/python/solutions/testdata'

Image = image.Image
ImageFormat = image_frame.ImageFormat


# TODO: Add unit tests specifically for memory management.
class ImageTest(absltest.TestCase):

  def test_create_image_from_gray_cv_mat(self):
    w, h = random.randrange(3, 100), random.randrange(3, 100)
    mat = cv2.cvtColor(
        np.random.randint(2**8 - 1, size=(h, w, 3), dtype=np.uint8),
        cv2.COLOR_RGB2GRAY)
    mat[2, 2] = 42
    gray8_image = Image(image_format=ImageFormat.GRAY8, data=mat)
    self.assertTrue(np.array_equal(mat, gray8_image.numpy_view()))
    with self.assertRaisesRegex(IndexError, 'index dimension mismatch'):
      print(gray8_image[w, h, 1])
    with self.assertRaisesRegex(IndexError, 'out of bounds'):
      print(gray8_image[w, h])
    self.assertEqual(42, gray8_image[2, 2])

  def test_create_image_from_rgb_cv_mat(self):
    w, h, channels = random.randrange(3, 100), random.randrange(3, 100), 3
    mat = cv2.cvtColor(
        np.random.randint(2**8 - 1, size=(h, w, channels), dtype=np.uint8),
        cv2.COLOR_RGB2BGR)
    mat[2, 2, 1] = 42
    rgb_image = Image(image_format=ImageFormat.SRGB, data=mat)
    self.assertTrue(np.array_equal(mat, rgb_image.numpy_view()))
    with self.assertRaisesRegex(IndexError, 'out of bounds'):
      print(rgb_image[w, h, channels])
    self.assertEqual(42, rgb_image[2, 2, 1])

  def test_create_image_from_rgb48_cv_mat(self):
    w, h, channels = random.randrange(3, 100), random.randrange(3, 100), 3
    mat = cv2.cvtColor(
        np.random.randint(2**16 - 1, size=(h, w, channels), dtype=np.uint16),
        cv2.COLOR_RGB2BGR)
    mat[2, 2, 1] = 42
    rgb48_image = Image(image_format=ImageFormat.SRGB48, data=mat)
    self.assertTrue(np.array_equal(mat, rgb48_image.numpy_view()))
    with self.assertRaisesRegex(IndexError, 'out of bounds'):
      print(rgb48_image[w, h, channels])
    self.assertEqual(42, rgb48_image[2, 2, 1])

  def test_create_image_from_gray_pil_image(self):
    w, h = random.randrange(3, 100), random.randrange(3, 100)
    img = PIL.Image.fromarray(
        np.random.randint(2**8 - 1, size=(h, w), dtype=np.uint8), 'L')
    gray8_image = Image(image_format=ImageFormat.GRAY8, data=np.asarray(img))
    self.assertTrue(np.array_equal(np.asarray(img), gray8_image.numpy_view()))
    with self.assertRaisesRegex(IndexError, 'index dimension mismatch'):
      print(gray8_image[w, h, 1])
    with self.assertRaisesRegex(IndexError, 'out of bounds'):
      print(gray8_image[w, h])

  def test_create_image_from_rgb_pil_image(self):
    w, h, channels = random.randrange(3, 100), random.randrange(3, 100), 3
    img = PIL.Image.fromarray(
        np.random.randint(2**8 - 1, size=(h, w, channels), dtype=np.uint8),
        'RGB')
    rgb_image = Image(image_format=ImageFormat.SRGB, data=np.asarray(img))
    self.assertTrue(np.array_equal(np.asarray(img), rgb_image.numpy_view()))
    with self.assertRaisesRegex(IndexError, 'out of bounds'):
      print(rgb_image[w, h, channels])

  def test_create_image_from_rgba64_pil_image(self):
    w, h, channels = random.randrange(3, 100), random.randrange(3, 100), 4
    img = PIL.Image.fromarray(
        np.random.randint(2**16 - 1, size=(h, w, channels), dtype=np.uint16),
        'RGBA')
    rgba_image = Image(
        image_format=ImageFormat.SRGBA64,
        data=np.asarray(img).astype(np.uint16))
    self.assertTrue(np.array_equal(np.asarray(img), rgba_image.numpy_view()))
    with self.assertRaisesRegex(IndexError, 'out of bounds'):
      print(rgba_image[1000, 1000, 1000])

  def test_image_numby_view(self):
    w, h, channels = random.randrange(3, 100), random.randrange(3, 100), 3
    mat = cv2.cvtColor(
        np.random.randint(2**8 - 1, size=(h, w, channels), dtype=np.uint8),
        cv2.COLOR_RGB2BGR)
    rgb_image = Image(image_format=ImageFormat.SRGB, data=mat)
    output_ndarray = rgb_image.numpy_view()
    self.assertTrue(np.array_equal(mat, rgb_image.numpy_view()))
    # The output of numpy_view() is a reference to the internal data and it's
    # unwritable after creation.
    with self.assertRaisesRegex(ValueError,
                                'assignment destination is read-only'):
      output_ndarray[0, 0, 0] = 0
    copied_ndarray = np.copy(output_ndarray)
    copied_ndarray[0, 0, 0] = 0

  def test_cropped_gray8_image(self):
    w, h = random.randrange(20, 100), random.randrange(20, 100)
    channels, offset = 3, 10
    mat = cv2.cvtColor(
        np.random.randint(2**8 - 1, size=(h, w, channels), dtype=np.uint8),
        cv2.COLOR_RGB2GRAY)
    gray8_image = Image(
        image_format=ImageFormat.GRAY8,
        data=np.ascontiguousarray(mat[offset:-offset, offset:-offset]))
    self.assertTrue(
        np.array_equal(mat[offset:-offset, offset:-offset],
                       gray8_image.numpy_view()))

  def test_cropped_rgb_image(self):
    w, h = random.randrange(20, 100), random.randrange(20, 100)
    channels, offset = 3, 10
    mat = cv2.cvtColor(
        np.random.randint(2**8 - 1, size=(h, w, channels), dtype=np.uint8),
        cv2.COLOR_RGB2BGR)
    rgb_image = Image(
        image_format=ImageFormat.SRGB,
        data=np.ascontiguousarray(mat[offset:-offset, offset:-offset, :]))
    self.assertTrue(
        np.array_equal(mat[offset:-offset, offset:-offset, :],
                       rgb_image.numpy_view()))

  # For image frames that store contiguous data, the output of numpy_view()
  # points to the pixel data of the original image frame object. The life cycle
  # of the data array should tie to the image frame object.
  def test_image_numpy_view_with_contiguous_data(self):
    w, h = 640, 480
    mat = np.random.randint(2**8 - 1, size=(h, w, 3), dtype=np.uint8)
    rgb_image = Image(image_format=ImageFormat.SRGB, data=mat)
    self.assertTrue(rgb_image.is_contiguous())
    initial_ref_count = sys.getrefcount(rgb_image)
    self.assertTrue(np.array_equal(mat, rgb_image.numpy_view()))
    # Get 2 data array objects and verify that the image frame's ref count is
    # increased by 2.
    np_view = rgb_image.numpy_view()
    self.assertEqual(sys.getrefcount(rgb_image), initial_ref_count + 1)
    np_view2 = rgb_image.numpy_view()
    self.assertEqual(sys.getrefcount(rgb_image), initial_ref_count + 2)
    del np_view
    del np_view2
    gc.collect()
    # After the two data array objects getting destroyed, the current ref count
    # should euqal to the initial ref count.
    self.assertEqual(sys.getrefcount(rgb_image), initial_ref_count)

  # For image frames that store non contiguous data, the output of numpy_view()
  # stores a copy of the pixel data of the image frame object. The life cycle of
  # the data array doesn't tie to the image frame object.
  def test_image_numpy_view_with_non_contiguous_data(self):
    w, h = 641, 481
    mat = np.random.randint(2**8 - 1, size=(h, w, 3), dtype=np.uint8)
    rgb_image = Image(image_format=ImageFormat.SRGB, data=mat)
    self.assertFalse(rgb_image.is_contiguous())
    initial_ref_count = sys.getrefcount(rgb_image)
    self.assertTrue(np.array_equal(mat, rgb_image.numpy_view()))
    np_view = rgb_image.numpy_view()
    self.assertEqual(sys.getrefcount(rgb_image), initial_ref_count)
    del np_view
    gc.collect()
    self.assertEqual(sys.getrefcount(rgb_image), initial_ref_count)

  def test_image_create_from_cvmat(self):
    image_path = os.path.join(os.path.dirname(__file__),
                              'solutions/testdata/hands.jpg')
    mat = cv2.imread(image_path).astype(np.uint8)
    mat = cv2.cvtColor(mat, cv2.COLOR_BGR2RGB)
    rgb_image = Image(image_format=ImageFormat.SRGB, data=mat)
    self.assertEqual(rgb_image.width, 720)
    self.assertEqual(rgb_image.height, 382)
    self.assertEqual(rgb_image.channels, 3)
    self.assertEqual(rgb_image.image_format, ImageFormat.SRGB)
    self.assertTrue(np.array_equal(mat, rgb_image.numpy_view()))

  def test_image_create_from_file(self):
    image_path = os.path.join(os.path.dirname(__file__),
                              'solutions/testdata/hands.jpg')
    loaded_image = Image.create_from_file(image_path)
    self.assertEqual(loaded_image.width, 720)
    self.assertEqual(loaded_image.height, 382)
    # On Mac w/ GPU support, images use 4 channels (SRGBA). Otherwise, all
    # images use 3 channels (SRGB).
    self.assertIn(loaded_image.channels, [3, 4])
    self.assertIn(
        loaded_image.image_format, [ImageFormat.SRGB, ImageFormat.SRGBA]
    )

if __name__ == '__main__':
  absltest.main()
