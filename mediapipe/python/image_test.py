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
import random
import sys
from absl.testing import absltest
import cv2
import mediapipe as mp
import numpy as np
import PIL.Image


# TODO: Add unit tests specifically for memory management.
class ImageTest(absltest.TestCase):

  def test_create_image_from_gray_cv_mat(self):
    w, h = random.randrange(3, 100), random.randrange(3, 100)
    mat = cv2.cvtColor(
        np.random.randint(2**8 - 1, size=(h, w, 3), dtype=np.uint8),
        cv2.COLOR_RGB2GRAY)
    mat[2, 2] = 42
    image = mp.Image(image_format=mp.ImageFormat.GRAY8, data=mat)
    self.assertTrue(np.array_equal(mat, image.numpy_view()))
    with self.assertRaisesRegex(IndexError, 'index dimension mismatch'):
      print(image[w, h, 1])
    with self.assertRaisesRegex(IndexError, 'out of bounds'):
      print(image[w, h])
    self.assertEqual(42, image[2, 2])

  def test_create_image_from_rgb_cv_mat(self):
    w, h, channels = random.randrange(3, 100), random.randrange(3, 100), 3
    mat = cv2.cvtColor(
        np.random.randint(2**8 - 1, size=(h, w, channels), dtype=np.uint8),
        cv2.COLOR_RGB2BGR)
    mat[2, 2, 1] = 42
    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=mat)
    self.assertTrue(np.array_equal(mat, image.numpy_view()))
    with self.assertRaisesRegex(IndexError, 'out of bounds'):
      print(image[w, h, channels])
    self.assertEqual(42, image[2, 2, 1])

  def test_create_image_from_rgb48_cv_mat(self):
    w, h, channels = random.randrange(3, 100), random.randrange(3, 100), 3
    mat = cv2.cvtColor(
        np.random.randint(2**16 - 1, size=(h, w, channels), dtype=np.uint16),
        cv2.COLOR_RGB2BGR)
    mat[2, 2, 1] = 42
    image = mp.Image(image_format=mp.ImageFormat.SRGB48, data=mat)
    self.assertTrue(np.array_equal(mat, image.numpy_view()))
    with self.assertRaisesRegex(IndexError, 'out of bounds'):
      print(image[w, h, channels])
    self.assertEqual(42, image[2, 2, 1])

  def test_create_image_from_gray_pil_image(self):
    w, h = random.randrange(3, 100), random.randrange(3, 100)
    img = PIL.Image.fromarray(
        np.random.randint(2**8 - 1, size=(h, w), dtype=np.uint8), 'L')
    image = mp.Image(image_format=mp.ImageFormat.GRAY8, data=np.asarray(img))
    self.assertTrue(np.array_equal(np.asarray(img), image.numpy_view()))
    with self.assertRaisesRegex(IndexError, 'index dimension mismatch'):
      print(image[w, h, 1])
    with self.assertRaisesRegex(IndexError, 'out of bounds'):
      print(image[w, h])

  def test_create_image_from_rgb_pil_image(self):
    w, h, channels = random.randrange(3, 100), random.randrange(3, 100), 3
    img = PIL.Image.fromarray(
        np.random.randint(2**8 - 1, size=(h, w, channels), dtype=np.uint8),
        'RGB')
    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.asarray(img))
    self.assertTrue(np.array_equal(np.asarray(img), image.numpy_view()))
    with self.assertRaisesRegex(IndexError, 'out of bounds'):
      print(image[w, h, channels])

  def test_create_image_from_rgba64_pil_image(self):
    w, h, channels = random.randrange(3, 100), random.randrange(3, 100), 4
    img = PIL.Image.fromarray(
        np.random.randint(2**16 - 1, size=(h, w, channels), dtype=np.uint16),
        'RGBA')
    image = mp.Image(
        image_format=mp.ImageFormat.SRGBA64,
        data=np.asarray(img).astype(np.uint16))
    self.assertTrue(np.array_equal(np.asarray(img), image.numpy_view()))
    with self.assertRaisesRegex(IndexError, 'out of bounds'):
      print(image[1000, 1000, 1000])

  def test_image_numby_view(self):
    w, h, channels = random.randrange(3, 100), random.randrange(3, 100), 3
    mat = cv2.cvtColor(
        np.random.randint(2**8 - 1, size=(h, w, channels), dtype=np.uint8),
        cv2.COLOR_RGB2BGR)
    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=mat)
    output_ndarray = image.numpy_view()
    self.assertTrue(np.array_equal(mat, image.numpy_view()))
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
    image = mp.Image(
        image_format=mp.ImageFormat.GRAY8,
        data=np.ascontiguousarray(mat[offset:-offset, offset:-offset]))
    self.assertTrue(
        np.array_equal(mat[offset:-offset, offset:-offset], image.numpy_view()))

  def test_cropped_rgb_image(self):
    w, h = random.randrange(20, 100), random.randrange(20, 100)
    channels, offset = 3, 10
    mat = cv2.cvtColor(
        np.random.randint(2**8 - 1, size=(h, w, channels), dtype=np.uint8),
        cv2.COLOR_RGB2BGR)
    image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=np.ascontiguousarray(mat[offset:-offset, offset:-offset, :]))
    self.assertTrue(
        np.array_equal(mat[offset:-offset, offset:-offset, :],
                       image.numpy_view()))

  # For image frames that store contiguous data, the output of numpy_view()
  # points to the pixel data of the original image frame object. The life cycle
  # of the data array should tie to the image frame object.
  def test_image_numpy_view_with_contiguous_data(self):
    w, h = 640, 480
    mat = np.random.randint(2**8 - 1, size=(h, w, 3), dtype=np.uint8)
    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=mat)
    self.assertTrue(image.is_contiguous())
    initial_ref_count = sys.getrefcount(image)
    self.assertTrue(np.array_equal(mat, image.numpy_view()))
    # Get 2 data array objects and verify that the image frame's ref count is
    # increased by 2.
    np_view = image.numpy_view()
    self.assertEqual(sys.getrefcount(image), initial_ref_count + 1)
    np_view2 = image.numpy_view()
    self.assertEqual(sys.getrefcount(image), initial_ref_count + 2)
    del np_view
    del np_view2
    gc.collect()
    # After the two data array objects getting destroyed, the current ref count
    # should euqal to the initial ref count.
    self.assertEqual(sys.getrefcount(image), initial_ref_count)

  # For image frames that store non contiguous data, the output of numpy_view()
  # stores a copy of the pixel data of the image frame object. The life cycle of
  # the data array doesn't tie to the image frame object.
  def test_image_numpy_view_with_non_contiguous_data(self):
    w, h = 641, 481
    mat = np.random.randint(2**8 - 1, size=(h, w, 3), dtype=np.uint8)
    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=mat)
    self.assertFalse(image.is_contiguous())
    initial_ref_count = sys.getrefcount(image)
    self.assertTrue(np.array_equal(mat, image.numpy_view()))
    np_view = image.numpy_view()
    self.assertEqual(sys.getrefcount(image), initial_ref_count)
    del np_view
    gc.collect()
    self.assertEqual(sys.getrefcount(image), initial_ref_count)


if __name__ == '__main__':
  absltest.main()
