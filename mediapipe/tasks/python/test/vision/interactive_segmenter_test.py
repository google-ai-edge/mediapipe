# Copyright 2023 The MediaPipe Authors.
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
"""Tests for interactive segmenter."""

import enum
import os

from absl.testing import absltest
from absl.testing import parameterized
import cv2
import numpy as np

from mediapipe.tasks.python.components.containers import keypoint as keypoint_module
from mediapipe.tasks.python.components.containers import rect as rect_module
from mediapipe.tasks.python.core import base_options as base_options_module
from mediapipe.tasks.python.test import test_utils
from mediapipe.tasks.python.vision import interactive_segmenter
from mediapipe.tasks.python.vision.core import image as image_module
from mediapipe.tasks.python.vision.core import image_processing_options as image_processing_options_module

InteractiveSegmenterResult = interactive_segmenter.InteractiveSegmenterResult
_BaseOptions = base_options_module.BaseOptions
_Image = image_module.Image
_ImageFormat = image_module.ImageFormat
_NormalizedKeypoint = keypoint_module.NormalizedKeypoint
_RectF = rect_module.RectF
_InteractiveSegmenter = interactive_segmenter.InteractiveSegmenter
_InteractiveSegmenterOptions = interactive_segmenter.InteractiveSegmenterOptions
_RegionOfInterest = interactive_segmenter.RegionOfInterest
_Format = interactive_segmenter.RegionOfInterest.Format
_ImageProcessingOptions = image_processing_options_module.ImageProcessingOptions

_MODEL_FILE = 'ptm_512_hdt_ptm_woid.tflite'
_CATS_AND_DOGS = 'cats_and_dogs.jpg'
_CATS_AND_DOGS_MASK_DOG_1 = 'cats_and_dogs_mask_dog1.png'
_CATS_AND_DOGS_MASK_DOG_2 = 'cats_and_dogs_mask_dog2.png'
_MASK_MAGNIFICATION_FACTOR = 255
_MASK_SIMILARITY_THRESHOLD = 0.96
_TEST_DATA_DIR = 'mediapipe/tasks/testdata/vision'


def _calculate_soft_iou(m1, m2):
  intersection_sum = np.sum(m1 * m2)
  union_sum = np.sum(m1 * m1) + np.sum(m2 * m2) - intersection_sum

  if union_sum > 0:
    return intersection_sum / union_sum
  else:
    return 0


def _similar_to_float_mask(
    actual_mask: _Image, expected_mask: _Image, similarity_threshold: float
):
  actual_mask = actual_mask.numpy_view()
  expected_mask = expected_mask.numpy_view() / 255.0

  return (
      actual_mask.shape == expected_mask.shape
      and _calculate_soft_iou(actual_mask, expected_mask) > similarity_threshold
  )


def _similar_to_uint8_mask(
    actual_mask: _Image, expected_mask: _Image, similarity_threshold: float
):
  actual_mask_pixels = actual_mask.numpy_view().flatten()
  expected_mask_pixels = expected_mask.numpy_view().flatten()

  consistent_pixels = 0
  num_pixels = len(expected_mask_pixels)

  for index in range(num_pixels):
    consistent_pixels += (
        actual_mask_pixels[index] * _MASK_MAGNIFICATION_FACTOR
        == expected_mask_pixels[index]
    )

  return consistent_pixels / num_pixels >= similarity_threshold


class ModelFileType(enum.Enum):
  FILE_CONTENT = 1
  FILE_NAME = 2


class InteractiveSegmenterTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    # Load the test input image.
    self.test_image = _Image.create_from_file(
        test_utils.get_test_data_path(
            os.path.join(_TEST_DATA_DIR, _CATS_AND_DOGS)
        )
    )
    # Loads ground truth segmentation file.
    self.test_seg_image = self._load_segmentation_mask(
        _CATS_AND_DOGS_MASK_DOG_1
    )
    self.model_path = test_utils.get_test_data_path(
        os.path.join(_TEST_DATA_DIR, _MODEL_FILE)
    )

  def _load_segmentation_mask(self, file_path: str):
    # Loads ground truth segmentation file.
    gt_segmentation_data = cv2.imread(
        test_utils.get_test_data_path(os.path.join(_TEST_DATA_DIR, file_path)),
        cv2.IMREAD_GRAYSCALE,
    )
    return _Image(_ImageFormat.GRAY8, gt_segmentation_data)

  def test_create_from_file_succeeds_with_valid_model_path(self):
    # Creates with default option and valid model file successfully.
    with _InteractiveSegmenter.create_from_model_path(
        self.model_path
    ) as segmenter:
      self.assertIsInstance(segmenter, _InteractiveSegmenter)

  def test_create_from_options_succeeds_with_valid_model_path(self):
    # Creates with options containing model file successfully.
    base_options = _BaseOptions(model_asset_path=self.model_path)
    options = _InteractiveSegmenterOptions(base_options=base_options)
    with _InteractiveSegmenter.create_from_options(options) as segmenter:
      self.assertIsInstance(segmenter, _InteractiveSegmenter)

  def test_create_from_options_fails_with_invalid_model_path(self):
    with self.assertRaisesRegex(
        FileNotFoundError,
        'Unable to open file at /path/to/invalid/model.tflite',
    ):
      base_options = _BaseOptions(
          model_asset_path='/path/to/invalid/model.tflite'
      )
      options = _InteractiveSegmenterOptions(base_options=base_options)
      segmenter = _InteractiveSegmenter.create_from_options(options)
      segmenter.close()

  def test_create_from_options_succeeds_with_valid_model_content(self):
    # Creates with options containing model content successfully.
    with open(self.model_path, 'rb') as f:
      base_options = _BaseOptions(model_asset_buffer=f.read())
      options = _InteractiveSegmenterOptions(base_options=base_options)
      segmenter = _InteractiveSegmenter.create_from_options(options)
      self.assertIsInstance(segmenter, _InteractiveSegmenter)
      segmenter.close()

  @parameterized.parameters(
      (
          ModelFileType.FILE_NAME,
          _RegionOfInterest.Format.KEYPOINT,
          _NormalizedKeypoint(0.44, 0.7),
          _CATS_AND_DOGS_MASK_DOG_1,
          0.84,
      ),
      (
          ModelFileType.FILE_CONTENT,
          _RegionOfInterest.Format.KEYPOINT,
          _NormalizedKeypoint(0.44, 0.7),
          _CATS_AND_DOGS_MASK_DOG_1,
          0.84,
      ),
      (
          ModelFileType.FILE_NAME,
          _RegionOfInterest.Format.KEYPOINT,
          _NormalizedKeypoint(0.66, 0.66),
          _CATS_AND_DOGS_MASK_DOG_2,
          _MASK_SIMILARITY_THRESHOLD,
      ),
      (
          ModelFileType.FILE_CONTENT,
          _RegionOfInterest.Format.KEYPOINT,
          _NormalizedKeypoint(0.66, 0.66),
          _CATS_AND_DOGS_MASK_DOG_2,
          _MASK_SIMILARITY_THRESHOLD,
      ),
  )
  def test_segment_succeeds_with_category_mask(
      self,
      model_file_type,
      roi_format,
      keypoint,
      output_mask,
      similarity_threshold,
  ):
    # Creates segmenter.
    if model_file_type is ModelFileType.FILE_NAME:
      base_options = _BaseOptions(model_asset_path=self.model_path)
    elif model_file_type is ModelFileType.FILE_CONTENT:
      with open(self.model_path, 'rb') as f:
        model_content = f.read()
      base_options = _BaseOptions(model_asset_buffer=model_content)
    else:
      # Should never happen
      raise ValueError('model_file_type is invalid.')

    options = _InteractiveSegmenterOptions(
        base_options=base_options,
        output_category_mask=True,
        output_confidence_masks=False,
    )
    segmenter = _InteractiveSegmenter.create_from_options(options)

    # Performs image segmentation on the input.
    roi = _RegionOfInterest(format=roi_format, keypoint=keypoint)
    segmentation_result = segmenter.segment(self.test_image, roi)
    category_mask = segmentation_result.category_mask
    assert category_mask is not None, 'Category mask was None'
    result_pixels = category_mask.numpy_view().flatten()

    # Check if data type of `category_mask` is correct.
    self.assertEqual(result_pixels.dtype, np.uint8)

    # Loads ground truth segmentation file.
    test_seg_image = self._load_segmentation_mask(output_mask)

    self.assertTrue(
        _similar_to_uint8_mask(
            category_mask, test_seg_image, similarity_threshold
        ),
        (
            'Number of pixels in the candidate mask differing from that of the'
            f' ground truth mask exceeds {similarity_threshold}.'
        ),
    )

    # Closes the segmenter explicitly when the segmenter is not used in
    # a context.
    segmenter.close()

  @parameterized.parameters(
      (
          _RegionOfInterest.Format.KEYPOINT,
          _NormalizedKeypoint(0.44, 0.7),
          _CATS_AND_DOGS_MASK_DOG_1,
          0.84,
      ),
      (
          _RegionOfInterest.Format.KEYPOINT,
          _NormalizedKeypoint(0.66, 0.66),
          _CATS_AND_DOGS_MASK_DOG_2,
          0.84,
      ),
  )
  def test_segment_succeeds_with_confidence_mask(
      self, roi_format, keypoint, output_mask, similarity_threshold
  ):
    # Creates segmenter.
    base_options = _BaseOptions(model_asset_path=self.model_path)
    roi = _RegionOfInterest(format=roi_format, keypoint=keypoint)

    # Run segmentation on the model in CONFIDENCE_MASK mode.
    options = _InteractiveSegmenterOptions(
        base_options=base_options,
        output_category_mask=False,
        output_confidence_masks=True,
    )

    with _InteractiveSegmenter.create_from_options(options) as segmenter:
      # Perform segmentation
      segmentation_result = segmenter.segment(self.test_image, roi)
      confidence_masks = segmentation_result.confidence_masks

      # Check if confidence mask shape is correct.
      self.assertLen(
          confidence_masks,
          2,
          'Number of confidence masks must match with number of categories.',
      )

      # Loads ground truth segmentation file.
      expected_mask = self._load_segmentation_mask(output_mask)

      self.assertTrue(
          _similar_to_float_mask(
              confidence_masks[1], expected_mask, similarity_threshold
          )
      )

  def test_segment_succeeds_with_rotation(self):
    # Creates segmenter.
    base_options = _BaseOptions(model_asset_path=self.model_path)
    roi = _RegionOfInterest(
        format=_RegionOfInterest.Format.KEYPOINT,
        keypoint=_NormalizedKeypoint(0.66, 0.66),
    )

    # Run segmentation on the model in CONFIDENCE_MASK mode.
    options = _InteractiveSegmenterOptions(
        base_options=base_options,
        output_category_mask=False,
        output_confidence_masks=True,
    )

    with _InteractiveSegmenter.create_from_options(options) as segmenter:
      # Perform segmentation
      image_processing_options = _ImageProcessingOptions(rotation_degrees=-90)
      segmentation_result = segmenter.segment(
          self.test_image, roi, image_processing_options
      )
      confidence_masks = segmentation_result.confidence_masks

      # Check if confidence mask shape is correct.
      self.assertLen(
          confidence_masks,
          2,
          'Number of confidence masks must match with number of categories.',
      )

  def test_segment_fails_with_roi_in_image_processing_options(self):
    # Creates segmenter.
    base_options = _BaseOptions(model_asset_path=self.model_path)
    roi = _RegionOfInterest(
        format=_RegionOfInterest.Format.KEYPOINT,
        keypoint=_NormalizedKeypoint(0.66, 0.66),
    )

    # Run segmentation on the model in CONFIDENCE_MASK mode.
    options = _InteractiveSegmenterOptions(
        base_options=base_options,
        output_category_mask=False,
        output_confidence_masks=True,
    )

    with self.assertRaises(ValueError):
      with _InteractiveSegmenter.create_from_options(options) as segmenter:
        # Perform segmentation
        image_processing_options = _ImageProcessingOptions(
            _RectF(left=0.1, top=0.0, right=0.9, bottom=1.0)
        )
        segmenter.segment(self.test_image, roi, image_processing_options)


if __name__ == '__main__':
  absltest.main()
