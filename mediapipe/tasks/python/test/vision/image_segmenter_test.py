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
"""Tests for image segmenter."""

import enum
import os
import threading
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import cv2
import numpy as np

from mediapipe.tasks.python.core import base_options as base_options_module
from mediapipe.tasks.python.test import test_utils
from mediapipe.tasks.python.vision import image_segmenter
from mediapipe.tasks.python.vision.core import image as image_module
from mediapipe.tasks.python.vision.core import vision_task_running_mode

ImageSegmenterResult = image_segmenter.ImageSegmenterResult
_BaseOptions = base_options_module.BaseOptions
_Image = image_module.Image
_ImageFormat = image_module.ImageFormat
_ImageSegmenter = image_segmenter.ImageSegmenter
_ImageSegmenterOptions = image_segmenter.ImageSegmenterOptions
_RUNNING_MODE = vision_task_running_mode.VisionTaskRunningMode

_MODEL_FILE = 'deeplabv3.tflite'
_IMAGE_FILE = 'segmentation_input_rotation0.jpg'
_SEGMENTATION_FILE = 'segmentation_golden_rotation0.png'
_CAT_IMAGE = 'cat.jpg'
_CAT_MASK = 'cat_mask.jpg'
_MASK_MAGNIFICATION_FACTOR = 10
_MASK_SIMILARITY_THRESHOLD = 0.98
_TEST_DATA_DIR = 'mediapipe/tasks/testdata/vision'
_EXPECTED_LABELS = [
    'background',
    'aeroplane',
    'bicycle',
    'bird',
    'boat',
    'bottle',
    'bus',
    'car',
    'cat',
    'chair',
    'cow',
    'dining table',
    'dog',
    'horse',
    'motorbike',
    'person',
    'potted plant',
    'sheep',
    'sofa',
    'train',
    'tv',
]


def _calculate_soft_iou(m1, m2):
  intersection_sum = np.sum(m1 * m2)
  union_sum = np.sum(m1 * m1) + np.sum(m2 * m2) - intersection_sum

  if union_sum > 0:
    return intersection_sum / union_sum
  else:
    return 0


def _similar_to_float_mask(actual_mask, expected_mask, similarity_threshold):
  if not actual_mask:
    raise ValueError('Result mask used for comparison was None')
  actual_mask = actual_mask.numpy_view()
  expected_mask = expected_mask.numpy_view() / 255.0

  return (
      actual_mask.shape == expected_mask.shape
      and _calculate_soft_iou(actual_mask, expected_mask) > similarity_threshold
  )


def _similar_to_uint8_mask(actual_mask, expected_mask):
  if not actual_mask:
    raise ValueError('Result mask used for comparison was None')
  actual_mask_pixels = actual_mask.numpy_view().flatten()
  expected_mask_pixels = expected_mask.numpy_view().flatten()

  consistent_pixels = 0
  num_pixels = len(expected_mask_pixels)

  for index in range(num_pixels):
    consistent_pixels += (
        actual_mask_pixels[index] * _MASK_MAGNIFICATION_FACTOR
        == expected_mask_pixels[index]
    )

  return consistent_pixels / num_pixels >= _MASK_SIMILARITY_THRESHOLD


class ModelFileType(enum.Enum):
  FILE_CONTENT = 1
  FILE_NAME = 2


class ImageSegmenterTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    # Load the test input image.
    self.test_image = _Image.create_from_file(
        test_utils.get_test_data_path(os.path.join(_TEST_DATA_DIR, _IMAGE_FILE))
    )
    # Loads ground truth segmentation file.
    gt_segmentation_data = cv2.imread(
        test_utils.get_test_data_path(
            os.path.join(_TEST_DATA_DIR, _SEGMENTATION_FILE)
        ),
        cv2.IMREAD_GRAYSCALE,
    )
    self.test_seg_image = _Image(_ImageFormat.GRAY8, gt_segmentation_data)
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
    with _ImageSegmenter.create_from_model_path(self.model_path) as segmenter:
      self.assertIsInstance(segmenter, _ImageSegmenter)

  def test_create_from_options_succeeds_with_valid_model_path(self):
    # Creates with options containing model file successfully.
    base_options = _BaseOptions(model_asset_path=self.model_path)
    options = _ImageSegmenterOptions(base_options=base_options)
    with _ImageSegmenter.create_from_options(options) as segmenter:
      self.assertIsInstance(segmenter, _ImageSegmenter)

  def test_create_from_options_fails_with_invalid_model_path(self):
    with self.assertRaisesRegex(
        FileNotFoundError,
        'Unable to open file at /path/to/invalid/model.tflite',
    ):
      base_options = _BaseOptions(
          model_asset_path='/path/to/invalid/model.tflite'
      )
      options = _ImageSegmenterOptions(base_options=base_options)
      segmenter = _ImageSegmenter.create_from_options(options)
      segmenter.close()

  def test_create_from_options_succeeds_with_valid_model_content(self):
    # Creates with options containing model content successfully.
    with open(self.model_path, 'rb') as f:
      base_options = _BaseOptions(model_asset_buffer=f.read())
      options = _ImageSegmenterOptions(base_options=base_options)
      segmenter = _ImageSegmenter.create_from_options(options)
      self.assertIsInstance(segmenter, _ImageSegmenter)
      segmenter.close()

  @parameterized.parameters(
      (ModelFileType.FILE_NAME,), (ModelFileType.FILE_CONTENT,)
  )
  def test_segment_succeeds_with_category_mask(self, model_file_type):
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

    options = _ImageSegmenterOptions(
        base_options=base_options,
        output_category_mask=True,
        output_confidence_masks=False,
    )
    segmenter = _ImageSegmenter.create_from_options(options)

    # Performs image segmentation on the input.
    segmentation_result = segmenter.segment(self.test_image)
    category_mask = segmentation_result.category_mask
    if not category_mask:
      raise ValueError('Result category mask was None')
    result_pixels = category_mask.numpy_view().flatten()

    # Check if data type of `category_mask` is correct.
    self.assertEqual(result_pixels.dtype, np.uint8)

    self.assertTrue(
        _similar_to_uint8_mask(category_mask, self.test_seg_image),
        (
            'Number of pixels in the candidate mask differing from that of the'
            f' ground truth mask exceeds {_MASK_SIMILARITY_THRESHOLD}.'
        ),
    )

    # Closes the segmenter explicitly when the segmenter is not used in
    # a context.
    segmenter.close()

  def test_segment_succeeds_with_confidence_mask(self):
    # Creates segmenter.
    base_options = _BaseOptions(model_asset_path=self.model_path)

    # Load the cat image.
    test_image = _Image.create_from_file(
        test_utils.get_test_data_path(os.path.join(_TEST_DATA_DIR, _CAT_IMAGE))
    )

    # Run segmentation on the model in CONFIDENCE_MASK mode.
    options = _ImageSegmenterOptions(
        base_options=base_options,
        output_category_mask=False,
        output_confidence_masks=True,
    )

    with _ImageSegmenter.create_from_options(options) as segmenter:
      segmentation_result = segmenter.segment(test_image)
      confidence_masks = segmentation_result.confidence_masks

      # Check if confidence mask shape is correct.
      self.assertLen(
          confidence_masks,
          21,
          'Number of confidence masks must match with number of categories.',
      )

      # Loads ground truth segmentation file.
      expected_mask = self._load_segmentation_mask(_CAT_MASK)

      self.assertTrue(
          _similar_to_float_mask(
              confidence_masks[8], expected_mask, _MASK_SIMILARITY_THRESHOLD
          )
      )

  @parameterized.parameters((True, False), (False, True))
  def test_labels_succeeds(self, output_category_mask, output_confidence_masks):
    expected_labels = _EXPECTED_LABELS
    base_options = _BaseOptions(model_asset_path=self.model_path)
    options = _ImageSegmenterOptions(
        base_options=base_options,
        output_category_mask=output_category_mask,
        output_confidence_masks=output_confidence_masks,
    )
    with _ImageSegmenter.create_from_options(options) as segmenter:
      # Performs image segmentation on the input.
      actual_labels = segmenter.labels
      self.assertListEqual(actual_labels, expected_labels)

  def test_missing_result_callback(self):
    options = _ImageSegmenterOptions(
        base_options=_BaseOptions(model_asset_path=self.model_path),
        running_mode=_RUNNING_MODE.LIVE_STREAM,
    )
    with self.assertRaisesRegex(
        ValueError, r'result callback must be provided'
    ):
      with _ImageSegmenter.create_from_options(options) as unused_segmenter:
        pass

  @parameterized.parameters((_RUNNING_MODE.IMAGE), (_RUNNING_MODE.VIDEO))
  def test_illegal_result_callback(self, running_mode):
    options = _ImageSegmenterOptions(
        base_options=_BaseOptions(model_asset_path=self.model_path),
        running_mode=running_mode,
        result_callback=mock.MagicMock(),
    )
    with self.assertRaisesRegex(
        ValueError, r'result callback should not be provided'
    ):
      with _ImageSegmenter.create_from_options(options) as unused_segmenter:
        pass

  def test_calling_segment_for_video_in_image_mode(self):
    options = _ImageSegmenterOptions(
        base_options=_BaseOptions(model_asset_path=self.model_path),
        running_mode=_RUNNING_MODE.IMAGE,
    )
    with _ImageSegmenter.create_from_options(options) as segmenter:
      with self.assertRaises(ValueError):
        segmenter.segment_for_video(self.test_image, 0)

  def test_calling_segment_async_in_image_mode(self):
    options = _ImageSegmenterOptions(
        base_options=_BaseOptions(model_asset_path=self.model_path),
        running_mode=_RUNNING_MODE.IMAGE,
    )
    with _ImageSegmenter.create_from_options(options) as segmenter:
      with self.assertRaises(ValueError):
        segmenter.segment_async(self.test_image, 0)

  def test_calling_segment_in_video_mode(self):
    options = _ImageSegmenterOptions(
        base_options=_BaseOptions(model_asset_path=self.model_path),
        running_mode=_RUNNING_MODE.VIDEO,
    )
    with _ImageSegmenter.create_from_options(options) as segmenter:
      with self.assertRaises(ValueError):
        segmenter.segment(self.test_image)

  def test_calling_segment_async_in_video_mode(self):
    options = _ImageSegmenterOptions(
        base_options=_BaseOptions(model_asset_path=self.model_path),
        running_mode=_RUNNING_MODE.VIDEO,
    )
    with _ImageSegmenter.create_from_options(options) as segmenter:
      with self.assertRaises(ValueError):
        segmenter.segment_async(self.test_image, 0)

  def test_segment_for_video_with_out_of_order_timestamp(self):
    options = _ImageSegmenterOptions(
        base_options=_BaseOptions(model_asset_path=self.model_path),
        running_mode=_RUNNING_MODE.VIDEO,
    )
    with _ImageSegmenter.create_from_options(options) as segmenter:
      unused_result = segmenter.segment_for_video(self.test_image, 1)
      with self.assertRaises(ValueError):
        segmenter.segment_for_video(self.test_image, 0)

  def test_segment_for_video_in_category_mask_mode(self):
    options = _ImageSegmenterOptions(
        base_options=_BaseOptions(model_asset_path=self.model_path),
        output_category_mask=True,
        output_confidence_masks=False,
        running_mode=_RUNNING_MODE.VIDEO,
    )
    with _ImageSegmenter.create_from_options(options) as segmenter:
      for timestamp in range(0, 300, 30):
        segmentation_result = segmenter.segment_for_video(
            self.test_image, timestamp
        )
        category_mask = segmentation_result.category_mask
        self.assertTrue(
            _similar_to_uint8_mask(category_mask, self.test_seg_image),
            (
                'Number of pixels in the candidate mask differing from that of'
                f' the ground truth mask exceeds {_MASK_SIMILARITY_THRESHOLD}.'
            ),
        )

  def test_segment_for_video_in_confidence_mask_mode(self):
    # Load the cat image.
    test_image = _Image.create_from_file(
        test_utils.get_test_data_path(os.path.join(_TEST_DATA_DIR, _CAT_IMAGE))
    )

    options = _ImageSegmenterOptions(
        base_options=_BaseOptions(model_asset_path=self.model_path),
        running_mode=_RUNNING_MODE.VIDEO,
        output_category_mask=False,
        output_confidence_masks=True,
    )
    with _ImageSegmenter.create_from_options(options) as segmenter:
      for timestamp in range(0, 300, 30):
        segmentation_result = segmenter.segment_for_video(test_image, timestamp)
        confidence_masks = segmentation_result.confidence_masks

        # Check if confidence mask shape is correct.
        self.assertLen(
            confidence_masks,
            21,
            'Number of confidence masks must match with number of categories.',
        )

        # Loads ground truth segmentation file.
        expected_mask = self._load_segmentation_mask(_CAT_MASK)
        self.assertTrue(
            _similar_to_float_mask(
                confidence_masks[8], expected_mask, _MASK_SIMILARITY_THRESHOLD
            )
        )

  def test_calling_segment_in_live_stream_mode(self):
    options = _ImageSegmenterOptions(
        base_options=_BaseOptions(model_asset_path=self.model_path),
        running_mode=_RUNNING_MODE.LIVE_STREAM,
        result_callback=mock.MagicMock(),
    )
    with _ImageSegmenter.create_from_options(options) as segmenter:
      with self.assertRaises(ValueError):
        segmenter.segment(self.test_image)

  def test_calling_segment_for_video_in_live_stream_mode(self):
    options = _ImageSegmenterOptions(
        base_options=_BaseOptions(model_asset_path=self.model_path),
        running_mode=_RUNNING_MODE.LIVE_STREAM,
        result_callback=mock.MagicMock(),
    )
    with _ImageSegmenter.create_from_options(options) as segmenter:
      with self.assertRaises(ValueError):
        segmenter.segment_for_video(self.test_image, 0)

  def test_segment_async_calls_with_illegal_timestamp(self):
    options = _ImageSegmenterOptions(
        base_options=_BaseOptions(model_asset_path=self.model_path),
        running_mode=_RUNNING_MODE.LIVE_STREAM,
        result_callback=mock.MagicMock(),
    )
    with _ImageSegmenter.create_from_options(options) as segmenter:
      segmenter.segment_async(self.test_image, 100)
      with self.assertRaises(ValueError):
        segmenter.segment_async(self.test_image, 0)

  def test_segment_async_calls_in_category_mask_mode(self):
    observed_timestamp_ms = -1
    callback_event = threading.Event()
    callback_exception = None

    def check_result(
        result: ImageSegmenterResult, output_image: _Image, timestamp_ms: int
    ):
      nonlocal callback_exception, observed_timestamp_ms
      try:
        # Get the output category mask.
        category_mask = result.category_mask
        self.assertEqual(output_image.width, self.test_image.width)
        self.assertEqual(output_image.height, self.test_image.height)
        self.assertEqual(output_image.width, self.test_seg_image.width)
        self.assertEqual(output_image.height, self.test_seg_image.height)
        self.assertTrue(
            _similar_to_uint8_mask(category_mask, self.test_seg_image),
            (
                'Number of pixels in the candidate mask differing from that of'
                f' the ground truth mask exceeds {_MASK_SIMILARITY_THRESHOLD}.'
            ),
        )
        self.assertLess(observed_timestamp_ms, timestamp_ms)
        self.observed_timestamp_ms = timestamp_ms
      except AssertionError as e:
        callback_exception = e
      finally:
        callback_event.set()

    options = _ImageSegmenterOptions(
        base_options=_BaseOptions(model_asset_path=self.model_path),
        output_category_mask=True,
        output_confidence_masks=False,
        running_mode=_RUNNING_MODE.LIVE_STREAM,
        result_callback=check_result,
    )
    with _ImageSegmenter.create_from_options(options) as segmenter:
      for timestamp in range(0, 300, 30):
        segmenter.segment_async(self.test_image, timestamp)
        callback_event.wait(3)
        if callback_exception is not None:
          raise callback_exception
        callback_event.clear()

  def test_segment_async_calls_in_confidence_mask_mode(self):
    # Load the cat image.
    test_image = _Image.create_from_file(
        test_utils.get_test_data_path(os.path.join(_TEST_DATA_DIR, _CAT_IMAGE))
    )

    # Loads ground truth segmentation file.
    expected_mask = self._load_segmentation_mask(_CAT_MASK)
    observed_timestamp_ms = -1
    callback_event = threading.Event()
    callback_exception = None

    def check_result(
        result: ImageSegmenterResult, output_image: _Image, timestamp_ms: int
    ):
      nonlocal callback_exception, observed_timestamp_ms
      try:
        # Get the output category mask.
        confidence_masks = result.confidence_masks

        # Check if confidence mask shape is correct.
        self.assertLen(
            confidence_masks,
            21,
            'Number of confidence masks must match with number of categories.',
        )
        self.assertEqual(output_image.width, test_image.width)
        self.assertEqual(output_image.height, test_image.height)
        self.assertTrue(
            _similar_to_float_mask(
                confidence_masks[8], expected_mask, _MASK_SIMILARITY_THRESHOLD
            )
        )
        self.assertLess(observed_timestamp_ms, timestamp_ms)
        self.observed_timestamp_ms = timestamp_ms
      except AssertionError as e:
        callback_exception = e
      finally:
        callback_event.set()

    options = _ImageSegmenterOptions(
        base_options=_BaseOptions(model_asset_path=self.model_path),
        running_mode=_RUNNING_MODE.LIVE_STREAM,
        output_category_mask=False,
        output_confidence_masks=True,
        result_callback=check_result,
    )
    with _ImageSegmenter.create_from_options(options) as segmenter:
      for timestamp in range(0, 300, 30):
        segmenter.segment_async(test_image, timestamp)
        callback_event.wait(3)
        if callback_exception is not None:
          raise callback_exception
        callback_event.clear()


if __name__ == '__main__':
  absltest.main()
