# Copyright 2022 The MediaPipe Authors. All Rights Reserved.
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
import numpy as np
import cv2

from absl.testing import absltest
from absl.testing import parameterized

from mediapipe.python._framework_bindings import image as image_module
from mediapipe.python._framework_bindings import image_frame as image_frame_module
from mediapipe.tasks.python.components import segmenter_options
from mediapipe.tasks.python.core import base_options as base_options_module
from mediapipe.tasks.python.test import test_util
from mediapipe.tasks.python.vision import image_segmenter
from mediapipe.tasks.python.vision.core import vision_task_running_mode as running_mode_module

_BaseOptions = base_options_module.BaseOptions
_Image = image_module.Image
_ImageFormat = image_frame_module.ImageFormat
_OutputType = segmenter_options.OutputType
_Activation = segmenter_options.Activation
_SegmenterOptions = segmenter_options.SegmenterOptions
_ImageSegmenter = image_segmenter.ImageSegmenter
_ImageSegmenterOptions = image_segmenter.ImageSegmenterOptions
_RUNNING_MODE = running_mode_module.VisionTaskRunningMode

_MODEL_FILE = 'deeplabv3.tflite'
_IMAGE_FILE = 'segmentation_input_rotation0.jpg'
_SEGMENTATION_FILE = 'segmentation_golden_rotation0.png'
_MASK_MAGNIFICATION_FACTOR = 10
_MATCH_PIXELS_THRESHOLD = 0.01


class ModelFileType(enum.Enum):
  FILE_CONTENT = 1
  FILE_NAME = 2


class ImageSegmenterTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.test_image = test_util.read_test_image(
        test_util.get_test_data_path(_IMAGE_FILE))
    self.test_seg_path = test_util.get_test_data_path(_SEGMENTATION_FILE)
    self.model_path = test_util.get_test_data_path(_MODEL_FILE)

  def test_create_from_file_succeeds_with_valid_model_path(self):
    # Creates with default option and valid model file successfully.
    with _ImageSegmenter.create_from_model_path(self.model_path) as segmenter:
      self.assertIsInstance(segmenter, _ImageSegmenter)

  def test_create_from_options_succeeds_with_valid_model_path(self):
    # Creates with options containing model file successfully.
    base_options = _BaseOptions(file_name=self.model_path)
    options = _ImageSegmenterOptions(base_options=base_options)
    with _ImageSegmenter.create_from_options(options) as segmenter:
      self.assertIsInstance(segmenter, _ImageSegmenter)

  def test_create_from_options_fails_with_invalid_model_path(self):
    # Invalid empty model path.
    with self.assertRaisesRegex(
        ValueError,
        r"ExternalFile must specify at least one of 'file_content', "
        r"'file_name' or 'file_descriptor_meta'."):
      base_options = _BaseOptions(file_name='')
      options = _ImageSegmenterOptions(base_options=base_options)
      _ImageSegmenter.create_from_options(options)

  def test_create_from_options_succeeds_with_valid_model_content(self):
    # Creates with options containing model content successfully.
    with open(self.model_path, 'rb') as f:
      base_options = _BaseOptions(file_content=f.read())
      options = _ImageSegmenterOptions(base_options=base_options)
      segmenter = _ImageSegmenter.create_from_options(options)
      self.assertIsInstance(segmenter, _ImageSegmenter)

  @parameterized.parameters(
      (ModelFileType.FILE_NAME,),
      (ModelFileType.FILE_CONTENT,))
  def test_succeeds_with_category_mask(self, model_file_type):
    # Creates segmenter.
    if model_file_type is ModelFileType.FILE_NAME:
      base_options = _BaseOptions(file_name=self.model_path)
    elif model_file_type is ModelFileType.FILE_CONTENT:
      with open(self.model_path, 'rb') as f:
        model_content = f.read()
      base_options = _BaseOptions(file_content=model_content)
    else:
      # Should never happen
      raise ValueError('model_file_type is invalid.')

    segmenter_options = _SegmenterOptions(output_type=_OutputType.CATEGORY_MASK)
    options = _ImageSegmenterOptions(base_options=base_options,
                                     segmenter_options=segmenter_options)
    segmenter = _ImageSegmenter.create_from_options(options)

    # Performs image segmentation on the input.
    category_masks = segmenter.segment(self.test_image)
    self.assertEqual(len(category_masks), 1)
    result_pixels = category_masks[0].numpy_view().flatten()

    # Check if data type of `category_masks` is correct.
    self.assertEqual(result_pixels.dtype, np.uint8)

    # Loads ground truth segmentation file.
    image_data = cv2.imread(self.test_seg_path, cv2.IMREAD_GRAYSCALE)
    gt_segmentation = _Image(_ImageFormat.GRAY8, image_data)
    gt_segmentation_array = gt_segmentation.numpy_view()
    gt_segmentation_shape = gt_segmentation_array.shape
    num_pixels = gt_segmentation_shape[0] * gt_segmentation_shape[1]
    ground_truth_pixels = gt_segmentation_array.flatten()

    self.assertEqual(
      len(result_pixels), len(ground_truth_pixels),
      'Segmentation mask size does not match the ground truth mask size.')

    inconsistent_pixels = 0

    for index in range(num_pixels):
      inconsistent_pixels += (
          result_pixels[index] * _MASK_MAGNIFICATION_FACTOR !=
          ground_truth_pixels[index])

    self.assertLessEqual(
      inconsistent_pixels / num_pixels, _MATCH_PIXELS_THRESHOLD,
      f'Number of pixels in the candidate mask differing from that of the '
      f'ground truth mask exceeds {_MATCH_PIXELS_THRESHOLD}.')

    # Closes the segmenter explicitly when the segmenter is not used in
    # a context.
    segmenter.close()

  def test_succeeds_with_confidence_mask(self):
    # Creates segmenter.
    base_options = _BaseOptions(file_name=self.model_path)

    # Run segmentation on the model in CATEGORY_MASK mode.
    segmenter_options = _SegmenterOptions(output_type=_OutputType.CATEGORY_MASK)
    options = _ImageSegmenterOptions(base_options=base_options,
                                     segmenter_options=segmenter_options)
    segmenter = _ImageSegmenter.create_from_options(options)
    category_masks = segmenter.segment(self.test_image)
    category_mask = category_masks[0].numpy_view()

    # Run segmentation on the model in CONFIDENCE_MASK mode.
    segmenter_options = _SegmenterOptions(
        output_type=_OutputType.CONFIDENCE_MASK,
        activation=_Activation.SOFTMAX)
    options = _ImageSegmenterOptions(base_options=base_options,
                                     segmenter_options=segmenter_options)
    segmenter = _ImageSegmenter.create_from_options(options)
    confidence_masks = segmenter.segment(self.test_image)

    # Check if confidence mask shape is correct.
    self.assertEqual(
        len(confidence_masks), 21,
        'Number of confidence masks must match with number of categories.')

    # Gather the confidence masks in a single array `confidence_mask_array`.
    confidence_mask_array = np.array(
      [confidence_mask.numpy_view() for confidence_mask in confidence_masks])

    # Check if data type of `confidence_masks` are correct.
    self.assertEqual(confidence_mask_array.dtype, np.float32)

    # Compute the category mask from the created confidence mask.
    calculated_category_mask = np.argmax(confidence_mask_array, axis=0)
    self.assertListEqual(
      calculated_category_mask.tolist(), category_mask.tolist(),
      'Confidence mask does not match with the category mask.')

    # Closes the segmenter explicitly when the segmenter is not used in
    # a context.
    segmenter.close()


if __name__ == '__main__':
  absltest.main()
