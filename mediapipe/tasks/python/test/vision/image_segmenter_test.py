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

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np

from mediapipe.python._framework_bindings import image as image_module
from mediapipe.tasks.python.components import segmenter_options
from mediapipe.tasks.python.core import base_options as base_options_module
from mediapipe.tasks.python.test import test_util
from mediapipe.tasks.python.vision import image_segmenter
from mediapipe.tasks.python.vision.core import vision_task_running_mode as running_mode_module

_BaseOptions = base_options_module.BaseOptions
_Image = image_module.Image
_OutputType = segmenter_options.OutputType
_Activation = segmenter_options.Activation
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
      (ModelFileType.FILE_NAME, 4),
      (ModelFileType.FILE_CONTENT, 4))
  def succeeds_with_category_mask(self, model_file_type, max_results):
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

    options = _ImageSegmenterOptions(base_options=base_options,
                                     output_type=_OutputType.CATEGORY_MASK)
    segmenter = _ImageSegmenter.create_from_options(options)

    # Performs image segmentation on the input.
    image_result = segmenter.segment(self.test_image)

    # Comparing results.
    print(image_result)

    # Closes the segmenter explicitly when the segmenter is not used in
    # a context.
    segmenter.close()


if __name__ == '__main__':
  absltest.main()
