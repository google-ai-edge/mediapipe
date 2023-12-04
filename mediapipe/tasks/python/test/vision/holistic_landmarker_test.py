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
"""Tests for holistic landmarker."""

import enum
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np

from google.protobuf import text_format
from mediapipe.framework.formats import classification_pb2
from mediapipe.framework.formats import landmark_pb2
from mediapipe.python._framework_bindings import image as image_module
from mediapipe.tasks.python.components.containers import category as category_module
from mediapipe.tasks.python.components.containers import landmark as landmark_module
from mediapipe.tasks.python.components.containers import rect as rect_module
from mediapipe.tasks.python.core import base_options as base_options_module
from mediapipe.tasks.python.test import test_utils
from mediapipe.tasks.python.vision import holistic_landmarker
from mediapipe.tasks.python.vision.core import image_processing_options as image_processing_options_module
from mediapipe.tasks.python.vision.core import vision_task_running_mode as running_mode_module


HolisticLandmarkerResult = holistic_landmarker.HolisticLandmarkerResult
_BaseOptions = base_options_module.BaseOptions
_Category = category_module.Category
_Rect = rect_module.Rect
_Landmark = landmark_module.Landmark
_NormalizedLandmark = landmark_module.NormalizedLandmark
_Image = image_module.Image
_HolisticLandmarker = holistic_landmarker.HolisticLandmarker
_HolisticLandmarkerOptions = holistic_landmarker.HolisticLandmarkerOptions
_RUNNING_MODE = running_mode_module.VisionTaskRunningMode
_ImageProcessingOptions = image_processing_options_module.ImageProcessingOptions

_HOLISTIC_LANDMARKER_BUNDLE_ASSET_FILE = 'face_landmarker.task'
_POSE_IMAGE = 'male_full_height_hands.jpg'
_CAT_IMAGE = 'cat.jpg'
_HOLISTIC_RESULT = "male_full_height_hands_result_cpu.pbtxt"
_LANDMARKS_MARGIN = 0.03
_BLENDSHAPES_MARGIN = 0.13


class ModelFileType(enum.Enum):
  FILE_CONTENT = 1
  FILE_NAME = 2


class HolisticLandmarkerTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.test_image = _Image.create_from_file(
        test_utils.get_test_data_path(_POSE_IMAGE)
    )
    self.model_path = test_utils.get_test_data_path(
        _HOLISTIC_LANDMARKER_BUNDLE_ASSET_FILE
    )

  @parameterized.parameters(
      (
          ModelFileType.FILE_NAME,
          _HOLISTIC_LANDMARKER_BUNDLE_ASSET_FILE
      ),
      (
          ModelFileType.FILE_CONTENT,
          _HOLISTIC_LANDMARKER_BUNDLE_ASSET_FILE
      ),
  )
  def test_detect(
      self,
      model_file_type,
      model_name
  ):
    # Creates holistic landmarker.
    model_path = test_utils.get_test_data_path(model_name)
    if model_file_type is ModelFileType.FILE_NAME:
      base_options = _BaseOptions(model_asset_path=model_path)
    elif model_file_type is ModelFileType.FILE_CONTENT:
      with open(model_path, 'rb') as f:
        model_content = f.read()
      base_options = _BaseOptions(model_asset_buffer=model_content)
    else:
      # Should never happen
      raise ValueError('model_file_type is invalid.')

    options = _HolisticLandmarkerOptions(
        base_options=base_options
    )
    landmarker = _HolisticLandmarker.create_from_options(options)

    # Performs holistic landmarks detection on the input.
    detection_result = landmarker.detect(self.test_image)

    # Closes the holistic landmarker explicitly when the holistic landmarker is not used
    # in a context.
    landmarker.close()


if __name__ == '__main__':
  absltest.main()
