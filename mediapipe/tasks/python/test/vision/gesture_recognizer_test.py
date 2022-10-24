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
"""Tests for gesture recognizer."""

import enum

from absl.testing import absltest
from absl.testing import parameterized

from mediapipe.python._framework_bindings import image as image_module
from mediapipe.tasks.python.components.containers import rect as rect_module
from mediapipe.tasks.python.components.containers import classification as classification_module
from mediapipe.tasks.python.components.containers import landmark as landmark_module
from mediapipe.tasks.python.core import base_options as base_options_module
from mediapipe.tasks.python.test import test_utils
from mediapipe.tasks.python.vision import gesture_recognizer
from mediapipe.tasks.python.vision.core import vision_task_running_mode as running_mode_module

_BaseOptions = base_options_module.BaseOptions
_NormalizedRect = rect_module.NormalizedRect
_ClassificationList = classification_module.ClassificationList
_LandmarkList = landmark_module.LandmarkList
_NormalizedLandmarkList = landmark_module.NormalizedLandmarkList
_Image = image_module.Image
_GestureRecognizer = gesture_recognizer.GestureRecognizer
_GestureRecognizerOptions = gesture_recognizer.GestureRecognizerOptions
_GestureRecognitionResult = gesture_recognizer.GestureRecognitionResult
_RUNNING_MODE = running_mode_module.VisionTaskRunningMode

_GESTURE_RECOGNIZER_MODEL_FILE = 'gesture_recognizer.task'
_IMAGE_FILE = 'right_hands.jpg'
_EXPECTED_DETECTION_RESULT = _GestureRecognitionResult([], [], [], [])


class ModelFileType(enum.Enum):
  FILE_CONTENT = 1
  FILE_NAME = 2


class GestureRecognizerTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.test_image = _Image.create_from_file(
        test_utils.get_test_data_path(_IMAGE_FILE))
    self.gesture_recognizer_model_path = test_utils.get_test_data_path(
        _GESTURE_RECOGNIZER_MODEL_FILE)

  @parameterized.parameters(
      (ModelFileType.FILE_NAME, _EXPECTED_DETECTION_RESULT),
      (ModelFileType.FILE_CONTENT, _EXPECTED_DETECTION_RESULT))
  def test_recognize(self, model_file_type, expected_recognition_result):
    # Creates gesture recognizer.
    if model_file_type is ModelFileType.FILE_NAME:
      gesture_recognizer_base_options = _BaseOptions(
          model_asset_path=self.gesture_recognizer_model_path)
    elif model_file_type is ModelFileType.FILE_CONTENT:
      with open(self.gesture_recognizer_model_path, 'rb') as f:
        model_content = f.read()
      gesture_recognizer_base_options = _BaseOptions(
          model_asset_buffer=model_content)
    else:
      # Should never happen
      raise ValueError('model_file_type is invalid.')

    options = _GestureRecognizerOptions(
        base_options=gesture_recognizer_base_options)
    recognizer = _GestureRecognizer.create_from_options(options)

    # Performs hand gesture recognition on the input.
    recognition_result = recognizer.recognize(self.test_image)
    # Comparing results.
    self.assertEqual(recognition_result, expected_recognition_result)
    # Closes the gesture recognizer explicitly when the detector is not used in
    # a context.
    recognizer.close()


if __name__ == '__main__':
  absltest.main()
