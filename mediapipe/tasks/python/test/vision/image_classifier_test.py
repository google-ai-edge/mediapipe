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
"""Tests for image classifier."""

import enum

import numpy as np
from absl.testing import absltest
from absl.testing import parameterized

from mediapipe.python._framework_bindings import image as image_module
from mediapipe.tasks.python.components.processors import classifier_options
from mediapipe.tasks.python.components.containers import category as category_module
from mediapipe.tasks.python.components.containers import classifications as classifications_module
from mediapipe.tasks.python.core import base_options as base_options_module
from mediapipe.tasks.python.test import test_utils
from mediapipe.tasks.python.vision import image_classifier
from mediapipe.tasks.python.vision.core import vision_task_running_mode as running_mode_module

_BaseOptions = base_options_module.BaseOptions
_ClassifierOptions = classifier_options.ClassifierOptions
_Category = category_module.Category
_ClassificationEntry = classifications_module.ClassificationEntry
_Classifications = classifications_module.Classifications
_ClassificationResult = classifications_module.ClassificationResult
_Image = image_module.Image
_ImageClassifier = image_classifier.ImageClassifier
_ImageClassifierOptions = image_classifier.ImageClassifierOptions
_RUNNING_MODE = running_mode_module.VisionTaskRunningMode

_MODEL_FILE = 'mobilenet_v2_1.0_224.tflite'
_IMAGE_FILE = 'burger.jpg'
_EXPECTED_CLASSIFICATION_RESULT = _ClassificationResult(
  classifications=[
    _Classifications(
      entries=[
        _ClassificationEntry(
          categories=[
            _Category(
              index=934,
              score=0.7939587831497192,
              display_name='',
              category_name='cheeseburger'),
            _Category(
              index=932,
              score=0.02739289402961731,
              display_name='',
              category_name='bagel'),
            _Category(
              index=925,
              score=0.01934075355529785,
              display_name='',
              category_name='guacamole'),
            _Category(
              index=963,
              score=0.006327860057353973,
              display_name='',
              category_name='meat loaf')
          ],
          timestamp_ms=0
        )
      ],
      head_index=0,
      head_name='probability')
  ])
_ALLOW_LIST = ['cheeseburger', 'guacamole']
_DENY_LIST = ['cheeseburger']
_SCORE_THRESHOLD = 0.5
_MAX_RESULTS = 3


class ModelFileType(enum.Enum):
  FILE_CONTENT = 1
  FILE_NAME = 2


class ImageClassifierTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.test_image = _Image.create_from_file(
        test_utils.get_test_data_path(_IMAGE_FILE))
    self.model_path = test_utils.get_test_data_path(_MODEL_FILE)

  @parameterized.parameters(
      (ModelFileType.FILE_NAME, 4, _EXPECTED_CLASSIFICATION_RESULT),
      (ModelFileType.FILE_CONTENT, 4, _EXPECTED_CLASSIFICATION_RESULT))
  def test_classify(self, model_file_type, max_results,
                    expected_classification_result):
    # Creates classifier.
    if model_file_type is ModelFileType.FILE_NAME:
      base_options = _BaseOptions(model_asset_path=self.model_path)
    elif model_file_type is ModelFileType.FILE_CONTENT:
      with open(self.model_path, 'rb') as f:
        model_content = f.read()
      base_options = _BaseOptions(model_asset_buffer=model_content)
    else:
      # Should never happen
      raise ValueError('model_file_type is invalid.')

    classifier_options = _ClassifierOptions(max_results=max_results)
    options = _ImageClassifierOptions(
        base_options=base_options, classifier_options=classifier_options)
    classifier = _ImageClassifier.create_from_options(options)

    # Performs image classification on the input.
    image_result = classifier.classify(self.test_image)
    # Comparing results.
    self.assertEqual(image_result, expected_classification_result)
    # Closes the classifier explicitly when the classifier is not used in
    # a context.
    classifier.close()

  def test_classify_for_video(self):
    classifier_options = _ClassifierOptions(max_results=4)
    options = _ImageClassifierOptions(
        base_options=_BaseOptions(model_asset_path=self.model_path),
        running_mode=_RUNNING_MODE.VIDEO,
        classifier_options=classifier_options)
    with _ImageClassifier.create_from_options(options) as classifier:
      for timestamp in range(0, 300, 30):
        classification_result = classifier.classify_for_video(
            self.test_image, timestamp)
        self.assertEqual(classification_result, _EXPECTED_CLASSIFICATION_RESULT)


if __name__ == '__main__':
  absltest.main()
