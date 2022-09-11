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

from absl.testing import absltest
from absl.testing import parameterized

from mediapipe.python._framework_bindings import image as image_module
from mediapipe.tasks.python.components.containers import category as category_module
from mediapipe.tasks.python.components.containers import classifications as classifications_module
from mediapipe.tasks.python.core import base_options as base_options_module
from mediapipe.tasks.python.test import test_util
from mediapipe.tasks.python.vision import image_classification
from mediapipe.tasks.python.vision.core import vision_task_running_mode as running_mode_module

_BaseOptions = base_options_module.BaseOptions
_Category = category_module.Category
_ClassificationEntry = classifications_module.ClassificationEntry
_Classifications = classifications_module.Classifications
_ClassificationResult = classifications_module.ClassificationResult
_Image = image_module.Image
_ImageClassifier = image_classification.ImageClassifier
_ImageClassifierOptions = image_classification.ImageClassifierOptions
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
              score=0.7952049970626831,
              display_name='',
              category_name='cheeseburger'),
            _Category(
              index=932,
              score=0.02732999622821808,
              display_name='',
              category_name='bagel'),
            _Category(
              index=925,
              score=0.01933487318456173,
              display_name='',
              category_name='guacamole'),
            _Category(
              index=963,
              score=0.006279350258409977,
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
    self.test_image = test_util.read_test_image(
        test_util.get_test_data_path(_IMAGE_FILE))
    self.model_path = test_util.get_test_data_path(_MODEL_FILE)

  def test_create_from_file_succeeds_with_valid_model_path(self):
    # Creates with default option and valid model file successfully.
    with _ImageClassifier.create_from_model_path(self.model_path) as classifier:
      self.assertIsInstance(classifier, _ImageClassifier)

  def test_create_from_options_succeeds_with_valid_model_path(self):
    # Creates with options containing model file successfully.
    base_options = _BaseOptions(file_name=self.model_path)
    options = _ImageClassifierOptions(base_options=base_options)
    with _ImageClassifier.create_from_options(options) as classifier:
      self.assertIsInstance(classifier, _ImageClassifier)

  def test_create_from_options_fails_with_invalid_model_path(self):
    # Invalid empty model path.
    with self.assertRaisesRegex(
        ValueError,
        r"ExternalFile must specify at least one of 'file_content', "
        r"'file_name' or 'file_descriptor_meta'."):
      base_options = _BaseOptions(file_name='')
      options = _ImageClassifierOptions(base_options=base_options)
      _ImageClassifier.create_from_options(options)

  def test_create_from_options_succeeds_with_valid_model_content(self):
    # Creates with options containing model content successfully.
    with open(self.model_path, 'rb') as f:
      base_options = _BaseOptions(file_content=f.read())
      options = _ImageClassifierOptions(base_options=base_options)
      classifier = _ImageClassifier.create_from_options(options)
      self.assertIsInstance(classifier, _ImageClassifier)

  @parameterized.parameters(
      (ModelFileType.FILE_NAME, 4, _EXPECTED_CLASSIFICATION_RESULT),
      (ModelFileType.FILE_CONTENT, 4, _EXPECTED_CLASSIFICATION_RESULT))
  def test_classify(self, model_file_type, max_results,
                    expected_classification_result):
    # Creates classifier.
    if model_file_type is ModelFileType.FILE_NAME:
      base_options = _BaseOptions(file_name=self.model_path)
    elif model_file_type is ModelFileType.FILE_CONTENT:
      with open(self.model_path, 'rb') as f:
        model_content = f.read()
      base_options = _BaseOptions(file_content=model_content)
    else:
      # Should never happen
      raise ValueError('model_file_type is invalid.')

    options = _ImageClassifierOptions(
        base_options=base_options, max_results=max_results)
    classifier = _ImageClassifier.create_from_options(options)

    # Performs image classification on the input.
    image_result = classifier.classify(self.test_image)
    # Comparing results.
    self.assertEqual(image_result, expected_classification_result)
    # Closes the classifier explicitly when the classifier is not used in
    # a context.
    classifier.close()

  @parameterized.parameters(
      (ModelFileType.FILE_NAME, 4, _EXPECTED_CLASSIFICATION_RESULT),
      (ModelFileType.FILE_CONTENT, 4, _EXPECTED_CLASSIFICATION_RESULT))
  def test_classify_in_context(self, model_file_type, max_results,
                               expected_classification_result):
    if model_file_type is ModelFileType.FILE_NAME:
      base_options = _BaseOptions(file_name=self.model_path)
    elif model_file_type is ModelFileType.FILE_CONTENT:
      with open(self.model_path, 'rb') as f:
        model_content = f.read()
      base_options = _BaseOptions(file_content=model_content)
    else:
      # Should never happen
      raise ValueError('model_file_type is invalid.')

    options = _ImageClassifierOptions(
        base_options=base_options, max_results=max_results)
    with _ImageClassifier.create_from_options(options) as classifier:
      # Performs object detection on the input.
      image_result = classifier.classify(self.test_image)
      # Comparing results.
      self.assertEqual(image_result, expected_classification_result)

  def test_score_threshold_option(self):
    options = _ImageClassifierOptions(
        base_options=_BaseOptions(file_name=self.model_path),
        score_threshold=_SCORE_THRESHOLD)
    with _ImageClassifier.create_from_options(options) as classifier:
      # Performs image classification on the input.
      image_result = classifier.classify(self.test_image)
      classifications = image_result.classifications

      for classification in classifications:
        for entry in classification.entries:
          score = entry.categories[0].score
          self.assertGreaterEqual(
              score, _SCORE_THRESHOLD,
              f'Classification with score lower than threshold found. '
              f'{classification}')

  def test_max_results_option(self):
    options = _ImageClassifierOptions(
        base_options=_BaseOptions(file_name=self.model_path),
        max_results=_MAX_RESULTS)
    with _ImageClassifier.create_from_options(options) as classifier:
      # Performs image classification on the input.
      image_result = classifier.classify(self.test_image)
      categories = image_result.classifications[0].entries[0].categories

      self.assertLessEqual(
          len(categories), _MAX_RESULTS, 'Too many results returned.')

  def test_allow_list_option(self):
    options = _ImageClassifierOptions(
        base_options=_BaseOptions(file_name=self.model_path),
        category_allowlist=_ALLOW_LIST)
    with _ImageClassifier.create_from_options(options) as classifier:
      # Performs image classification on the input.
      image_result = classifier.classify(self.test_image)
      classifications = image_result.classifications

      for classification in classifications:
        for entry in classification.entries:
          label = entry.categories[0].category_name
          self.assertIn(label, _ALLOW_LIST,
                        f'Label {label} found but not in label allow list')

  def test_deny_list_option(self):
    options = _ImageClassifierOptions(
      base_options=_BaseOptions(file_name=self.model_path),
      category_denylist=_DENY_LIST)
    with _ImageClassifier.create_from_options(options) as classifier:
      # Performs image classification on the input.
      image_result = classifier.classify(self.test_image)
      classifications = image_result.classifications

      for classification in classifications:
        for entry in classification.entries:
          label = entry.categories[0].category_name
          self.assertNotIn(label, _DENY_LIST,
                        f'Label {label} found but in deny list.')

  def test_combined_allowlist_and_denylist(self):
    # Fails with combined allowlist and denylist
    with self.assertRaisesRegex(
        ValueError,
        r'`category_allowlist` and `category_denylist` are mutually '
        r'exclusive options.'):
      options = _ImageClassifierOptions(
          base_options=_BaseOptions(file_name=self.model_path),
          category_allowlist=['foo'],
          category_denylist=['bar'])
      with _ImageClassifier.create_from_options(options) as unused_classifier:
        pass

  def test_empty_classification_outputs(self):
    options = _ImageClassifierOptions(
        base_options=_BaseOptions(file_name=self.model_path), score_threshold=1)
    with _ImageClassifier.create_from_options(options) as classifier:
      # Performs image classification on the input.
      image_result = classifier.classify(self.test_image)
      self.assertEmpty(image_result.classifications[0].entries[0].categories)

  def test_missing_result_callback(self):
    options = _ImageClassifierOptions(
        base_options=_BaseOptions(file_name=self.model_path),
        running_mode=_RUNNING_MODE.LIVE_STREAM)
    with self.assertRaisesRegex(ValueError,
                                r'result callback must be provided'):
      with _ImageClassifier.create_from_options(options) as unused_classifier:
        pass

  @parameterized.parameters((_RUNNING_MODE.IMAGE), (_RUNNING_MODE.VIDEO))
  def test_illegal_result_callback(self, running_mode):

    def pass_through(unused_result: _ClassificationResult):
      pass

    options = _ImageClassifierOptions(
        base_options=_BaseOptions(file_name=self.model_path),
        running_mode=running_mode,
        result_callback=pass_through)
    with self.assertRaisesRegex(ValueError,
                                r'result callback should not be provided'):
      with _ImageClassifier.create_from_options(options) as unused_classifier:
        pass


if __name__ == '__main__':
  absltest.main()
