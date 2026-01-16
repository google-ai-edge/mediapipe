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
"""Tests for image classifier."""

import enum
import os
import threading
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np

from mediapipe.tasks.python.components.containers import category as category_module
from mediapipe.tasks.python.components.containers import classification_result as classification_result_module
from mediapipe.tasks.python.components.containers import rect as rect_module
from mediapipe.tasks.python.core import base_options as base_options_module
from mediapipe.tasks.python.test import test_utils
from mediapipe.tasks.python.vision import image_classifier
from mediapipe.tasks.python.vision.core import image as image_module
from mediapipe.tasks.python.vision.core import image_processing_options as image_processing_options_module
from mediapipe.tasks.python.vision.core import vision_task_running_mode

ImageClassifierResult = classification_result_module.ClassificationResult
_RectF = rect_module.RectF
_BaseOptions = base_options_module.BaseOptions
_Category = category_module.Category
_Classifications = classification_result_module.Classifications
_Image = image_module.Image
_ImageClassifier = image_classifier.ImageClassifier
_ImageClassifierOptions = image_classifier.ImageClassifierOptions
_RUNNING_MODE = vision_task_running_mode.VisionTaskRunningMode
_ImageProcessingOptions = image_processing_options_module.ImageProcessingOptions

_MODEL_FILE = 'mobilenet_v2_1.0_224.tflite'
_IMAGE_FILE = 'burger.jpg'
_ALLOW_LIST = ['cheeseburger', 'guacamole']
_DENY_LIST = ['cheeseburger']
_SCORE_THRESHOLD = 0.5
_MAX_RESULTS = 3
_TEST_DATA_DIR = 'mediapipe/tasks/testdata/vision'


def _generate_empty_results() -> ImageClassifierResult:
  return ImageClassifierResult(
      classifications=[
          _Classifications(categories=[], head_index=0, head_name='probability')
      ],
      timestamp_ms=0,
  )


def _generate_burger_results(timestamp_ms=0) -> ImageClassifierResult:
  return ImageClassifierResult(
      classifications=[
          _Classifications(
              categories=[
                  _Category(
                      index=934,
                      score=0.793959,
                      display_name='',
                      category_name='cheeseburger',
                  ),
                  _Category(
                      index=932,
                      score=0.0273929,
                      display_name='',
                      category_name='bagel',
                  ),
                  _Category(
                      index=925,
                      score=0.0193408,
                      display_name='',
                      category_name='guacamole',
                  ),
                  _Category(
                      index=963,
                      score=0.00632786,
                      display_name='',
                      category_name='meat loaf',
                  ),
              ],
              head_index=0,
              head_name='probability',
          )
      ],
      timestamp_ms=timestamp_ms,
  )


def _generate_soccer_ball_results(timestamp_ms=0) -> ImageClassifierResult:
  return ImageClassifierResult(
      classifications=[
          _Classifications(
              categories=[
                  _Category(
                      index=806,
                      score=0.996527,
                      display_name='',
                      category_name='soccer ball',
                  )
              ],
              head_index=0,
              head_name='probability',
          )
      ],
      timestamp_ms=timestamp_ms,
  )


class ModelFileType(enum.Enum):
  FILE_CONTENT = 1
  FILE_NAME = 2


class ImageClassifierTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.test_image = _Image.create_from_file(
        test_utils.get_test_data_path(os.path.join(_TEST_DATA_DIR, _IMAGE_FILE))
    )
    self.model_path = test_utils.get_test_data_path(
        os.path.join(_TEST_DATA_DIR, _MODEL_FILE)
    )

  def assertCategoryAlmostEqual(
      self, actual: _Category, expected: _Category, delta: float = 1e-6
  ):
    self.assertEqual(actual.index, expected.index)
    self.assertAlmostEqual(actual.score, expected.score, delta=delta)
    self.assertEqual(actual.display_name or '', expected.display_name)
    self.assertEqual(actual.category_name, expected.category_name)

  def assertClassificationsAlmostEqual(
      self, actual: _Classifications, expected: _Classifications
  ):
    self.assertEqual(actual.head_index, expected.head_index)
    self.assertEqual(actual.head_name, expected.head_name)
    self.assertLen(actual.categories, len(expected.categories))
    for i, actual_category in enumerate(actual.categories):
      self.assertCategoryAlmostEqual(actual_category, expected.categories[i])

  def assertClassificationResultCorrect(
      self, actual: ImageClassifierResult, expected: ImageClassifierResult
  ):
    self.assertEqual(actual.timestamp_ms, expected.timestamp_ms)
    self.assertLen(actual.classifications, len(expected.classifications))
    for i, actual_classifications in enumerate(actual.classifications):
      self.assertClassificationsAlmostEqual(
          actual_classifications, expected.classifications[i]
      )

  def test_create_from_file_succeeds_with_valid_model_path(self):
    # Creates with default option and valid model file successfully.
    with _ImageClassifier.create_from_model_path(self.model_path) as classifier:
      self.assertIsInstance(classifier, _ImageClassifier)

  def test_create_from_options_succeeds_with_valid_model_path(self):
    # Creates with options containing model file successfully.
    base_options = _BaseOptions(model_asset_path=self.model_path)
    options = _ImageClassifierOptions(base_options=base_options)
    with _ImageClassifier.create_from_options(options) as classifier:
      self.assertIsInstance(classifier, _ImageClassifier)

  def test_create_from_options_fails_with_invalid_model_path(self):
    with self.assertRaisesRegex(
        FileNotFoundError,
        'Unable to open file at /path/to/invalid/model.tflite',
    ):
      base_options = _BaseOptions(
          model_asset_path='/path/to/invalid/model.tflite'
      )
      options = _ImageClassifierOptions(base_options=base_options)
      _ImageClassifier.create_from_options(options)

  def test_create_from_options_succeeds_with_valid_model_content(self):
    # Creates with options containing model content successfully.
    with open(self.model_path, 'rb') as f:
      base_options = _BaseOptions(model_asset_buffer=f.read())
      options = _ImageClassifierOptions(base_options=base_options)
      with _ImageClassifier.create_from_options(options) as classifier:
        self.assertIsInstance(classifier, _ImageClassifier)

  @parameterized.parameters(
      (ModelFileType.FILE_NAME, 4, _generate_burger_results()),
      (ModelFileType.FILE_CONTENT, 4, _generate_burger_results()),
  )
  def test_classify(
      self, model_file_type, max_results, expected_classification_result
  ):
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

    options = _ImageClassifierOptions(
        base_options=base_options, max_results=max_results
    )
    classifier = _ImageClassifier.create_from_options(options)

    # Performs image classification on the input.
    image_result = classifier.classify(self.test_image)
    # Comparing results.
    self.assertClassificationResultCorrect(
        image_result, expected_classification_result
    )
    # Closes the classifier explicitly when the classifier is not used in
    # a context.
    classifier.close()

  @parameterized.parameters(
      (ModelFileType.FILE_NAME, 4, _generate_burger_results()),
      (ModelFileType.FILE_CONTENT, 4, _generate_burger_results()),
  )
  def test_classify_in_context(
      self, model_file_type, max_results, expected_classification_result
  ):
    if model_file_type is ModelFileType.FILE_NAME:
      base_options = _BaseOptions(model_asset_path=self.model_path)
    elif model_file_type is ModelFileType.FILE_CONTENT:
      with open(self.model_path, 'rb') as f:
        model_content = f.read()
      base_options = _BaseOptions(model_asset_buffer=model_content)
    else:
      # Should never happen
      raise ValueError('model_file_type is invalid.')

    options = _ImageClassifierOptions(
        base_options=base_options, max_results=max_results
    )
    with _ImageClassifier.create_from_options(options) as classifier:
      # Performs image classification on the input.
      image_result = classifier.classify(self.test_image)
      # Comparing results.
      self.assertClassificationResultCorrect(
          image_result, expected_classification_result
      )

  def test_classify_succeeds_with_region_of_interest(self):
    base_options = _BaseOptions(model_asset_path=self.model_path)
    options = _ImageClassifierOptions(base_options=base_options, max_results=1)
    with _ImageClassifier.create_from_options(options) as classifier:
      # Load the test image.
      test_image = _Image.create_from_file(
          test_utils.get_test_data_path(
              os.path.join(_TEST_DATA_DIR, 'multi_objects.jpg')
          )
      )
      # Region-of-interest around the soccer ball.
      roi = _RectF(left=0.45, top=0.3075, right=0.614, bottom=0.7345)
      image_processing_options = _ImageProcessingOptions(roi)
      # Performs image classification on the input.
      image_result = classifier.classify(test_image, image_processing_options)
      # Comparing results.
      self.assertClassificationResultCorrect(
          image_result, _generate_soccer_ball_results()
      )

  def test_classify_succeeds_with_rotation(self):
    base_options = _BaseOptions(model_asset_path=self.model_path)
    options = _ImageClassifierOptions(base_options=base_options, max_results=3)
    with _ImageClassifier.create_from_options(options) as classifier:
      # Load the test image.
      test_image = _Image.create_from_file(
          test_utils.get_test_data_path(
              os.path.join(_TEST_DATA_DIR, 'burger_rotated.jpg')
          )
      )
      # Specify a 90° anti-clockwise rotation.
      image_processing_options = _ImageProcessingOptions(None, -90)
      # Performs image classification on the input.
      image_result = classifier.classify(test_image, image_processing_options)
      # Comparing results.
      expected = ImageClassifierResult(
          classifications=[
              _Classifications(
                  categories=[
                      _Category(
                          index=934,
                          score=0.754467,
                          display_name='',
                          category_name='cheeseburger',
                      ),
                      _Category(
                          index=925,
                          score=0.0288028,
                          display_name='',
                          category_name='guacamole',
                      ),
                      _Category(
                          index=932,
                          score=0.0286119,
                          display_name='',
                          category_name='bagel',
                      ),
                  ],
                  head_index=0,
                  head_name='probability',
              )
          ],
          timestamp_ms=0,
      )
      self.assertClassificationResultCorrect(image_result, expected)

  def test_classify_succeeds_with_region_of_interest_and_rotation(self):
    base_options = _BaseOptions(model_asset_path=self.model_path)
    options = _ImageClassifierOptions(base_options=base_options, max_results=1)
    with _ImageClassifier.create_from_options(options) as classifier:
      # Load the test image.
      test_image = _Image.create_from_file(
          test_utils.get_test_data_path(
              os.path.join(_TEST_DATA_DIR, 'multi_objects_rotated.jpg')
          )
      )
      # Region-of-interest around the soccer ball, with 90° anti-clockwise
      # rotation.
      roi = _RectF(left=0.2655, top=0.45, right=0.6925, bottom=0.614)
      image_processing_options = _ImageProcessingOptions(roi, -90)
      # Performs image classification on the input.
      image_result = classifier.classify(test_image, image_processing_options)
      # Comparing results.
      expected = ImageClassifierResult(
          classifications=[
              _Classifications(
                  categories=[
                      _Category(
                          index=806,
                          score=0.997684,
                          display_name='',
                          category_name='soccer ball',
                      ),
                  ],
                  head_index=0,
                  head_name='probability',
              )
          ],
          timestamp_ms=0,
      )
      self.assertClassificationResultCorrect(image_result, expected)

  def test_score_threshold_option(self):
    options = _ImageClassifierOptions(
        base_options=_BaseOptions(model_asset_path=self.model_path),
        score_threshold=_SCORE_THRESHOLD,
    )
    with _ImageClassifier.create_from_options(options) as classifier:
      # Performs image classification on the input.
      image_result = classifier.classify(self.test_image)
      classifications = image_result.classifications

      for classification in classifications:
        for category in classification.categories:
          score = category.score
          self.assertGreaterEqual(
              score,
              _SCORE_THRESHOLD,
              (
                  'Classification with score lower than threshold found. '
                  f'{classification}'
              ),
          )

  def test_max_results_option(self):
    options = _ImageClassifierOptions(
        base_options=_BaseOptions(model_asset_path=self.model_path),
        score_threshold=_SCORE_THRESHOLD,
    )
    with _ImageClassifier.create_from_options(options) as classifier:
      # Performs image classification on the input.
      image_result = classifier.classify(self.test_image)
      categories = image_result.classifications[0].categories

      self.assertLessEqual(
          len(categories), _MAX_RESULTS, 'Too many results returned.'
      )

  def test_allow_list_option(self):
    options = _ImageClassifierOptions(
        base_options=_BaseOptions(model_asset_path=self.model_path),
        category_allowlist=_ALLOW_LIST,
    )
    with _ImageClassifier.create_from_options(options) as classifier:
      # Performs image classification on the input.
      image_result = classifier.classify(self.test_image)
      classifications = image_result.classifications

      for classification in classifications:
        for category in classification.categories:
          label = category.category_name
          self.assertIn(
              label,
              _ALLOW_LIST,
              f'Label {label} found but not in label allow list',
          )

  def test_deny_list_option(self):
    options = _ImageClassifierOptions(
        base_options=_BaseOptions(model_asset_path=self.model_path),
        category_denylist=_DENY_LIST,
    )
    with _ImageClassifier.create_from_options(options) as classifier:
      # Performs image classification on the input.
      image_result = classifier.classify(self.test_image)
      classifications = image_result.classifications

      for classification in classifications:
        for category in classification.categories:
          label = category.category_name
          self.assertNotIn(
              label, _DENY_LIST, f'Label {label} found but in deny list.'
          )

  def test_combined_allowlist_and_denylist(self):
    # Fails with combined allowlist and denylist
    with self.assertRaises(ValueError):
      options = _ImageClassifierOptions(
          base_options=_BaseOptions(model_asset_path=self.model_path),
          category_allowlist=['foo'],
          category_denylist=['bar'],
      )
      with _ImageClassifier.create_from_options(options) as unused_classifier:
        pass

  def test_empty_classification_outputs(self):
    options = _ImageClassifierOptions(
        base_options=_BaseOptions(model_asset_path=self.model_path),
        score_threshold=1,
    )
    with _ImageClassifier.create_from_options(options) as classifier:
      # Performs image classification on the input.
      image_result = classifier.classify(self.test_image)
      self.assertEmpty(image_result.classifications[0].categories)

  def test_missing_result_callback(self):
    options = _ImageClassifierOptions(
        base_options=_BaseOptions(model_asset_path=self.model_path),
        running_mode=_RUNNING_MODE.LIVE_STREAM,
    )
    with self.assertRaisesRegex(
        ValueError, r'result callback must be provided'
    ):
      with _ImageClassifier.create_from_options(options) as unused_classifier:
        pass

  @parameterized.parameters((_RUNNING_MODE.IMAGE), (_RUNNING_MODE.VIDEO))
  def test_illegal_result_callback(self, running_mode):
    options = _ImageClassifierOptions(
        base_options=_BaseOptions(model_asset_path=self.model_path),
        running_mode=running_mode,
        result_callback=mock.MagicMock(),
    )
    with self.assertRaisesRegex(
        ValueError, r'result callback should not be provided'
    ):
      with _ImageClassifier.create_from_options(options) as unused_classifier:
        pass

  def test_calling_classify_for_video_in_image_mode(self):
    options = _ImageClassifierOptions(
        base_options=_BaseOptions(model_asset_path=self.model_path),
        running_mode=_RUNNING_MODE.IMAGE,
    )
    with _ImageClassifier.create_from_options(options) as classifier:
      with self.assertRaises(ValueError):
        classifier.classify_for_video(self.test_image, 0)

  def test_calling_classify_async_in_image_mode(self):
    options = _ImageClassifierOptions(
        base_options=_BaseOptions(model_asset_path=self.model_path),
        running_mode=_RUNNING_MODE.IMAGE,
    )
    with _ImageClassifier.create_from_options(options) as classifier:
      with self.assertRaises(ValueError):
        classifier.classify_async(self.test_image, 0)

  def test_calling_classify_in_video_mode(self):
    options = _ImageClassifierOptions(
        base_options=_BaseOptions(model_asset_path=self.model_path),
        running_mode=_RUNNING_MODE.VIDEO,
    )
    with _ImageClassifier.create_from_options(options) as classifier:
      with self.assertRaises(ValueError):
        classifier.classify(self.test_image)

  def test_calling_classify_async_in_video_mode(self):
    options = _ImageClassifierOptions(
        base_options=_BaseOptions(model_asset_path=self.model_path),
        running_mode=_RUNNING_MODE.VIDEO,
    )
    with _ImageClassifier.create_from_options(options) as classifier:
      with self.assertRaises(ValueError):
        classifier.classify_async(self.test_image, 0)

  def test_classify_for_video_with_out_of_order_timestamp(self):
    options = _ImageClassifierOptions(
        base_options=_BaseOptions(model_asset_path=self.model_path),
        running_mode=_RUNNING_MODE.VIDEO,
    )
    with _ImageClassifier.create_from_options(options) as classifier:
      unused_result = classifier.classify_for_video(self.test_image, 1)
      with self.assertRaises(ValueError):
        classifier.classify_for_video(self.test_image, 0)

  def test_classify_for_video(self):
    options = _ImageClassifierOptions(
        base_options=_BaseOptions(model_asset_path=self.model_path),
        running_mode=_RUNNING_MODE.VIDEO,
        max_results=4,
    )
    with _ImageClassifier.create_from_options(options) as classifier:
      for timestamp in range(0, 300, 30):
        classification_result = classifier.classify_for_video(
            self.test_image, timestamp
        )
        self.assertClassificationResultCorrect(
            classification_result, _generate_burger_results(timestamp)
        )

  def test_classify_for_video_succeeds_with_region_of_interest(self):
    options = _ImageClassifierOptions(
        base_options=_BaseOptions(model_asset_path=self.model_path),
        running_mode=_RUNNING_MODE.VIDEO,
        max_results=1,
    )
    with _ImageClassifier.create_from_options(options) as classifier:
      # Load the test image.
      test_image = _Image.create_from_file(
          test_utils.get_test_data_path(
              os.path.join(_TEST_DATA_DIR, 'multi_objects.jpg')
          )
      )
      # Region-of-interest around the soccer ball.
      roi = _RectF(left=0.45, top=0.3075, right=0.614, bottom=0.7345)
      image_processing_options = _ImageProcessingOptions(roi)
      for timestamp in range(0, 300, 30):
        classification_result = classifier.classify_for_video(
            test_image, timestamp, image_processing_options
        )
        self.assertClassificationResultCorrect(
            classification_result, _generate_soccer_ball_results(timestamp)
        )

  def test_calling_classify_in_live_stream_mode(self):
    options = _ImageClassifierOptions(
        base_options=_BaseOptions(model_asset_path=self.model_path),
        running_mode=_RUNNING_MODE.LIVE_STREAM,
        result_callback=mock.MagicMock(),
    )
    with _ImageClassifier.create_from_options(options) as classifier:
      with self.assertRaises(ValueError):
        classifier.classify(self.test_image)

  def test_calling_classify_for_video_in_live_stream_mode(self):
    options = _ImageClassifierOptions(
        base_options=_BaseOptions(model_asset_path=self.model_path),
        running_mode=_RUNNING_MODE.LIVE_STREAM,
        result_callback=mock.MagicMock(),
    )
    with _ImageClassifier.create_from_options(options) as classifier:
      with self.assertRaises(ValueError):
        classifier.classify_for_video(self.test_image, 0)

  def test_classify_async_calls_with_illegal_timestamp(self):
    options = _ImageClassifierOptions(
        base_options=_BaseOptions(model_asset_path=self.model_path),
        running_mode=_RUNNING_MODE.LIVE_STREAM,
        max_results=4,
        result_callback=mock.MagicMock(),
    )
    with _ImageClassifier.create_from_options(options) as classifier:
      classifier.classify_async(self.test_image, 100)
      with self.assertRaises(ValueError):
        classifier.classify_async(self.test_image, 0)

  @parameterized.parameters(
      (0, _generate_burger_results()), (1, _generate_empty_results())
  )
  def test_classify_async_calls(self, threshold, expected_result):
    observed_timestamp_ms = -1
    callback_event = threading.Event()
    callback_exception = None

    def check_result(
        result: ImageClassifierResult, output_image: _Image, timestamp_ms: int
    ):
      nonlocal callback_exception, observed_timestamp_ms
      try:
        self.assertClassificationResultCorrect(result, expected_result)
        self.assertTrue(
            np.array_equal(
                output_image.numpy_view(), self.test_image.numpy_view()
            )
        )
        self.assertLess(observed_timestamp_ms, timestamp_ms)
        observed_timestamp_ms = timestamp_ms
      except AssertionError as e:
        callback_exception = e
      finally:
        callback_event.set()

    options = _ImageClassifierOptions(
        base_options=_BaseOptions(model_asset_path=self.model_path),
        running_mode=_RUNNING_MODE.LIVE_STREAM,
        max_results=4,
        score_threshold=threshold,
        result_callback=check_result,
    )
    with _ImageClassifier.create_from_options(options) as classifier:
      classifier.classify_async(self.test_image, 0)
      callback_event.wait(3)
      if callback_exception is not None:
        raise callback_exception
      callback_event.clear()

  def test_classify_async_succeeds_with_region_of_interest(self):
    # Load the test image.
    test_image = _Image.create_from_file(
        test_utils.get_test_data_path(
            os.path.join(_TEST_DATA_DIR, 'multi_objects.jpg')
        )
    )
    # Region-of-interest around the soccer ball.
    roi = _RectF(left=0.45, top=0.3075, right=0.614, bottom=0.7345)
    image_processing_options = _ImageProcessingOptions(roi)
    observed_timestamp_ms = -1
    callback_event = threading.Event()
    callback_exception = None

    def check_result(
        result: ImageClassifierResult, output_image: _Image, timestamp_ms: int
    ):
      nonlocal callback_exception, observed_timestamp_ms
      try:
        self.assertClassificationResultCorrect(
            result, _generate_soccer_ball_results(100)
        )
        self.assertEqual(output_image.width, test_image.width)
        self.assertEqual(output_image.height, test_image.height)
        self.assertLess(observed_timestamp_ms, timestamp_ms)
        observed_timestamp_ms = timestamp_ms
      except AssertionError as e:
        callback_exception = e
      finally:
        callback_event.set()

    options = _ImageClassifierOptions(
        base_options=_BaseOptions(model_asset_path=self.model_path),
        running_mode=_RUNNING_MODE.LIVE_STREAM,
        max_results=1,
        result_callback=check_result,
    )
    with _ImageClassifier.create_from_options(options) as classifier:
      classifier.classify_async(test_image, 100, image_processing_options)
      callback_event.wait(3)
      if callback_exception is not None:
        raise callback_exception
      callback_event.clear()


if __name__ == '__main__':
  absltest.main()
