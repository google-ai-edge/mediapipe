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
"""Tests for language detector."""

import enum
import os

from absl.testing import absltest
from absl.testing import parameterized

from mediapipe.tasks.python.components.containers import category
from mediapipe.tasks.python.components.containers import classification_result as classification_result_module
from mediapipe.tasks.python.core import base_options as base_options_module
from mediapipe.tasks.python.test import test_utils
from mediapipe.tasks.python.text import language_detector

LanguageDetectorResult = language_detector.LanguageDetectorResult
LanguageDetectorPrediction = (
    language_detector.LanguageDetectorResult.Detection
)
_BaseOptions = base_options_module.BaseOptions
_Category = category.Category
_Classifications = classification_result_module.Classifications
_LanguageDetector = language_detector.LanguageDetector
_LanguageDetectorOptions = language_detector.LanguageDetectorOptions

_LANGUAGE_DETECTOR_MODEL = "language_detector.tflite"
_TEST_DATA_DIR = "mediapipe/tasks/testdata/text"

_SCORE_THRESHOLD = 0.3
_EN_TEXT = "To be, or not to be, that is the question"
_EN_EXPECTED_RESULT = LanguageDetectorResult(
    [LanguageDetectorPrediction("en", 0.999856)]
)
_FR_TEXT = (
    "Il y a beaucoup de bouches qui parlent et fort peu de têtes qui pensent."
)
_FR_EXPECTED_RESULT = LanguageDetectorResult(
    [LanguageDetectorPrediction("fr", 0.999781)]
)
_RU_TEXT = "это какой-то английский язык"
_RU_EXPECTED_RESULT = LanguageDetectorResult(
    [LanguageDetectorPrediction("ru", 0.993362)]
)
_MIXED_TEXT = "分久必合合久必分"
_MIXED_EXPECTED_RESULT = LanguageDetectorResult([
    LanguageDetectorPrediction("zh", 0.505424),
    LanguageDetectorPrediction("ja", 0.481617),
])
_TOLERANCE = 1e-6


class ModelFileType(enum.Enum):
  FILE_CONTENT = 1
  FILE_NAME = 2


class LanguageDetectorTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.model_path = test_utils.get_test_data_path(
        os.path.join(_TEST_DATA_DIR, _LANGUAGE_DETECTOR_MODEL)
    )

  def _expect_language_detector_result_correct(
      self,
      actual_result: LanguageDetectorResult,
      expect_result: LanguageDetectorResult,
  ):
    for i, prediction in enumerate(actual_result.detections):
      expected_prediction = expect_result.detections[i]
      self.assertEqual(
          prediction.language_code,
          expected_prediction.language_code,
      )
      self.assertAlmostEqual(
          prediction.probability,
          expected_prediction.probability,
          delta=_TOLERANCE,
      )

  def test_create_from_file_succeeds_with_valid_model_path(self):
    # Creates with default option and valid model file successfully.
    with _LanguageDetector.create_from_model_path(self.model_path) as detector:
      self.assertIsInstance(detector, _LanguageDetector)

  def test_create_from_options_succeeds_with_valid_model_path(self):
    # Creates with options containing model file successfully.
    base_options = _BaseOptions(model_asset_path=self.model_path)
    options = _LanguageDetectorOptions(base_options=base_options)
    with _LanguageDetector.create_from_options(options) as detector:
      self.assertIsInstance(detector, _LanguageDetector)

  def test_create_from_options_fails_with_invalid_model_path(self):
    with self.assertRaises(FileNotFoundError):
      base_options = _BaseOptions(
          model_asset_path="/path/to/invalid/model.tflite"
      )
      options = _LanguageDetectorOptions(base_options=base_options)
      _LanguageDetector.create_from_options(options)

  def test_create_from_options_succeeds_with_valid_model_content(self):
    # Creates with options containing model content successfully.
    with open(self.model_path, "rb") as f:
      base_options = _BaseOptions(model_asset_buffer=f.read())
      options = _LanguageDetectorOptions(base_options=base_options)
      detector = _LanguageDetector.create_from_options(options)
      self.assertIsInstance(detector, _LanguageDetector)

  @parameterized.parameters(
      (ModelFileType.FILE_NAME, _EN_TEXT, _EN_EXPECTED_RESULT),
      (ModelFileType.FILE_CONTENT, _EN_TEXT, _EN_EXPECTED_RESULT),
      (ModelFileType.FILE_NAME, _FR_TEXT, _FR_EXPECTED_RESULT),
      (ModelFileType.FILE_CONTENT, _FR_TEXT, _FR_EXPECTED_RESULT),
      (ModelFileType.FILE_NAME, _RU_TEXT, _RU_EXPECTED_RESULT),
      (ModelFileType.FILE_CONTENT, _RU_TEXT, _RU_EXPECTED_RESULT),
      (ModelFileType.FILE_NAME, _MIXED_TEXT, _MIXED_EXPECTED_RESULT),
      (ModelFileType.FILE_CONTENT, _MIXED_TEXT, _MIXED_EXPECTED_RESULT),
  )
  def test_detect(self, model_file_type, text, expected_result):
    # Creates detector.
    if model_file_type is ModelFileType.FILE_NAME:
      base_options = _BaseOptions(model_asset_path=self.model_path)
    elif model_file_type is ModelFileType.FILE_CONTENT:
      with open(self.model_path, "rb") as f:
        model_content = f.read()
      base_options = _BaseOptions(model_asset_buffer=model_content)
    else:
      # Should never happen
      raise ValueError("model_file_type is invalid.")

    options = _LanguageDetectorOptions(
        base_options=base_options, score_threshold=_SCORE_THRESHOLD
    )
    detector = _LanguageDetector.create_from_options(options)

    # Performs language detection on the input.
    text_result = detector.detect(text)
    # Comparing results.
    self._expect_language_detector_result_correct(text_result, expected_result)
    # Closes the detector explicitly when the detector is not used in
    # a context.
    detector.close()

  @parameterized.parameters(
      (ModelFileType.FILE_NAME, _EN_TEXT, _EN_EXPECTED_RESULT),
      (ModelFileType.FILE_NAME, _FR_TEXT, _FR_EXPECTED_RESULT),
      (ModelFileType.FILE_NAME, _RU_TEXT, _RU_EXPECTED_RESULT),
      (ModelFileType.FILE_CONTENT, _MIXED_TEXT, _MIXED_EXPECTED_RESULT),
  )
  def test_detect_in_context(self, model_file_type, text, expected_result):
    # Creates detector.
    if model_file_type is ModelFileType.FILE_NAME:
      base_options = _BaseOptions(model_asset_path=self.model_path)
    elif model_file_type is ModelFileType.FILE_CONTENT:
      with open(self.model_path, "rb") as f:
        model_content = f.read()
      base_options = _BaseOptions(model_asset_buffer=model_content)
    else:
      # Should never happen
      raise ValueError("model_file_type is invalid.")

    options = _LanguageDetectorOptions(
        base_options=base_options, score_threshold=_SCORE_THRESHOLD
    )
    with _LanguageDetector.create_from_options(options) as detector:
      # Performs language detection on the input.
      text_result = detector.detect(text)
      # Comparing results.
      self._expect_language_detector_result_correct(
          text_result, expected_result
      )

  def test_allowlist_option(self):
    # Creates detector.
    base_options = _BaseOptions(model_asset_path=self.model_path)
    options = _LanguageDetectorOptions(
        base_options=base_options,
        score_threshold=_SCORE_THRESHOLD,
        category_allowlist=["ja"],
    )
    with _LanguageDetector.create_from_options(options) as detector:
      # Performs language detection on the input.
      text_result = detector.detect(_MIXED_TEXT)
      # Comparing results.
      expected_result = LanguageDetectorResult(
          [LanguageDetectorPrediction("ja", 0.481617)]
      )
      self._expect_language_detector_result_correct(
          text_result, expected_result
      )

  def test_denylist_option(self):
    # Creates detector.
    base_options = _BaseOptions(model_asset_path=self.model_path)
    options = _LanguageDetectorOptions(
        base_options=base_options,
        score_threshold=_SCORE_THRESHOLD,
        category_denylist=["ja"],
    )
    with _LanguageDetector.create_from_options(options) as detector:
      # Performs language detection on the input.
      text_result = detector.detect(_MIXED_TEXT)
      # Comparing results.
      expected_result = LanguageDetectorResult(
          [LanguageDetectorPrediction("zh", 0.505424)]
      )
      self._expect_language_detector_result_correct(
          text_result, expected_result
      )


if __name__ == "__main__":
  absltest.main()
