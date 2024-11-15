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
"""Tests for text classifier."""

import enum
import os

from absl.testing import absltest
from absl.testing import parameterized

from mediapipe.tasks.python.components.containers import category
from mediapipe.tasks.python.components.containers import classification_result as classification_result_module
from mediapipe.tasks.python.core import base_options as base_options_module
from mediapipe.tasks.python.test import test_utils
from mediapipe.tasks.python.text import text_classifier

TextClassifierResult = classification_result_module.ClassificationResult
_BaseOptions = base_options_module.BaseOptions
_Category = category.Category
_Classifications = classification_result_module.Classifications
_TextClassifier = text_classifier.TextClassifier
_TextClassifierOptions = text_classifier.TextClassifierOptions

_BERT_MODEL_FILE = 'bert_text_classifier.tflite'
_REGEX_MODEL_FILE = 'test_model_text_classifier_with_regex_tokenizer.tflite'
_TEST_DATA_DIR = 'mediapipe/tasks/testdata/text'

_NEGATIVE_TEXT = 'What a waste of my time.'
_POSITIVE_TEXT = ('This is the best movie I’ve seen in recent years.'
                  'Strongly recommend it!')

_BERT_NEGATIVE_RESULTS = TextClassifierResult(
    classifications=[
        _Classifications(
            categories=[
                _Category(
                    index=0,
                    score=0.9995,
                    display_name='',
                    category_name='negative'),
                _Category(
                    index=1,
                    score=0.0005,
                    display_name='',
                    category_name='positive')
            ],
            head_index=0,
            head_name='probability')
    ],
    timestamp_ms=0)
_BERT_POSITIVE_RESULTS = TextClassifierResult(
    classifications=[
        _Classifications(
            categories=[
                _Category(
                    index=1,
                    score=0.9994,
                    display_name='',
                    category_name='positive',
                ),
                _Category(
                    index=0,
                    score=0.0006,
                    display_name='',
                    category_name='negative',
                ),
            ],
            head_index=0,
            head_name='probability',
        )
    ],
    timestamp_ms=0,
)
_REGEX_NEGATIVE_RESULTS = TextClassifierResult(
    classifications=[
        _Classifications(
            categories=[
                _Category(
                    index=0,
                    score=0.81313,
                    display_name='',
                    category_name='Negative'),
                _Category(
                    index=1,
                    score=0.1868704,
                    display_name='',
                    category_name='Positive')
            ],
            head_index=0,
            head_name='probability')
    ],
    timestamp_ms=0)
_REGEX_POSITIVE_RESULTS = TextClassifierResult(
    classifications=[
        _Classifications(
            categories=[
                _Category(
                    index=1,
                    score=0.5134273,
                    display_name='',
                    category_name='Positive'),
                _Category(
                    index=0,
                    score=0.486573,
                    display_name='',
                    category_name='Negative')
            ],
            head_index=0,
            head_name='probability')
    ],
    timestamp_ms=0)


class ModelFileType(enum.Enum):
  FILE_CONTENT = 1
  FILE_NAME = 2


class TextClassifierTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.model_path = test_utils.get_test_data_path(
        os.path.join(_TEST_DATA_DIR, _BERT_MODEL_FILE))

  def test_create_from_file_succeeds_with_valid_model_path(self):
    # Creates with default option and valid model file successfully.
    with _TextClassifier.create_from_model_path(self.model_path) as classifier:
      self.assertIsInstance(classifier, _TextClassifier)

  def test_create_from_options_succeeds_with_valid_model_path(self):
    # Creates with options containing model file successfully.
    base_options = _BaseOptions(model_asset_path=self.model_path)
    options = _TextClassifierOptions(base_options=base_options)
    with _TextClassifier.create_from_options(options) as classifier:
      self.assertIsInstance(classifier, _TextClassifier)

  def test_create_from_options_fails_with_invalid_model_path(self):
    with self.assertRaisesRegex(
        RuntimeError, 'Unable to open file at /path/to/invalid/model.tflite'):
      base_options = _BaseOptions(
          model_asset_path='/path/to/invalid/model.tflite')
      options = _TextClassifierOptions(base_options=base_options)
      _TextClassifier.create_from_options(options)

  def test_create_from_options_succeeds_with_valid_model_content(self):
    # Creates with options containing model content successfully.
    with open(self.model_path, 'rb') as f:
      base_options = _BaseOptions(model_asset_buffer=f.read())
      options = _TextClassifierOptions(base_options=base_options)
      classifier = _TextClassifier.create_from_options(options)
      self.assertIsInstance(classifier, _TextClassifier)

  @parameterized.parameters(
      (ModelFileType.FILE_NAME, _BERT_MODEL_FILE, _NEGATIVE_TEXT,
       _BERT_NEGATIVE_RESULTS), (ModelFileType.FILE_CONTENT, _BERT_MODEL_FILE,
                                 _NEGATIVE_TEXT, _BERT_NEGATIVE_RESULTS),
      (ModelFileType.FILE_NAME, _BERT_MODEL_FILE, _POSITIVE_TEXT,
       _BERT_POSITIVE_RESULTS), (ModelFileType.FILE_CONTENT, _BERT_MODEL_FILE,
                                 _POSITIVE_TEXT, _BERT_POSITIVE_RESULTS),
      (ModelFileType.FILE_NAME, _REGEX_MODEL_FILE, _NEGATIVE_TEXT,
       _REGEX_NEGATIVE_RESULTS), (ModelFileType.FILE_CONTENT, _REGEX_MODEL_FILE,
                                  _NEGATIVE_TEXT, _REGEX_NEGATIVE_RESULTS),
      (ModelFileType.FILE_NAME, _REGEX_MODEL_FILE, _POSITIVE_TEXT,
       _REGEX_POSITIVE_RESULTS), (ModelFileType.FILE_CONTENT, _REGEX_MODEL_FILE,
                                  _POSITIVE_TEXT, _REGEX_POSITIVE_RESULTS))
  def test_classify(self, model_file_type, model_name, text,
                    expected_classification_result):
    # Creates classifier.
    model_path = test_utils.get_test_data_path(
        os.path.join(_TEST_DATA_DIR, model_name))
    if model_file_type is ModelFileType.FILE_NAME:
      base_options = _BaseOptions(model_asset_path=model_path)
    elif model_file_type is ModelFileType.FILE_CONTENT:
      with open(model_path, 'rb') as f:
        model_content = f.read()
      base_options = _BaseOptions(model_asset_buffer=model_content)
    else:
      # Should never happen
      raise ValueError('model_file_type is invalid.')

    options = _TextClassifierOptions(base_options=base_options)
    classifier = _TextClassifier.create_from_options(options)

    # Performs text classification on the input.
    text_result = classifier.classify(text)
    # Comparing results.
    test_utils.assert_proto_equals(self, text_result.to_pb2(),
                                   expected_classification_result.to_pb2())
    # Closes the classifier explicitly when the classifier is not used in
    # a context.
    classifier.close()

  @parameterized.parameters((ModelFileType.FILE_NAME, _BERT_MODEL_FILE,
                             _NEGATIVE_TEXT, _BERT_NEGATIVE_RESULTS),
                            (ModelFileType.FILE_CONTENT, _BERT_MODEL_FILE,
                             _NEGATIVE_TEXT, _BERT_NEGATIVE_RESULTS))
  def test_classify_in_context(self, model_file_type, model_name, text,
                               expected_classification_result):
    # Creates classifier.
    model_path = test_utils.get_test_data_path(
        os.path.join(_TEST_DATA_DIR, model_name))
    if model_file_type is ModelFileType.FILE_NAME:
      base_options = _BaseOptions(model_asset_path=model_path)
    elif model_file_type is ModelFileType.FILE_CONTENT:
      with open(model_path, 'rb') as f:
        model_content = f.read()
      base_options = _BaseOptions(model_asset_buffer=model_content)
    else:
      # Should never happen
      raise ValueError('model_file_type is invalid.')

    options = _TextClassifierOptions(base_options=base_options)

    with _TextClassifier.create_from_options(options) as classifier:
      # Performs text classification on the input.
      text_result = classifier.classify(text)
      # Comparing results.
      test_utils.assert_proto_equals(self, text_result.to_pb2(),
                                     expected_classification_result.to_pb2())


if __name__ == '__main__':
  absltest.main()
