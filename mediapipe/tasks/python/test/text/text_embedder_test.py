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
"""Tests for text embedder."""

import enum
import os

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np

from mediapipe.tasks.python.components.containers import embedding_result as embedding_result_module
from mediapipe.tasks.python.core import base_options as base_options_module
from mediapipe.tasks.python.test import test_utils
from mediapipe.tasks.python.text import text_embedder

_BaseOptions = base_options_module.BaseOptions
_Embedding = embedding_result_module.Embedding
_TextEmbedder = text_embedder.TextEmbedder
_TextEmbedderOptions = text_embedder.TextEmbedderOptions

_BERT_MODEL_FILE = 'mobilebert_embedding_with_metadata.tflite'
_REGEX_MODEL_FILE = 'regex_one_embedding_with_metadata.tflite'
_TEST_DATA_DIR = 'mediapipe/tasks/testdata/text'
# Tolerance for embedding vector coordinate values.
_EPSILON = 1e-4
# Tolerance for cosine similarity evaluation.
_SIMILARITY_TOLERANCE = 1e-6


class ModelFileType(enum.Enum):
  FILE_CONTENT = 1
  FILE_NAME = 2


class TextEmbedderTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.model_path = test_utils.get_test_data_path(
        os.path.join(_TEST_DATA_DIR, _BERT_MODEL_FILE))

  def test_create_from_file_succeeds_with_valid_model_path(self):
    # Creates with default option and valid model file successfully.
    with _TextEmbedder.create_from_model_path(self.model_path) as embedder:
      self.assertIsInstance(embedder, _TextEmbedder)

  def test_create_from_options_succeeds_with_valid_model_path(self):
    # Creates with options containing model file successfully.
    base_options = _BaseOptions(model_asset_path=self.model_path)
    options = _TextEmbedderOptions(base_options=base_options)
    with _TextEmbedder.create_from_options(options) as embedder:
      self.assertIsInstance(embedder, _TextEmbedder)

  def test_create_from_options_fails_with_invalid_model_path(self):
    with self.assertRaisesRegex(
        RuntimeError, 'Unable to open file at /path/to/invalid/model.tflite'):
      base_options = _BaseOptions(
          model_asset_path='/path/to/invalid/model.tflite')
      options = _TextEmbedderOptions(base_options=base_options)
      _TextEmbedder.create_from_options(options)

  def test_create_from_options_succeeds_with_valid_model_content(self):
    # Creates with options containing model content successfully.
    with open(self.model_path, 'rb') as f:
      base_options = _BaseOptions(model_asset_buffer=f.read())
      options = _TextEmbedderOptions(base_options=base_options)
      embedder = _TextEmbedder.create_from_options(options)
      self.assertIsInstance(embedder, _TextEmbedder)

  def _check_embedding_value(self, result, expected_first_value):
    # Check embedding first value.
    self.assertAlmostEqual(
        result.embeddings[0].embedding[0], expected_first_value, delta=_EPSILON)

  def _check_embedding_size(self, result, quantize, expected_embedding_size):
    # Check embedding size.
    self.assertLen(result.embeddings, 1)
    embedding_result = result.embeddings[0]
    self.assertLen(embedding_result.embedding, expected_embedding_size)
    if quantize:
      self.assertEqual(embedding_result.embedding.dtype, np.uint8)
    else:
      self.assertEqual(embedding_result.embedding.dtype, float)

  def _check_cosine_similarity(self, result0, result1, expected_similarity):
    # Checks cosine similarity.
    similarity = _TextEmbedder.cosine_similarity(result0.embeddings[0],
                                                 result1.embeddings[0])
    self.assertAlmostEqual(
        similarity, expected_similarity, delta=_SIMILARITY_TOLERANCE)

  @parameterized.parameters(
      (False, False, _BERT_MODEL_FILE, ModelFileType.FILE_NAME, 0.969514, 512,
       (19.9016, 22.626251)),
      (True, False, _BERT_MODEL_FILE, ModelFileType.FILE_NAME, 0.969514, 512,
       (0.0585837, 0.0723035)),
      (False, False, _REGEX_MODEL_FILE, ModelFileType.FILE_NAME, 0.999937, 16,
       (0.0309356, 0.0312863)),
      (True, False, _REGEX_MODEL_FILE, ModelFileType.FILE_CONTENT, 0.999937, 16,
       (0.549632, 0.552879)),
  )
  def test_embed(self, l2_normalize, quantize, model_name, model_file_type,
                 expected_similarity, expected_size, expected_first_values):
    # Creates embedder.
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

    options = _TextEmbedderOptions(
        base_options=base_options, l2_normalize=l2_normalize, quantize=quantize)
    embedder = _TextEmbedder.create_from_options(options)

    # Extracts both embeddings.
    positive_text0 = "it's a charming and often affecting journey"
    positive_text1 = 'what a great and fantastic trip'

    result0 = embedder.embed(positive_text0)
    result1 = embedder.embed(positive_text1)

    # Checks embeddings and cosine similarity.
    expected_result0_value, expected_result1_value = expected_first_values
    self._check_embedding_size(result0, quantize, expected_size)
    self._check_embedding_size(result1, quantize, expected_size)
    self._check_embedding_value(result0, expected_result0_value)
    self._check_embedding_value(result1, expected_result1_value)
    self._check_cosine_similarity(result0, result1, expected_similarity)
    # Closes the embedder explicitly when the embedder is not used in
    # a context.
    embedder.close()

  @parameterized.parameters(
      (False, False, _BERT_MODEL_FILE, ModelFileType.FILE_NAME, 0.969514, 512,
       (19.9016, 22.626251)),
      (True, False, _BERT_MODEL_FILE, ModelFileType.FILE_NAME, 0.969514, 512,
       (0.0585837, 0.0723035)),
      (False, False, _REGEX_MODEL_FILE, ModelFileType.FILE_NAME, 0.999937, 16,
       (0.0309356, 0.0312863)),
      (True, False, _REGEX_MODEL_FILE, ModelFileType.FILE_CONTENT, 0.999937, 16,
       (0.549632, 0.552879)),
  )
  def test_embed_in_context(self, l2_normalize, quantize, model_name,
                            model_file_type, expected_similarity, expected_size,
                            expected_first_values):
    # Creates embedder.
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

    options = _TextEmbedderOptions(
        base_options=base_options, l2_normalize=l2_normalize, quantize=quantize)
    with _TextEmbedder.create_from_options(options) as embedder:
      # Extracts both embeddings.
      positive_text0 = "it's a charming and often affecting journey"
      positive_text1 = 'what a great and fantastic trip'

      result0 = embedder.embed(positive_text0)
      result1 = embedder.embed(positive_text1)

      # Checks embeddings and cosine similarity.
      expected_result0_value, expected_result1_value = expected_first_values
      self._check_embedding_size(result0, quantize, expected_size)
      self._check_embedding_size(result1, quantize, expected_size)
      self._check_embedding_value(result0, expected_result0_value)
      self._check_embedding_value(result1, expected_result1_value)
      self._check_cosine_similarity(result0, result1, expected_similarity)


if __name__ == '__main__':
  absltest.main()
