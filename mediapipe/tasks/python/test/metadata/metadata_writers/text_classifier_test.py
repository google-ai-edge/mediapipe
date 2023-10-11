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
# ==============================================================================
"""Tests for metadata_writer.text_classifier."""

from absl.testing import absltest

from mediapipe.tasks.python.metadata.metadata_writers import metadata_writer
from mediapipe.tasks.python.metadata.metadata_writers import text_classifier
from mediapipe.tasks.python.test import test_utils

_TEST_DIR = "mediapipe/tasks/testdata/metadata/"
_REGEX_MODEL = test_utils.get_test_data_path(_TEST_DIR + "movie_review.tflite")
_LABEL_FILE = test_utils.get_test_data_path(_TEST_DIR +
                                            "movie_review_labels.txt")
_REGEX_VOCAB_FILE = test_utils.get_test_data_path(_TEST_DIR + "regex_vocab.txt")
_DELIM_REGEX_PATTERN = r"[^\w\']+"
_REGEX_JSON_FILE = test_utils.get_test_data_path("movie_review.json")

_BERT_MODEL = test_utils.get_test_data_path(
    _TEST_DIR + "bert_text_classifier_no_metadata.tflite")
_BERT_VOCAB_FILE = test_utils.get_test_data_path(_TEST_DIR +
                                                 "mobilebert_vocab.txt")
_SP_MODEL_FILE = test_utils.get_test_data_path(_TEST_DIR + "30k-clean.model")
_BERT_JSON_FILE = test_utils.get_test_data_path(
    _TEST_DIR + "bert_text_classifier_with_bert_tokenizer.json")
_SENTENCE_PIECE_JSON_FILE = test_utils.get_test_data_path(
    _TEST_DIR + "bert_text_classifier_with_sentence_piece.json")


class TextClassifierTest(absltest.TestCase):

  def test_write_metadata_for_regex_model(self):
    with open(_REGEX_MODEL, "rb") as f:
      model_buffer = f.read()
    writer = text_classifier.MetadataWriter.create_for_regex_model(
        model_buffer,
        regex_tokenizer=metadata_writer.RegexTokenizer(
            delim_regex_pattern=_DELIM_REGEX_PATTERN,
            vocab_file_path=_REGEX_VOCAB_FILE),
        labels=metadata_writer.Labels().add_from_file(_LABEL_FILE))
    _, metadata_json = writer.populate()

    with open(_REGEX_JSON_FILE, "r") as f:
      expected_json = f.read()
    self.assertEqual(metadata_json, expected_json)

  def test_write_metadata_for_bert(self):
    with open(_BERT_MODEL, "rb") as f:
      model_buffer = f.read()
    writer = text_classifier.MetadataWriter.create_for_bert_model(
        model_buffer,
        tokenizer=metadata_writer.BertTokenizer(_BERT_VOCAB_FILE),
        labels=metadata_writer.Labels().add_from_file(_LABEL_FILE))
    _, metadata_json = writer.populate()

    with open(_BERT_JSON_FILE, "r") as f:
      expected_json = f.read()
    self.assertEqual(metadata_json, expected_json)

  def test_write_metadata_for_sentence_piece(self):
    with open(_BERT_MODEL, "rb") as f:
      model_buffer = f.read()
    writer = text_classifier.MetadataWriter.create_for_bert_model(
        model_buffer,
        tokenizer=metadata_writer.SentencePieceTokenizer(_SP_MODEL_FILE),
        labels=metadata_writer.Labels().add_from_file(_LABEL_FILE))
    _, metadata_json = writer.populate()

    with open(_SENTENCE_PIECE_JSON_FILE, "r") as f:
      expected_json = f.read()
    self.assertEqual(metadata_json, expected_json)


if __name__ == "__main__":
  absltest.main()
