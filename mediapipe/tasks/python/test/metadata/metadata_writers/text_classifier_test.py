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
# ==============================================================================
"""Tests for metadata_writer.text_classifier."""

from absl.testing import absltest

from mediapipe.tasks.python.metadata.metadata_writers import metadata_writer
from mediapipe.tasks.python.metadata.metadata_writers import text_classifier
from mediapipe.tasks.python.test import test_utils

_TEST_DIR = "mediapipe/tasks/testdata/metadata/"
_MODEL = test_utils.get_test_data_path(_TEST_DIR + "movie_review.tflite")
_LABEL_FILE = test_utils.get_test_data_path(_TEST_DIR +
                                            "movie_review_labels.txt")
_VOCAB_FILE = test_utils.get_test_data_path(_TEST_DIR + "regex_vocab.txt")
_DELIM_REGEX_PATTERN = r"[^\w\']+"
_JSON_FILE = test_utils.get_test_data_path("movie_review.json")


class TextClassifierTest(absltest.TestCase):

  def test_write_metadata(self,):
    with open(_MODEL, "rb") as f:
      model_buffer = f.read()
    writer = text_classifier.MetadataWriter.create_for_regex_model(
        model_buffer,
        regex_tokenizer=metadata_writer.RegexTokenizer(
            delim_regex_pattern=_DELIM_REGEX_PATTERN,
            vocab_file_path=_VOCAB_FILE),
        labels=metadata_writer.Labels().add_from_file(_LABEL_FILE))
    _, metadata_json = writer.populate()

    with open(_JSON_FILE, "r") as f:
      expected_json = f.read()
    self.assertEqual(metadata_json, expected_json)


if __name__ == "__main__":
  absltest.main()
