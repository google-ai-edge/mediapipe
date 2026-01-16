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
"""Tests for metadata_writer.image_segmenter."""

import os

from absl.testing import absltest

from mediapipe.tasks.python.metadata import metadata
from mediapipe.tasks.python.metadata.metadata_writers import image_segmenter
from mediapipe.tasks.python.metadata.metadata_writers import metadata_writer
from mediapipe.tasks.python.test import test_utils

_TEST_DATA_DIR = "mediapipe/tasks/testdata/metadata"
_MODEL_FILE = test_utils.get_test_data_path(
    os.path.join(_TEST_DATA_DIR, "deeplabv3_without_metadata.tflite")
)
_LABEL_FILE_NAME = "labels.txt"
_LABEL_FILE = test_utils.get_test_data_path(
    os.path.join(_TEST_DATA_DIR, "segmenter_labelmap.txt")
)
_NORM_MEAN = 127.5
_NORM_STD = 127.5
_JSON_FILE = test_utils.get_test_data_path(
    os.path.join(_TEST_DATA_DIR, "deeplabv3.json")
)
_JSON_FILE_WITHOUT_LABELS = test_utils.get_test_data_path(
    os.path.join(_TEST_DATA_DIR, "deeplabv3_without_labels.json")
)
_JSON_FILE_WITH_ACTIVATION = test_utils.get_test_data_path(
    os.path.join(_TEST_DATA_DIR, "deeplabv3_with_activation.json")
)


class ImageSegmenterTest(absltest.TestCase):

  def test_write_metadata(self):
    with open(_MODEL_FILE, "rb") as f:
      model_buffer = f.read()
      writer = image_segmenter.MetadataWriter.create(
          bytearray(model_buffer),
          [_NORM_MEAN],
          [_NORM_STD],
          labels=metadata_writer.Labels().add_from_file(_LABEL_FILE),
      )
    tflite_content, metadata_json = writer.populate()
    with open(_JSON_FILE, "r") as f:
      expected_json = f.read().strip()
    self.assertEqual(metadata_json, expected_json)

    displayer = metadata.MetadataDisplayer.with_model_buffer(tflite_content)
    label_file_buffer = displayer.get_associated_file_buffer(_LABEL_FILE_NAME)
    with open(_LABEL_FILE, "rb") as f:
      expected_labelfile_buffer = f.read()
    self.assertEqual(label_file_buffer, expected_labelfile_buffer)

  def test_write_metadata_without_labels(self):
    with open(_MODEL_FILE, "rb") as f:
      model_buffer = f.read()
      writer = image_segmenter.MetadataWriter.create(
          bytearray(model_buffer),
          [_NORM_MEAN],
          [_NORM_STD],
      )
    _, metadata_json = writer.populate()
    with open(_JSON_FILE_WITHOUT_LABELS, "r") as f:
      expected_json = f.read().strip()
    self.assertEqual(metadata_json, expected_json)

  def test_write_metadata_with_activation(self):
    with open(_MODEL_FILE, "rb") as f:
      model_buffer = f.read()
      writer = image_segmenter.MetadataWriter.create(
          bytearray(model_buffer),
          [_NORM_MEAN],
          [_NORM_STD],
          activation=image_segmenter.Activation.SIGMOID,
      )
    _, metadata_json = writer.populate()
    with open(_JSON_FILE_WITH_ACTIVATION, "r") as f:
      expected_json = f.read().strip()
    self.assertEqual(metadata_json, expected_json)


if __name__ == "__main__":
  absltest.main()
