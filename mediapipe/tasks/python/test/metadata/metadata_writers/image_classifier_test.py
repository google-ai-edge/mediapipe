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
"""Tests for metadata_writer.image_classifier."""

import os

from absl.testing import absltest
from absl.testing import parameterized

from mediapipe.tasks.metadata import metadata_schema_py_generated as metadata_fb
from mediapipe.tasks.python.metadata import metadata
from mediapipe.tasks.python.metadata.metadata_writers import image_classifier
from mediapipe.tasks.python.metadata.metadata_writers import metadata_writer
from mediapipe.tasks.python.test import test_utils

_TEST_DATA_DIR = "mediapipe/tasks/testdata/metadata"
_FLOAT_MODEL = test_utils.get_test_data_path(
    os.path.join(_TEST_DATA_DIR,
                 "mobilenet_v2_1.0_224_without_metadata.tflite"))
_QUANT_MODEL = test_utils.get_test_data_path(
    os.path.join(_TEST_DATA_DIR,
                 "mobilenet_v2_1.0_224_quant_without_metadata.tflite"))
_LABEL_FILE = test_utils.get_test_data_path(
    os.path.join(_TEST_DATA_DIR, "labels.txt"))
_SCORE_CALIBRATION_FILE = test_utils.get_test_data_path(
    os.path.join(_TEST_DATA_DIR, "score_calibration.txt"))
_SCORE_CALIBRATION_FILENAME = "score_calibration.txt"
_DEFAULT_SCORE_CALIBRATION_VALUE = 0.2
_NORM_MEAN = 127.5
_NORM_STD = 127.5
_FLOAT_JSON = test_utils.get_test_data_path(
    os.path.join(_TEST_DATA_DIR, "mobilenet_v2_1.0_224.json"))
_QUANT_JSON = test_utils.get_test_data_path(
    os.path.join(_TEST_DATA_DIR, "mobilenet_v2_1.0_224_quant.json"))


class ImageClassifierTest(parameterized.TestCase):

  @parameterized.named_parameters(
      {
          "testcase_name": "test_float_model",
          "model_file": _FLOAT_MODEL,
          "golden_json": _FLOAT_JSON
      }, {
          "testcase_name": "test_quant_model",
          "model_file": _QUANT_MODEL,
          "golden_json": _QUANT_JSON
      })
  def test_write_metadata(self, model_file: str, golden_json: str):
    with open(model_file, "rb") as f:
      model_buffer = f.read()
    writer = image_classifier.MetadataWriter.create(
        model_buffer, [_NORM_MEAN], [_NORM_STD],
        labels=metadata_writer.Labels().add_from_file(_LABEL_FILE),
        score_calibration=metadata_writer.ScoreCalibration.create_from_file(
            metadata_fb.ScoreTransformationType.LOG, _SCORE_CALIBRATION_FILE,
            _DEFAULT_SCORE_CALIBRATION_VALUE))
    tflite_content, metadata_json = writer.populate()

    with open(golden_json, "r") as f:
      expected_json = f.read()
    self.assertEqual(metadata_json, expected_json)

    displayer = metadata.MetadataDisplayer.with_model_buffer(tflite_content)
    file_buffer = displayer.get_associated_file_buffer(
        _SCORE_CALIBRATION_FILENAME)
    with open(_SCORE_CALIBRATION_FILE, "rb") as f:
      expected_file_buffer = f.read()
    self.assertEqual(file_buffer, expected_file_buffer)


if __name__ == "__main__":
  absltest.main()
