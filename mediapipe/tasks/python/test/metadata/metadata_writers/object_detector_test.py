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
# ==============================================================================
"""Tests for metadata_writer.object_detector."""

import csv
import os

from absl.testing import absltest
from absl.testing import parameterized

from mediapipe.tasks.metadata import metadata_schema_py_generated as metadata_fb
from mediapipe.tasks.python.metadata import metadata
from mediapipe.tasks.python.metadata.metadata_writers import metadata_info
from mediapipe.tasks.python.metadata.metadata_writers import metadata_writer
from mediapipe.tasks.python.metadata.metadata_writers import object_detector
from mediapipe.tasks.python.test import test_utils

_TEST_DATA_DIR = "mediapipe/tasks/testdata/metadata"
_LABEL_FILE = test_utils.get_test_data_path(
    os.path.join(_TEST_DATA_DIR, "labelmap.txt")
)
_LABEL_FILE_NAME = "labels.txt"
_NORM_MEAN = 127.5
_NORM_STD = 127.5

_MODEL_COCO = test_utils.get_test_data_path(
    os.path.join(
        _TEST_DATA_DIR,
        "coco_ssd_mobilenet_v1_1.0_quant_2018_06_29_no_metadata.tflite",
    )
)
_SCORE_CALIBRATION_FILE = test_utils.get_test_data_path(
    os.path.join(_TEST_DATA_DIR, "score_calibration.csv")
)
_SCORE_CALIBRATION_FILENAME = "score_calibration.txt"
_SCORE_CALIBRATION_DEFAULT_SCORE = 0.2
_JSON_FOR_SCORE_CALIBRATION = test_utils.get_test_data_path(
    os.path.join(_TEST_DATA_DIR, "coco_ssd_mobilenet_v1_score_calibration.json")
)

_EFFICIENTDET_LITE0_ANCHORS_FILE = test_utils.get_test_data_path(
    os.path.join(_TEST_DATA_DIR, "efficientdet_lite0_fp16_no_nms_anchors.csv")
)


def read_ssd_anchors_from_csv(file_path):
  with open(file_path, "r") as anchors_file:
    csv_reader = csv.reader(anchors_file, delimiter=",")
    parameters = []
    for row in csv_reader:
      if not row:
        parameters.append(None)
        continue
      if len(row) != 4:
        raise ValueError(
            "Expected empty lines or 4 parameters per line in "
            f"anchors file, but got {len(row)}."
        )
      parameters.append(row)
  anchors = []
  for parameter in parameters:
    anchors.append(
        object_detector.FixedAnchor(
            x_center=float(parameter[1]),
            y_center=float(parameter[0]),
            width=float(parameter[3]),
            height=float(parameter[2]),
        )
    )
  return object_detector.SsdAnchorsOptions(
      fixed_anchors_schema=object_detector.FixedAnchorsSchema(anchors)
  )


class MetadataWriterTest(parameterized.TestCase, absltest.TestCase):

  @parameterized.parameters(
      "ssd_mobilenet_v1_no_metadata",
      "efficientdet_lite0_v1",
  )
  def test_create_should_succeed(self, model_name):
    model_path = test_utils.get_test_data_path(
        os.path.join(_TEST_DATA_DIR, model_name + ".tflite")
    )
    with open(model_path, "rb") as f:
      model_buffer = f.read()
    writer = (
        object_detector.MetadataWriter.create_for_models_with_nms(
            model_buffer,
            [_NORM_MEAN],
            [_NORM_STD],
            labels=metadata_writer.Labels().add_from_file(_LABEL_FILE),
        )
    )
    _, metadata_json = writer.populate()
    expected_json_path = test_utils.get_test_data_path(
        os.path.join(_TEST_DATA_DIR, model_name + ".json")
    )
    with open(expected_json_path, "r") as f:
      expected_json = f.read().strip()
    self.assertEqual(metadata_json, expected_json)

  def test_create_with_score_calibration_should_succeed(self):
    with open(_MODEL_COCO, "rb") as f:
      model_buffer = f.read()
    writer = (
        object_detector.MetadataWriter.create_for_models_with_nms(
            model_buffer,
            [_NORM_MEAN],
            [_NORM_STD],
            labels=metadata_writer.Labels().add_from_file(_LABEL_FILE),
            score_calibration=metadata_writer.ScoreCalibration.create_from_file(
                metadata_fb.ScoreTransformationType.INVERSE_LOGISTIC,
                _SCORE_CALIBRATION_FILE,
                _SCORE_CALIBRATION_DEFAULT_SCORE,
            ),
        )
    )
    tflite_content, metadata_json = writer.populate()
    with open(_JSON_FOR_SCORE_CALIBRATION, "r") as f:
      expected_json = f.read().strip()
    self.assertEqual(metadata_json, expected_json)

    displayer = metadata.MetadataDisplayer.with_model_buffer(tflite_content)
    calibration_file_buffer = displayer.get_associated_file_buffer(
        _SCORE_CALIBRATION_FILENAME
    )
    with open(_SCORE_CALIBRATION_FILE, "rb") as f:
      expected_calibration_file_buffer = f.read()
    self.assertEqual(calibration_file_buffer, expected_calibration_file_buffer)

    label_file_buffer = displayer.get_associated_file_buffer(_LABEL_FILE_NAME)
    with open(_LABEL_FILE, "rb") as f:
      expected_labelfile_buffer = f.read()
    self.assertEqual(label_file_buffer, expected_labelfile_buffer)


if __name__ == "__main__":
  absltest.main()
