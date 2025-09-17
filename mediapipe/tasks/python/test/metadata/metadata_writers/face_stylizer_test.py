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
"""Tests for metadata_writer.face_stylizer."""

import os
import tempfile
import zipfile

from absl.testing import absltest
from absl.testing import parameterized

from mediapipe.tasks.python.metadata.metadata_writers import face_stylizer
from mediapipe.tasks.python.test import test_utils

_TEST_DATA_DIR = "mediapipe/tasks/testdata/metadata"
_NORM_MEAN = 0
_NORM_STD = 255
_TFLITE = test_utils.get_test_data_path(
    os.path.join(_TEST_DATA_DIR, "dummy_face_stylizer.tflite")
)
_EXPECTED_JSON = test_utils.get_test_data_path(
    os.path.join(_TEST_DATA_DIR, "face_stylizer.json")
)


class FaceStylizerTest(parameterized.TestCase):

  def test_write_metadata_and_create_model_asset_bundle_successful(self):
    # Use dummy model buffer for unit test only.
    with open(_TFLITE, "rb") as f:
      face_stylizer_model_buffer = f.read()
    face_detector_model_buffer = b"\x33\x44"
    face_landmarks_detector_model_buffer = b"\x55\x66"
    writer = face_stylizer.MetadataWriter.create(
        face_stylizer_model_buffer,
        face_detector_model_buffer,
        face_landmarks_detector_model_buffer,
        input_norm_mean=[_NORM_MEAN],
        input_norm_std=[_NORM_STD],
    )
    model_bundle_content, metadata_json = writer.populate()
    with open(_EXPECTED_JSON, "r") as f:
      expected_json = f.read()
    self.assertEqual(metadata_json, expected_json)

    temp_folder = tempfile.TemporaryDirectory()

    # Checks the model bundle can be extracted successfully.
    model_bundle_filepath = os.path.join(temp_folder.name, "face_stylizer.task")

    with open(model_bundle_filepath, "wb") as f:
      f.write(model_bundle_content)

    with zipfile.ZipFile(model_bundle_filepath) as zf:
      self.assertEqual(
          set(zf.namelist()),
          set([
              "face_detector.tflite",
              "face_landmarks_detector.tflite",
              "face_stylizer.tflite",
          ]),
      )
      zf.extractall(temp_folder.name)
    temp_folder.cleanup()


if __name__ == "__main__":
  absltest.main()
