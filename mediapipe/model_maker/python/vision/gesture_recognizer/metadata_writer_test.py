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
"""Tests for metadata_writer."""

import os
import zipfile

import tensorflow as tf

from mediapipe.model_maker.python.vision.gesture_recognizer import metadata_writer
from mediapipe.tasks.python.metadata.metadata_writers import metadata_writer as base_metadata_writer
from mediapipe.tasks.python.test import test_utils

_TEST_DATA_DIR = "mediapipe/model_maker/python/vision/gesture_recognizer/testdata/metadata"

_EXPECTED_JSON = test_utils.get_test_data_path(
    os.path.join(_TEST_DATA_DIR, "custom_gesture_classifier_meta.json"))
_CUSTOM_GESTURE_CLASSIFIER_PATH = test_utils.get_test_data_path(
    os.path.join(_TEST_DATA_DIR, "custom_gesture_classifier.tflite"))


class MetadataWriterTest(tf.test.TestCase):

  def test_hand_landmarker_metadata_writer(self):
    # Use dummy model buffer for unit test only.
    hand_detector_model_buffer = b"\x11\x12"
    hand_landmarks_detector_model_buffer = b"\x22"
    writer = metadata_writer.HandLandmarkerMetadataWriter(
        hand_detector_model_buffer, hand_landmarks_detector_model_buffer)
    model_bundle_content = writer.populate()
    model_bundle_filepath = os.path.join(self.get_temp_dir(),
                                         "hand_landmarker.task")
    with open(model_bundle_filepath, "wb") as f:
      f.write(model_bundle_content)

    with zipfile.ZipFile(model_bundle_filepath) as zf:
      self.assertEqual(
          set(zf.namelist()),
          set(["hand_landmarks_detector.tflite", "hand_detector.tflite"]))

  def test_write_metadata_and_create_model_asset_bundle_successful(self):
    # Use dummy model buffer for unit test only.
    hand_detector_model_buffer = b"\x11\x12"
    hand_landmarks_detector_model_buffer = b"\x22"
    gesture_embedder_model_buffer = b"\x33"
    canned_gesture_classifier_model_buffer = b"\x44"
    custom_gesture_classifier_metadata_writer = metadata_writer.GestureClassifierOptions(
        model_buffer=metadata_writer.read_file(_CUSTOM_GESTURE_CLASSIFIER_PATH),
        labels=base_metadata_writer.Labels().add(
            ["None", "Paper", "Rock", "Scissors"]),
        score_thresholding=base_metadata_writer.ScoreThresholding(
            global_score_threshold=0.5))
    writer = metadata_writer.MetadataWriter.create(
        hand_detector_model_buffer, hand_landmarks_detector_model_buffer,
        gesture_embedder_model_buffer, canned_gesture_classifier_model_buffer,
        custom_gesture_classifier_metadata_writer)
    model_bundle_content, metadata_json = writer.populate()
    with open(_EXPECTED_JSON, "r") as f:
      expected_json = f.read()
    self.assertEqual(metadata_json, expected_json)

    # Checks the top-level model bundle can be extracted successfully.
    model_bundle_filepath = os.path.join(self.get_temp_dir(),
                                         "gesture_recognition.task")

    with open(model_bundle_filepath, "wb") as f:
      f.write(model_bundle_content)

    with zipfile.ZipFile(model_bundle_filepath) as zf:
      self.assertEqual(
          set(zf.namelist()),
          set(["hand_landmarker.task", "hand_gesture_recognizer.task"]))
      zf.extractall(self.get_temp_dir())

    # Checks the model bundles for sub-task can be extracted successfully.
    hand_landmarker_bundle_filepath = os.path.join(self.get_temp_dir(),
                                                   "hand_landmarker.task")
    with zipfile.ZipFile(hand_landmarker_bundle_filepath) as zf:
      self.assertEqual(
          set(zf.namelist()),
          set(["hand_landmarks_detector.tflite", "hand_detector.tflite"]))

    hand_gesture_recognizer_bundle_filepath = os.path.join(
        self.get_temp_dir(), "hand_gesture_recognizer.task")
    with zipfile.ZipFile(hand_gesture_recognizer_bundle_filepath) as zf:
      self.assertEqual(
          set(zf.namelist()),
          set([
              "canned_gesture_classifier.tflite",
              "custom_gesture_classifier.tflite", "gesture_embedder.tflite"
          ]))


if __name__ == "__main__":
  tf.test.main()
