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
"""Tests for gesture recognizer."""

import enum

from google.protobuf import text_format
from absl.testing import absltest
from absl.testing import parameterized

from mediapipe.python._framework_bindings import image as image_module
from mediapipe.tasks.cc.components.containers.proto import landmarks_detection_result_pb2
from mediapipe.tasks.python.components.containers import rect as rect_module
from mediapipe.tasks.python.components.containers import classification as classification_module
from mediapipe.tasks.python.components.containers import landmark as landmark_module
from mediapipe.tasks.python.components.containers import landmark_detection_result as landmark_detection_result_module
from mediapipe.tasks.python.core import base_options as base_options_module
from mediapipe.tasks.python.test import test_utils
from mediapipe.tasks.python.vision import gesture_recognizer
from mediapipe.tasks.python.vision.core import vision_task_running_mode as running_mode_module

_LandmarksDetectionResultProto = landmarks_detection_result_pb2.LandmarksDetectionResult
_BaseOptions = base_options_module.BaseOptions
_NormalizedRect = rect_module.NormalizedRect
_Classification = classification_module.Classification
_ClassificationList = classification_module.ClassificationList
_Landmark = landmark_module.Landmark
_LandmarkList = landmark_module.LandmarkList
_NormalizedLandmark = landmark_module.NormalizedLandmark
_NormalizedLandmarkList = landmark_module.NormalizedLandmarkList
_LandmarksDetectionResult = landmark_detection_result_module.LandmarksDetectionResult
_Image = image_module.Image
_GestureRecognizer = gesture_recognizer.GestureRecognizer
_GestureRecognizerOptions = gesture_recognizer.GestureRecognizerOptions
_GestureRecognitionResult = gesture_recognizer.GestureRecognitionResult
_RUNNING_MODE = running_mode_module.VisionTaskRunningMode

_GESTURE_RECOGNIZER_MODEL_FILE = 'gesture_recognizer.task'
_THUMB_UP_IMAGE = 'thumb_up.jpg'
_THUMB_UP_LANDMARKS = "thumb_up_landmarks.pbtxt"
_THUMB_UP_LABEL = "Thumb_Up"
_THUMB_UP_INDEX = 5
_LANDMARKS_ERROR_TOLERANCE = 0.03


def _get_expected_gesture_recognition_result(
    file_path: str, gesture_label: str, gesture_index: int
) -> _GestureRecognitionResult:
  landmarks_detection_result_file_path = test_utils.get_test_data_path(
    file_path)
  with open(landmarks_detection_result_file_path, "rb") as f:
    landmarks_detection_result_proto = _LandmarksDetectionResultProto()
    # # Use this if a .pb file is available.
    # landmarks_detection_result_proto.ParseFromString(f.read())
    text_format.Parse(f.read(), landmarks_detection_result_proto)
    landmarks_detection_result = _LandmarksDetectionResult.create_from_pb2(
        landmarks_detection_result_proto)
  gesture = _ClassificationList(
      classifications=[
        _Classification(label=gesture_label, index=gesture_index,
                        display_name='')
      ], tensor_index=0, tensor_name='')
  return _GestureRecognitionResult(
      gestures=[gesture],
      handedness=[landmarks_detection_result.classifications],
      hand_landmarks=[landmarks_detection_result.landmarks],
      hand_world_landmarks=[landmarks_detection_result.world_landmarks])


class ModelFileType(enum.Enum):
  FILE_CONTENT = 1
  FILE_NAME = 2


class GestureRecognizerTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.test_image = _Image.create_from_file(
        test_utils.get_test_data_path(_THUMB_UP_IMAGE))
    self.gesture_recognizer_model_path = test_utils.get_test_data_path(
        _GESTURE_RECOGNIZER_MODEL_FILE)

  def _assert_actual_result_approximately_matches_expected_result(
      self,
      actual_result: _GestureRecognitionResult,
      expected_result: _GestureRecognitionResult
  ):
    # Expects to have the same number of hands detected.
    self.assertLen(actual_result.hand_landmarks,
                   len(expected_result.hand_landmarks))
    self.assertLen(actual_result.hand_world_landmarks,
                   len(expected_result.hand_world_landmarks))
    self.assertLen(actual_result.handedness, len(expected_result.handedness))
    self.assertLen(actual_result.gestures, len(expected_result.gestures))
    # Actual landmarks match expected landmarks.
    self.assertEqual(actual_result.hand_landmarks,
                     expected_result.hand_landmarks)
    # Actual handedness matches expected handedness.
    actual_top_handedness = actual_result.handedness[0].classifications[0]
    expected_top_handedness = expected_result.handedness[0].classifications[0]
    self.assertEqual(actual_top_handedness.index, expected_top_handedness.index)
    self.assertEqual(actual_top_handedness.label, expected_top_handedness.label)
    # Actual gesture with top score matches expected gesture.
    actual_top_gesture = actual_result.gestures[0].classifications[0]
    expected_top_gesture = expected_result.gestures[0].classifications[0]
    self.assertEqual(actual_top_gesture.index, expected_top_gesture.index)
    self.assertEqual(actual_top_gesture.label, expected_top_gesture.label)

  @parameterized.parameters(
      (ModelFileType.FILE_NAME, 0.3, _get_expected_gesture_recognition_result(
          _THUMB_UP_LANDMARKS, _THUMB_UP_LABEL, _THUMB_UP_INDEX
      )),
      (ModelFileType.FILE_CONTENT, 0.3, _get_expected_gesture_recognition_result(
          _THUMB_UP_LANDMARKS, _THUMB_UP_LABEL, _THUMB_UP_INDEX
      )))
  def test_recognize(self, model_file_type, min_gesture_confidence,
                     expected_recognition_result):
    # Creates gesture recognizer.
    if model_file_type is ModelFileType.FILE_NAME:
      gesture_recognizer_base_options = _BaseOptions(
          model_asset_path=self.gesture_recognizer_model_path)
    elif model_file_type is ModelFileType.FILE_CONTENT:
      with open(self.gesture_recognizer_model_path, 'rb') as f:
        model_content = f.read()
      gesture_recognizer_base_options = _BaseOptions(
          model_asset_buffer=model_content)
    else:
      # Should never happen
      raise ValueError('model_file_type is invalid.')

    options = _GestureRecognizerOptions(
        base_options=gesture_recognizer_base_options,
        min_gesture_confidence=min_gesture_confidence
    )
    recognizer = _GestureRecognizer.create_from_options(options)

    # Performs hand gesture recognition on the input.
    recognition_result = recognizer.recognize(self.test_image)
    # Comparing results.
    self._assert_actual_result_approximately_matches_expected_result(
        recognition_result, expected_recognition_result)
    # Closes the gesture recognizer explicitly when the detector is not used in
    # a context.
    recognizer.close()


if __name__ == '__main__':
  absltest.main()
