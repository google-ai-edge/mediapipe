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
from unittest import mock

import numpy as np
from google.protobuf import text_format
from absl.testing import absltest
from absl.testing import parameterized

from mediapipe.python._framework_bindings import image as image_module
from mediapipe.tasks.cc.components.containers.proto import landmarks_detection_result_pb2
from mediapipe.tasks.python.components.containers import rect as rect_module
from mediapipe.tasks.python.components.containers import category as category_module
from mediapipe.tasks.python.components.containers import landmark as landmark_module
from mediapipe.tasks.python.components.containers import landmark_detection_result as landmark_detection_result_module
from mediapipe.tasks.python.core import base_options as base_options_module
from mediapipe.tasks.python.test import test_utils
from mediapipe.tasks.python.vision import gesture_recognizer
from mediapipe.tasks.python.vision.core import vision_task_running_mode as running_mode_module
from mediapipe.tasks.python.vision.core import image_processing_options as image_processing_options_module

_LandmarksDetectionResultProto = landmarks_detection_result_pb2.LandmarksDetectionResult
_BaseOptions = base_options_module.BaseOptions
_Rect = rect_module.Rect
_Category = category_module.Category
_Landmark = landmark_module.Landmark
_NormalizedLandmark = landmark_module.NormalizedLandmark
_LandmarksDetectionResult = landmark_detection_result_module.LandmarksDetectionResult
_Image = image_module.Image
_GestureRecognizer = gesture_recognizer.GestureRecognizer
_GestureRecognizerOptions = gesture_recognizer.GestureRecognizerOptions
_GestureRecognitionResult = gesture_recognizer.GestureRecognitionResult
_RUNNING_MODE = running_mode_module.VisionTaskRunningMode
_ImageProcessingOptions = image_processing_options_module.ImageProcessingOptions

_GESTURE_RECOGNIZER_BUNDLE_ASSET_FILE = 'gesture_recognizer.task'
_GESTURE_RECOGNIZER_WITH_CUSTOM_CLASSIFIER_BUNDLE_ASSET_FILE = 'gesture_recognizer_with_custom_classifier.task'
_NO_HANDS_IMAGE = 'cats_and_dogs.jpg'
_TWO_HANDS_IMAGE = 'right_hands.jpg'
_FIST_IMAGE = 'fist.jpg'
_FIST_LANDMARKS = 'fist_landmarks.pbtxt'
_VICTORY_IMAGE = 'victory.jpg'
_VICTORY_LANDMARKS = 'victory_landmarks.pbtxt'
_VICTORY_LABEL = 'Victory'
_THUMB_UP_IMAGE = 'thumb_up.jpg'
_THUMB_UP_LANDMARKS = 'thumb_up_landmarks.pbtxt'
_THUMB_UP_LABEL = 'Thumb_Up'
_POINTING_UP_ROTATED_IMAGE = 'pointing_up_rotated.jpg'
_POINTING_UP_LANDMARKS = 'pointing_up_rotated_landmarks.pbtxt'
_POINTING_UP_LABEL = 'Pointing_Up'
_ROCK_LABEL = "Rock"
_LANDMARKS_ERROR_TOLERANCE = 0.03
_GESTURE_EXPECTED_INDEX = -1


def _get_expected_gesture_recognition_result(
    file_path: str, gesture_label: str
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
  gesture = _Category(category_name=gesture_label,
                      index=_GESTURE_EXPECTED_INDEX,
                      display_name='')
  return _GestureRecognitionResult(
      gestures=[[gesture]],
      handedness=[landmarks_detection_result.categories],
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
    self.model_path = test_utils.get_test_data_path(
        _GESTURE_RECOGNIZER_BUNDLE_ASSET_FILE)

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
    self.assertLen(actual_result.hand_landmarks[0],
                   len(expected_result.hand_landmarks[0]))
    actual_landmarks = actual_result.hand_landmarks[0]
    expected_landmarks = expected_result.hand_landmarks[0]
    for i in range(len(actual_landmarks)):
      self.assertAlmostEqual(actual_landmarks[i].x, expected_landmarks[i].x,
                             delta=_LANDMARKS_ERROR_TOLERANCE)
      self.assertAlmostEqual(actual_landmarks[i].y, expected_landmarks[i].y,
                             delta=_LANDMARKS_ERROR_TOLERANCE)
    # Actual handedness matches expected handedness.
    actual_top_handedness = actual_result.handedness[0][0]
    expected_top_handedness = expected_result.handedness[0][0]
    self.assertEqual(actual_top_handedness.index, expected_top_handedness.index)
    self.assertEqual(actual_top_handedness.category_name,
                     expected_top_handedness.category_name)
    # Actual gesture with top score matches expected gesture.
    actual_top_gesture = actual_result.gestures[0][0]
    expected_top_gesture = expected_result.gestures[0][0]
    self.assertEqual(actual_top_gesture.index, _GESTURE_EXPECTED_INDEX)
    self.assertEqual(actual_top_gesture.category_name,
                     expected_top_gesture.category_name)

  def test_create_from_file_succeeds_with_valid_model_path(self):
    # Creates with default option and valid model file successfully.
    with _GestureRecognizer.create_from_model_path(self.model_path) as recognizer:
      self.assertIsInstance(recognizer, _GestureRecognizer)

  def test_create_from_options_succeeds_with_valid_model_path(self):
    # Creates with options containing model file successfully.
    base_options = _BaseOptions(model_asset_path=self.model_path)
    options = _GestureRecognizerOptions(base_options=base_options)
    with _GestureRecognizer.create_from_options(options) as recognizer:
      self.assertIsInstance(recognizer, _GestureRecognizer)

  def test_create_from_options_fails_with_invalid_model_path(self):
    # Invalid empty model path.
    with self.assertRaisesRegex(
        ValueError,
        r"ExternalFile must specify at least one of 'file_content', "
        r"'file_name', 'file_pointer_meta' or 'file_descriptor_meta'."):
      base_options = _BaseOptions(model_asset_path='')
      options = _GestureRecognizerOptions(base_options=base_options)
      _GestureRecognizer.create_from_options(options)

  def test_create_from_options_succeeds_with_valid_model_content(self):
    # Creates with options containing model content successfully.
    with open(self.model_path, 'rb') as f:
      base_options = _BaseOptions(model_asset_buffer=f.read())
      options = _GestureRecognizerOptions(base_options=base_options)
      recognizer = _GestureRecognizer.create_from_options(options)
      self.assertIsInstance(recognizer, _GestureRecognizer)

  @parameterized.parameters(
      (ModelFileType.FILE_NAME, _get_expected_gesture_recognition_result(
          _THUMB_UP_LANDMARKS, _THUMB_UP_LABEL
      )),
      (ModelFileType.FILE_CONTENT, _get_expected_gesture_recognition_result(
          _THUMB_UP_LANDMARKS, _THUMB_UP_LABEL
      )))
  def test_recognize(self, model_file_type, expected_recognition_result):
    # Creates gesture recognizer.
    if model_file_type is ModelFileType.FILE_NAME:
      base_options = _BaseOptions(model_asset_path=self.model_path)
    elif model_file_type is ModelFileType.FILE_CONTENT:
      with open(self.model_path, 'rb') as f:
        model_content = f.read()
      base_options = _BaseOptions(model_asset_buffer=model_content)
    else:
      # Should never happen
      raise ValueError('model_file_type is invalid.')

    options = _GestureRecognizerOptions(base_options=base_options)
    recognizer = _GestureRecognizer.create_from_options(options)

    # Performs hand gesture recognition on the input.
    recognition_result = recognizer.recognize(self.test_image)
    # Comparing results.
    self._assert_actual_result_approximately_matches_expected_result(
        recognition_result, expected_recognition_result)
    # Closes the gesture recognizer explicitly when the gesture recognizer is
    # not used in a context.
    recognizer.close()

  @parameterized.parameters(
      (ModelFileType.FILE_NAME, _get_expected_gesture_recognition_result(
          _THUMB_UP_LANDMARKS, _THUMB_UP_LABEL
      )),
      (ModelFileType.FILE_CONTENT, _get_expected_gesture_recognition_result(
          _THUMB_UP_LANDMARKS, _THUMB_UP_LABEL
      )))
  def test_recognize_in_context(self, model_file_type,
                                expected_recognition_result):
    # Creates gesture recognizer.
    if model_file_type is ModelFileType.FILE_NAME:
      base_options = _BaseOptions(model_asset_path=self.model_path)
    elif model_file_type is ModelFileType.FILE_CONTENT:
      with open(self.model_path, 'rb') as f:
        model_content = f.read()
      base_options = _BaseOptions(model_asset_buffer=model_content)
    else:
      # Should never happen
      raise ValueError('model_file_type is invalid.')

    options = _GestureRecognizerOptions(base_options=base_options)
    with _GestureRecognizer.create_from_options(options) as recognizer:
      # Performs hand gesture recognition on the input.
      recognition_result = recognizer.recognize(self.test_image)
      # Comparing results.
      self._assert_actual_result_approximately_matches_expected_result(
        recognition_result, expected_recognition_result)

  def test_recognize_succeeds_with_min_gesture_confidence(self):
    # Creates gesture recognizer.
    base_options = _BaseOptions(model_asset_path=self.model_path)
    options = _GestureRecognizerOptions(base_options=base_options,
                                        min_gesture_confidence=0.5)
    with _GestureRecognizer.create_from_options(options) as recognizer:
      # Performs hand gesture recognition on the input.
      recognition_result = recognizer.recognize(self.test_image)
      expected_result = _get_expected_gesture_recognition_result(
          _THUMB_UP_LANDMARKS, _THUMB_UP_LABEL)
      # Only contains one top scoring gesture.
      self.assertLen(recognition_result.gestures[0], 1)
      # Actual gesture with top score matches expected gesture.
      actual_top_gesture = recognition_result.gestures[0][0]
      expected_top_gesture = expected_result.gestures[0][0]
      self.assertEqual(actual_top_gesture.index, expected_top_gesture.index)
      self.assertEqual(actual_top_gesture.category_name,
                       expected_top_gesture.category_name)

  def test_recognize_succeeds_with_num_hands(self):
    # Creates gesture recognizer.
    base_options = _BaseOptions(model_asset_path=self.model_path)
    options = _GestureRecognizerOptions(base_options=base_options, num_hands=2)
    with _GestureRecognizer.create_from_options(options) as recognizer:
      # Load the pointing up rotated image.
      test_image = _Image.create_from_file(
        test_utils.get_test_data_path(_TWO_HANDS_IMAGE))
      # Performs hand gesture recognition on the input.
      recognition_result = recognizer.recognize(test_image)
      # Comparing results.
      self.assertLen(recognition_result.handedness, 2)

  def test_recognize_succeeds_with_rotation(self):
    # Creates gesture recognizer.
    base_options = _BaseOptions(model_asset_path=self.model_path)
    options = _GestureRecognizerOptions(base_options=base_options, num_hands=1)
    with _GestureRecognizer.create_from_options(options) as recognizer:
      # Load the pointing up rotated image.
      test_image = _Image.create_from_file(
          test_utils.get_test_data_path(_POINTING_UP_ROTATED_IMAGE))
      # Set rotation parameters using ImageProcessingOptions.
      image_processing_options = _ImageProcessingOptions(rotation_degrees=-90)
      # Performs hand gesture recognition on the input.
      recognition_result = recognizer.recognize(test_image,
                                                image_processing_options)
      expected_recognition_result = _get_expected_gesture_recognition_result(
          _POINTING_UP_LANDMARKS, _POINTING_UP_LABEL)
      # Comparing results.
      self._assert_actual_result_approximately_matches_expected_result(
          recognition_result, expected_recognition_result)

  def test_recognize_succeeds_with_canned_gesture_victory(self):
    # Creates gesture recognizer.
    base_options = _BaseOptions(model_asset_path=self.model_path)
    options = _GestureRecognizerOptions(base_options=base_options, num_hands=1)
    with _GestureRecognizer.create_from_options(options) as recognizer:
      # Load the fist image.
      test_image = _Image.create_from_file(
        test_utils.get_test_data_path(_VICTORY_IMAGE))
      # Performs hand gesture recognition on the input.
      recognition_result = recognizer.recognize(test_image)
      expected_recognition_result = _get_expected_gesture_recognition_result(
        _VICTORY_LANDMARKS, _VICTORY_LABEL)
      # Comparing results.
      self._assert_actual_result_approximately_matches_expected_result(
        recognition_result, expected_recognition_result)

  def test_recognize_succeeds_with_custom_gesture_fist(self):
    # Creates gesture recognizer.
    model_path = test_utils.get_test_data_path(
        _GESTURE_RECOGNIZER_WITH_CUSTOM_CLASSIFIER_BUNDLE_ASSET_FILE)
    base_options = _BaseOptions(model_asset_path=model_path)
    options = _GestureRecognizerOptions(base_options=base_options, num_hands=1)
    with _GestureRecognizer.create_from_options(options) as recognizer:
      # Load the fist image.
      test_image = _Image.create_from_file(
          test_utils.get_test_data_path(_FIST_IMAGE))
      # Performs hand gesture recognition on the input.
      recognition_result = recognizer.recognize(test_image)
      expected_recognition_result = _get_expected_gesture_recognition_result(
          _FIST_LANDMARKS, _ROCK_LABEL)
      # Comparing results.
      self._assert_actual_result_approximately_matches_expected_result(
        recognition_result, expected_recognition_result)

  def test_recognize_fails_with_region_of_interest(self):
    # Creates gesture recognizer.
    base_options = _BaseOptions(model_asset_path=self.model_path)
    options = _GestureRecognizerOptions(base_options=base_options, num_hands=1)
    with self.assertRaisesRegex(
        ValueError, "This task doesn't support region-of-interest."):
      with _GestureRecognizer.create_from_options(options) as recognizer:
        # Set the `region_of_interest` parameter using `ImageProcessingOptions`.
        image_processing_options = _ImageProcessingOptions(
            region_of_interest=_Rect(0, 0, 1, 1))
        # Attempt to perform hand gesture recognition on the cropped input.
        recognizer.recognize(self.test_image, image_processing_options)

  def test_empty_recognition_outputs(self):
    options = _GestureRecognizerOptions(
      base_options=_BaseOptions(model_asset_path=self.model_path))
    with _GestureRecognizer.create_from_options(options) as recognizer:
      # Load the image with no hands.
      no_hands_test_image = _Image.create_from_file(
        test_utils.get_test_data_path(_NO_HANDS_IMAGE))
      # Performs gesture recognition on the input.
      recognition_result = recognizer.recognize(no_hands_test_image)
      self.assertEmpty(recognition_result.hand_landmarks)
      self.assertEmpty(recognition_result.hand_world_landmarks)
      self.assertEmpty(recognition_result.handedness)
      self.assertEmpty(recognition_result.gestures)

  def test_missing_result_callback(self):
    options = _GestureRecognizerOptions(
      base_options=_BaseOptions(model_asset_path=self.model_path),
      running_mode=_RUNNING_MODE.LIVE_STREAM)
    with self.assertRaisesRegex(ValueError,
                                r'result callback must be provided'):
      with _GestureRecognizer.create_from_options(options) as unused_recognizer:
        pass

  @parameterized.parameters((_RUNNING_MODE.IMAGE), (_RUNNING_MODE.VIDEO))
  def test_illegal_result_callback(self, running_mode):
    options = _GestureRecognizerOptions(
      base_options=_BaseOptions(model_asset_path=self.model_path),
      running_mode=running_mode,
      result_callback=mock.MagicMock())
    with self.assertRaisesRegex(ValueError,
                                r'result callback should not be provided'):
      with _GestureRecognizer.create_from_options(options) as unused_recognizer:
        pass

  def test_calling_recognize_for_video_in_image_mode(self):
    options = _GestureRecognizerOptions(
      base_options=_BaseOptions(model_asset_path=self.model_path),
      running_mode=_RUNNING_MODE.IMAGE)
    with _GestureRecognizer.create_from_options(options) as recognizer:
      with self.assertRaisesRegex(ValueError,
                                  r'not initialized with the video mode'):
        recognizer.recognize_for_video(self.test_image, 0)

  def test_calling_recognize_async_in_image_mode(self):
    options = _GestureRecognizerOptions(
      base_options=_BaseOptions(model_asset_path=self.model_path),
      running_mode=_RUNNING_MODE.IMAGE)
    with _GestureRecognizer.create_from_options(options) as recognizer:
      with self.assertRaisesRegex(ValueError,
                                  r'not initialized with the live stream mode'):
        recognizer.recognize_async(self.test_image, 0)

  def test_calling_recognize_in_video_mode(self):
    options = _GestureRecognizerOptions(
      base_options=_BaseOptions(model_asset_path=self.model_path),
      running_mode=_RUNNING_MODE.VIDEO)
    with _GestureRecognizer.create_from_options(options) as recognizer:
      with self.assertRaisesRegex(ValueError,
                                  r'not initialized with the image mode'):
        recognizer.recognize(self.test_image)

  def test_calling_recognize_async_in_video_mode(self):
    options = _GestureRecognizerOptions(
      base_options=_BaseOptions(model_asset_path=self.model_path),
      running_mode=_RUNNING_MODE.VIDEO)
    with _GestureRecognizer.create_from_options(options) as recognizer:
      with self.assertRaisesRegex(ValueError,
                                  r'not initialized with the live stream mode'):
        recognizer.recognize_async(self.test_image, 0)

  def test_recognize_for_video_with_out_of_order_timestamp(self):
    options = _GestureRecognizerOptions(
      base_options=_BaseOptions(model_asset_path=self.model_path),
      running_mode=_RUNNING_MODE.VIDEO)
    with _GestureRecognizer.create_from_options(options) as recognizer:
      unused_result = recognizer.recognize_for_video(self.test_image, 1)
      with self.assertRaisesRegex(
          ValueError, r'Input timestamp must be monotonically increasing'):
        recognizer.recognize_for_video(self.test_image, 0)

  def test_recognize_for_video(self):
    options = _GestureRecognizerOptions(
      base_options=_BaseOptions(model_asset_path=self.model_path),
      running_mode=_RUNNING_MODE.VIDEO)
    with _GestureRecognizer.create_from_options(options) as recognizer:
      for timestamp in range(0, 300, 30):
        recognition_result = recognizer.recognize_for_video(self.test_image,
                                                            timestamp)
        expected_recognition_result = _get_expected_gesture_recognition_result(
           _THUMB_UP_LANDMARKS, _THUMB_UP_LABEL)
        self._assert_actual_result_approximately_matches_expected_result(
            recognition_result, expected_recognition_result)

  def test_calling_recognize_in_live_stream_mode(self):
    options = _GestureRecognizerOptions(
      base_options=_BaseOptions(model_asset_path=self.model_path),
      running_mode=_RUNNING_MODE.LIVE_STREAM,
      result_callback=mock.MagicMock())
    with _GestureRecognizer.create_from_options(options) as recognizer:
      with self.assertRaisesRegex(ValueError,
                                  r'not initialized with the image mode'):
        recognizer.recognize(self.test_image)

  def test_calling_recognize_for_video_in_live_stream_mode(self):
    options = _GestureRecognizerOptions(
      base_options=_BaseOptions(model_asset_path=self.model_path),
      running_mode=_RUNNING_MODE.LIVE_STREAM,
      result_callback=mock.MagicMock())
    with _GestureRecognizer.create_from_options(options) as recognizer:
      with self.assertRaisesRegex(ValueError,
                                  r'not initialized with the video mode'):
        recognizer.recognize_for_video(self.test_image, 0)

  def test_recognize_async_calls_with_illegal_timestamp(self):
    options = _GestureRecognizerOptions(
      base_options=_BaseOptions(model_asset_path=self.model_path),
      running_mode=_RUNNING_MODE.LIVE_STREAM,
      result_callback=mock.MagicMock())
    with _GestureRecognizer.create_from_options(options) as recognizer:
      recognizer.recognize_async(self.test_image, 100)
      with self.assertRaisesRegex(
          ValueError, r'Input timestamp must be monotonically increasing'):
        recognizer.recognize_async(self.test_image, 0)

  @parameterized.parameters(
      (_THUMB_UP_IMAGE, _get_expected_gesture_recognition_result(
          _THUMB_UP_LANDMARKS, _THUMB_UP_LABEL)),
      (_NO_HANDS_IMAGE, _GestureRecognitionResult([], [], [], [])))
  def test_recognize_async_calls(self, image_path, expected_result):
    test_image = _Image.create_from_file(
        test_utils.get_test_data_path(image_path))
    observed_timestamp_ms = -1

    def check_result(result: _GestureRecognitionResult, output_image: _Image,
                     timestamp_ms: int):
      if result.hand_landmarks and result.hand_world_landmarks and \
        result.handedness and result.gestures:
        self._assert_actual_result_approximately_matches_expected_result(
            result, expected_result)
      else:
        self.assertEqual(result, expected_result)
      self.assertTrue(
          np.array_equal(output_image.numpy_view(),
                         test_image.numpy_view()))
      self.assertLess(observed_timestamp_ms, timestamp_ms)
      self.observed_timestamp_ms = timestamp_ms

    options = _GestureRecognizerOptions(
      base_options=_BaseOptions(model_asset_path=self.model_path),
      running_mode=_RUNNING_MODE.LIVE_STREAM,
      result_callback=check_result)
    with _GestureRecognizer.create_from_options(options) as recognizer:
      for timestamp in range(0, 300, 30):
        recognizer.recognize_async(test_image, timestamp)


if __name__ == '__main__':
  absltest.main()
