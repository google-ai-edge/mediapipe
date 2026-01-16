# Copyright 2020 The MediaPipe Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for mediapipe.python.solutions.hands."""

import json
import os
import tempfile  # pylint: disable=unused-import
from typing import NamedTuple

from absl.testing import absltest
from absl.testing import parameterized
import cv2
import numpy as np
import numpy.testing as npt

# resources dependency
# undeclared dependency
from mediapipe.python.solutions import drawing_styles
from mediapipe.python.solutions import drawing_utils as mp_drawing
from mediapipe.python.solutions import hands as mp_hands


TEST_IMAGE_PATH = 'mediapipe/python/solutions/testdata'
LITE_MODEL_DIFF_THRESHOLD = 25  # pixels
FULL_MODEL_DIFF_THRESHOLD = 20  # pixels
EXPECTED_HAND_COORDINATES_PREDICTION = [[[580, 34], [504, 50], [459, 94],
                                         [429, 146], [397, 182], [507, 167],
                                         [479, 245], [469, 292], [464, 330],
                                         [545, 180], [534, 265], [533, 319],
                                         [536, 360], [581, 172], [587, 252],
                                         [593, 304], [599, 346], [615, 168],
                                         [628, 223], [638, 258], [648, 288]],
                                        [[138, 343], [211, 330], [257, 286],
                                         [289, 237], [322, 203], [219, 216],
                                         [238, 138], [249, 90], [253, 51],
                                         [177, 204], [184, 115], [187, 60],
                                         [185, 19], [138, 208], [131, 127],
                                         [124, 77], [117, 36], [106, 222],
                                         [92, 159], [79, 124], [68, 93]]]


class HandsTest(parameterized.TestCase):

  def _get_output_path(self, name):
    return os.path.join(tempfile.gettempdir(), self.id().split('.')[-1] + name)

  def _landmarks_list_to_array(self, landmark_list, image_shape):
    rows, cols, _ = image_shape
    return np.asarray([(lmk.x * cols, lmk.y * rows, lmk.z * cols)
                       for lmk in landmark_list.landmark])

  def _world_landmarks_list_to_array(self, landmark_list):
    return np.asarray([(lmk.x, lmk.y, lmk.z)
                       for lmk in landmark_list.landmark])

  def _assert_diff_less(self, array1, array2, threshold):
    npt.assert_array_less(np.abs(array1 - array2), threshold)

  def _annotate(self, frame: np.ndarray, results: NamedTuple, idx: int):
    for hand_landmarks in results.multi_hand_landmarks:
      mp_drawing.draw_landmarks(
          frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
          drawing_styles.get_default_hand_landmarks_style(),
          drawing_styles.get_default_hand_connections_style())
    path = os.path.join(tempfile.gettempdir(), self.id().split('.')[-1] +
                                              '_frame_{}.png'.format(idx))
    cv2.imwrite(path, frame)

  def test_invalid_image_shape(self):
    with mp_hands.Hands() as hands:
      with self.assertRaisesRegex(
          ValueError, 'Input image must contain three channel rgb data.'):
        hands.process(np.arange(36, dtype=np.uint8).reshape(3, 3, 4))

  def test_blank_image(self):
    with mp_hands.Hands() as hands:
      image = np.zeros([100, 100, 3], dtype=np.uint8)
      image.fill(255)
      results = hands.process(image)
      self.assertIsNone(results.multi_hand_landmarks)
      self.assertIsNone(results.multi_handedness)

  @parameterized.named_parameters(
      ('static_image_mode_with_lite_model', True, 0, 5),
      ('video_mode_with_lite_model', False, 0, 10),
      ('static_image_mode_with_full_model', True, 1, 5),
      ('video_mode_with_full_model', False, 1, 10))
  def test_multi_hands(self, static_image_mode, model_complexity, num_frames):
    image_path = os.path.join(os.path.dirname(__file__), 'testdata/hands.jpg')
    image = cv2.imread(image_path)
    with mp_hands.Hands(
        static_image_mode=static_image_mode,
        max_num_hands=2,
        model_complexity=model_complexity,
        min_detection_confidence=0.5) as hands:
      for idx in range(num_frames):
        results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        self._annotate(image.copy(), results, idx)
        handedness = [
            handedness.classification[0].label
            for handedness in results.multi_handedness
        ]
        multi_hand_coordinates = []
        rows, cols, _ = image.shape
        for landmarks in results.multi_hand_landmarks:
          self.assertLen(landmarks.landmark, 21)
          x = [landmark.x * cols for landmark in landmarks.landmark]
          y = [landmark.y * rows for landmark in landmarks.landmark]
          hand_coordinates = np.column_stack((x, y))
          multi_hand_coordinates.append(hand_coordinates)
        self.assertLen(handedness, 2)
        self.assertLen(multi_hand_coordinates, 2)
        prediction_error = np.abs(
            np.asarray(multi_hand_coordinates) -
            np.asarray(EXPECTED_HAND_COORDINATES_PREDICTION))
        diff_threshold = LITE_MODEL_DIFF_THRESHOLD if model_complexity == 0 else FULL_MODEL_DIFF_THRESHOLD
        npt.assert_array_less(prediction_error, diff_threshold)

  def _process_video(self, model_complexity, video_path,
                     max_num_hands=1,
                     num_landmarks=21,
                     num_dimensions=3):
    # Predict pose landmarks for each frame.
    video_cap = cv2.VideoCapture(video_path)
    landmarks_per_frame = []
    w_landmarks_per_frame = []
    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=max_num_hands,
        model_complexity=model_complexity,
        min_detection_confidence=0.5) as hands:
      while True:
        # Get next frame of the video.
        success, input_frame = video_cap.read()
        if not success:
          break

        # Run pose tracker.
        input_frame = cv2.cvtColor(input_frame, cv2.COLOR_BGR2RGB)
        frame_shape = input_frame.shape
        result = hands.process(image=input_frame)
        frame_landmarks = np.zeros([max_num_hands,
                                    num_landmarks, num_dimensions]) * np.nan
        frame_w_landmarks = np.zeros([max_num_hands,
                                      num_landmarks, num_dimensions]) * np.nan

        if result.multi_hand_landmarks:
          for idx, landmarks in enumerate(result.multi_hand_landmarks):
            landmarks = self._landmarks_list_to_array(landmarks, frame_shape)
            frame_landmarks[idx] = landmarks
        if result.multi_hand_world_landmarks:
          for idx, w_landmarks in enumerate(result.multi_hand_world_landmarks):
            w_landmarks = self._world_landmarks_list_to_array(w_landmarks)
            frame_w_landmarks[idx] = w_landmarks

        landmarks_per_frame.append(frame_landmarks)
        w_landmarks_per_frame.append(frame_w_landmarks)
    return (np.array(landmarks_per_frame), np.array(w_landmarks_per_frame))

  @parameterized.named_parameters(
      ('full', 1, 'asl_hand.full.npz'))
  def test_on_video(self, model_complexity, expected_name):
    """Tests hand models on a video."""
    video_path = os.path.join(os.path.dirname(__file__),
                              'testdata/asl_hand.25fps.mp4')
    expected_path = os.path.join(os.path.dirname(__file__),
                                 'testdata/{}'.format(expected_name))
    actual, actual_world = self._process_video(model_complexity, video_path)

    # Dump actual .npz.
    npz_path = self._get_output_path(expected_name)
    np.savez(npz_path, predictions=actual, w_predictions=actual_world)

    # Dump actual JSON.
    json_path = self._get_output_path(expected_name.replace('.npz', '.json'))
    with open(json_path, 'w') as fl:
      dump_data = {
          'predictions': np.around(actual, 3).tolist(),
          'predictions_world': np.around(actual_world, 3).tolist(),
      }
      fl.write(json.dumps(dump_data, indent=2, separators=(',', ': ')))

    # Validate actual vs. expected landmarks.
    expected = np.load(expected_path)['predictions']
    assert (
        actual.shape == expected.shape
    ), 'Unexpected shape of predictions: {} instead of {}'.format(
        actual.shape, expected.shape
    )
    # large values, use relative tolerance for testing.
    np.testing.assert_allclose(actual[..., :2], expected[..., :2], rtol=0.1)

    # Validate actual vs. expected world landmarks.
    expected_world = np.load(expected_path)['w_predictions']
    assert (
        actual_world.shape == expected_world.shape
    ), 'Unexpected shape of world predictions: {} instead of {}'.format(
        actual_world.shape, expected_world.shape
    )
    # small values, use absolute tolerance for testing.
    np.testing.assert_array_almost_equal(
        actual_world, expected_world, decimal=1
    )


if __name__ == '__main__':
  absltest.main()
