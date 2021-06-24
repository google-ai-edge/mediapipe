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
from mediapipe.python.solutions import drawing_utils as mp_drawing
from mediapipe.python.solutions import hands as mp_hands

TEST_IMAGE_PATH = 'mediapipe/python/solutions/testdata'
DIFF_THRESHOLD = 15  # pixels
EXPECTED_HAND_COORDINATES_PREDICTION = [[[144, 345], [211, 323], [257, 286],
                                         [289, 237], [322, 203], [219, 216],
                                         [238, 138], [249, 90], [253, 51],
                                         [177, 204], [184, 115], [187, 60],
                                         [185, 19], [138, 208], [131, 127],
                                         [124, 77], [117, 36], [106, 222],
                                         [92, 159], [79, 124], [68, 93]],
                                        [[577, 40], [504, 56], [459, 94],
                                         [429, 146], [397, 182], [496, 167],
                                         [479, 245], [469, 292], [464, 330],
                                         [540, 177], [534, 265], [533, 319],
                                         [536, 360], [581, 172], [587, 252],
                                         [593, 304], [599, 346], [615, 157],
                                         [628, 223], [638, 258], [648, 288]]]


class HandsTest(parameterized.TestCase):

  def _annotate(self, frame: np.ndarray, results: NamedTuple, idx: int):
    for hand_landmarks in results.multi_hand_landmarks:
      mp_drawing.draw_landmarks(frame, hand_landmarks,
                                mp_hands.HAND_CONNECTIONS)
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

  @parameterized.named_parameters(('static_image_mode', True, 1),
                                  ('video_mode', False, 5))
  def test_multi_hands(self, static_image_mode, num_frames):
    image_path = os.path.join(os.path.dirname(__file__), 'testdata/hands.jpg')
    image = cv2.imread(image_path)
    with mp_hands.Hands(
        static_image_mode=static_image_mode,
        max_num_hands=2,
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
        npt.assert_array_less(prediction_error, DIFF_THRESHOLD)


if __name__ == '__main__':
  absltest.main()
