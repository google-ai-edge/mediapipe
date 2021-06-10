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
"""Tests for mediapipe.python.solutions.pose."""

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
from mediapipe.python.solutions import holistic as mp_holistic

TEST_IMAGE_PATH = 'mediapipe/python/solutions/testdata'
POSE_DIFF_THRESHOLD = 30  # pixels
HAND_DIFF_THRESHOLD = 30  # pixels
EXPECTED_POSE_LANDMARKS = np.array([[782, 243], [791, 232], [796, 233],
                                    [801, 233], [773, 231], [766, 231],
                                    [759, 232], [802, 242], [751, 239],
                                    [791, 258], [766, 258], [830, 301],
                                    [708, 298], [910, 248], [635, 234],
                                    [954, 161], [593, 136], [961, 137],
                                    [583, 110], [952, 132], [592, 106],
                                    [950, 141], [596, 115], [793, 500],
                                    [724, 502], [874, 626], [640, 629],
                                    [965, 756], [542, 760], [962, 779],
                                    [533, 781], [1025, 797], [487, 803]])
EXPECTED_LEFT_HAND_LANDMARKS = np.array([[958, 167], [950, 161], [945, 151],
                                         [945, 141], [947, 134], [945, 136],
                                         [939, 122], [935, 113], [931, 106],
                                         [951, 134], [946, 118], [942, 108],
                                         [938, 100], [957, 135], [954, 120],
                                         [951, 111], [948, 103], [964, 138],
                                         [964, 128], [965, 122], [965, 117]])
EXPECTED_RIGHT_HAND_LANDMARKS = np.array([[590, 135], [602, 125], [609, 114],
                                          [613, 103], [617, 96], [596, 100],
                                          [595, 84], [594, 74], [593, 68],
                                          [588, 100], [586, 84], [585, 73],
                                          [584, 65], [581, 103], [579, 89],
                                          [579, 79], [579, 72], [575, 109],
                                          [571, 99], [570, 93], [569, 87]])


class PoseTest(parameterized.TestCase):

  def _landmarks_list_to_array(self, landmark_list, image_shape):
    rows, cols, _ = image_shape
    return np.asarray([(lmk.x * cols, lmk.y * rows)
                       for lmk in landmark_list.landmark])

  def _assert_diff_less(self, array1, array2, threshold):
    npt.assert_array_less(np.abs(array1 - array2), threshold)

  def _annotate(self, frame: np.ndarray, results: NamedTuple, idx: int):
    drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
    mp_drawing.draw_landmarks(
        image=frame,
        landmark_list=results.face_landmarks,
        landmark_drawing_spec=drawing_spec)
    mp_drawing.draw_landmarks(frame, results.left_hand_landmarks,
                              mp_holistic.HAND_CONNECTIONS)
    mp_drawing.draw_landmarks(frame, results.right_hand_landmarks,
                              mp_holistic.HAND_CONNECTIONS)
    mp_drawing.draw_landmarks(frame, results.pose_landmarks,
                              mp_holistic.POSE_CONNECTIONS)
    path = os.path.join(tempfile.gettempdir(), self.id().split('.')[-1] +
                                              '_frame_{}.png'.format(idx))
    cv2.imwrite(path, frame)

  def test_invalid_image_shape(self):
    with mp_holistic.Holistic() as holistic:
      with self.assertRaisesRegex(
          ValueError, 'Input image must contain three channel rgb data.'):
        holistic.process(np.arange(36, dtype=np.uint8).reshape(3, 3, 4))

  def test_blank_image(self):
    with mp_holistic.Holistic() as holistic:
      image = np.zeros([100, 100, 3], dtype=np.uint8)
      image.fill(255)
      results = holistic.process(image)
      self.assertIsNone(results.pose_landmarks)

  @parameterized.named_parameters(('static_lite', True, 0, 3),
                                  ('static_full', True, 1, 3),
                                  ('static_heavy', True, 2, 3),
                                  ('video_lite', False, 0, 3),
                                  ('video_full', False, 1, 3),
                                  ('video_heavy', False, 2, 3))
  def test_on_image(self, static_image_mode, model_complexity, num_frames):
    image_path = os.path.join(os.path.dirname(__file__),
                              'testdata/holistic.jpg')
    image = cv2.imread(image_path)
    with mp_holistic.Holistic(static_image_mode=static_image_mode,
                              model_complexity=model_complexity) as holistic:
      for idx in range(num_frames):
        results = holistic.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        self._annotate(image.copy(), results, idx)
        self._assert_diff_less(
            self._landmarks_list_to_array(results.pose_landmarks, image.shape),
            EXPECTED_POSE_LANDMARKS,
            POSE_DIFF_THRESHOLD)
        self._assert_diff_less(
            self._landmarks_list_to_array(results.left_hand_landmarks,
                                          image.shape),
            EXPECTED_LEFT_HAND_LANDMARKS,
            HAND_DIFF_THRESHOLD)
        self._assert_diff_less(
            self._landmarks_list_to_array(results.right_hand_landmarks,
                                          image.shape),
            EXPECTED_RIGHT_HAND_LANDMARKS,
            HAND_DIFF_THRESHOLD)
        # TODO: Verify the correctness of the face landmarks.
        self.assertLen(results.face_landmarks.landmark, 468)


if __name__ == '__main__':
  absltest.main()
