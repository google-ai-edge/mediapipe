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

from absl.testing import absltest
from absl.testing import parameterized
import cv2
import numpy as np
import numpy.testing as npt

# resources dependency
from mediapipe.python.solutions import holistic as mp_holistic

TEST_IMAGE_PATH = 'mediapipe/python/solutions/testdata'
POSE_DIFF_THRESHOLD = 30  # pixels
HAND_DIFF_THRESHOLD = 30  # pixels
EXPECTED_UPPER_BODY_LANDMARKS = np.array([[457, 289], [465, 278], [467, 278],
                                          [470, 277], [461, 279], [461, 279],
                                          [461, 279], [485, 277], [474, 278],
                                          [468, 296], [463, 297], [542, 324],
                                          [449, 327], [614, 321], [376, 318],
                                          [680, 322], [312, 310], [697, 320],
                                          [293, 305], [699, 314], [289, 302],
                                          [693, 316], [296, 305], [515, 451],
                                          [467, 453]])
EXPECTED_FULL_BODY_LANDMARKS = np.array([[460, 287], [469, 277], [472, 276],
                                         [475, 276], [464, 277], [463, 277],
                                         [463, 276], [492, 277], [472, 277],
                                         [471, 295], [465, 295], [542, 323],
                                         [448, 318], [619, 319], [372, 313],
                                         [695, 316], [296, 308], [717, 313],
                                         [273, 304], [718, 304], [280, 298],
                                         [709, 307], [289, 303], [521, 470],
                                         [459, 466], [626, 533], [364, 500],
                                         [704, 616], [347, 614], [710, 631],
                                         [357, 633], [737, 625], [306, 639]])
EXPECTED_LEFT_HAND_LANDMARKS = np.array([[698, 314], [712, 314], [721, 314],
                                         [727, 314], [732, 313], [728, 309],
                                         [738, 309], [745, 308], [751, 307],
                                         [724, 310], [735, 309], [742, 309],
                                         [747, 307], [719, 312], [727, 313],
                                         [729, 312], [731, 311], [713, 315],
                                         [717, 315], [719, 314], [719, 313]])
EXPECTED_RIGHT_HAND_LANDMARKS = np.array([[293, 307], [284, 306], [277, 304],
                                          [271, 303], [266, 303], [271, 302],
                                          [261, 302], [254, 301], [247, 299],
                                          [272, 303], [261, 303], [253, 301],
                                          [245, 299], [275, 304], [266, 303],
                                          [258, 302], [252, 300], [279, 305],
                                          [273, 305], [268, 304], [263, 303]])


class PoseTest(parameterized.TestCase):

  def _landmarks_list_to_array(self, landmark_list, image_shape):
    rows, cols, _ = image_shape
    return np.asarray([(lmk.x * cols, lmk.y * rows)
                       for lmk in landmark_list.landmark])

  def _assert_diff_less(self, array1, array2, threshold):
    npt.assert_array_less(np.abs(array1 - array2), threshold)

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

  @parameterized.named_parameters(('static_image_mode', True, 3),
                                  ('video_mode', False, 3))
  def test_upper_body_model(self, static_image_mode, num_frames):
    image_path = os.path.join(os.path.dirname(__file__), 'testdata/pose.jpg')
    with mp_holistic.Holistic(
        static_image_mode=static_image_mode, upper_body_only=True) as holistic:
      image = cv2.imread(image_path)
      for _ in range(num_frames):
        results = holistic.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        self._assert_diff_less(
            self._landmarks_list_to_array(results.pose_landmarks, image.shape),
            EXPECTED_UPPER_BODY_LANDMARKS,
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

  @parameterized.named_parameters(('static_image_mode', True, 3),
                                  ('video_mode', False, 3))
  def test_full_body_model(self, static_image_mode, num_frames):
    image_path = os.path.join(os.path.dirname(__file__), 'testdata/pose.jpg')
    image = cv2.imread(image_path)

    with mp_holistic.Holistic(static_image_mode=static_image_mode) as holistic:
      for _ in range(num_frames):
        results = holistic.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        self._assert_diff_less(
            self._landmarks_list_to_array(results.pose_landmarks, image.shape),
            EXPECTED_FULL_BODY_LANDMARKS,
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
