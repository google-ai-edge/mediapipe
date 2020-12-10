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

import math
import os

from absl.testing import absltest
from absl.testing import parameterized
import cv2
import numpy as np
import numpy.testing as npt

# resources dependency
from mediapipe.python.solutions import holistic as mp_holistic

TEST_IMAGE_PATH = 'mediapipe/python/solutions/testdata'
POSE_DIFF_THRESHOLOD = 30  # pixels
HAND_DIFF_THRESHOLOD = 10  # pixels
EXPECTED_POSE_COORDINATES_PREDICTION = [[593, 645], [593, 626], [599, 621],
                                        [605, 617], [575, 637], [569, 640],
                                        [563, 643], [621, 616], [565, 652],
                                        [617, 652], [595, 667], [714, 662],
                                        [567, 749], [792, 559], [497, 844],
                                        [844, 435], [407, 906], [866, 403],
                                        [381, 921], [859, 392], [366, 922],
                                        [850, 405], [381, 918], [707, 948],
                                        [631, 940], [582, 1122], [599, 1097],
                                        [495, 1277], [641, 1239], [485, 1300],
                                        [658, 1257], [453, 1332], [626, 1308]]
EXPECTED_LEFT_HAND_COORDINATES_PREDICTION = [[843, 404], [862, 395], [876, 383],
                                             [887, 369], [896, 359], [854, 367],
                                             [868, 347], [879, 346], [885, 349],
                                             [843, 362], [859, 341], [871, 340],
                                             [878, 344], [837, 361], [849, 341],
                                             [859, 338], [867, 339], [834, 361],
                                             [841, 346], [848, 342], [854, 341]]
EXPECTED_RIGHT_HAND_COORDINATES_PREDICTION = [[391, 934], [371,
                                                           930], [354, 930],
                                              [340, 934], [328,
                                                           939], [350, 938],
                                              [339, 946], [347,
                                                           951], [355, 952],
                                              [356, 946], [346,
                                                           955], [358, 956],
                                              [366, 953], [361,
                                                           952], [354, 959],
                                              [364, 958], [372,
                                                           954], [366, 957],
                                              [359, 963], [364, 962],
                                              [368, 960]]


class PoseTest(parameterized.TestCase):

  def _verify_output_landmarks(self, landmark_list, image_shape, num_landmarks,
                               expected_results, diff_thresholds):
    self.assertLen(landmark_list.landmark, num_landmarks)
    image_rows, image_cols, _ = image_shape
    pose_coordinates = [(math.floor(landmark.x * image_cols),
                         math.floor(landmark.y * image_rows))
                        for landmark in landmark_list.landmark]
    prediction_error = np.abs(
        np.asarray(pose_coordinates) -
        np.asarray(expected_results[:num_landmarks]))
    npt.assert_array_less(prediction_error, diff_thresholds)

  def test_invalid_image_shape(self):
    holistic = mp_holistic.Holistic()
    with self.assertRaisesRegex(
        ValueError, 'Input image must contain three channel rgb data.'):
      holistic.process(np.arange(36, dtype=np.uint8).reshape(3, 3, 4))

  def test_blank_image(self):
    holistic = mp_holistic.Holistic()
    image = np.zeros([100, 100, 3], dtype=np.uint8)
    image.fill(255)
    results = holistic.process(image)
    self.assertIsNone(results.pose_landmarks)
    holistic.close()

  @parameterized.named_parameters(('static_image_mode', True, 3),
                                  ('video_mode', False, 3))
  def test_upper_body_model(self, static_image_mode, num_frames):
    image_path = os.path.join(os.path.dirname(__file__), 'testdata/pose.jpg')
    holistic = mp_holistic.Holistic(
        static_image_mode=static_image_mode, upper_body_only=True)
    image = cv2.imread(image_path)
    for _ in range(num_frames):
      results = holistic.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
      self._verify_output_landmarks(results.pose_landmarks, image.shape, 25,
                                    EXPECTED_POSE_COORDINATES_PREDICTION,
                                    POSE_DIFF_THRESHOLOD)
      self._verify_output_landmarks(results.left_hand_landmarks, image.shape,
                                    21,
                                    EXPECTED_LEFT_HAND_COORDINATES_PREDICTION,
                                    HAND_DIFF_THRESHOLOD)
      self._verify_output_landmarks(results.right_hand_landmarks, image.shape,
                                    21,
                                    EXPECTED_RIGHT_HAND_COORDINATES_PREDICTION,
                                    HAND_DIFF_THRESHOLOD)
      # TODO: Verify the correctness of the face landmarks.
      self.assertLen(results.face_landmarks.landmark, 468)
    holistic.close()

  @parameterized.named_parameters(('static_image_mode', True, 3),
                                  ('video_mode', False, 3))
  def test_full_body_model(self, static_image_mode, num_frames):
    image_path = os.path.join(os.path.dirname(__file__), 'testdata/pose.jpg')
    holistic = mp_holistic.Holistic(static_image_mode=static_image_mode)
    image = cv2.imread(image_path)

    for _ in range(num_frames):
      results = holistic.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
      self._verify_output_landmarks(results.pose_landmarks, image.shape, 33,
                                    EXPECTED_POSE_COORDINATES_PREDICTION,
                                    POSE_DIFF_THRESHOLOD)
      self._verify_output_landmarks(results.left_hand_landmarks, image.shape,
                                    21,
                                    EXPECTED_LEFT_HAND_COORDINATES_PREDICTION,
                                    HAND_DIFF_THRESHOLOD)
      self._verify_output_landmarks(results.right_hand_landmarks, image.shape,
                                    21,
                                    EXPECTED_RIGHT_HAND_COORDINATES_PREDICTION,
                                    HAND_DIFF_THRESHOLOD)
      # TODO: Verify the correctness of the face landmarks.
      self.assertLen(results.face_landmarks.landmark, 468)
    holistic.close()


if __name__ == '__main__':
  absltest.main()
