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
from mediapipe.python.solutions import pose as mp_pose

TEST_IMAGE_PATH = 'mediapipe/python/solutions/testdata'
DIFF_THRESHOLOD = 30  # pixels
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


class PoseTest(parameterized.TestCase):

  def _verify_output_landmarks(self, landmark_list, image_shape, num_landmarks):
    self.assertLen(landmark_list.landmark, num_landmarks)
    image_rows, image_cols, _ = image_shape
    pose_coordinates = [(math.floor(landmark.x * image_cols),
                         math.floor(landmark.y * image_rows))
                        for landmark in landmark_list.landmark]
    prediction_error = np.abs(
        np.asarray(pose_coordinates) -
        np.asarray(EXPECTED_POSE_COORDINATES_PREDICTION[:num_landmarks]))
    npt.assert_array_less(prediction_error, DIFF_THRESHOLOD)

  def test_invalid_image_shape(self):
    pose = mp_pose.Pose()
    with self.assertRaisesRegex(
        ValueError, 'Input image must contain three channel rgb data.'):
      pose.process(np.arange(36, dtype=np.uint8).reshape(3, 3, 4))

  def test_blank_image(self):
    pose = mp_pose.Pose()
    image = np.zeros([100, 100, 3], dtype=np.uint8)
    image.fill(255)
    results = pose.process(image)
    self.assertIsNone(results.pose_landmarks)
    pose.close()

  @parameterized.named_parameters(('static_image_mode', True, 3),
                                  ('video_mode', False, 3))
  def test_upper_body_model(self, static_image_mode, num_frames):
    image_path = os.path.join(os.path.dirname(__file__), 'testdata/pose.jpg')
    pose = mp_pose.Pose(static_image_mode=static_image_mode,
                        upper_body_only=True)
    image = cv2.imread(image_path)

    for _ in range(num_frames):
      results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
      self._verify_output_landmarks(results.pose_landmarks, image.shape, 25)
    pose.close()

  @parameterized.named_parameters(('static_image_mode', True, 3),
                                  ('video_mode', False, 3))
  def test_full_body_model(self, static_image_mode, num_frames):
    image_path = os.path.join(os.path.dirname(__file__), 'testdata/pose.jpg')
    pose = mp_pose.Pose(static_image_mode=static_image_mode)
    image = cv2.imread(image_path)

    for _ in range(num_frames):
      results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
      self._verify_output_landmarks(results.pose_landmarks, image.shape, 33)
    pose.close()


if __name__ == '__main__':
  absltest.main()
