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

import json
import os
import tempfile

from absl.testing import absltest
from absl.testing import parameterized
import cv2
import numpy as np
import numpy.testing as npt

# resources dependency
from mediapipe.python.solutions import pose as mp_pose

TEST_IMAGE_PATH = 'mediapipe/python/solutions/testdata'
DIFF_THRESHOLD = 30  # pixels
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


class PoseTest(parameterized.TestCase):

  def _landmarks_list_to_array(self, landmark_list, image_shape):
    rows, cols, _ = image_shape
    return np.asarray([(lmk.x * cols, lmk.y * rows, lmk.z * cols)
                       for lmk in landmark_list.landmark])

  def _assert_diff_less(self, array1, array2, threshold):
    npt.assert_array_less(np.abs(array1 - array2), threshold)

  def test_invalid_image_shape(self):
    with mp_pose.Pose() as pose:
      with self.assertRaisesRegex(
          ValueError, 'Input image must contain three channel rgb data.'):
        pose.process(np.arange(36, dtype=np.uint8).reshape(3, 3, 4))

  def test_blank_image(self):
    with mp_pose.Pose() as pose:
      image = np.zeros([100, 100, 3], dtype=np.uint8)
      image.fill(255)
      results = pose.process(image)
      self.assertIsNone(results.pose_landmarks)

  @parameterized.named_parameters(('static_image_mode', True, 3),
                                  ('video_mode', False, 3))
  def test_upper_body_model(self, static_image_mode, num_frames):
    image_path = os.path.join(os.path.dirname(__file__), 'testdata/pose.jpg')
    with mp_pose.Pose(
        static_image_mode=static_image_mode, upper_body_only=True) as pose:
      image = cv2.imread(image_path)
      for _ in range(num_frames):
        results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        self._assert_diff_less(
            self._landmarks_list_to_array(results.pose_landmarks,
                                          image.shape)[:, :2],
            EXPECTED_UPPER_BODY_LANDMARKS, DIFF_THRESHOLD)

  @parameterized.named_parameters(('static_image_mode', True, 3),
                                  ('video_mode', False, 3))
  def test_full_body_model(self, static_image_mode, num_frames):
    image_path = os.path.join(os.path.dirname(__file__), 'testdata/pose.jpg')
    image = cv2.imread(image_path)

    with mp_pose.Pose(static_image_mode=static_image_mode) as pose:
      for _ in range(num_frames):
        results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        self._assert_diff_less(
            self._landmarks_list_to_array(results.pose_landmarks,
                                          image.shape)[:, :2],
            EXPECTED_FULL_BODY_LANDMARKS, DIFF_THRESHOLD)

  @parameterized.named_parameters(
      ('full_body', False, 'pose_squats.full_body.npz'),
      ('upper_body', True, 'pose_squats.upper_body.npz'))
  def test_on_video(self, upper_body_only, expected_name):
    """Tests pose models on a video."""
    # If set to `True` will dump actual predictions to .npz and JSON files.
    dump_predictions = False

    # Set threshold for comparing actual and expected predictions in pixels.
    diff_threshold = 50

    video_path = os.path.join(os.path.dirname(__file__),
                              'testdata/pose_squats.mp4')
    expected_path = os.path.join(os.path.dirname(__file__),
                                 'testdata/{}'.format(expected_name))

    # Predict pose landmarks for each frame.
    video_cap = cv2.VideoCapture(video_path)
    actual_per_frame = []
    with mp_pose.Pose(
        static_image_mode=False, upper_body_only=upper_body_only) as pose:
      while True:
        # Get next frame of the video.
        success, input_frame = video_cap.read()
        if not success:
          break

        # Run pose tracker.
        input_frame = cv2.cvtColor(input_frame, cv2.COLOR_BGR2RGB)
        result = pose.process(image=input_frame)
        pose_landmarks = self._landmarks_list_to_array(result.pose_landmarks,
                                                       input_frame.shape)

        actual_per_frame.append(pose_landmarks)
    actual = np.asarray(actual_per_frame)

    if dump_predictions:
      # Dump .npz
      with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        np.savez(tmp_file, predictions=np.array(actual))
        print('Predictions saved as .npz to {}'.format(tmp_file.name))

      # Dump JSON
      with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        with open(tmp_file.name, 'w') as fl:
          dump_data = {'predictions': np.around(actual, 3).tolist()}
          fl.write(json.dumps(dump_data, indent=2, separators=(',', ': ')))
          print('Predictions saved as JSON to {}'.format(tmp_file.name))

    # Validate actual vs. expected predictions.
    expected = np.load(expected_path)['predictions']
    assert actual.shape == expected.shape, (
        'Unexpected shape of predictions: {} instead of {}'.format(
            actual.shape, expected.shape))
    self._assert_diff_less(
        actual[..., :2], expected[..., :2], threshold=diff_threshold)


if __name__ == '__main__':
  absltest.main()
