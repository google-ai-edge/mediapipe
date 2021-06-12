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
from typing import NamedTuple

from absl.testing import absltest
from absl.testing import parameterized
import cv2
import numpy as np
import numpy.testing as npt

# resources dependency
# undeclared dependency
from mediapipe.python.solutions import drawing_utils as mp_drawing
from mediapipe.python.solutions import pose as mp_pose

TEST_IMAGE_PATH = 'mediapipe/python/solutions/testdata'
DIFF_THRESHOLD = 15  # pixels
EXPECTED_POSE_LANDMARKS = np.array([[460, 283], [467, 273], [471, 273],
                                    [474, 273], [465, 273], [465, 273],
                                    [466, 273], [491, 277], [480, 277],
                                    [470, 294], [465, 294], [545, 319],
                                    [453, 329], [622, 323], [375, 316],
                                    [696, 316], [299, 307], [719, 316],
                                    [278, 306], [721, 311], [274, 304],
                                    [713, 313], [283, 306], [520, 476],
                                    [467, 471], [612, 550], [358, 490],
                                    [701, 613], [349, 611], [709, 624],
                                    [363, 630], [730, 633], [303, 628]])


class PoseTest(parameterized.TestCase):

  def _landmarks_list_to_array(self, landmark_list, image_shape):
    rows, cols, _ = image_shape
    return np.asarray([(lmk.x * cols, lmk.y * rows, lmk.z * cols)
                       for lmk in landmark_list.landmark])

  def _assert_diff_less(self, array1, array2, threshold):
    npt.assert_array_less(np.abs(array1 - array2), threshold)

  def _annotate(self, frame: np.ndarray, results: NamedTuple, idx: int):
    mp_drawing.draw_landmarks(frame, results.pose_landmarks,
                              mp_pose.POSE_CONNECTIONS)
    path = os.path.join(tempfile.gettempdir(), self.id().split('.')[-1] +
                                              '_frame_{}.png'.format(idx))
    cv2.imwrite(path, frame)

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

  @parameterized.named_parameters(('static_lite', True, 0, 3),
                                  ('static_full', True, 1, 3),
                                  ('static_heavy', True, 2, 3),
                                  ('video_lite', False, 0, 3),
                                  ('video_full', False, 1, 3),
                                  ('video_heavy', False, 2, 3))
  def test_on_image(self, static_image_mode, model_complexity, num_frames):
    image_path = os.path.join(os.path.dirname(__file__), 'testdata/pose.jpg')
    image = cv2.imread(image_path)
    with mp_pose.Pose(static_image_mode=static_image_mode,
                      model_complexity=model_complexity) as pose:
      for idx in range(num_frames):
        results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        self._annotate(image.copy(), results, idx)
        self._assert_diff_less(
            self._landmarks_list_to_array(results.pose_landmarks,
                                          image.shape)[:, :2],
            EXPECTED_POSE_LANDMARKS, DIFF_THRESHOLD)

  @parameterized.named_parameters(
      ('full', 1, 'pose_squats.full.npz'))
  def test_on_video(self, model_complexity, expected_name):
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
    frame_idx = 0
    with mp_pose.Pose(static_image_mode=False,
                      model_complexity=model_complexity) as pose:
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

        input_frame = cv2.cvtColor(input_frame, cv2.COLOR_RGB2BGR)
        self._annotate(input_frame, result, frame_idx)
        frame_idx += 1
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
