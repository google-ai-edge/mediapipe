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
# pylint: disable=unused-import
import tempfile
# pylint: enable=unused-import
from typing import NamedTuple

from absl.testing import absltest
from absl.testing import parameterized
import cv2
import numpy as np
import numpy.testing as npt
from PIL import Image

# resources dependency
# undeclared dependency
from mediapipe.python.solutions import drawing_styles
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
WORLD_DIFF_THRESHOLD = 0.2  # meters
EXPECTED_POSE_WORLD_LANDMARKS = np.array([
    [-0.11, -0.59, -0.15], [-0.09, -0.64, -0.16], [-0.09, -0.64, -0.16],
    [-0.09, -0.64, -0.16], [-0.11, -0.64, -0.14], [-0.11, -0.64, -0.14],
    [-0.11, -0.64, -0.14], [0.01, -0.65, -0.15], [-0.06, -0.64, -0.05],
    [-0.07, -0.57, -0.15], [-0.09, -0.57, -0.12], [0.18, -0.49, -0.09],
    [-0.14, -0.5, -0.03], [0.41, -0.48, -0.11], [-0.42, -0.5, -0.02],
    [0.64, -0.49, -0.17], [-0.63, -0.51, -0.13], [0.7, -0.5, -0.19],
    [-0.71, -0.53, -0.15], [0.72, -0.51, -0.23], [-0.69, -0.54, -0.19],
    [0.66, -0.49, -0.19], [-0.64, -0.52, -0.15], [0.09, 0., -0.04],
    [-0.09, -0., 0.03], [0.41, 0.23, -0.09], [-0.43, 0.1, -0.11],
    [0.69, 0.49, -0.04], [-0.48, 0.47, -0.02], [0.72, 0.52, -0.04],
    [-0.48, 0.51, -0.02], [0.8, 0.5, -0.14], [-0.59, 0.52, -0.11],
])
IOU_THRESHOLD = 0.85  # percents


class PoseTest(parameterized.TestCase):

  def _landmarks_list_to_array(self, landmark_list, image_shape):
    rows, cols, _ = image_shape
    return np.asarray([(lmk.x * cols, lmk.y * rows, lmk.z * cols)
                       for lmk in landmark_list.landmark])

  def _world_landmarks_list_to_array(self, landmark_list):
    return np.asarray([(lmk.x, lmk.y, lmk.z)
                       for lmk in landmark_list.landmark])

  def _assert_diff_less(self, array1, array2, threshold):
    npt.assert_array_less(np.abs(array1 - array2), threshold)

  def _get_output_path(self, name):
    return os.path.join(tempfile.gettempdir(), self.id().split('.')[-1] + name)

  def _annotate(self, frame: np.ndarray, results: NamedTuple, idx: int):
    mp_drawing.draw_landmarks(
        frame,
        results.pose_landmarks,
        mp_pose.POSE_CONNECTIONS,
        landmark_drawing_spec=drawing_styles.get_default_pose_landmarks_style())
    path = self._get_output_path('_frame_{}.png'.format(idx))
    cv2.imwrite(path, frame)

  def _annotate_segmentation(self, segmentation, expected_segmentation,
                             idx: int):
    path = self._get_output_path('_segmentation_{}.png'.format(idx))
    self._segmentation_to_rgb(segmentation).save(path)
    path = self._get_output_path('_segmentation_diff_{}.png'.format(idx))
    self._segmentation_diff_to_rgb(
        expected_segmentation, segmentation).save(path)

  def _rgb_to_segmentation(self, img, back_color=(255, 0, 0),
                           front_color=(0, 0, 255)):
    img = np.array(img)
    # Check all pixels are either front or back.
    is_back = (img == back_color).all(axis=2)
    is_front = (img == front_color).all(axis=2)
    np.logical_or(is_back, is_front).all()
    segm = np.zeros(img.shape[:2], dtype=np.uint8)
    segm[is_front] = 1
    return segm

  def _segmentation_to_rgb(self, segm, back_color=(255, 0, 0),
                           front_color=(0, 0, 255)):
    height, width = segm.shape
    img = np.zeros((height, width, 3), dtype=np.uint8)
    img[:, :] = back_color
    img[segm == 1] = front_color
    return Image.fromarray(img)

  def _segmentation_iou(self, segm_expected, segm_actual):
    intersection = segm_expected * segm_actual
    expected_dot = segm_expected * segm_expected
    actual_dot = segm_actual * segm_actual
    eps = np.finfo(np.float32).eps
    result = intersection.sum() / (expected_dot.sum() +
                                   actual_dot.sum() -
                                   intersection.sum() + eps)
    return result

  def _segmentation_diff_to_rgb(self, segm_expected, segm_actual,
                                expected_color=(0, 255, 0),
                                actual_color=(255, 0, 0)):
    height, width = segm_expected.shape
    img = np.zeros((height, width, 3), dtype=np.uint8)
    img[np.logical_and(segm_expected == 1, segm_actual == 0)] = expected_color
    img[np.logical_and(segm_expected == 0, segm_actual == 1)] = actual_color
    return Image.fromarray(img)

  def test_invalid_image_shape(self):
    with mp_pose.Pose() as pose:
      with self.assertRaisesRegex(
          ValueError, 'Input image must contain three channel rgb data.'):
        pose.process(np.arange(36, dtype=np.uint8).reshape(3, 3, 4))

  def test_blank_image(self):
    with mp_pose.Pose(enable_segmentation=True) as pose:
      image = np.zeros([100, 100, 3], dtype=np.uint8)
      image.fill(255)
      results = pose.process(image)
      self.assertIsNone(results.pose_landmarks)
      self.assertIsNone(results.segmentation_mask)

  def test_blank_image_with_extra_settings(self):
    with mp_pose.Pose(
        enable_segmentation=True,
        extra_settings=mp_pose.ExtraSettings(
            disallow_service_default_initialization=True
        ),
    ) as pose:
      image = np.zeros([100, 100, 3], dtype=np.uint8)
      image.fill(255)
      results = pose.process(image)
      self.assertIsNone(results.pose_landmarks)
      self.assertIsNone(results.segmentation_mask)

  @parameterized.named_parameters(('static_lite', True, 0, 3),
                                  ('static_full', True, 1, 3),
                                  ('static_heavy', True, 2, 3),
                                  ('video_lite', False, 0, 3),
                                  ('video_full', False, 1, 3),
                                  ('video_heavy', False, 2, 3))
  def test_on_image(self, static_image_mode, model_complexity, num_frames):
    image_path = os.path.join(os.path.dirname(__file__), 'testdata/pose.jpg')
    expected_segmentation_path = os.path.join(
        os.path.dirname(__file__), 'testdata/pose_segmentation.png')
    image = cv2.imread(image_path)
    expected_segmentation = self._rgb_to_segmentation(
        Image.open(expected_segmentation_path).convert('RGB'))

    with mp_pose.Pose(static_image_mode=static_image_mode,
                      model_complexity=model_complexity,
                      enable_segmentation=True) as pose:
      for idx in range(num_frames):
        results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        segmentation = results.segmentation_mask.round().astype(np.uint8)

        # TODO: Add rendering of world 3D when supported.
        self._annotate(image.copy(), results, idx)
        self._annotate_segmentation(segmentation, expected_segmentation, idx)

        self._assert_diff_less(
            self._landmarks_list_to_array(results.pose_landmarks,
                                          image.shape)[:, :2],
            EXPECTED_POSE_LANDMARKS, DIFF_THRESHOLD)
        self._assert_diff_less(
            self._world_landmarks_list_to_array(results.pose_world_landmarks),
            EXPECTED_POSE_WORLD_LANDMARKS, WORLD_DIFF_THRESHOLD)
        self.assertGreaterEqual(
            self._segmentation_iou(expected_segmentation, segmentation),
            IOU_THRESHOLD)

  @parameterized.named_parameters(
      ('full', 1, 'pose_squats.full.npz'))
  def test_on_video(self, model_complexity, expected_name):
    """Tests pose models on a video."""
    # Set threshold for comparing actual and expected predictions in pixels.
    diff_threshold = 15
    world_diff_threshold = 0.1

    video_path = os.path.join(os.path.dirname(__file__),
                              'testdata/pose_squats.mp4')
    expected_path = os.path.join(os.path.dirname(__file__),
                                 'testdata/{}'.format(expected_name))

    # Predict pose landmarks for each frame.
    video_cap = cv2.VideoCapture(video_path)
    actual_per_frame = []
    actual_world_per_frame = []
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
        pose_world_landmarks = self._world_landmarks_list_to_array(
            result.pose_world_landmarks)

        actual_per_frame.append(pose_landmarks)
        actual_world_per_frame.append(pose_world_landmarks)

        input_frame = cv2.cvtColor(input_frame, cv2.COLOR_RGB2BGR)
        self._annotate(input_frame, result, frame_idx)
        frame_idx += 1
    actual = np.array(actual_per_frame)
    actual_world = np.array(actual_world_per_frame)

    # Dump actual .npz.
    npz_path = self._get_output_path(expected_name)
    np.savez(npz_path, predictions=actual, predictions_world=actual_world)

    # Dump actual JSON.
    json_path = self._get_output_path(expected_name.replace('.npz', '.json'))
    with open(json_path, 'w') as fl:
      dump_data = {
          'predictions': np.around(actual, 3).tolist(),
          'predictions_world': np.around(actual_world, 3).tolist()
      }
      fl.write(json.dumps(dump_data, indent=2, separators=(',', ': ')))

    # Validate actual vs. expected landmarks.
    expected = np.load(expected_path)['predictions']
    assert actual.shape == expected.shape, (
        'Unexpected shape of predictions: {} instead of {}'.format(
            actual.shape, expected.shape))
    self._assert_diff_less(
        actual[..., :2], expected[..., :2], threshold=diff_threshold)

    # Validate actual vs. expected world landmarks.
    expected_world = np.load(expected_path)['predictions_world']
    assert actual_world.shape == expected_world.shape, (
        'Unexpected shape of world predictions: {} instead of {}'.format(
            actual_world.shape, expected_world.shape))
    self._assert_diff_less(
        actual_world, expected_world, threshold=world_diff_threshold)


if __name__ == '__main__':
  absltest.main()
