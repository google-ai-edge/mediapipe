# Copyright 2021 The MediaPipe Authors.
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
"""Tests for mediapipe.python.solutions.face_detection."""

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
from mediapipe.python.solutions import face_detection as mp_faces

TEST_IMAGE_PATH = 'mediapipe/python/solutions/testdata'
SHORT_RANGE_EXPECTED_FACE_KEY_POINTS = [[363, 182], [460, 186], [420, 241],
                                        [417, 284], [295, 199], [502, 198]]
FULL_RANGE_EXPECTED_FACE_KEY_POINTS = [[363, 181], [455, 181], [413, 233],
                                       [411, 278], [306, 204], [499, 207]]
DIFF_THRESHOLD = 5  # pixels


class FaceDetectionTest(parameterized.TestCase):

  def _annotate(self, frame: np.ndarray, results: NamedTuple, idx: int):
    for detection in results.detections:
      mp_drawing.draw_detection(frame, detection)
    path = os.path.join(tempfile.gettempdir(), self.id().split('.')[-1] +
                                              '_frame_{}.png'.format(idx))
    cv2.imwrite(path, frame)

  def test_invalid_image_shape(self):
    with mp_faces.FaceDetection() as faces:
      with self.assertRaisesRegex(
          ValueError, 'Input image must contain three channel rgb data.'):
        faces.process(np.arange(36, dtype=np.uint8).reshape(3, 3, 4))

  def test_blank_image(self):
    image = np.zeros([100, 100, 3], dtype=np.uint8)
    image.fill(255)
    with mp_faces.FaceDetection(min_detection_confidence=0.5) as faces:
      results = faces.process(image)
      self.assertIsNone(results.detections)

  @parameterized.named_parameters(('short_range_model', 0),
                                  ('full_range_model', 1))
  def test_face(self, model_selection):
    image_path = os.path.join(os.path.dirname(__file__),
                              'testdata/portrait.jpg')
    image = cv2.imread(image_path)
    rows, cols, _ = image.shape
    with mp_faces.FaceDetection(
        min_detection_confidence=0.5, model_selection=model_selection) as faces:
      for idx in range(5):
        results = faces.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        self._annotate(image.copy(), results, idx)
        location_data = results.detections[0].location_data
        x = [keypoint.x * cols for keypoint in location_data.relative_keypoints]
        y = [keypoint.y * rows for keypoint in location_data.relative_keypoints]
        face_keypoints = np.column_stack((x, y))
        if model_selection == 0:
          prediction_error = np.abs(
              np.asarray(face_keypoints) -
              np.asarray(SHORT_RANGE_EXPECTED_FACE_KEY_POINTS))
        else:
          prediction_error = np.abs(
              np.asarray(face_keypoints) -
              np.asarray(FULL_RANGE_EXPECTED_FACE_KEY_POINTS))

        self.assertLen(results.detections, 1)
        self.assertLen(location_data.relative_keypoints, 6)
        npt.assert_array_less(prediction_error, DIFF_THRESHOLD)


if __name__ == '__main__':
  absltest.main()
