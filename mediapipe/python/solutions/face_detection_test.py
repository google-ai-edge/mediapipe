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

from absl.testing import absltest
import cv2
import numpy as np
import numpy.testing as npt

# resources dependency
from mediapipe.python.solutions import face_detection as mp_faces

TEST_IMAGE_PATH = 'mediapipe/python/solutions/testdata'
EXPECTED_FACE_KEY_POINTS = [[182, 368], [186, 467], [236, 416], [284, 415],
                            [203, 310], [212, 521]]
DIFF_THRESHOLD = 10  # pixels


class FaceDetectionTest(absltest.TestCase):

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

  def test_face(self):
    image_path = os.path.join(os.path.dirname(__file__), 'testdata/face.jpg')
    image = cv2.flip(cv2.imread(image_path), 1)

    with mp_faces.FaceDetection(min_detection_confidence=0.5) as faces:
      for _ in range(5):
        results = faces.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        location_data = results.detections[0].location_data
        x = [keypoint.x for keypoint in location_data.relative_keypoints]
        y = [keypoint.y for keypoint in location_data.relative_keypoints]
        face_keypoints = np.transpose(np.stack((y, x))) * image.shape[0:2]
        prediction_error = np.abs(
            np.asarray(face_keypoints) - np.asarray(EXPECTED_FACE_KEY_POINTS))

        self.assertLen(results.detections, 1)
        self.assertLen(location_data.relative_keypoints, 6)
        npt.assert_array_less(prediction_error, DIFF_THRESHOLD)


if __name__ == '__main__':
  absltest.main()
