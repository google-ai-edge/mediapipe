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

"""Tests for mediapipe.python.solutions.face_mesh."""

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
from mediapipe.python.solutions import drawing_styles
from mediapipe.python.solutions import drawing_utils as mp_drawing
from mediapipe.python.solutions import face_mesh as mp_faces

TEST_IMAGE_PATH = 'mediapipe/python/solutions/testdata'
DIFF_THRESHOLD = 5  # pixels
EYE_INDICES_TO_LANDMARKS = {
    33: [345, 178],
    7: [348, 179],
    163: [352, 178],
    144: [357, 179],
    145: [365, 179],
    153: [371, 179],
    154: [378, 178],
    155: [381, 177],
    133: [383, 177],
    246: [347, 175],
    161: [350, 174],
    160: [355, 172],
    159: [362, 170],
    158: [368, 171],
    157: [375, 172],
    173: [380, 175],
    263: [467, 176],
    249: [464, 177],
    390: [460, 177],
    373: [455, 178],
    374: [448, 179],
    380: [441, 179],
    381: [435, 178],
    382: [432, 177],
    362: [430, 177],
    466: [465, 175],
    388: [462, 173],
    387: [457, 171],
    386: [450, 170],
    385: [444, 171],
    384: [437, 172],
    398: [432, 175]
}


class FaceMeshTest(parameterized.TestCase):

  def _annotate(self, frame: np.ndarray, results: NamedTuple, idx: int):
    for face_landmarks in results.multi_face_landmarks:
      mp_drawing.draw_landmarks(
          frame,
          face_landmarks,
          mp_faces.FACEMESH_TESSELATION,
          landmark_drawing_spec=None,
          connection_drawing_spec=drawing_styles
          .get_default_face_mesh_tesselation_style())
      mp_drawing.draw_landmarks(
          frame,
          face_landmarks,
          mp_faces.FACEMESH_CONTOURS,
          landmark_drawing_spec=None,
          connection_drawing_spec=drawing_styles
          .get_default_face_mesh_contours_style())
    path = os.path.join(tempfile.gettempdir(), self.id().split('.')[-1] +
                                              '_frame_{}.png'.format(idx))
    cv2.imwrite(path, frame)

  def test_invalid_image_shape(self):
    with mp_faces.FaceMesh() as faces:
      with self.assertRaisesRegex(
          ValueError, 'Input image must contain three channel rgb data.'):
        faces.process(np.arange(36, dtype=np.uint8).reshape(3, 3, 4))

  def test_blank_image(self):
    with mp_faces.FaceMesh() as faces:
      image = np.zeros([100, 100, 3], dtype=np.uint8)
      image.fill(255)
      results = faces.process(image)
      self.assertIsNone(results.multi_face_landmarks)

  @parameterized.named_parameters(('static_image_mode', True, 1),
                                  ('video_mode', False, 5))
  def test_face(self, static_image_mode: bool, num_frames: int):
    image_path = os.path.join(os.path.dirname(__file__),
                              'testdata/portrait.jpg')
    image = cv2.imread(image_path)
    rows, cols, _ = image.shape
    with mp_faces.FaceMesh(
        static_image_mode=static_image_mode,
        min_detection_confidence=0.5) as faces:
      for idx in range(num_frames):
        results = faces.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        self._annotate(image.copy(), results, idx)
        multi_face_landmarks = []
        for landmarks in results.multi_face_landmarks:
          self.assertLen(landmarks.landmark, 468)
          x = [landmark.x * cols for landmark in landmarks.landmark]
          y = [landmark.y * rows for landmark in landmarks.landmark]
          face_landmarks = np.column_stack((x, y))
          multi_face_landmarks.append(face_landmarks)
        self.assertLen(multi_face_landmarks, 1)
        # Verify the eye landmarks are correct as sanity check.
        for eye_idx, gt_lds in EYE_INDICES_TO_LANDMARKS.items():
          prediction_error = np.abs(
              np.asarray(multi_face_landmarks[0][eye_idx]) - np.asarray(gt_lds))
          npt.assert_array_less(prediction_error, DIFF_THRESHOLD)


if __name__ == '__main__':
  absltest.main()
