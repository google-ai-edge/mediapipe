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
from mediapipe.python.solutions import drawing_utils as mp_drawing
from mediapipe.python.solutions import face_mesh as mp_faces

TEST_IMAGE_PATH = 'mediapipe/python/solutions/testdata'
DIFF_THRESHOLD = 5  # pixels
EYE_INDICES_TO_LANDMARKS = {
    33: [178, 345],
    7: [179, 348],
    163: [178, 352],
    144: [179, 357],
    145: [179, 365],
    153: [179, 371],
    154: [178, 378],
    155: [177, 381],
    133: [177, 383],
    246: [175, 347],
    161: [174, 350],
    160: [172, 355],
    159: [170, 362],
    158: [171, 368],
    157: [172, 375],
    173: [175, 380],
    263: [176, 467],
    249: [177, 464],
    390: [177, 460],
    373: [178, 455],
    374: [179, 448],
    380: [179, 441],
    381: [178, 435],
    382: [177, 432],
    362: [177, 430],
    466: [175, 465],
    388: [173, 462],
    387: [171, 457],
    386: [170, 450],
    385: [171, 444],
    384: [172, 437],
    398: [175, 432]
}


class FaceMeshTest(parameterized.TestCase):

  def _annotate(self, frame: np.ndarray, results: NamedTuple, idx: int):
    drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
    for face_landmarks in results.multi_face_landmarks:
      mp_drawing.draw_landmarks(
          image=frame,
          landmark_list=face_landmarks,
          landmark_drawing_spec=drawing_spec)
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
    with mp_faces.FaceMesh(
        static_image_mode=static_image_mode,
        min_detection_confidence=0.5) as faces:
      for idx in range(num_frames):
        results = faces.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        self._annotate(image.copy(), results, idx)
        multi_face_landmarks = []
        for landmarks in results.multi_face_landmarks:
          self.assertLen(landmarks.landmark, 468)
          x = [landmark.x for landmark in landmarks.landmark]
          y = [landmark.y for landmark in landmarks.landmark]
          face_landmarks = np.transpose(np.stack((y, x))) * image.shape[0:2]
          multi_face_landmarks.append(face_landmarks)
        self.assertLen(multi_face_landmarks, 1)
        # Verify the eye landmarks are correct as sanity check.
        for eye_idx, gt_lds in EYE_INDICES_TO_LANDMARKS.items():
          prediction_error = np.abs(
              np.asarray(multi_face_landmarks[0][eye_idx]) - np.asarray(gt_lds))
          npt.assert_array_less(prediction_error, DIFF_THRESHOLD)


if __name__ == '__main__':
  absltest.main()
