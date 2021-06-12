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

"""Tests for mediapipe.python.solutions.objectron."""

import os

from absl.testing import absltest
from absl.testing import parameterized
import cv2
import numpy as np
import numpy.testing as npt

# resources dependency
from mediapipe.python.solutions import objectron as mp_objectron

TEST_IMAGE_PATH = 'mediapipe/python/solutions/testdata'
DIFF_THRESHOLD = 30  # pixels
EXPECTED_BOX_COORDINATES_PREDICTION = [[[236, 413], [408, 474], [135, 457],
                                        [383, 505], [80, 478], [408, 345],
                                        [130, 347], [384, 355], [72, 353]],
                                       [[241, 206], [411, 279], [131, 280],
                                        [392, 249], [78, 252], [412, 155],
                                        [140, 178], [396, 105], [89, 137]]]


class ObjectronTest(parameterized.TestCase):

  def test_invalid_image_shape(self):
    with mp_objectron.Objectron() as objectron:
      with self.assertRaisesRegex(
          ValueError, 'Input image must contain three channel rgb data.'):
        objectron.process(np.arange(36, dtype=np.uint8).reshape(3, 3, 4))

  def test_blank_image(self):
    with mp_objectron.Objectron() as objectron:
      image = np.zeros([100, 100, 3], dtype=np.uint8)
      image.fill(255)
      results = objectron.process(image)
      self.assertIsNone(results.detected_objects)

  @parameterized.named_parameters(('static_image_mode', True, 1),
                                  ('video_mode', False, 5))
  def test_multi_objects(self, static_image_mode, num_frames):
    image_path = os.path.join(os.path.dirname(__file__), 'testdata/shoes.jpg')
    image = cv2.imread(image_path)

    with mp_objectron.Objectron(
        static_image_mode=static_image_mode,
        max_num_objects=2,
        min_detection_confidence=0.5) as objectron:
      for _ in range(num_frames):
        results = objectron.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        multi_box_coordinates = []
        for detected_object in results.detected_objects:
          landmarks = detected_object.landmarks_2d
          self.assertLen(landmarks.landmark, 9)
          x = [landmark.x for landmark in landmarks.landmark]
          y = [landmark.y for landmark in landmarks.landmark]
          box_coordinates = np.transpose(np.stack((y, x))) * image.shape[0:2]
          multi_box_coordinates.append(box_coordinates)
        self.assertLen(multi_box_coordinates, 2)
        prediction_error = np.abs(
            np.asarray(multi_box_coordinates) -
            np.asarray(EXPECTED_BOX_COORDINATES_PREDICTION))
        npt.assert_array_less(prediction_error, DIFF_THRESHOLD)


if __name__ == '__main__':
  absltest.main()
