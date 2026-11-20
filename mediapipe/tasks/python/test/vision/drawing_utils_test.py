# Copyright 2026 The MediaPipe Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tests for drawing_utils."""

from absl.testing import absltest
import numpy as np

from mediapipe.tasks.python.components.containers import bounding_box as bounding_box_module
from mediapipe.tasks.python.components.containers import detections as detections_module
from mediapipe.tasks.python.components.containers import keypoint as keypoint_module
from mediapipe.tasks.python.vision import drawing_utils

_BoundingBox = bounding_box_module.BoundingBox
_Detection = detections_module.Detection
_NormalizedKeypoint = keypoint_module.NormalizedKeypoint


class DrawingUtilsTest(absltest.TestCase):

  def test_draw_detection_skips_keypoints_outside_the_image(self):
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    detection = _Detection(
        bounding_box=_BoundingBox(
            origin_x=40, origin_y=40, width=20, height=20
        ),
        categories=[],
        keypoints=[_NormalizedKeypoint(x=1.5, y=0.5)],
    )

    drawing_utils.draw_detection(image, detection)

    # Only the bounding box is drawn: nothing may land in the top-left corner.
    self.assertFalse(image[:20, :20].any())
    self.assertTrue(image[40:60, 40:60].any())

  def test_draw_detection_draws_keypoints_inside_the_image(self):
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    detection = _Detection(
        bounding_box=_BoundingBox(
            origin_x=40, origin_y=40, width=20, height=20
        ),
        categories=[],
        keypoints=[_NormalizedKeypoint(x=0.1, y=0.1)],
    )

    drawing_utils.draw_detection(image, detection)

    self.assertTrue(image[5:15, 5:15].any())


if __name__ == '__main__':
  absltest.main()
