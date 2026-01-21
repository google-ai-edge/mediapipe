# Copyright 2025 The MediaPipe Authors.
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
"""Tests for BoundingBox conversion between Python and C."""

from typing import Mapping
import dataclasses

from absl.testing import absltest

from mediapipe.tasks.python.components.containers import bounding_box as bounding_box_lib
from mediapipe.tasks.python.components.containers import rect_c as rect_c_lib


class BoundingBoxTest(absltest.TestCase):

  def _expect_bounding_box_equal(
      self,
      actual: bounding_box_lib.BoundingBox,
      expected_values: Mapping[str, int],
  ) -> None:
    actual_values = dataclasses.asdict(actual)
    self.assertDictAlmostEqual(actual_values, expected_values)

  def test_create_bounding_box_from_ctypes_converts_values(self):
    c_rect = rect_c_lib.RectC(left=10, top=20, right=50, bottom=70)

    actual_bounding_box = bounding_box_lib.BoundingBox.from_ctypes(c_rect)

    expected_values = {
        'origin_x': 10,
        'origin_y': 20,
        'width': 40,
        'height': 50,
    }
    self._expect_bounding_box_equal(actual_bounding_box, expected_values)


if __name__ == '__main__':
  absltest.main()
