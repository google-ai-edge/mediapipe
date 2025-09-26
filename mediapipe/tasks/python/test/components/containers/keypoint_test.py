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
"""Tests for NormalizedKeypoint conversion between Python and C."""

from typing import Mapping
import dataclasses
from typing import Any

from absl.testing import absltest
from absl.testing import parameterized

from mediapipe.tasks.python.components.containers import keypoint as keypoint_lib
from mediapipe.tasks.python.components.containers import keypoint_c as keypoint_c_lib


class NormalizedKeypointTest(parameterized.TestCase):

  def _expect_keypoint_equal(
      self,
      actual: keypoint_lib.NormalizedKeypoint,
      expected_values: Mapping[str, Any],
  ):
    actual_values = dataclasses.asdict(actual)
    self.assertDictAlmostEqual(actual_values, expected_values)

  @parameterized.named_parameters(
      dict(
          testcase_name='with_optional_fields',
          x=0.1,
          y=0.2,
          label='test_label',
          score=0.9,
      ),
      dict(
          testcase_name='without_optional_fields',
          x=0.1,
          y=0.2,
          label=None,
          score=0.0,
      ),
  )
  def test_create_from_ctypes_succeeds(
      self, x: float, y: float, label: str | None, score: float
  ):
    c_keypoint = keypoint_c_lib.NormalizedKeypointC(
        x=x, y=y, label=label.encode('utf-8') if label else None, score=score
    )
    actual_keypoint = keypoint_lib.NormalizedKeypoint.from_ctypes(c_keypoint)

    expected_keypoint_values = {
        'x': x,
        'y': y,
        'label': label,
        'score': score,
    }
    self._expect_keypoint_equal(actual_keypoint, expected_keypoint_values)


if __name__ == '__main__':
  absltest.main()
