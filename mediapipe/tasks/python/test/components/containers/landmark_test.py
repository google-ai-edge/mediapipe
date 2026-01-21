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
"""Tests for Landmark conversion between Python and C."""

import dataclasses
from typing import Any

from absl.testing import absltest

from mediapipe.tasks.python.components.containers import landmark as landmark_lib
from mediapipe.tasks.python.components.containers import landmark_c as landmark_c_lib


class LandmarkTest(absltest.TestCase):

  def _expect_landmark_equal(
      self,
      actual: landmark_lib.Landmark,
      expected_values: dict[str, Any],
  ):
    actual_values = dataclasses.asdict(actual)
    self.assertDictAlmostEqual(actual_values, expected_values)

  def test_create_landmark_from_ctypes(self):
    c_landmark = landmark_c_lib.LandmarkC(
        x=0.1,
        y=0.2,
        z=0.3,
        has_visibility=True,
        visibility=0.4,
        has_presence=True,
        presence=0.5,
        name=b'test_landmark',
    )

    actual_landmark = landmark_lib.Landmark.from_ctypes(c_landmark)

    expected_values = {
        'x': 0.1,
        'y': 0.2,
        'z': 0.3,
        'visibility': 0.4,
        'presence': 0.5,
        'name': 'test_landmark',
    }
    self._expect_landmark_equal(actual_landmark, expected_values)

  def test_create_landmark_from_ctypes_without_optional_fields(self):
    c_landmark = landmark_c_lib.LandmarkC(
        x=0.1,
        y=0.2,
        z=0.3,
        has_visibility=False,
        visibility=0.0,
        has_presence=False,
        presence=0.0,
        name=None,
    )

    actual_landmark = landmark_lib.Landmark.from_ctypes(c_landmark)

    expected_values = {
        'x': 0.1,
        'y': 0.2,
        'z': 0.3,
        'visibility': None,
        'presence': None,
        'name': None,
    }
    self._expect_landmark_equal(actual_landmark, expected_values)


class NormalizedLandmarkTest(absltest.TestCase):

  def _expect_landmark_equal(
      self,
      actual: landmark_lib.NormalizedLandmark,
      expected_values: dict[str, Any],
  ):
    actual_values = dataclasses.asdict(actual)
    self.assertDictAlmostEqual(actual_values, expected_values)

  def test_create_normalized_landmark_from_ctypes(self):
    c_landmark = landmark_c_lib.NormalizedLandmarkC(
        x=0.1,
        y=0.2,
        z=0.3,
        has_visibility=True,
        visibility=0.4,
        has_presence=True,
        presence=0.5,
        name=b'test_landmark',
    )

    actual_landmark = landmark_lib.NormalizedLandmark.from_ctypes(c_landmark)

    expected_values = {
        'x': 0.1,
        'y': 0.2,
        'z': 0.3,
        'visibility': 0.4,
        'presence': 0.5,
        'name': 'test_landmark',
    }
    self._expect_landmark_equal(actual_landmark, expected_values)

  def test_create_normalized_landmark_from_ctypes_without_optional_fields(self):
    c_landmark = landmark_c_lib.NormalizedLandmarkC(
        x=0.1,
        y=0.2,
        z=0.3,
        has_visibility=False,
        visibility=0.0,
        has_presence=False,
        presence=0.0,
        name=None,
    )

    actual_landmark = landmark_lib.NormalizedLandmark.from_ctypes(c_landmark)

    expected_values = {
        'x': 0.1,
        'y': 0.2,
        'z': 0.3,
        'visibility': None,
        'presence': None,
        'name': None,
    }
    self._expect_landmark_equal(actual_landmark, expected_values)


if __name__ == '__main__':
  absltest.main()
