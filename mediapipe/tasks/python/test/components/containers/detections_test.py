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
"""Tests for Detections conversion between Python and C."""

from typing import Mapping
import dataclasses
from typing import Any

from absl.testing import absltest

from mediapipe.tasks.python.components.containers import bounding_box as bounding_box_lib
from mediapipe.tasks.python.components.containers import category as category_lib
from mediapipe.tasks.python.components.containers import category_c as category_c_lib
from mediapipe.tasks.python.components.containers import detections as detections_lib
from mediapipe.tasks.python.components.containers import detections_c as detections_c_lib
from mediapipe.tasks.python.components.containers import keypoint as keypoint_lib
from mediapipe.tasks.python.components.containers import keypoint_c as keypoint_c_lib
from mediapipe.tasks.python.components.containers import rect_c as rect_c_lib


_CATEGORY_WITH_NAMES = category_c_lib.CategoryC(
    index=1,
    score=0.9,
    category_name=b'test_category_WITH_NAMES',
    display_name=b'Test Category 1',
)
_CATEGORY_WITH_NAMES_DICT = {
    'index': 1,
    'score': 0.9,
    'category_name': 'test_category_WITH_NAMES',
    'display_name': 'Test Category 1',
}

_CATEGORY_WITHOUT_NAMES = category_c_lib.CategoryC(
    index=2,
    score=0.8,
    category_name=b'test_category_WITHOUT_NAMES',
    display_name=b'Test Category 2',
)
_CATEGORY_WITHOUT_NAMES_DICT = {
    'index': 2,
    'score': 0.8,
    'category_name': 'test_category_WITHOUT_NAMES',
    'display_name': 'Test Category 2',
}

_KEYPOINT_1 = keypoint_c_lib.NormalizedKeypointC(
    x=0.1, y=0.2, label=b'keypoint1', score=0.9
)
_KEYPOINT_1_DICT = {
    'x': 0.1,
    'y': 0.2,
    'label': 'keypoint1',
    'score': 0.9,
}

_KEYPOINT_2 = keypoint_c_lib.NormalizedKeypointC(
    x=0.3, y=0.4, label=b'keypoint2', score=0.8
)
_KEYPOINT_2_DICT = {
    'x': 0.3,
    'y': 0.4,
    'label': 'keypoint2',
    'score': 0.8,
}

_RECT_1 = rect_c_lib.RectC(left=10, top=20, right=50, bottom=70)
_RECT_1_DICT = {
    'origin_x': 10,
    'origin_y': 20,
    'width': 40,
    'height': 50,
}

_RECT_2 = rect_c_lib.RectC(left=15, top=25, right=55, bottom=75)
_RECT_2_DICT = {
    'origin_x': 15,
    'origin_y': 25,
    'width': 40,
    'height': 50,
}


class DetectionsTest(absltest.TestCase):

  def _assert_categories_equal(
      self,
      actual_categories: list[category_lib.Category],
      expected_categories: list[Mapping[str, Any]],
  ):
    self.assertEqual(len(actual_categories), len(expected_categories))
    for i, actual_category in enumerate(actual_categories):
      actual_values = dataclasses.asdict(actual_category)
      self.assertDictAlmostEqual(actual_values, expected_categories[i])

  def _assert_keypoints_equal(
      self,
      actual_keypoints: list[keypoint_lib.NormalizedKeypoint],
      expected_keypoints: list[Mapping[str, Any]],
  ):
    self.assertEqual(len(actual_keypoints), len(expected_keypoints))
    for i, actual_keypoint in enumerate(actual_keypoints):
      actual_values = dataclasses.asdict(actual_keypoint)
      self.assertDictAlmostEqual(actual_values, expected_keypoints[i])

  def _assert_bounding_box_equal(
      self,
      actual: bounding_box_lib.BoundingBox,
      expected_values: Mapping[str, Any],
  ):
    actual_values = dataclasses.asdict(actual)
    self.assertDictAlmostEqual(actual_values, expected_values)

  def _assert_detection_matches(
      self,
      actual: detections_lib.Detection,
      expected_bounding_box: Mapping[str, Any],
      expected_categories: list[Mapping[str, Any]],
      expected_keypoints: list[Mapping[str, Any]] | None = None,
  ):
    self._assert_bounding_box_equal(actual.bounding_box, expected_bounding_box)
    self._assert_categories_equal(actual.categories, expected_categories)

    if expected_keypoints:
      self._assert_keypoints_equal(actual.keypoints, expected_keypoints)
    else:
      self.assertIsNone(actual.keypoints)

  def test_create_detection_from_ctypes(self):
    c_categories = (category_c_lib.CategoryC * 2)(
        _CATEGORY_WITH_NAMES, _CATEGORY_WITHOUT_NAMES
    )
    c_keypoints = (keypoint_c_lib.NormalizedKeypointC * 2)(
        _KEYPOINT_1, _KEYPOINT_2
    )
    c_detection = detections_c_lib.DetectionC(
        categories=c_categories,
        categories_count=2,
        bounding_box=_RECT_1,
        keypoints=c_keypoints,
        keypoints_count=2,
    )

    actual_detection = detections_lib.Detection.from_ctypes(c_detection)

    self._assert_detection_matches(
        actual_detection,
        _RECT_1_DICT,
        [_CATEGORY_WITH_NAMES_DICT, _CATEGORY_WITHOUT_NAMES_DICT],
        [_KEYPOINT_1_DICT, _KEYPOINT_2_DICT],
    )

  def test_create_detection_from_ctypes_without_keypoints(self):
    c_categories = (category_c_lib.CategoryC * 1)(_CATEGORY_WITH_NAMES)
    c_detection = detections_c_lib.DetectionC(
        categories=c_categories,
        categories_count=1,
        bounding_box=_RECT_2,
        keypoints=None,
        keypoints_count=0,
    )

    actual_detection = detections_lib.Detection.from_ctypes(c_detection)

    self._assert_detection_matches(
        actual_detection,
        _RECT_2_DICT,
        [_CATEGORY_WITH_NAMES_DICT],
        None,
    )

  def test_create_detection_result_from_ctypes(self):
    c_categories_1 = (category_c_lib.CategoryC * 1)(_CATEGORY_WITH_NAMES)
    c_keypoints_1 = (keypoint_c_lib.NormalizedKeypointC * 1)(_KEYPOINT_1)
    c_detection_1 = detections_c_lib.DetectionC(
        categories=c_categories_1,
        categories_count=1,
        bounding_box=_RECT_1,
        keypoints=c_keypoints_1,
        keypoints_count=1,
    )

    c_categories_2 = (category_c_lib.CategoryC * 1)(_CATEGORY_WITHOUT_NAMES)
    c_keypoints_2 = (keypoint_c_lib.NormalizedKeypointC * 1)(_KEYPOINT_2)
    c_detection_2 = detections_c_lib.DetectionC(
        categories=c_categories_2,
        categories_count=1,
        bounding_box=_RECT_2,
        keypoints=c_keypoints_2,
        keypoints_count=1,
    )

    c_detections = (detections_c_lib.DetectionC * 2)(
        c_detection_1, c_detection_2
    )
    c_detection_result = detections_c_lib.DetectionResultC(
        detections=c_detections,
        detections_count=2,
    )

    actual_detection_result = detections_lib.DetectionResult.from_ctypes(
        c_detection_result
    )

    self.assertLen(actual_detection_result.detections, 2)
    with self.subTest('FirstDetectionConvertedCorrectly'):
      self._assert_detection_matches(
          actual_detection_result.detections[0],
          _RECT_1_DICT,
          [_CATEGORY_WITH_NAMES_DICT],
          [_KEYPOINT_1_DICT],
      )
    with self.subTest('SecondDetectionConvertedCorrectly'):
      self._assert_detection_matches(
          actual_detection_result.detections[1],
          _RECT_2_DICT,
          [_CATEGORY_WITHOUT_NAMES_DICT],
          [_KEYPOINT_2_DICT],
      )


if __name__ == '__main__':
  absltest.main()
