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
"""Tests for Category conversion between Python and C."""

import dataclasses
from typing import Any, Mapping

from absl.testing import absltest
import immutabledict

from mediapipe.tasks.python.components.containers import category as category_lib
from mediapipe.tasks.python.components.containers import category_c as category_c_lib

_CATEGORY_WITH_NAMES = category_c_lib.CategoryC(
    index=1,
    score=0.9,
    category_name=b'test_category_with_names',
    display_name=b'Test Category 1',
)
_DICT_WITH_NAMES = immutabledict.immutabledict({
    'index': 1,
    'score': 0.9,
    'category_name': 'test_category_with_names',
    'display_name': 'Test Category 1',
})
_CATEGORY_WITHOUT_NAMES = category_c_lib.CategoryC(
    index=2,
    score=0.8,
    category_name=None,
    display_name=None,
)
_DICT_WITHOUT_NAMES = immutabledict.immutabledict({
    'index': 2,
    'score': 0.8,
    'category_name': None,
    'display_name': None,
})


class CategoryTest(absltest.TestCase):

  def _expect_category_equal(
      self,
      actual: category_lib.Category,
      expected_values: Mapping[str, Any],
  ):
    actual_values = dataclasses.asdict(actual)
    self.assertDictAlmostEqual(actual_values, expected_values)

  def test_create_category_from_ctypes(self):
    actual_category = category_lib.Category.from_ctypes(_CATEGORY_WITH_NAMES)

    self._expect_category_equal(actual_category, _DICT_WITH_NAMES)

  def test_create_category_from_ctypes_without_name_fields(self):
    actual_category = category_lib.Category.from_ctypes(_CATEGORY_WITHOUT_NAMES)

    self._expect_category_equal(actual_category, _DICT_WITHOUT_NAMES)

  def test_create_category_from_ctypes_with_unknown_index(self):
    values = category_c_lib.CategoryC(
        index=-1,
        score=0.8,
        category_name=None,
        display_name=None,
    )
    category = category_lib.Category.from_ctypes(values)

    self._expect_category_equal(category, {
        'index': None,
        'score': 0.8,
        'category_name': None,
        'display_name': None,
    })

  def test_create_categories_from_ctypes(self):
    c_categories = (category_c_lib.CategoryC * 2)(
        _CATEGORY_WITH_NAMES, _CATEGORY_WITHOUT_NAMES
    )
    c_categories_ptr = category_c_lib.CategoriesC(
        categories=c_categories,
        categories_count=2,
    )

    actual_categories = category_lib.create_list_of_categories_from_ctypes(
        c_categories_ptr
    )

    self.assertLen(actual_categories, 2)
    with self.subTest('CategoryWithNamesConvertedCorrectly'):
      self._expect_category_equal(actual_categories[0], _DICT_WITH_NAMES)
    with self.subTest('CategoryWithoutNamesConvertedCorrectly'):
      self._expect_category_equal(actual_categories[1], _DICT_WITHOUT_NAMES)
