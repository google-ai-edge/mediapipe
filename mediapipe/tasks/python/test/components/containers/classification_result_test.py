"""Tests for ClassificationResult conversion between Python and C."""

from absl.testing import absltest
from mediapipe.tasks.python.components.containers import category
from mediapipe.tasks.python.components.containers import category_c
from mediapipe.tasks.python.components.containers import classification_result
from mediapipe.tasks.python.components.containers import classification_result_c


_MOCK_CATEGORY_NAME = "test_category"
_MOCK_DISPLAY_NAME = "Test Category"
_MOCK_TIMESTAMP_MS = 1000
_MOCK_HEAD_NAME = "Head"
_MOCK_HEAD_INDEX = 0
_MOCK_CATEGORY = category.Category(
    index=1,
    score=0.95,
    category_name=_MOCK_CATEGORY_NAME,
    display_name=_MOCK_DISPLAY_NAME,
)


def _create_classification_result_c(
    categories_count: int, classifications_count: int
):
  """Creates a ClassificationResultC struct.

  Args:
    categories_count: The number of categories to create.
    classifications_count: The number of classifications to create.

  Returns:
    A ClassificationResultC struct with the given properties.
  """
  c_category = category_c.CategoryC(
      index=_MOCK_CATEGORY.index,
      score=_MOCK_CATEGORY.score,
      category_name=_MOCK_CATEGORY_NAME.encode("utf-8"),
      display_name=_MOCK_DISPLAY_NAME.encode("utf-8"),
  )
  c_categories = (category_c.CategoryC * categories_count)(
      *[c_category for _ in range(categories_count)]
  )
  c_classifications = classification_result_c.ClassificationsC(
      categories=c_categories,
      categories_count=categories_count,
      head_index=_MOCK_HEAD_INDEX,
      head_name=_MOCK_HEAD_NAME.encode("utf-8"),
  )
  c_classifications_array = (
      classification_result_c.ClassificationsC * classifications_count
  )(*[c_classifications for _ in range(classifications_count)])

  return classification_result_c.ClassificationResultC(
      classifications=c_classifications_array,
      classifications_count=classifications_count,
      timestamp_ms=_MOCK_TIMESTAMP_MS,
      has_timestamp_ms=True,
  )


class ClassificationResultTest(absltest.TestCase):

  def _assert_category_matches(
      self,
      actual: category.Category,
      expected: category.Category,
  ):
    self.assertEqual(expected.index, actual.index)
    self.assertAlmostEqual(expected.score, actual.score)
    self.assertEqual(expected.category_name, actual.category_name)
    self.assertEqual(expected.display_name, actual.display_name)

  def _assert_classsification_matches(
      self,
      actual_result: classification_result.Classifications,
      expected_categories: list[category.Category],
  ):
    self.assertEqual(actual_result.head_index, _MOCK_HEAD_INDEX)
    self.assertEqual(actual_result.head_name, _MOCK_HEAD_NAME)
    self.assertLen(actual_result.categories, len(expected_categories))
    for i, expected_category in enumerate(expected_categories):
      self._assert_category_matches(
          actual_result.categories[i], expected_category
      )

  def test_converts_fully_populated_classification_result_to_python(self):
    c_classsification_result = _create_classification_result_c(
        categories_count=2, classifications_count=1
    )

    actual_result = classification_result.ClassificationResult.from_ctypes(
        c_classsification_result
    )

    self.assertEqual(actual_result.timestamp_ms, _MOCK_TIMESTAMP_MS)
    self.assertLen(actual_result.classifications, 1)
    self._assert_classsification_matches(
        actual_result=actual_result.classifications[0],
        expected_categories=[_MOCK_CATEGORY, _MOCK_CATEGORY],
    )

  def test_converts_empty_classification_result_to_python(self):
    c_classsification_result = _create_classification_result_c(
        categories_count=0, classifications_count=0
    )

    actual_result = classification_result.ClassificationResult.from_ctypes(
        c_classsification_result
    )

    self.assertEqual(actual_result.timestamp_ms, _MOCK_TIMESTAMP_MS)
    self.assertEmpty(actual_result.classifications)


if __name__ == "__main__":
  absltest.main()
