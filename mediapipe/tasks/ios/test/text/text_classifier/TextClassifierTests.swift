// Copyright 2023 The MediaPipe Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

import MPPCommon
import XCTest

@testable import MPPTextClassifier

class TextClassifierTests: XCTestCase {

  static let bundle = Bundle(for: TextClassifierTests.self)

  static let bertModelPath = bundle.path(
    forResource: "bert_text_classifier",
    ofType: "tflite")

  static let positiveText = "it's a charming and often affecting journey"

  static let negativeText = "unflinchingly bleak and desperate"

  static let bertNegativeTextResults = [
    ResultCategory(
      index: 0,
      score: 0.9633251,
      categoryName: "negative",
      displayName: nil),
    ResultCategory(
      index: 1,
      score: 0.036674,
      categoryName: "positive",
      displayName: nil),
  ]

  static let bertNegativeTextResultsForEdgeTestCases = [
    ResultCategory(
      index: 0,
      score: 0.963325,
      categoryName: "negative",
      displayName: nil)
  ]

  func assertEqualErrorDescriptions(
    _ error: Error, expectedLocalizedDescription: String
  ) {
    XCTAssertEqual(
      error.localizedDescription,
      expectedLocalizedDescription)
  }

  func assertCategoriesAreEqual(
    category: ResultCategory,
    expectedCategory: ResultCategory,
    indexInCategoryList: Int
  ) {
    XCTAssertEqual(
      category.index,
      expectedCategory.index,
      String(
        format: """
          category[%d].index and expectedCategory[%d].index are not equal.
          """, indexInCategoryList))
    XCTAssertEqual(
      category.score,
      expectedCategory.score,
      accuracy: 1e-3,
      String(
        format: """
          category[%d].score and expectedCategory[%d].score are not equal.
          """, indexInCategoryList))
    XCTAssertEqual(
      category.categoryName,
      expectedCategory.categoryName,
      String(
        format: """
          category[%d].categoryName and expectedCategory[%d].categoryName are \
          not equal.
          """, indexInCategoryList))
    XCTAssertEqual(
      category.displayName,
      expectedCategory.displayName,
      String(
        format: """
          category[%d].displayName and expectedCategory[%d].displayName are \
          not equal.
          """, indexInCategoryList))
  }

  func assertEqualCategoryArrays(
    categoryArray: [ResultCategory],
    expectedCategoryArray: [ResultCategory]
  ) {
    XCTAssertEqual(
      categoryArray.count,
      expectedCategoryArray.count)

    for (index, (category, expectedCategory)) in zip(categoryArray, expectedCategoryArray)
      .enumerated()
    {
      assertCategoriesAreEqual(
        category: category,
        expectedCategory: expectedCategory,
        indexInCategoryList: index)
    }
  }

  func assertTextClassifierResultHasOneHead(
    _ textClassifierResult: TextClassifierResult
  ) {
    XCTAssertEqual(textClassifierResult.classificationResult.classifications.count, 1)
    XCTAssertEqual(textClassifierResult.classificationResult.classifications[0].headIndex, 0)
  }

  func textClassifierOptionsWithModelPath(
    _ modelPath: String?
  ) throws -> TextClassifierOptions {
    let modelPath = try XCTUnwrap(modelPath)

    let textClassifierOptions = TextClassifierOptions()
    textClassifierOptions.baseOptions.modelAssetPath = modelPath

    return textClassifierOptions
  }

  func assertCreateTextClassifierThrowsError(
    textClassifierOptions: TextClassifierOptions,
    expectedErrorDescription: String
  ) {
    do {
      let textClassifier = try TextClassifier(options: textClassifierOptions)
      XCTAssertNil(textClassifier)
    } catch {
      assertEqualErrorDescriptions(
        error,
        expectedLocalizedDescription: expectedErrorDescription)
    }
  }

  func assertResultsForClassify(
    text: String,
    using textClassifier: TextClassifier,
    equals expectedCategories: [ResultCategory]
  ) throws {
    let textClassifierResult =
      try XCTUnwrap(
        textClassifier.classify(text: text))
    assertTextClassifierResultHasOneHead(textClassifierResult)
    assertEqualCategoryArrays(
      categoryArray:
        textClassifierResult.classificationResult.classifications[0].categories,
      expectedCategoryArray: expectedCategories)
  }

  func testCreateTextClassifierWithInvalidMaxResultsFails() throws {
    let textClassifierOptions =
      try XCTUnwrap(
        textClassifierOptionsWithModelPath(TextClassifierTests.bertModelPath))
    textClassifierOptions.maxResults = 0

    assertCreateTextClassifierThrowsError(
      textClassifierOptions: textClassifierOptions,
      expectedErrorDescription: """
        INVALID_ARGUMENT: Invalid `max_results` option: value must be != 0.
        """)
  }

  func testCreateTextClassifierWithCategoryAllowlistAndDenylistFails() throws {

    let textClassifierOptions =
      try XCTUnwrap(
        textClassifierOptionsWithModelPath(TextClassifierTests.bertModelPath))
    textClassifierOptions.categoryAllowlist = ["positive"]
    textClassifierOptions.categoryDenylist = ["positive"]

    assertCreateTextClassifierThrowsError(
      textClassifierOptions: textClassifierOptions,
      expectedErrorDescription: """
        INVALID_ARGUMENT: `category_allowlist` and `category_denylist` are \
        mutually exclusive options.
        """)
  }

  func testClassifyWithBertSucceeds() throws {

    let modelPath = try XCTUnwrap(TextClassifierTests.bertModelPath)
    let textClassifier = try XCTUnwrap(TextClassifier(modelPath: modelPath))

    try assertResultsForClassify(
      text: TextClassifierTests.negativeText,
      using: textClassifier,
      equals: TextClassifierTests.bertNegativeTextResults)
  }

  func testClassifyWithMaxResultsSucceeds() throws {
    let textClassifierOptions =
      try XCTUnwrap(
        textClassifierOptionsWithModelPath(TextClassifierTests.bertModelPath))
    textClassifierOptions.maxResults = 1

    let textClassifier =
      try XCTUnwrap(TextClassifier(options: textClassifierOptions))

    try assertResultsForClassify(
      text: TextClassifierTests.negativeText,
      using: textClassifier,
      equals: TextClassifierTests.bertNegativeTextResultsForEdgeTestCases)
  }

  func testClassifyWithCategoryAllowlistSucceeds() throws {
    let textClassifierOptions =
      try XCTUnwrap(
        textClassifierOptionsWithModelPath(TextClassifierTests.bertModelPath))
    textClassifierOptions.categoryAllowlist = ["negative"]

    let textClassifier =
      try XCTUnwrap(TextClassifier(options: textClassifierOptions))

    try assertResultsForClassify(
      text: TextClassifierTests.negativeText,
      using: textClassifier,
      equals: TextClassifierTests.bertNegativeTextResultsForEdgeTestCases)
  }

  func testClassifyWithCategoryDenylistSucceeds() throws {
    let textClassifierOptions =
      try XCTUnwrap(
        textClassifierOptionsWithModelPath(TextClassifierTests.bertModelPath))
    textClassifierOptions.categoryDenylist = ["positive"]

    let textClassifier =
      try XCTUnwrap(TextClassifier(options: textClassifierOptions))

    try assertResultsForClassify(
      text: TextClassifierTests.negativeText,
      using: textClassifier,
      equals: TextClassifierTests.bertNegativeTextResultsForEdgeTestCases)
  }

  func testClassifyWithScoreThresholdSucceeds() throws {
    let textClassifierOptions =
      try XCTUnwrap(
        textClassifierOptionsWithModelPath(TextClassifierTests.bertModelPath))
    textClassifierOptions.scoreThreshold = 0.5

    let textClassifier =
      try XCTUnwrap(TextClassifier(options: textClassifierOptions))

    try assertResultsForClassify(
      text: TextClassifierTests.negativeText,
      using: textClassifier,
      equals: TextClassifierTests.bertNegativeTextResultsForEdgeTestCases)
  }

}
