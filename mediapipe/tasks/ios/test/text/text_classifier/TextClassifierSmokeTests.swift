// Copyright 2026 The MediaPipe Authors.
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
import MPPTextClassifier
import XCTest

extension TasksErrorCode {
  fileprivate static let errorDomain = "com.google.mediapipe.tasks"
}

final class TextClassifierSmokeTests: XCTestCase {

  func testOptionsCoverage() {
    let options = TextClassifierOptions()
    options.scoreThreshold = 0.5
    options.maxResults = 3
    options.displayNamesLocale = "en"
    options.categoryAllowlist = ["positive"]
    options.categoryDenylist = ["negative"]

    XCTAssertEqual(options.scoreThreshold, 0.5, accuracy: 1e-3)
    XCTAssertEqual(options.maxResults, 3)
    XCTAssertEqual(options.displayNamesLocale, "en")
    XCTAssertEqual(options.categoryAllowlist, ["positive"])
    XCTAssertEqual(options.categoryDenylist, ["negative"])
  }

  func testRuntimeRegistrationAndInitialization() {
    let options = TextClassifierOptions()
    options.baseOptions.modelAssetPath = "/nonexistent/text_classifier.task"

    XCTAssertThrowsError(try TextClassifier(options: options)) { error in
      let nsError = error as NSError
      XCTAssertEqual(nsError.domain, TasksErrorCode.errorDomain)
      XCTAssertTrue(
        nsError.code == TasksErrorCode.invalidArgumentError.rawValue
          || nsError.code == TasksErrorCode.notFoundError.rawValue
      )
      XCTAssertFalse(
        nsError.localizedDescription.contains("No registered object with name"),
        "TextClassifier subgraph not linked"
      )
      XCTAssertTrue(
        nsError.localizedDescription.contains("Unable to open file"),
        "Unexpected error: \(nsError.localizedDescription)"
      )
    }

    XCTAssertThrowsError(
      try TextClassifier(modelPath: "/nonexistent/text_classifier.task")
    ) { error in
      let nsError = error as NSError
      XCTAssertEqual(nsError.domain, TasksErrorCode.errorDomain)
      XCTAssertTrue(
        nsError.code == TasksErrorCode.invalidArgumentError.rawValue
          || nsError.code == TasksErrorCode.notFoundError.rawValue
      )
    }
  }

  func testExecutionMethodSignaturesCompile() {
    let _: (TextClassifier) -> (String) throws -> TextClassifierResult =
      TextClassifier.classify(text:)
  }

  func testResultContainerTypes() {
    let category = ResultCategory(
      index: 0,
      score: 0.95,
      categoryName: "positive",
      displayName: "Positive"
    )
    XCTAssertEqual(category.index, 0)
    XCTAssertEqual(category.score, 0.95, accuracy: 1e-3)
    XCTAssertEqual(category.categoryName, "positive")
    XCTAssertEqual(category.displayName, "Positive")

    let classifications = Classifications(
      head: 0,
      headName: "head",
      categories: [category]
    )
    XCTAssertEqual(classifications.headIndex, 0)
    XCTAssertEqual(classifications.headName, "head")
    XCTAssertEqual(classifications.categories.count, 1)

    let classificationResult = ClassificationResult(
      classifications: [classifications],
      timestampInMilliseconds: 1000
    )
    XCTAssertEqual(classificationResult.classifications.count, 1)
    XCTAssertEqual(classificationResult.timestampInMilliseconds, 1000)

    let result = TextClassifierResult(
      classificationResult: classificationResult,
      timestampInMilliseconds: 1000
    )
    XCTAssertEqual(
      result.classificationResult.classifications.count,
      1
    )
    XCTAssertEqual(result.timestampInMilliseconds, 1000)
  }
}
