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

import Foundation
import MediaPipeTasksTextClassifier
import XCTest

final class TextClassifierSmokeTests: XCTestCase {

  func testOptionsCoverage() {
    var options = TextClassifierOptions()
    options.scoreThreshold = 0.5
    options.maxResults = 3
    options.displayNamesLocale = "de"
    options.categoryAllowlist = ["positive"]
    options.categoryDenylist = ["negative"]

    XCTAssertEqual(options.scoreThreshold, 0.5, accuracy: 1e-3)
    XCTAssertEqual(options.maxResults, 3)
    XCTAssertEqual(options.displayNamesLocale, "de")
    XCTAssertEqual(options.categoryAllowlist, ["positive"])
    XCTAssertEqual(options.categoryDenylist, ["negative"])
  }

  func testRuntimeRegistrationAndInitialization() {
    var options = TextClassifierOptions()
    options.baseOptions.modelAssetPath = "/nonexistent/text_classifier.task"

    XCTAssertThrowsError(try TextClassifier(options: options)) { error in
      guard let tasksError = error as? TasksError else {
        XCTFail("Expected TasksError, got \(error)")
        return
      }
      XCTAssertTrue(tasksError.code == .invalidArgument || tasksError.code == .notFound)
      let nsError = error as NSError
      XCTAssertEqual(nsError.domain, TasksError.errorDomain)
      XCTAssertTrue(
        nsError.code == TasksErrorCode.invalidArgument.rawValue
          || nsError.code == TasksErrorCode.notFound.rawValue
      )
      XCTAssertTrue(
        error.localizedDescription.contains("Unable to open file"),
        "Unexpected error: \(error.localizedDescription)"
      )
    }

    XCTAssertThrowsError(
      try TextClassifier(modelPath: "/nonexistent/text_classifier.task")
    ) { error in
      guard let tasksError = error as? TasksError else {
        XCTFail("Expected TasksError, got \(error)")
        return
      }
      XCTAssertTrue(tasksError.code == .invalidArgument || tasksError.code == .notFound)
      XCTAssertTrue(
        error.localizedDescription.contains("Unable to open file"),
        "Unexpected error: \(error.localizedDescription)"
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
      headIndex: 0,
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
    XCTAssertEqual(result.classificationResult.classifications.count, 1)
    XCTAssertEqual(result.timestampInMilliseconds, 1000)
  }
}
