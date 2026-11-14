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
import MediaPipeTasksImageClassifier
import XCTest

final class ImageClassifierSmokeTests: XCTestCase {

  func testOptionsAndDelegateCoverage() {
    var options = ImageClassifierOptions()
    options.runningMode = .liveStream
    options.scoreThreshold = 0.5
    options.maxResults = 3
    options.displayNamesLocale = "fr"
    options.categoryAllowlist = ["cat", "dog"]
    options.categoryDenylist = ["bird"]
    options.liveStreamResultHandler = { _, _ in }

    XCTAssertNotNil(options.liveStreamResultHandler)
    XCTAssertEqual(options.runningMode, .liveStream)
    XCTAssertEqual(options.scoreThreshold, 0.5, accuracy: 1e-3)
    XCTAssertEqual(options.maxResults, 3)
    XCTAssertEqual(options.displayNamesLocale, "fr")
    XCTAssertEqual(options.categoryAllowlist, ["cat", "dog"])
    XCTAssertEqual(options.categoryDenylist, ["bird"])
  }

  func testRuntimeRegistrationAndInitialization() {
    var options = ImageClassifierOptions()
    options.baseOptions.modelAssetPath = "/nonexistent/classifier.task"

    XCTAssertThrowsError(try ImageClassifier(options: options)) { error in
      let nsError = error as NSError
      XCTAssertEqual(nsError.domain, TasksError.errorDomain)
      XCTAssertTrue(
        nsError.code == TasksErrorCode.invalidArgument.rawValue
          || nsError.code == TasksErrorCode.notFound.rawValue
      )
      XCTAssertTrue(
        nsError.localizedDescription.contains("Unable to open file"),
        "Unexpected error: \(nsError.localizedDescription)"
      )
    }

    XCTAssertThrowsError(
      try ImageClassifier(modelPath: "/nonexistent/classifier.task")
    ) { error in
      let nsError = error as NSError
      XCTAssertEqual(nsError.domain, TasksError.errorDomain)
      XCTAssertTrue(
        nsError.code == TasksErrorCode.invalidArgument.rawValue
          || nsError.code == TasksErrorCode.notFound.rawValue
      )
    }
  }

  func testExecutionMethodSignaturesCompile() {
    let _:
      (ImageClassifier) -> (MPImage, ImageProcessingOptions?) throws ->
        ImageClassifierResult =
        ImageClassifier.classify(image:imageProcessingOptions:)
    let _:
      (ImageClassifier) -> (MPImage, Int, ImageProcessingOptions?) throws ->
        ImageClassifierResult =
        ImageClassifier.classify(videoFrame:timestampInMilliseconds:imageProcessingOptions:)
    let _: (ImageClassifier) -> (MPImage, Int, ImageProcessingOptions?) throws -> Void =
      ImageClassifier.classifyAsync(image:timestampInMilliseconds:imageProcessingOptions:)
  }

  func testResultContainerTypes() {
    let category = ResultCategory(
      index: 0,
      score: 0.9,
      categoryName: "cat",
      displayName: "Cat"
    )
    XCTAssertEqual(category.index, 0)
    XCTAssertEqual(category.score, 0.9, accuracy: 1e-3)
    XCTAssertEqual(category.categoryName, "cat")
    XCTAssertEqual(category.displayName, "Cat")

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

    let result = ImageClassifierResult(
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
