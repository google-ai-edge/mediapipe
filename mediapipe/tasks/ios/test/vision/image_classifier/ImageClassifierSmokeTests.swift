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
import MPPImageClassifier
import XCTest

extension TasksErrorCode {
  fileprivate static let errorDomain = "com.google.mediapipe.tasks"
}

final class ImageClassifierSmokeTests: XCTestCase {

  private final class MockLiveStreamDelegate: NSObject,
    ImageClassifierLiveStreamDelegate
  {
    func imageClassifier(
      _ imageClassifier: ImageClassifier,
      didFinishClassification result: ImageClassifierResult?,
      timestampInMilliseconds: Int,
      error: (any Error)?
    ) {}
  }

  func testOptionsAndDelegateCoverage() {
    let options = ImageClassifierOptions()
    options.runningMode = .liveStream
    options.scoreThreshold = 0.5
    options.maxResults = 3
    options.displayNamesLocale = "en"
    options.categoryAllowlist = ["cat", "dog"]
    options.categoryDenylist = ["bird"]

    let delegate = MockLiveStreamDelegate()
    options.imageClassifierLiveStreamDelegate = delegate
    XCTAssertNotNil(options.imageClassifierLiveStreamDelegate)
    XCTAssertEqual(options.runningMode, .liveStream)
    XCTAssertEqual(options.scoreThreshold, 0.5, accuracy: 1e-3)
    XCTAssertEqual(options.maxResults, 3)
    XCTAssertEqual(options.displayNamesLocale, "en")
    XCTAssertEqual(options.categoryAllowlist, ["cat", "dog"])
    XCTAssertEqual(options.categoryDenylist, ["bird"])
  }

  func testRuntimeRegistrationAndInitialization() {
    let options = ImageClassifierOptions()
    options.baseOptions.modelAssetPath = "/nonexistent/classifier.task"

    XCTAssertThrowsError(try ImageClassifier(options: options)) { error in
      let nsError = error as NSError
      XCTAssertEqual(nsError.domain, TasksErrorCode.errorDomain)
      XCTAssertTrue(
        nsError.code == TasksErrorCode.invalidArgumentError.rawValue
          || nsError.code == TasksErrorCode.notFoundError.rawValue
      )
      XCTAssertFalse(
        nsError.localizedDescription.contains("No registered object with name"),
        "ImageClassifier subgraph not linked"
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
      XCTAssertEqual(nsError.domain, TasksErrorCode.errorDomain)
      XCTAssertTrue(
        nsError.code == TasksErrorCode.invalidArgumentError.rawValue
          || nsError.code == TasksErrorCode.notFoundError.rawValue
      )
    }
  }

  func testExecutionMethodSignaturesCompile() {
    let _: (ImageClassifier) -> (MPImage) throws -> ImageClassifierResult =
      ImageClassifier.classify(image:)
    let _: (ImageClassifier) -> (MPImage, CGRect) throws -> ImageClassifierResult =
      ImageClassifier.classify(image:regionOfInterest:)
    let _: (ImageClassifier) -> (MPImage, Int) throws -> ImageClassifierResult =
      ImageClassifier.classify(videoFrame:timestampInMilliseconds:)
    let _: (ImageClassifier) -> (MPImage, Int, CGRect) throws -> ImageClassifierResult =
      ImageClassifier.classify(videoFrame:timestampInMilliseconds:regionOfInterest:)
    let _: (ImageClassifier) -> (MPImage, Int) throws -> Void =
      ImageClassifier.classifyAsync(image:timestampInMilliseconds:)
    let _: (ImageClassifier) -> (MPImage, Int, CGRect) throws -> Void =
      ImageClassifier.classifyAsync(image:timestampInMilliseconds:regionOfInterest:)
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
