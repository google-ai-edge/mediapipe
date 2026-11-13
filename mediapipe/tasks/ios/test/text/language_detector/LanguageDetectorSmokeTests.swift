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
import MediaPipeTasksLanguageDetector
import XCTest

final class LanguageDetectorSmokeTests: XCTestCase {

  func testOptionsCoverage() {
    var options = LanguageDetectorOptions()
    options.displayNamesLocale = "fr"
    options.maxResults = 3
    options.scoreThreshold = 0.5
    options.categoryAllowlist = ["en", "fr"]
    options.categoryDenylist = ["de"]

    XCTAssertEqual(options.displayNamesLocale, "fr")
    XCTAssertEqual(options.maxResults, 3)
    XCTAssertEqual(options.scoreThreshold, 0.5, accuracy: 1e-3)
    XCTAssertEqual(options.categoryAllowlist, ["en", "fr"])
    XCTAssertEqual(options.categoryDenylist, ["de"])
  }

  func testRuntimeRegistrationAndInitialization() {
    var options = LanguageDetectorOptions()
    options.baseOptions.modelAssetPath = "/nonexistent/language_detector.tflite"

    XCTAssertThrowsError(try LanguageDetector(options: options)) { error in
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
      try LanguageDetector(modelPath: "/nonexistent/language_detector.tflite")
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
    let _: (LanguageDetector) -> (String) throws -> LanguageDetectorResult =
      LanguageDetector.detect(text:)
  }

  func testResultContainerTypes() throws {
    let prediction = LanguagePrediction(languageCode: "en", probability: 0.95)
    XCTAssertEqual(prediction.languageCode, "en")
    XCTAssertEqual(prediction.probability, 0.95, accuracy: 1e-3)

    let result = LanguageDetectorResult(
      languages: [prediction],
      timestampInMilliseconds: 1000
    )
    XCTAssertEqual(result.languages.count, 1)
    XCTAssertEqual(try XCTUnwrap(result.languages.first).languageCode, "en")
    XCTAssertEqual(result.timestampInMilliseconds, 1000)
  }
}
