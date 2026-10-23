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
import MPPLanguageDetector
import XCTest

extension TasksErrorCode {
  fileprivate static let errorDomain = "com.google.mediapipe.tasks"
}

final class LanguageDetectorSmokeTests: XCTestCase {

  func testOptionsCoverage() {
    let options = LanguageDetectorOptions()
    options.displayNamesLocale = "en"
    options.maxResults = 3
    options.scoreThreshold = 0.5
    options.categoryAllowlist = ["en", "fr"]
    options.categoryDenylist = ["de"]

    XCTAssertEqual(options.displayNamesLocale, "en")
    XCTAssertEqual(options.maxResults, 3)
    XCTAssertEqual(options.scoreThreshold, 0.5, accuracy: 1e-3)
    XCTAssertEqual(options.categoryAllowlist, ["en", "fr"])
    XCTAssertEqual(options.categoryDenylist, ["de"])
  }

  func testRuntimeRegistrationAndInitialization() {
    let options = LanguageDetectorOptions()
    options.baseOptions.modelAssetPath = "/nonexistent/language_detector.tflite"

    XCTAssertThrowsError(try LanguageDetector(options: options)) { error in
      let nsError = error as NSError
      XCTAssertEqual(nsError.domain, TasksErrorCode.errorDomain)
      XCTAssertTrue(
        nsError.code == TasksErrorCode.invalidArgumentError.rawValue
          || nsError.code == TasksErrorCode.notFoundError.rawValue
      )
      XCTAssertFalse(
        nsError.localizedDescription.contains("No registered object with name"),
        "LanguageDetector subgraph not linked"
      )
      XCTAssertTrue(
        nsError.localizedDescription.contains("Unable to open file"),
        "Unexpected error: \(nsError.localizedDescription)"
      )
    }

    XCTAssertThrowsError(
      try LanguageDetector(modelPath: "/nonexistent/language_detector.tflite")
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
    let _: (LanguageDetector) -> (String) throws -> LanguageDetectorResult =
      LanguageDetector.detect(text:)
  }

  func testResultContainerTypes() {
    let prediction = LanguagePrediction(languageCode: "en", probability: 0.95)
    XCTAssertEqual(prediction.languageCode, "en")
    XCTAssertEqual(prediction.probability, 0.95, accuracy: 1e-3)

    let result = LanguageDetectorResult(
      languagePredictions: [prediction],
      timestampInMilliseconds: 1000
    )
    XCTAssertEqual(result.languagePredictions.count, 1)
    XCTAssertEqual(result.languagePredictions[0].languageCode, "en")
    XCTAssertEqual(result.timestampInMilliseconds, 1000)
  }
}
