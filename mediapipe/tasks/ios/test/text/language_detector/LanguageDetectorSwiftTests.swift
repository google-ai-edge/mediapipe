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
import third_party_mediapipe_tasks_c_text_language_detector_language_detector_lib

final class LanguageDetectorSwiftTests: XCTestCase {

  private enum TestFixture {
    static let modelName = "language_detector"
    static let modelExtension = "tflite"
    static let probabilityAccuracy: Float = 1e-3
    static let englishText = "To be, or not to be, that is the question"
    static let frenchText =
      "Il y a beaucoup de bouches qui parlent et fort peu de têtes qui pensent."
    static let russianText = "это какой-то английский язык"
    static let mixedText = "分久必合合久必分"
  }

  private func resolveModelPath(
    file: StaticString = #filePath,
    line: UInt = #line
  ) throws -> String {
    try XCTUnwrap(
      Bundle(for: Self.self).path(
        forResource: TestFixture.modelName,
        ofType: TestFixture.modelExtension
      ),
      file: file,
      line: line
    )
  }

  private func assertPredictions(
    _ actual: [LanguagePrediction],
    match expected: [LanguagePrediction],
    file: StaticString = #filePath,
    line: UInt = #line
  ) {
    XCTAssertEqual(actual.count, expected.count, file: file, line: line)
    guard actual.count == expected.count else { return }
    for (lhs, rhs) in zip(actual, expected) {
      XCTAssertEqual(lhs.languageCode, rhs.languageCode, file: file, line: line)
      XCTAssertEqual(
        lhs.probability, rhs.probability, accuracy: TestFixture.probabilityAccuracy,
        file: file, line: line
      )
    }
  }

  private func assertThrowsTasksError<T>(
    _ expression: @autoclosure () throws -> T,
    expectedCode: TasksErrorCode,
    messageSubstring: String? = nil,
    file: StaticString = #filePath,
    line: UInt = #line
  ) {
    XCTAssertThrowsError(try expression(), file: file, line: line) { error in
      guard let tasksError = error as? TasksError else {
        XCTFail("Expected TasksError, got \(error)", file: file, line: line)
        return
      }
      XCTAssertEqual(tasksError.code, expectedCode, file: file, line: line)
      let nsError = error as NSError
      XCTAssertEqual(nsError.domain, TasksError.errorDomain, file: file, line: line)
      XCTAssertEqual(nsError.code, expectedCode.rawValue, file: file, line: line)
      if let messageSubstring {
        XCTAssertTrue(tasksError.message.contains(messageSubstring), file: file, line: line)
      }
    }
  }

  func testOptionsAndResultValueSemanticsAndCBridging() {
    let defaultOptions = LanguageDetectorOptions()
    XCTAssertEqual(defaultOptions.baseOptions.modelAssetPath, "")
    XCTAssertNil(defaultOptions.baseOptions.modelAssetBuffer)
    XCTAssertEqual(defaultOptions.baseOptions.delegate, .cpu)
    XCTAssertEqual(defaultOptions.displayNamesLocale, "en")
    XCTAssertEqual(defaultOptions.maxResults, -1)
    XCTAssertEqual(defaultOptions.scoreThreshold, 0.0)
    XCTAssertEqual(defaultOptions.categoryAllowlist, [])
    XCTAssertEqual(defaultOptions.categoryDenylist, [])

    var mutated = defaultOptions
    mutated.baseOptions.modelAssetPath = "/tmp/language_detector.tflite"
    mutated.maxResults = 2
    mutated.displayNamesLocale = "fr"
    XCTAssertNotEqual(defaultOptions, mutated)
    XCTAssertEqual(Set([defaultOptions, mutated, defaultOptions]).count, 2)

    let emptyResult = LanguageDetectorResult(
      cResult: MpLanguageDetectorResult(predictions: nil, predictions_count: 0)
    )
    XCTAssertTrue(emptyResult.languages.isEmpty)
    XCTAssertEqual(emptyResult.timestampInMilliseconds, 0)

    var frChars = Array("fr".utf8CString)
    frChars.withUnsafeMutableBufferPointer { frBuffer in
      var cPredictions = [
        MpLanguageDetectorPrediction(language_code: frBuffer.baseAddress, probability: 0.8),
        MpLanguageDetectorPrediction(language_code: nil, probability: 0.2),
      ]
      cPredictions.withUnsafeMutableBufferPointer { buffer in
        let cResult = MpLanguageDetectorResult(
          predictions: buffer.baseAddress, predictions_count: UInt32(buffer.count)
        )
        let bridged = LanguageDetectorResult(cResult: cResult, timestampInMilliseconds: 100)
        XCTAssertEqual(bridged.timestampInMilliseconds, 100)
        XCTAssertEqual(
          bridged.languages,
          [
            LanguagePrediction(languageCode: "fr", probability: 0.8),
            LanguagePrediction(languageCode: "", probability: 0.2),
          ]
        )
      }
    }
  }

  func testInitFailsWithInvalidModelOrOptions() throws {
    assertThrowsTasksError(
      try LanguageDetector(modelPath: ""),
      expectedCode: .invalidArgument,
      messageSubstring: "ExternalFile"
    )
    assertThrowsTasksError(
      try LanguageDetector(options: LanguageDetectorOptions()),
      expectedCode: .invalidArgument,
      messageSubstring: "ExternalFile"
    )
    XCTAssertThrowsError(try LanguageDetector(modelPath: "/nonexistent/language_detector.tflite")) {
      error in
      XCTAssertTrue(error is TasksError)
      XCTAssertEqual((error as NSError).domain, TasksError.errorDomain)
    }

    let modelPath = try resolveModelPath()
    var exclusiveOptions = LanguageDetectorOptions(
      baseOptions: BaseOptions(modelAssetPath: modelPath),
      categoryAllowlist: ["en"],
      categoryDenylist: ["fr"]
    )
    assertThrowsTasksError(
      try LanguageDetector(options: exclusiveOptions),
      expectedCode: .invalidArgument,
      messageSubstring: "mutually exclusive"
    )

    let zeroMaxResultsOptions = LanguageDetectorOptions(
      baseOptions: BaseOptions(modelAssetPath: modelPath),
      maxResults: 0
    )
    assertThrowsTasksError(
      try LanguageDetector(options: zeroMaxResultsOptions),
      expectedCode: .invalidArgument,
      messageSubstring: "max_results"
    )
  }

  func testDetectLanguagesAndModelAssetBufferSucceeds() throws {
    let modelPath = try resolveModelPath()
    let detector = try LanguageDetector(modelPath: modelPath)

    let enResult = try detector.detect(text: TestFixture.englishText)
    XCTAssertEqual(enResult.timestampInMilliseconds, 0)
    assertPredictions(
      enResult.languages,
      match: [LanguagePrediction(languageCode: "en", probability: 0.9998559)]
    )
    assertPredictions(
      try detector.detect(text: TestFixture.frenchText).languages,
      match: [LanguagePrediction(languageCode: "fr", probability: 0.9997813)]
    )
    assertPredictions(
      try detector.detect(text: TestFixture.russianText).languages,
      match: [LanguagePrediction(languageCode: "ru", probability: 0.9933616)]
    )
    assertPredictions(
      try detector.detect(text: TestFixture.mixedText).languages,
      match: [
        LanguagePrediction(languageCode: "zh", probability: 0.505424),
        LanguagePrediction(languageCode: "ja", probability: 0.481617),
      ]
    )

    let modelData = try Data(contentsOf: URL(fileURLWithPath: modelPath))
    let bufferDetector = try LanguageDetector(
      options: LanguageDetectorOptions(baseOptions: BaseOptions(modelAssetBuffer: modelData))
    )
    assertPredictions(
      try bufferDetector.detect(text: TestFixture.englishText).languages,
      match: [LanguagePrediction(languageCode: "en", probability: 0.9998559)]
    )
  }

  func testDetectFilteringOptionsSucceed() throws {
    let modelPath = try resolveModelPath()
    let base = BaseOptions(modelAssetPath: modelPath)

    let maxResultsDetector = try LanguageDetector(
      options: LanguageDetectorOptions(baseOptions: base, maxResults: 1)
    )
    assertPredictions(
      try maxResultsDetector.detect(text: TestFixture.mixedText).languages,
      match: [LanguagePrediction(languageCode: "zh", probability: 0.505424)]
    )

    let thresholdDetector = try LanguageDetector(
      options: LanguageDetectorOptions(baseOptions: base, scoreThreshold: 0.5)
    )
    assertPredictions(
      try thresholdDetector.detect(text: TestFixture.mixedText).languages,
      match: [LanguagePrediction(languageCode: "zh", probability: 0.505424)]
    )

    let allowlistDetector = try LanguageDetector(
      options: LanguageDetectorOptions(baseOptions: base, categoryAllowlist: ["zh"])
    )
    assertPredictions(
      try allowlistDetector.detect(text: TestFixture.mixedText).languages,
      match: [LanguagePrediction(languageCode: "zh", probability: 0.505424)]
    )

    let denylistDetector = try LanguageDetector(
      options: LanguageDetectorOptions(baseOptions: base, categoryDenylist: ["zh"])
    )
    assertPredictions(
      try denylistDetector.detect(text: TestFixture.mixedText).languages,
      match: [LanguagePrediction(languageCode: "ja", probability: 0.481617)]
    )
  }
}
