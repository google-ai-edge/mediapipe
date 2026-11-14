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
import MediaPipeTasksAudioClassifier
import XCTest

final class AudioClassifierSmokeTests: XCTestCase {

  func testOptionsAndCallbackCoverage() {
    var options = AudioClassifierOptions()
    options.runningMode = .audioStream
    options.scoreThreshold = 0.5
    options.maxResults = 3
    options.displayNamesLocale = "fr"
    options.categoryAllowlist = ["speech", "music"]
    options.categoryDenylist = ["noise"]
    options.audioStreamResultHandler = { _ in }

    XCTAssertNotNil(options.audioStreamResultHandler)
    XCTAssertEqual(options.runningMode, .audioStream)
    XCTAssertEqual(options.scoreThreshold, 0.5, accuracy: 1e-3)
    XCTAssertEqual(options.maxResults, 3)
    XCTAssertEqual(options.displayNamesLocale, "fr")
    XCTAssertEqual(options.categoryAllowlist, ["speech", "music"])
    XCTAssertEqual(options.categoryDenylist, ["noise"])
  }

  func testRuntimeRegistrationAndInitialization() {
    var options = AudioClassifierOptions()
    options.baseOptions.modelAssetPath = "/nonexistent/audio_classifier.task"

    XCTAssertThrowsError(try AudioClassifier(options: options)) { error in
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
      try AudioClassifier(modelPath: "/nonexistent/audio_classifier.task")
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
    let _: (AudioClassifier) -> (AudioData) throws -> AudioClassifierResult =
      AudioClassifier.classify(audioClip:)
    let _: (AudioClassifier) -> (AudioData, Int) throws -> Void =
      AudioClassifier.classifyAsync(audioBlock:timestampInMilliseconds:)
    let _: (AudioClassifier) -> () throws -> Void = AudioClassifier.close
    let _: (Int, Double, Int) throws -> AudioRecord =
      AudioClassifier.createAudioRecord(channelCount:sampleRate:bufferLength:)
    let _: (AudioRecord) -> () throws -> Void = AudioRecord.startRecording
    let _: (AudioRecord) -> () throws -> Void = AudioRecord.stop
    let _: (AudioRecord) -> (Int, Int) throws -> FloatBuffer =
      AudioRecord.read(offset:length:)
  }

  func testResultContainerTypes() throws {
    let category = ResultCategory(
      index: 0,
      score: 0.8,
      categoryName: "speech",
      displayName: "Speech"
    )
    XCTAssertEqual(category.index, 0)
    XCTAssertEqual(category.score, 0.8, accuracy: 1e-3)
    XCTAssertEqual(category.categoryName, "speech")
    XCTAssertEqual(category.displayName, "Speech")

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

    let result = AudioClassifierResult(
      classificationResults: [classificationResult],
      timestampInMilliseconds: 1000
    )
    XCTAssertEqual(result.classificationResults.count, 1)
    XCTAssertEqual(result.timestampInMilliseconds, 1000)

    let format = AudioDataFormat(sampleRate: 16000, channelCount: 1)
    XCTAssertEqual(format.channelCount, 1)
    XCTAssertEqual(format.sampleRate, 16000, accuracy: 1e-3)

    let formatMono = AudioDataFormat(sampleRate: 16000)
    XCTAssertEqual(formatMono.channelCount, 1)
    XCTAssertEqual(formatMono.sampleRate, 16000, accuracy: 1e-3)

    let audioData = AudioData(format: format, sampleCount: 100)
    XCTAssertEqual(audioData.format.channelCount, 1)
    XCTAssertEqual(audioData.bufferLength, 100)

    let buffer = FloatBuffer(length: 10)
    XCTAssertNoThrow(try audioData.load(buffer: buffer, offset: 0, length: 10))

    let audioRecord = try AudioRecord(audioDataFormat: formatMono, bufferLength: 100)
    XCTAssertEqual(audioRecord.bufferLength, 100)
    XCTAssertEqual(audioRecord.audioDataFormat.channelCount, 1)
    XCTAssertThrowsError(try audioData.load(audioRecord: audioRecord))
  }
}
