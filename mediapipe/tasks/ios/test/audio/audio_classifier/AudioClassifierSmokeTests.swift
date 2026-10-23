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

import MPPAudioClassifier
import MPPCommon
import XCTest

extension TasksErrorCode {
  fileprivate static let errorDomain = "com.google.mediapipe.tasks"
}

final class AudioClassifierSmokeTests: XCTestCase {

  private final class MockAudioClassifierStreamDelegate: NSObject,
    AudioClassifierStreamDelegate
  {
    func audioClassifier(
      _ audioClassifier: AudioClassifier,
      didFinishClassification result: AudioClassifierResult?,
      timestampInMilliseconds: Int,
      error: (any Error)?
    ) {}

    func audioClassifier(
      _ audioClassifier: AudioClassifier,
      didFinishClassification result: AudioClassifierResult,
      timestampInMilliseconds: Int
    ) {}
  }

  func testOptionsAndDelegateCoverage() {
    let options = AudioClassifierOptions()
    options.runningMode = .audioStream
    options.scoreThreshold = 0.5
    options.maxResults = 3
    options.displayNamesLocale = "en"
    options.categoryAllowlist = ["speech", "music"]
    options.categoryDenylist = ["noise"]

    let delegate = MockAudioClassifierStreamDelegate()
    options.audioClassifierStreamDelegate = delegate
    XCTAssertNotNil(options.audioClassifierStreamDelegate)
    XCTAssertEqual(options.runningMode, .audioStream)
    XCTAssertEqual(options.scoreThreshold, 0.5, accuracy: 1e-3)
    XCTAssertEqual(options.maxResults, 3)
    XCTAssertEqual(options.displayNamesLocale, "en")
    XCTAssertEqual(options.categoryAllowlist, ["speech", "music"])
    XCTAssertEqual(options.categoryDenylist, ["noise"])
  }

  func testRuntimeRegistrationAndInitialization() {
    let options = AudioClassifierOptions()
    options.baseOptions.modelAssetPath = "/nonexistent/audio_classifier.task"

    XCTAssertThrowsError(try AudioClassifier(options: options)) { error in
      let nsError = error as NSError
      XCTAssertEqual(nsError.domain, TasksErrorCode.errorDomain)
      XCTAssertTrue(
        nsError.code == TasksErrorCode.invalidArgumentError.rawValue
          || nsError.code == TasksErrorCode.notFoundError.rawValue
      )
      XCTAssertFalse(
        nsError.localizedDescription.contains("No registered object with name"),
        "AudioClassifier subgraph not linked"
      )
      XCTAssertTrue(
        nsError.localizedDescription.contains("Unable to open file"),
        "Unexpected error: \(nsError.localizedDescription)"
      )
    }

    XCTAssertThrowsError(
      try AudioClassifier(modelPath: "/nonexistent/audio_classifier.task")
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
    let _: (AudioClassifier) -> (AudioData) throws -> AudioClassifierResult =
      AudioClassifier.classify(audioClip:)
    let _: (AudioClassifier) -> (AudioData, Int) throws -> Void =
      AudioClassifier.classifyAsync(audioBlock:timestampInMilliseconds:)
    let _: (AudioClassifier) -> () throws -> Void = AudioClassifier.close
    let _: (AudioRecord) -> () throws -> Void = AudioRecord.startRecording
    let _: (AudioRecord) -> () throws -> Void = AudioRecord.stop
    let _: (AudioRecord) -> (UInt, UInt) throws -> MPPFloatBuffer =
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

    let result = AudioClassifierResult(
      classificationResults: [classificationResult],
      timestampInMilliseconds: 1000
    )
    XCTAssertEqual(result.classificationResults.count, 1)
    XCTAssertEqual(result.timestampInMilliseconds, 1000)

    let format = AudioDataFormat(channelCount: 1, sampleRate: 16000)
    XCTAssertEqual(format.channelCount, 1)
    XCTAssertEqual(format.sampleRate, 16000, accuracy: 1e-3)

    let formatMono = AudioDataFormat(sampleRate: 16000)
    XCTAssertEqual(formatMono.channelCount, 1)
    XCTAssertEqual(formatMono.sampleRate, 16000, accuracy: 1e-3)

    let audioData = AudioData(format: format, sampleCount: 100)
    XCTAssertEqual(audioData.format.channelCount, 1)
    XCTAssertEqual(audioData.bufferLength, 100)

    let buffer = MPPFloatBuffer(length: 10)
    XCTAssertNoThrow(try audioData.load(buffer: buffer, offset: 0, length: 10))

    let audioRecord = try XCTUnwrap(
      AudioRecord(audioDataFormat: formatMono, bufferLength: 100)
    )
    XCTAssertEqual(audioRecord.bufferLength, 100)
    XCTAssertEqual(audioRecord.audioDataFormat.channelCount, 1)
    XCTAssertThrowsError(try audioData.load(audioRecord: audioRecord))
  }
}
