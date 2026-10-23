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
import MPPGestureRecognizer
import XCTest

extension TasksErrorCode {
  fileprivate static let errorDomain = "com.google.mediapipe.tasks"
}

final class GestureRecognizerSmokeTests: XCTestCase {

  private final class MockLiveStreamDelegate: NSObject,
    GestureRecognizerLiveStreamDelegate
  {
    func gestureRecognizer(
      _ gestureRecognizer: GestureRecognizer,
      didFinishGestureRecognition result: GestureRecognizerResult?,
      timestampInMilliseconds: Int,
      error: (any Error)?
    ) {}
  }

  func testOptionsAndDelegateCoverage() {
    let options = GestureRecognizerOptions()
    options.runningMode = .liveStream
    options.numHands = 2
    options.minHandDetectionConfidence = 0.6
    options.minHandPresenceConfidence = 0.6
    options.minTrackingConfidence = 0.6

    let cannedOptions = ClassifierOptions()
    cannedOptions.scoreThreshold = 0.5
    options.cannedGesturesClassifierOptions = cannedOptions

    let customOptions = ClassifierOptions()
    customOptions.scoreThreshold = 0.5
    options.customGesturesClassifierOptions = customOptions

    let delegate = MockLiveStreamDelegate()
    options.gestureRecognizerLiveStreamDelegate = delegate
    XCTAssertNotNil(options.gestureRecognizerLiveStreamDelegate)
    XCTAssertEqual(options.runningMode, .liveStream)
    XCTAssertEqual(options.numHands, 2)
    XCTAssertEqual(options.minHandDetectionConfidence, 0.6, accuracy: 1e-3)
    XCTAssertEqual(options.minHandPresenceConfidence, 0.6, accuracy: 1e-3)
    XCTAssertEqual(options.minTrackingConfidence, 0.6, accuracy: 1e-3)
    XCTAssertNotNil(options.cannedGesturesClassifierOptions)
    XCTAssertNotNil(options.customGesturesClassifierOptions)
  }

  func testRuntimeRegistrationAndInitialization() {
    let options = GestureRecognizerOptions()
    options.baseOptions.modelAssetPath =
      "/nonexistent/gesture_recognizer.task"

    XCTAssertThrowsError(try GestureRecognizer(options: options)) { error in
      let nsError = error as NSError
      XCTAssertEqual(nsError.domain, TasksErrorCode.errorDomain)
      XCTAssertTrue(
        nsError.code == TasksErrorCode.invalidArgumentError.rawValue
          || nsError.code == TasksErrorCode.notFoundError.rawValue
      )
      XCTAssertFalse(
        nsError.localizedDescription.contains("No registered object with name"),
        "GestureRecognizer subgraph not linked"
      )
      XCTAssertTrue(
        nsError.localizedDescription.contains("Unable to open file"),
        "Unexpected error: \(nsError.localizedDescription)"
      )
    }

    XCTAssertThrowsError(
      try GestureRecognizer(modelPath: "/nonexistent/gesture_recognizer.task")
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
    let _: (GestureRecognizer) -> (MPImage) throws -> GestureRecognizerResult =
      GestureRecognizer.recognize(image:)
    let _: (GestureRecognizer) -> (MPImage, Int) throws -> GestureRecognizerResult =
      GestureRecognizer.recognize(videoFrame:timestampInMilliseconds:)
    let _: (GestureRecognizer) -> (MPImage, Int) throws -> Void =
      GestureRecognizer.recognizeAsync(image:timestampInMilliseconds:)
  }

  func testResultContainerTypes() throws {
    let landmark = NormalizedLandmark(
      x: 0.1,
      y: 0.2,
      z: 0.3,
      visibility: 0.9,
      presence: 0.8
    )
    XCTAssertEqual(landmark.x, 0.1, accuracy: 1e-3)
    XCTAssertEqual(landmark.y, 0.2, accuracy: 1e-3)
    XCTAssertEqual(landmark.z, 0.3, accuracy: 1e-3)
    let landmarkVisibility = try XCTUnwrap(landmark.visibility)
    XCTAssertEqual(landmarkVisibility.floatValue, 0.9, accuracy: 1e-3)
    let landmarkPresence = try XCTUnwrap(landmark.presence)
    XCTAssertEqual(landmarkPresence.floatValue, 0.8, accuracy: 1e-3)

    let worldLandmark = Landmark(
      x: 1.0,
      y: 2.0,
      z: 3.0,
      visibility: 0.9,
      presence: 0.8
    )
    XCTAssertEqual(worldLandmark.x, 1.0, accuracy: 1e-3)
    XCTAssertEqual(worldLandmark.y, 2.0, accuracy: 1e-3)
    XCTAssertEqual(worldLandmark.z, 3.0, accuracy: 1e-3)
    let worldVisibility = try XCTUnwrap(worldLandmark.visibility)
    XCTAssertEqual(worldVisibility.floatValue, 0.9, accuracy: 1e-3)
    let worldPresence = try XCTUnwrap(worldLandmark.presence)
    XCTAssertEqual(worldPresence.floatValue, 0.8, accuracy: 1e-3)

    let gesture = ResultCategory(
      index: -1,
      score: 0.9,
      categoryName: "Victory",
      displayName: "Victory"
    )
    XCTAssertEqual(gesture.index, -1)
    XCTAssertEqual(gesture.score, 0.9, accuracy: 1e-3)
    XCTAssertEqual(gesture.categoryName, "Victory")
    XCTAssertEqual(gesture.displayName, "Victory")

    let handedness = ResultCategory(
      index: 0,
      score: 0.9,
      categoryName: "Left",
      displayName: "Left"
    )
    XCTAssertEqual(handedness.index, 0)
    XCTAssertEqual(handedness.score, 0.9, accuracy: 1e-3)
    XCTAssertEqual(handedness.categoryName, "Left")
    XCTAssertEqual(handedness.displayName, "Left")

    let result = GestureRecognizerResult(
      gestures: [[gesture]],
      handedness: [[handedness]],
      landmarks: [[landmark]],
      worldLandmarks: [[worldLandmark]],
      timestampInMilliseconds: 1000
    )
    XCTAssertEqual(result.gestures.count, 1)
    XCTAssertEqual(result.handedness.count, 1)
    XCTAssertEqual(result.landmarks.count, 1)
    XCTAssertEqual(result.worldLandmarks.count, 1)
    XCTAssertEqual(result.timestampInMilliseconds, 1000)
  }
}
