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
import MPPHandLandmarker
import XCTest

extension TasksErrorCode {
  fileprivate static let errorDomain = "com.google.mediapipe.tasks"
}

final class HandLandmarkerSmokeTests: XCTestCase {

  private final class MockLiveStreamDelegate: NSObject,
    HandLandmarkerLiveStreamDelegate
  {
    func handLandmarker(
      _ handLandmarker: HandLandmarker,
      didFinishDetection result: HandLandmarkerResult?,
      timestampInMilliseconds: Int,
      error: (any Error)?
    ) {}
  }

  func testOptionsAndDelegateCoverage() {
    let options = HandLandmarkerOptions()
    options.runningMode = .liveStream
    options.numHands = 2
    options.minHandDetectionConfidence = 0.6
    options.minHandPresenceConfidence = 0.6
    options.minTrackingConfidence = 0.6

    let delegate = MockLiveStreamDelegate()
    options.handLandmarkerLiveStreamDelegate = delegate
    XCTAssertNotNil(options.handLandmarkerLiveStreamDelegate)
    XCTAssertEqual(options.runningMode, .liveStream)
    XCTAssertEqual(options.numHands, 2)
    XCTAssertEqual(options.minHandDetectionConfidence, 0.6, accuracy: 1e-3)
    XCTAssertEqual(options.minHandPresenceConfidence, 0.6, accuracy: 1e-3)
    XCTAssertEqual(options.minTrackingConfidence, 0.6, accuracy: 1e-3)
  }

  func testRuntimeRegistrationAndInitialization() {
    let options = HandLandmarkerOptions()
    options.baseOptions.modelAssetPath = "/nonexistent/hand_landmarker.task"

    XCTAssertThrowsError(try HandLandmarker(options: options)) { error in
      let nsError = error as NSError
      XCTAssertEqual(nsError.domain, TasksErrorCode.errorDomain)
      XCTAssertTrue(
        nsError.code == TasksErrorCode.invalidArgumentError.rawValue
          || nsError.code == TasksErrorCode.notFoundError.rawValue
      )
      XCTAssertFalse(
        nsError.localizedDescription.contains("No registered object with name"),
        "HandLandmarker subgraph not linked"
      )
      XCTAssertTrue(
        nsError.localizedDescription.contains("Unable to open file"),
        "Unexpected error: \(nsError.localizedDescription)"
      )
    }

    XCTAssertThrowsError(
      try HandLandmarker(modelPath: "/nonexistent/hand_landmarker.task")
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
    let _: (HandLandmarker) -> (MPImage) throws -> HandLandmarkerResult =
      HandLandmarker.detect(image:)
    let _: (HandLandmarker) -> (MPImage, Int) throws -> HandLandmarkerResult =
      HandLandmarker.detect(videoFrame:timestampInMilliseconds:)
    let _: (HandLandmarker) -> (MPImage, Int) throws -> Void =
      HandLandmarker.detectAsync(image:timestampInMilliseconds:)
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

    let result = HandLandmarkerResult(
      landmarks: [[landmark]],
      worldLandmarks: [[worldLandmark]],
      handedness: [[handedness]],
      timestampInMilliseconds: 1000
    )
    XCTAssertEqual(result.landmarks.count, 1)
    XCTAssertEqual(result.worldLandmarks.count, 1)
    XCTAssertEqual(result.handedness.count, 1)
    XCTAssertEqual(result.timestampInMilliseconds, 1000)

    let landmarkIndex = HandLandmark.wrist.rawValue
    XCTAssertEqual(landmarkIndex, 0)
    let connection = Connection(start: 0, end: 1)
    XCTAssertEqual(connection.start, 0)
    XCTAssertEqual(connection.end, 1)
  }

  func testTopologyConnections() {
    XCTAssertFalse(HandLandmarker.handPalmConnections.isEmpty)
    XCTAssertFalse(HandLandmarker.handThumbConnections.isEmpty)
    XCTAssertFalse(HandLandmarker.handIndexFingerConnections.isEmpty)
    XCTAssertFalse(HandLandmarker.handMiddleFingerConnections.isEmpty)
    XCTAssertFalse(HandLandmarker.handRingFingerConnections.isEmpty)
    XCTAssertFalse(HandLandmarker.handPinkyConnections.isEmpty)
    XCTAssertFalse(HandLandmarker.handConnections.isEmpty)
  }
}
