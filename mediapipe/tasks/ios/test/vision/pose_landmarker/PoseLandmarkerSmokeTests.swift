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
import MPPPoseLandmarker
import XCTest

extension TasksErrorCode {
  fileprivate static let errorDomain = "com.google.mediapipe.tasks"
}

final class PoseLandmarkerSmokeTests: XCTestCase {

  private final class MockLiveStreamDelegate: NSObject,
    PoseLandmarkerLiveStreamDelegate
  {
    func poseLandmarker(
      _ poseLandmarker: PoseLandmarker,
      didFinishDetection result: PoseLandmarkerResult?,
      timestampInMilliseconds: Int,
      error: (any Error)?
    ) {}
  }

  func testOptionsAndDelegateCoverage() {
    let options = PoseLandmarkerOptions()
    options.runningMode = .liveStream
    options.numPoses = 2
    options.minPoseDetectionConfidence = 0.6
    options.minPosePresenceConfidence = 0.6
    options.minTrackingConfidence = 0.6
    options.shouldOutputSegmentationMasks = true

    let delegate = MockLiveStreamDelegate()
    options.poseLandmarkerLiveStreamDelegate = delegate
    XCTAssertNotNil(options.poseLandmarkerLiveStreamDelegate)
    XCTAssertEqual(options.runningMode, .liveStream)
    XCTAssertEqual(options.numPoses, 2)
    XCTAssertEqual(options.minPoseDetectionConfidence, 0.6, accuracy: 1e-3)
    XCTAssertEqual(options.minPosePresenceConfidence, 0.6, accuracy: 1e-3)
    XCTAssertEqual(options.minTrackingConfidence, 0.6, accuracy: 1e-3)
    XCTAssertTrue(options.shouldOutputSegmentationMasks)
  }

  func testRuntimeRegistrationAndInitialization() {
    let options = PoseLandmarkerOptions()
    options.baseOptions.modelAssetPath = "/nonexistent/pose_landmarker.task"

    XCTAssertThrowsError(try PoseLandmarker(options: options)) { error in
      let nsError = error as NSError
      XCTAssertEqual(nsError.domain, TasksErrorCode.errorDomain)
      XCTAssertTrue(
        nsError.code == TasksErrorCode.invalidArgumentError.rawValue
          || nsError.code == TasksErrorCode.notFoundError.rawValue
      )
      XCTAssertFalse(
        nsError.localizedDescription.contains("No registered object with name"),
        "PoseLandmarker subgraph not linked"
      )
      XCTAssertTrue(
        nsError.localizedDescription.contains("Unable to open file"),
        "Unexpected error: \(nsError.localizedDescription)"
      )
    }

    XCTAssertThrowsError(
      try PoseLandmarker(modelPath: "/nonexistent/pose_landmarker.task")
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
    let _: (PoseLandmarker) -> (MPImage) throws -> PoseLandmarkerResult =
      PoseLandmarker.detect(image:)
    let _: (PoseLandmarker) -> (MPImage, Int) throws -> PoseLandmarkerResult =
      PoseLandmarker.detect(videoFrame:timestampInMilliseconds:)
    let _: (PoseLandmarker) -> (MPImage, Int) throws -> Void =
      PoseLandmarker.detectAsync(image:timestampInMilliseconds:)
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
    let visibility = try XCTUnwrap(landmark.visibility)
    XCTAssertEqual(visibility.floatValue, 0.9, accuracy: 1e-3)
    let presence = try XCTUnwrap(landmark.presence)
    XCTAssertEqual(presence.floatValue, 0.8, accuracy: 1e-3)

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

    let result = PoseLandmarkerResult(
      landmarks: [[landmark]],
      worldLandmarks: [[worldLandmark]],
      segmentationMasks: nil,
      timestampInMilliseconds: 1000
    )
    XCTAssertEqual(result.landmarks.count, 1)
    XCTAssertEqual(result.worldLandmarks.count, 1)
    XCTAssertTrue(result.segmentationMasks.isEmpty)
    XCTAssertEqual(result.timestampInMilliseconds, 1000)
  }

  func testTopologyConnections() {
    XCTAssertFalse(PoseLandmarker.poseLandmarks.isEmpty)
  }
}
