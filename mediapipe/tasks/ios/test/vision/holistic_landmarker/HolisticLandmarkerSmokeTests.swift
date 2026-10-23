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
import MPPHolisticLandmarker
import XCTest

extension TasksErrorCode {
  fileprivate static let errorDomain = "com.google.mediapipe.tasks"
}

final class HolisticLandmarkerSmokeTests: XCTestCase {

  private final class MockLiveStreamDelegate: NSObject,
    HolisticLandmarkerLiveStreamDelegate
  {
    func holisticLandmarker(
      _ holisticLandmarker: HolisticLandmarker,
      didFinishDetection result: HolisticLandmarkerResult?,
      timestampInMilliseconds: Int,
      error: (any Error)?
    ) {}
  }

  func testOptionsAndDelegateCoverage() {
    let options = HolisticLandmarkerOptions()
    options.runningMode = .liveStream
    options.minFaceDetectionConfidence = 0.6
    options.minFaceSuppressionThreshold = 0.4
    options.minFacePresenceConfidence = 0.6
    options.minPoseDetectionConfidence = 0.6
    options.minPoseSuppressionThreshold = 0.4
    options.minPosePresenceConfidence = 0.6
    options.outputFaceBlendshapes = true
    options.outputPoseSegmentationMasks = true
    options.minHandLandmarksConfidence = 0.6

    let delegate = MockLiveStreamDelegate()
    options.holisticLandmarkerLiveStreamDelegate = delegate
    XCTAssertNotNil(options.holisticLandmarkerLiveStreamDelegate)
    XCTAssertEqual(options.runningMode, .liveStream)
    XCTAssertEqual(options.minFaceDetectionConfidence, 0.6, accuracy: 1e-3)
    XCTAssertEqual(options.minFaceSuppressionThreshold, 0.4, accuracy: 1e-3)
    XCTAssertEqual(options.minFacePresenceConfidence, 0.6, accuracy: 1e-3)
    XCTAssertEqual(options.minPoseDetectionConfidence, 0.6, accuracy: 1e-3)
    XCTAssertEqual(options.minPoseSuppressionThreshold, 0.4, accuracy: 1e-3)
    XCTAssertEqual(options.minPosePresenceConfidence, 0.6, accuracy: 1e-3)
    XCTAssertTrue(options.outputFaceBlendshapes)
    XCTAssertTrue(options.outputPoseSegmentationMasks)
    XCTAssertEqual(options.minHandLandmarksConfidence, 0.6, accuracy: 1e-3)
  }

  func testRuntimeRegistrationAndInitialization() {
    let options = HolisticLandmarkerOptions()
    options.baseOptions.modelAssetPath =
      "/nonexistent/holistic_landmarker.task"

    XCTAssertThrowsError(try HolisticLandmarker(options: options)) { error in
      let nsError = error as NSError
      XCTAssertEqual(nsError.domain, TasksErrorCode.errorDomain)
      XCTAssertTrue(
        nsError.code == TasksErrorCode.invalidArgumentError.rawValue
          || nsError.code == TasksErrorCode.notFoundError.rawValue
      )
      XCTAssertFalse(
        nsError.localizedDescription.contains("No registered object with name"),
        "HolisticLandmarker subgraph not linked"
      )
      XCTAssertTrue(
        nsError.localizedDescription.contains("Unable to open file"),
        "Unexpected error: \(nsError.localizedDescription)"
      )
    }

    XCTAssertThrowsError(
      try HolisticLandmarker(modelPath: "/nonexistent/holistic_landmarker.task")
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
    let _: (HolisticLandmarker) -> (MPImage) throws -> HolisticLandmarkerResult =
      HolisticLandmarker.detect(image:)
    let _: (HolisticLandmarker) -> (MPImage, Int) throws -> HolisticLandmarkerResult =
      HolisticLandmarker.detect(videoFrame:timestampInMilliseconds:)
    let _: (HolisticLandmarker) -> (MPImage, Int) throws -> Void =
      HolisticLandmarker.detectAsync(image:timestampInMilliseconds:)
  }

  func testResultContainerTypes() throws {
    let normalizedLandmark = NormalizedLandmark(
      x: 0.1,
      y: 0.2,
      z: 0.3,
      visibility: 0.9,
      presence: 0.8
    )
    XCTAssertEqual(normalizedLandmark.x, 0.1, accuracy: 1e-3)
    XCTAssertEqual(normalizedLandmark.y, 0.2, accuracy: 1e-3)
    XCTAssertEqual(normalizedLandmark.z, 0.3, accuracy: 1e-3)
    let normalizedVisibility = try XCTUnwrap(normalizedLandmark.visibility)
    XCTAssertEqual(normalizedVisibility.floatValue, 0.9, accuracy: 1e-3)
    let normalizedPresence = try XCTUnwrap(normalizedLandmark.presence)
    XCTAssertEqual(normalizedPresence.floatValue, 0.8, accuracy: 1e-3)

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

    let result = HolisticLandmarkerResult(
      face: [normalizedLandmark],
      faceBlendshapes: nil,
      poseLandmarks: [normalizedLandmark],
      poseWorldLandmarks: [worldLandmark],
      poseSegmentationMask: nil,
      leftHand: [normalizedLandmark],
      leftHandWorldLandmarks: [worldLandmark],
      rightHand: [normalizedLandmark],
      rightHandWorldLandmarks: [worldLandmark],
      timestampInMilliseconds: 1000
    )
    XCTAssertEqual(result.faceLandmarks.count, 1)
    XCTAssertNil(result.faceBlendshapes)
    XCTAssertEqual(result.poseLandmarks.count, 1)
    XCTAssertEqual(result.poseWorldLandmarks.count, 1)
    XCTAssertNil(result.poseSegmentationMask)
    XCTAssertEqual(result.leftHandLandmarks.count, 1)
    XCTAssertEqual(result.leftHandWorldLandmarks.count, 1)
    XCTAssertEqual(result.rightHandLandmarks.count, 1)
    XCTAssertEqual(result.rightHandWorldLandmarks.count, 1)
    XCTAssertEqual(result.timestampInMilliseconds, 1000)
  }
}
