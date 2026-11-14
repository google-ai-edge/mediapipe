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
import MediaPipeTasksPoseLandmarker
import XCTest

final class PoseLandmarkerSmokeTests: XCTestCase {

  func testOptionsAndHandlerCoverage() {
    var options = PoseLandmarkerOptions()
    options.runningMode = .liveStream
    options.numPoses = 2
    options.minPoseDetectionConfidence = 0.6
    options.minPosePresenceConfidence = 0.6
    options.minTrackingConfidence = 0.6
    options.outputSegmentationMasks = true
    options.liveStreamResultHandler = { _, _ in }

    XCTAssertNotNil(options.liveStreamResultHandler)
    XCTAssertEqual(options.runningMode, .liveStream)
    XCTAssertEqual(options.numPoses, 2)
    XCTAssertEqual(options.minPoseDetectionConfidence, 0.6, accuracy: 1e-3)
    XCTAssertEqual(options.minPosePresenceConfidence, 0.6, accuracy: 1e-3)
    XCTAssertEqual(options.minTrackingConfidence, 0.6, accuracy: 1e-3)
    XCTAssertTrue(options.outputSegmentationMasks)
  }

  func testRuntimeRegistrationAndInitialization() {
    var options = PoseLandmarkerOptions()
    options.baseOptions.modelAssetPath = "/nonexistent/pose_landmarker.task"

    XCTAssertThrowsError(try PoseLandmarker(options: options)) { error in
      guard let tasksError = error as? TasksError else {
        XCTFail("Expected TasksError, got \(error)")
        return
      }
      XCTAssertEqual((error as NSError).domain, TasksError.errorDomain)
      XCTAssertTrue(
        tasksError.code == .invalidArgument || tasksError.code == .notFound
      )
      XCTAssertTrue(
        error.localizedDescription.contains("Unable to open file"),
        "Unexpected error: \(error.localizedDescription)"
      )
    }

    XCTAssertThrowsError(
      try PoseLandmarker(modelPath: "/nonexistent/pose_landmarker.task")
    ) { error in
      guard let tasksError = error as? TasksError else {
        XCTFail("Expected TasksError, got \(error)")
        return
      }
      XCTAssertEqual((error as NSError).domain, TasksError.errorDomain)
      XCTAssertTrue(
        tasksError.code == .invalidArgument || tasksError.code == .notFound
      )
    }
  }

  func testExecutionMethodSignaturesCompile() {
    let _:
      (PoseLandmarker) -> (MPImage, ImageProcessingOptions?) throws ->
        PoseLandmarkerResult =
        PoseLandmarker.detect(image:imageProcessingOptions:)
    let _:
      (PoseLandmarker) -> (MPImage, Int, ImageProcessingOptions?) throws ->
        PoseLandmarkerResult =
        PoseLandmarker.detect(
          videoFrame:timestampInMilliseconds:imageProcessingOptions:
        )
    let _:
      (PoseLandmarker) -> (MPImage, Int, ImageProcessingOptions?) throws ->
        Void =
        PoseLandmarker.detectAsync(
          image:timestampInMilliseconds:imageProcessingOptions:
        )
    let _: (PoseLandmarker, MPImage) throws -> PoseLandmarkerResult = {
      try $0.detect(image: $1)
    }
    let _: (PoseLandmarker, MPImage, Int) throws -> PoseLandmarkerResult = {
      try $0.detect(videoFrame: $1, timestampInMilliseconds: $2)
    }
    let _: (PoseLandmarker, MPImage, Int) throws -> Void = {
      try $0.detectAsync(image: $1, timestampInMilliseconds: $2)
    }
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
    XCTAssertEqual(visibility, 0.9, accuracy: 1e-3)
    let presence = try XCTUnwrap(landmark.presence)
    XCTAssertEqual(presence, 0.8, accuracy: 1e-3)

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
    XCTAssertEqual(worldVisibility, 0.9, accuracy: 1e-3)
    let worldPresence = try XCTUnwrap(worldLandmark.presence)
    XCTAssertEqual(worldPresence, 0.8, accuracy: 1e-3)

    let result = PoseLandmarkerResult(
      landmarks: [[landmark]],
      worldLandmarks: [[worldLandmark]],
      segmentationMasks: nil,
      timestampInMilliseconds: 1000
    )
    XCTAssertEqual(result.landmarks.count, 1)
    XCTAssertEqual(result.worldLandmarks.count, 1)
    XCTAssertNil(result.segmentationMasks)
    XCTAssertEqual(result.timestampInMilliseconds, 1000)
  }

  func testTopologyConnections() {
    XCTAssertFalse(PoseLandmarker.poseLandmarks.isEmpty)
  }
}
