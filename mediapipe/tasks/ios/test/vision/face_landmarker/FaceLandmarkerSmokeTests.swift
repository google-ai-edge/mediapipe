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
import MediaPipeTasksFaceLandmarker
import XCTest

final class FaceLandmarkerSmokeTests: XCTestCase {

  func testOptionsAndHandlerCoverage() {
    var options = FaceLandmarkerOptions()
    options.runningMode = .liveStream
    options.numFaces = 2
    options.minFaceDetectionConfidence = 0.6
    options.minFacePresenceConfidence = 0.6
    options.minTrackingConfidence = 0.6
    options.outputFaceBlendshapes = true
    options.outputFacialTransformationMatrixes = true
    options.liveStreamResultHandler = { _, _ in }

    XCTAssertNotNil(options.liveStreamResultHandler)
    XCTAssertEqual(options.runningMode, .liveStream)
    XCTAssertEqual(options.numFaces, 2)
    XCTAssertEqual(options.minFaceDetectionConfidence, 0.6, accuracy: 1e-3)
    XCTAssertEqual(options.minFacePresenceConfidence, 0.6, accuracy: 1e-3)
    XCTAssertEqual(options.minTrackingConfidence, 0.6, accuracy: 1e-3)
    XCTAssertTrue(options.outputFaceBlendshapes)
    XCTAssertTrue(options.outputFacialTransformationMatrixes)
  }

  func testRuntimeRegistrationAndInitialization() {
    var options = FaceLandmarkerOptions()
    options.baseOptions.modelAssetPath = "/nonexistent/face_landmarker.task"

    XCTAssertThrowsError(try FaceLandmarker(options: options)) { error in
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
      try FaceLandmarker(modelPath: "/nonexistent/face_landmarker.task")
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
      (FaceLandmarker) -> (MPImage, ImageProcessingOptions?) throws ->
        FaceLandmarkerResult =
        FaceLandmarker.detect(image:imageProcessingOptions:)
    let _:
      (FaceLandmarker) -> (MPImage, Int, ImageProcessingOptions?) throws ->
        FaceLandmarkerResult =
        FaceLandmarker.detect(
          videoFrame:timestampInMilliseconds:imageProcessingOptions:
        )
    let _:
      (FaceLandmarker) -> (MPImage, Int, ImageProcessingOptions?) throws ->
        Void =
        FaceLandmarker.detectAsync(
          image:timestampInMilliseconds:imageProcessingOptions:
        )
    let _: (FaceLandmarker, MPImage) throws -> FaceLandmarkerResult = {
      try $0.detect(image: $1)
    }
    let _: (FaceLandmarker, MPImage, Int) throws -> FaceLandmarkerResult = {
      try $0.detect(videoFrame: $1, timestampInMilliseconds: $2)
    }
    let _: (FaceLandmarker, MPImage, Int) throws -> Void = {
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

    let category = ResultCategory(
      index: 0,
      score: 0.8,
      categoryName: "neutral",
      displayName: "Neutral"
    )
    let blendshapes = Classifications(
      headIndex: 0,
      headName: "blendshapes",
      categories: [category]
    )

    let matrixData: [Float] = [
      1.0, 0.0, 0.0, 0.0,
      0.0, 1.0, 0.0, 0.0,
      0.0, 0.0, 1.0, 0.0,
      0.0, 0.0, 0.0, 1.0,
    ]
    let matrix = try TransformMatrix(data: matrixData, rows: 4, columns: 4)
    XCTAssertEqual(matrix.rows, 4)
    XCTAssertEqual(matrix.columns, 4)
    XCTAssertEqual(matrix.value(atRow: 0, column: 0), 1.0, accuracy: 1e-3)

    let result = FaceLandmarkerResult(
      faceLandmarks: [[landmark]],
      faceBlendshapes: [blendshapes],
      facialTransformationMatrixes: [matrix],
      timestampInMilliseconds: 1000
    )
    XCTAssertEqual(result.faceLandmarks.count, 1)
    XCTAssertEqual(result.faceLandmarks[0].count, 1)
    XCTAssertEqual(result.faceBlendshapes.count, 1)
    XCTAssertEqual(result.facialTransformationMatrixes.count, 1)
    XCTAssertEqual(result.timestampInMilliseconds, 1000)
  }

  func testTopologyConnections() {
    XCTAssertFalse(FaceLandmarker.lipsConnections.isEmpty)
    XCTAssertFalse(FaceLandmarker.leftEyeConnections.isEmpty)
    XCTAssertFalse(FaceLandmarker.leftEyebrowConnections.isEmpty)
    XCTAssertFalse(FaceLandmarker.leftIrisConnections.isEmpty)
    XCTAssertFalse(FaceLandmarker.rightEyeConnections.isEmpty)
    XCTAssertFalse(FaceLandmarker.rightEyebrowConnections.isEmpty)
    XCTAssertFalse(FaceLandmarker.rightIrisConnections.isEmpty)
    XCTAssertFalse(FaceLandmarker.faceOvalConnections.isEmpty)
    XCTAssertFalse(FaceLandmarker.contoursConnections.isEmpty)
    XCTAssertFalse(FaceLandmarker.tesselationConnections.isEmpty)
    XCTAssertFalse(FaceLandmarker.faceConnections.isEmpty)
  }
}
