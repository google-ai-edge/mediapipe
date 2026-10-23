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
import MPPFaceDetector
import XCTest

extension TasksErrorCode {
  fileprivate static let errorDomain = "com.google.mediapipe.tasks"
}

final class FaceDetectorSmokeTests: XCTestCase {

  private final class MockLiveStreamDelegate: NSObject,
    FaceDetectorLiveStreamDelegate
  {
    func faceDetector(
      _ faceDetector: FaceDetector,
      didFinishDetection result: FaceDetectorResult?,
      timestampInMilliseconds: Int,
      error: (any Error)?
    ) {}
  }

  func testOptionsAndDelegateCoverage() {
    let options = FaceDetectorOptions()
    options.runningMode = .liveStream
    options.minDetectionConfidence = 0.7
    options.minSuppressionThreshold = 0.4

    let delegate = MockLiveStreamDelegate()
    options.faceDetectorLiveStreamDelegate = delegate
    XCTAssertNotNil(options.faceDetectorLiveStreamDelegate)
    XCTAssertEqual(options.runningMode, .liveStream)
    XCTAssertEqual(options.minDetectionConfidence, 0.7, accuracy: 1e-3)
    XCTAssertEqual(options.minSuppressionThreshold, 0.4, accuracy: 1e-3)
  }

  func testRuntimeRegistrationAndInitialization() {
    let options = FaceDetectorOptions()
    options.baseOptions.modelAssetPath = "/nonexistent/face_detector.task"

    XCTAssertThrowsError(try FaceDetector(options: options)) { error in
      let nsError = error as NSError
      XCTAssertEqual(nsError.domain, TasksErrorCode.errorDomain)
      XCTAssertTrue(
        nsError.code == TasksErrorCode.invalidArgumentError.rawValue
          || nsError.code == TasksErrorCode.notFoundError.rawValue
      )
      XCTAssertFalse(
        nsError.localizedDescription.contains("No registered object with name"),
        "FaceDetector subgraph not linked"
      )
      XCTAssertTrue(
        nsError.localizedDescription.contains("Unable to open file"),
        "Unexpected error: \(nsError.localizedDescription)"
      )
    }

    XCTAssertThrowsError(
      try FaceDetector(modelPath: "/nonexistent/face_detector.task")
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
    let _: (FaceDetector) -> (MPImage) throws -> FaceDetectorResult =
      FaceDetector.detect(image:)
    let _: (FaceDetector) -> (MPImage, Int) throws -> FaceDetectorResult =
      FaceDetector.detect(videoFrame:timestampInMilliseconds:)
    let _: (FaceDetector) -> (MPImage, Int) throws -> Void =
      FaceDetector.detectAsync(image:timestampInMilliseconds:)
  }

  func testResultContainerTypes() throws {
    let keypoint = NormalizedKeypoint(
      location: CGPoint(x: 0.5, y: 0.5),
      label: "nose_tip",
      score: 0.9
    )
    XCTAssertEqual(keypoint.location.x, 0.5, accuracy: 1e-3)
    XCTAssertEqual(keypoint.location.y, 0.5, accuracy: 1e-3)
    XCTAssertEqual(keypoint.label, "nose_tip")
    XCTAssertEqual(keypoint.score, 0.9, accuracy: 1e-3)

    let category = ResultCategory(
      index: 0,
      score: 0.9,
      categoryName: "face",
      displayName: "Face"
    )
    XCTAssertEqual(category.index, 0)
    XCTAssertEqual(category.score, 0.9, accuracy: 1e-3)
    XCTAssertEqual(category.categoryName, "face")
    XCTAssertEqual(category.displayName, "Face")

    let detection = Detection(
      categories: [category],
      boundingBox: CGRect(x: 0, y: 0, width: 100, height: 100),
      keypoints: [keypoint]
    )
    XCTAssertEqual(detection.categories.count, 1)
    XCTAssertEqual(detection.boundingBox.origin.x, 0)
    XCTAssertEqual(detection.boundingBox.origin.y, 0)
    XCTAssertEqual(detection.boundingBox.size.width, 100)
    XCTAssertEqual(detection.boundingBox.size.height, 100)
    let unwrappedKeypoints = try XCTUnwrap(detection.keypoints)
    XCTAssertEqual(unwrappedKeypoints.count, 1)

    let result = FaceDetectorResult(
      detections: [detection],
      timestampInMilliseconds: 1000
    )
    XCTAssertEqual(result.detections.count, 1)
    XCTAssertEqual(result.timestampInMilliseconds, 1000)
  }
}
