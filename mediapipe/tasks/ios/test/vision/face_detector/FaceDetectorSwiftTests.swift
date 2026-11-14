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

import CoreGraphics
import Foundation
import MediaPipeTasksComponents
import MediaPipeTasksCore
import MediaPipeTasksFaceDetector
import MediaPipeTasksVisionCore
import UIKit
import XCTest
import third_party_mediapipe_tasks_c_components_containers_detection_result
import third_party_mediapipe_tasks_c_components_containers_rect
import third_party_mediapipe_tasks_c_vision_core_image
import third_party_mediapipe_tasks_c_vision_face_detector_face_detector_lib

final class FaceDetectorSwiftTests: XCTestCase {
  private enum TestFixture {
    static let shortRangeModel = "face_detection_short_range"
    static let keypointTolerance: CGFloat = 0.02
    static let portraitKeypoints: [CGPoint] = [
      CGPoint(x: 0.44416, y: 0.17643), CGPoint(x: 0.55514, y: 0.17731),
      CGPoint(x: 0.50467, y: 0.22657), CGPoint(x: 0.50227, y: 0.27199),
      CGPoint(x: 0.36063, y: 0.20143), CGPoint(x: 0.60841, y: 0.20409),
    ]
    static let rotatedPortraitKeypoints: [CGPoint] = [
      CGPoint(x: 0.82075, y: 0.44679), CGPoint(x: 0.81965, y: 0.56261),
      CGPoint(x: 0.76194, y: 0.51719), CGPoint(x: 0.71993, y: 0.51719),
      CGPoint(x: 0.80700, y: 0.36298), CGPoint(x: 0.80882, y: 0.61204),
    ]
  }

  private static func modelPath(
    file: StaticString = #filePath, line: UInt = #line
  ) throws -> String {
    let bundle = Bundle(for: FaceDetectorSwiftTests.self)
    return try XCTUnwrap(
      bundle.path(forResource: TestFixture.shortRangeModel, ofType: "tflite"),
      file: file, line: line)
  }

  private static func loadImage(
    _ name: String, orientation: UIImage.Orientation = .up,
    file: StaticString = #filePath, line: UInt = #line
  ) throws -> MPImage {
    let bundle = Bundle(for: FaceDetectorSwiftTests.self)
    let path = try XCTUnwrap(bundle.path(forResource: name, ofType: "jpg"), file: file, line: line)
    return try MPImage(
      uiImage: try XCTUnwrap(UIImage(contentsOfFile: path), file: file, line: line),
      orientation: orientation)
  }

  private func assertPortraitDetections(
    _ result: FaceDetectorResult,
    expectedKeypoints: [CGPoint] = TestFixture.portraitKeypoints,
    file: StaticString = #filePath,
    line: UInt = #line
  ) throws {
    XCTAssertEqual(result.detections.count, 1, file: file, line: line)
    let detection = try XCTUnwrap(result.detections.first, file: file, line: line)
    XCTAssertFalse(detection.boundingBox.isEmpty, file: file, line: line)
    let keypoints = try XCTUnwrap(detection.keypoints, file: file, line: line)
    XCTAssertEqual(keypoints.count, expectedKeypoints.count, file: file, line: line)
    for (actual, expected) in zip(keypoints, expectedKeypoints) {
      XCTAssertEqual(
        actual.location.x, expected.x, accuracy: TestFixture.keypointTolerance, file: file,
        line: line)
      XCTAssertEqual(
        actual.location.y, expected.y, accuracy: TestFixture.keypointTolerance, file: file,
        line: line)
    }
  }

  func testOptionsAndResultValueSemantics() throws {
    var options = FaceDetectorOptions()
    XCTAssertEqual(options.runningMode, .image)
    XCTAssertEqual(options.minDetectionConfidence, 0.5)
    XCTAssertEqual(options.minSuppressionThreshold, 0.3)
    XCTAssertNil(options.liveStreamResultHandler)

    var copy = options
    copy.runningMode = .video
    copy.minDetectionConfidence = 0.7
    copy.minSuppressionThreshold = 0.4
    XCTAssertEqual(options.runningMode, .image)
    try copy.withCOptions { cOptions in
      XCTAssertEqual(cOptions.running_mode, MP_RUNNING_MODE_VIDEO)
      XCTAssertEqual(cOptions.min_detection_confidence, 0.7, accuracy: 1e-6)
      XCTAssertEqual(cOptions.min_suppression_threshold, 0.4, accuracy: 1e-6)
    }

    var cDetection = MpDetection(
      categories: nil, categories_count: 0,
      bounding_box: MpRect(left: 10, top: 20, bottom: 70, right: 50),
      keypoints: nil, keypoints_count: 0
    )
    withUnsafeMutablePointer(to: &cDetection) { ptr in
      let bridged = FaceDetectorResult(
        cResult: MpDetectionResult(detections: ptr, detections_count: 1),
        timestampInMilliseconds: 42
      )
      let manual = FaceDetectorResult(
        detections: [
          Detection(categories: [], boundingBox: CGRect(x: 10, y: 20, width: 40, height: 50))
        ],
        timestampInMilliseconds: 42
      )
      XCTAssertEqual(bridged, manual)
      XCTAssertEqual(Set([bridged, manual]).count, 1)
    }
  }

  func testValidationAndWrongRunningModeErrors() throws {
    XCTAssertThrowsError(try FaceDetector(options: FaceDetectorOptions())) { error in
      XCTAssertEqual((error as? TasksError)?.code, .invalidArgument)
      XCTAssertEqual((error as NSError).domain, TasksError.errorDomain)
    }
    let path = try Self.modelPath()
    XCTAssertThrowsError(
      try FaceDetector(
        options: FaceDetectorOptions(
          baseOptions: BaseOptions(modelAssetPath: path), runningMode: .liveStream)))
    XCTAssertThrowsError(
      try FaceDetector(
        options: FaceDetectorOptions(
          baseOptions: BaseOptions(modelAssetPath: path), runningMode: .image,
          liveStreamResultHandler: { _, _ in })))

    let detector = try FaceDetector(modelPath: path)
    let portrait = try Self.loadImage("portrait")
    XCTAssertThrowsError(try detector.detect(videoFrame: portrait, timestampInMilliseconds: 0))
    XCTAssertThrowsError(try detector.detectAsync(image: portrait, timestampInMilliseconds: 0))
    let roi = ImageProcessingOptions(
      regionOfInterest: CGRect(x: 0.1, y: 0.1, width: 0.5, height: 0.5))
    XCTAssertThrowsError(try detector.detect(image: portrait, imageProcessingOptions: roi)) {
      XCTAssertEqual(($0 as? TasksError)?.code, .invalidArgument)
    }
  }

  func testImageAndVideoModeDetection() throws {
    let path = try Self.modelPath()
    let imageDetector = try FaceDetector(modelPath: path)
    try assertPortraitDetections(try imageDetector.detect(image: try Self.loadImage("portrait")))
    try assertPortraitDetections(
      try imageDetector.detect(image: try Self.loadImage("portrait_rotated", orientation: .left)),
      expectedKeypoints: TestFixture.rotatedPortraitKeypoints
    )
    XCTAssertTrue(try imageDetector.detect(image: try Self.loadImage("cat")).detections.isEmpty)

    let videoDetector = try FaceDetector(
      options: FaceDetectorOptions(
        baseOptions: BaseOptions(modelAssetPath: path), runningMode: .video)
    )
    let portrait = try Self.loadImage("portrait")
    XCTAssertThrowsError(
      try videoDetector.detect(videoFrame: portrait, timestampInMilliseconds: -1))
    XCTAssertThrowsError(
      try videoDetector.detect(videoFrame: portrait, timestampInMilliseconds: .max))
    for timestamp in [0, 33, 66] {
      let videoResult = try videoDetector.detect(
        videoFrame: portrait, timestampInMilliseconds: timestamp)
      XCTAssertEqual(videoResult.timestampInMilliseconds, timestamp)
      try assertPortraitDetections(videoResult)
    }
  }

  func testLiveStreamDetectionAndOutOfOrderTimestamp() throws {
    final class CapturedLiveResult: @unchecked Sendable {
      private let lock = NSLock()
      private let semaphore = DispatchSemaphore(value: 0)
      private var outcome: Result<FaceDetectorResult, any Error>?

      func record(_ result: Result<FaceDetectorResult, any Error>) {
        let shouldSignal = lock.withLock { () -> Bool in
          guard outcome == nil else { return false }
          outcome = result
          return true
        }
        if shouldSignal { semaphore.signal() }
      }

      func wait(timeoutSeconds: TimeInterval) -> Result<FaceDetectorResult, any Error>? {
        semaphore.wait(timeout: .now() + timeoutSeconds) == .success
          ? lock.withLock { outcome } : nil
      }
    }

    let captured = CapturedLiveResult()
    let liveDetector = try FaceDetector(
      options: FaceDetectorOptions(
        baseOptions: BaseOptions(modelAssetPath: try Self.modelPath()),
        runningMode: .liveStream,
        liveStreamResultHandler: { outcome, timestamp in
          if timestamp == 10 { captured.record(outcome) }
        }
      )
    )
    let portrait = try Self.loadImage("portrait")
    try liveDetector.detectAsync(image: portrait, timestampInMilliseconds: 10)
    XCTAssertThrowsError(try liveDetector.detectAsync(image: portrait, timestampInMilliseconds: 5))
    let delivered = try XCTUnwrap(captured.wait(timeoutSeconds: 5.0)).get()
    XCTAssertEqual(delivered.timestampInMilliseconds, 10)
    try assertPortraitDetections(delivered)
  }
}
