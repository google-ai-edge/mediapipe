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
import MediaPipeTasksObjectDetector
import MediaPipeTasksVisionCore
import UIKit
import XCTest

final class ObjectDetectorSwiftTests: XCTestCase {
  private enum TestFixture {
    static let modelName = "coco_ssd_mobilenet_v1_1.0_quant_2018_06_29"
    static let modelExtension = "tflite"
    static let imageName = "cats_and_dogs"
    static let rotatedImageName = "cats_and_dogs_rotated"
    static let imageExtension = "jpg"
    static let scoreTolerance: Float = 0.02
    static let boxTolerance: CGFloat = 10.0
  }

  private func resourcePath(
    name: String, ext: String, file: StaticString = #filePath, line: UInt = #line
  ) throws -> String {
    try XCTUnwrap(
      Bundle(for: Self.self).path(forResource: name, ofType: ext), file: file, line: line)
  }

  private func loadImage(
    name: String = TestFixture.imageName,
    orientation: UIImage.Orientation? = nil,
    file: StaticString = #filePath,
    line: UInt = #line
  ) throws -> MPImage {
    let path = try resourcePath(name: name, ext: TestFixture.imageExtension, file: file, line: line)
    let uiImage = try XCTUnwrap(UIImage(contentsOfFile: path), file: file, line: line)
    return try MPImage(uiImage: uiImage, orientation: orientation)
  }

  private func assertBox(
    _ actual: CGRect, near expected: CGRect, file: StaticString = #filePath, line: UInt = #line
  ) {
    XCTAssertEqual(
      actual.origin.x, expected.origin.x, accuracy: TestFixture.boxTolerance, file: file, line: line)
    XCTAssertEqual(
      actual.origin.y, expected.origin.y, accuracy: TestFixture.boxTolerance, file: file, line: line)
    XCTAssertEqual(
      actual.width, expected.width, accuracy: TestFixture.boxTolerance, file: file, line: line)
    XCTAssertEqual(
      actual.height, expected.height, accuracy: TestFixture.boxTolerance, file: file, line: line)
  }

  func testDefaultOptionsValues() {
    let defaults = ObjectDetectorOptions()
    XCTAssertEqual(defaults.runningMode, .image)
    XCTAssertEqual(defaults.displayNamesLocale, "en")
    XCTAssertEqual(defaults.maxResults, -1)
    XCTAssertEqual(defaults.scoreThreshold, 0.0)
  }

  func testInitFailsWithEmptyModelPath() {
    XCTAssertThrowsError(try ObjectDetector(modelPath: "")) { error in
      XCTAssertEqual((error as? TasksError)?.code, .invalidArgument)
    }
  }

  func testInitFailsWithConflictingCategoryAllowlistAndDenylist() throws {
    let modelPath = try resourcePath(name: TestFixture.modelName, ext: TestFixture.modelExtension)
    let conflictingLists = ObjectDetectorOptions(
      baseOptions: BaseOptions(modelAssetPath: modelPath),
      categoryAllowlist: ["cat"],
      categoryDenylist: ["dog"]
    )
    XCTAssertThrowsError(try ObjectDetector(options: conflictingLists)) { error in
      XCTAssertEqual((error as? TasksError)?.code, .invalidArgument)
    }
  }

  func testInitFailsInLiveStreamModeWithoutResultHandler() throws {
    let modelPath = try resourcePath(name: TestFixture.modelName, ext: TestFixture.modelExtension)
    let missingHandler = ObjectDetectorOptions(
      baseOptions: BaseOptions(modelAssetPath: modelPath), runningMode: .liveStream)
    XCTAssertThrowsError(try ObjectDetector(options: missingHandler)) { error in
      XCTAssertEqual((error as? TasksError)?.code, .invalidArgument)
    }
  }

  func testInitFailsInImageModeWithResultHandler() throws {
    let modelPath = try resourcePath(name: TestFixture.modelName, ext: TestFixture.modelExtension)
    let unexpectedHandler = ObjectDetectorOptions(
      baseOptions: BaseOptions(modelAssetPath: modelPath),
      runningMode: .image,
      liveStreamResultHandler: { _, _ in }
    )
    XCTAssertThrowsError(try ObjectDetector(options: unexpectedHandler)) { error in
      XCTAssertEqual((error as? TasksError)?.code, .invalidArgument)
    }
  }

  func testDetectImageReturnsExpectedDetections() throws {
    let modelPath = try resourcePath(name: TestFixture.modelName, ext: TestFixture.modelExtension)
    let image = try loadImage()

    let defaultDetector = try ObjectDetector(modelPath: modelPath)
    let fullResult = try defaultDetector.detect(image: image)
    XCTAssertEqual(fullResult.timestampInMilliseconds, 0)
    XCTAssertEqual(fullResult.detections.count, 10)
    XCTAssertEqual(fullResult.detections[0].categories.first?.categoryName, "cat")
    XCTAssertEqual(
      try XCTUnwrap(fullResult.detections[0].categories.first?.score), 0.6992,
      accuracy: TestFixture.scoreTolerance)
    assertBox(
      fullResult.detections[0].boundingBox, near: CGRect(x: 608, y: 161, width: 381, height: 439))
    XCTAssertEqual(fullResult, Set([fullResult]).first)
  }

  func testDetectImageAppliesScoreThresholdAndDenylist() throws {
    let modelPath = try resourcePath(name: TestFixture.modelName, ext: TestFixture.modelExtension)
    let image = try loadImage()

    let thresholdDetector = try ObjectDetector(
      options: ObjectDetectorOptions(
        baseOptions: BaseOptions(modelAssetPath: modelPath), scoreThreshold: 0.68))
    XCTAssertEqual(try thresholdDetector.detect(image: image).detections.count, 1)

    let denylistDetector = try ObjectDetector(
      options: ObjectDetectorOptions(
        baseOptions: BaseOptions(modelAssetPath: modelPath), categoryDenylist: ["cat"]))
    let deniedResult = try denylistDetector.detect(image: image)
    XCTAssertEqual(deniedResult.detections.first?.categories.first?.categoryName, "teddy bear")
  }

  func testDetectImageAppliesRotatedOrientation() throws {
    let modelPath = try resourcePath(name: TestFixture.modelName, ext: TestFixture.modelExtension)
    let rotatedDetector = try ObjectDetector(
      options: ObjectDetectorOptions(
        baseOptions: BaseOptions(modelAssetPath: modelPath), maxResults: 1))
    let rotatedImage = try loadImage(name: TestFixture.rotatedImageName, orientation: .left)
    let rotatedResult = try rotatedDetector.detect(image: rotatedImage)
    XCTAssertEqual(rotatedResult.detections.count, 1)
    assertBox(
      rotatedResult.detections[0].boundingBox, near: CGRect(x: 0, y: 608, width: 439, height: 387))
  }

  func testDetectRejectsUnsupportedAndOutOfBoundsRegionOfInterest() throws {
    let modelPath = try resourcePath(name: TestFixture.modelName, ext: TestFixture.modelExtension)
    let detector = try ObjectDetector(modelPath: modelPath)
    let image = try loadImage()

    XCTAssertThrowsError(
      try detector.detect(
        image: image,
        imageProcessingOptions: ImageProcessingOptions(
          regionOfInterest: CGRect(x: 0.1, y: 0.1, width: 0.5, height: 0.5)))
    ) { error in
      XCTAssertEqual((error as? TasksError)?.code, .invalidArgument)
    }

    XCTAssertThrowsError(
      try detector.detect(
        image: image,
        imageProcessingOptions: ImageProcessingOptions(
          regionOfInterest: CGRect(x: -0.1, y: 0.0, width: 0.5, height: 0.5)))
    ) { error in
      XCTAssertEqual((error as? TasksError)?.code, .invalidArgument)
    }
  }

  func testDetectRejectsMismatchedRunningModes() throws {
    let modelPath = try resourcePath(name: TestFixture.modelName, ext: TestFixture.modelExtension)
    let image = try loadImage()

    let imageDetector = try ObjectDetector(modelPath: modelPath)
    XCTAssertThrowsError(try imageDetector.detect(videoFrame: image, timestampInMilliseconds: 0)) {
      XCTAssertEqual(($0 as? TasksError)?.code, .invalidArgument)
    }
    XCTAssertThrowsError(try imageDetector.detectAsync(image: image, timestampInMilliseconds: 0)) {
      XCTAssertEqual(($0 as? TasksError)?.code, .invalidArgument)
    }

    let videoDetector = try ObjectDetector(
      options: ObjectDetectorOptions(
        baseOptions: BaseOptions(modelAssetPath: modelPath), runningMode: .video))
    XCTAssertThrowsError(try videoDetector.detect(image: image)) {
      XCTAssertEqual(($0 as? TasksError)?.code, .invalidArgument)
    }
    XCTAssertThrowsError(try videoDetector.detectAsync(image: image, timestampInMilliseconds: 0)) {
      XCTAssertEqual(($0 as? TasksError)?.code, .invalidArgument)
    }

    let liveDetector = try ObjectDetector(
      options: ObjectDetectorOptions(
        baseOptions: BaseOptions(modelAssetPath: modelPath),
        runningMode: .liveStream,
        liveStreamResultHandler: { _, _ in }
      ))
    XCTAssertThrowsError(try liveDetector.detect(image: image)) {
      XCTAssertEqual(($0 as? TasksError)?.code, .invalidArgument)
    }
    XCTAssertThrowsError(try liveDetector.detect(videoFrame: image, timestampInMilliseconds: 0)) {
      XCTAssertEqual(($0 as? TasksError)?.code, .invalidArgument)
    }
  }

  func testDetectVideoFramesAtMonotonicTimestamps() throws {
    let modelPath = try resourcePath(name: TestFixture.modelName, ext: TestFixture.modelExtension)
    let image = try loadImage()
    let videoDetector = try ObjectDetector(
      options: ObjectDetectorOptions(
        baseOptions: BaseOptions(modelAssetPath: modelPath), runningMode: .video, maxResults: 3))

    let first = try videoDetector.detect(videoFrame: image, timestampInMilliseconds: 0)
    XCTAssertEqual(first.timestampInMilliseconds, 0)
    XCTAssertEqual(first.detections.count, 3)
    XCTAssertEqual(first.detections.first?.categories.first?.categoryName, "cat")

    let second = try videoDetector.detect(videoFrame: image, timestampInMilliseconds: 300)
    XCTAssertEqual(second.timestampInMilliseconds, 300)
    XCTAssertEqual(second.detections.count, 3)
    XCTAssertEqual(second.detections.first?.categories.first?.categoryName, "cat")
  }

  func testDetectVideoRejectsNonMonotonicAndOutOfRangeTimestamps() throws {
    let modelPath = try resourcePath(name: TestFixture.modelName, ext: TestFixture.modelExtension)
    let image = try loadImage()
    let videoDetector = try ObjectDetector(
      options: ObjectDetectorOptions(
        baseOptions: BaseOptions(modelAssetPath: modelPath), runningMode: .video))

    _ = try videoDetector.detect(videoFrame: image, timestampInMilliseconds: 300)
    XCTAssertThrowsError(
      try videoDetector.detect(videoFrame: image, timestampInMilliseconds: 100)
    )
    XCTAssertThrowsError(try videoDetector.detect(videoFrame: image, timestampInMilliseconds: -1)) {
      XCTAssertEqual(($0 as? TasksError)?.code, .invalidArgument)
    }
    XCTAssertThrowsError(
      try videoDetector.detect(videoFrame: image, timestampInMilliseconds: .max)
    ) {
      XCTAssertEqual(($0 as? TasksError)?.code, .invalidArgument)
    }
  }

  func testDetectAsyncInLiveStreamModeDeliversResult() throws {
    let modelPath = try resourcePath(name: TestFixture.modelName, ext: TestFixture.modelExtension)
    let image = try loadImage()
    let expectation = expectation(description: "liveStream callback")
    final class CallbackState: @unchecked Sendable {
      let lock = NSLock()
      var received: (Result<ObjectDetectorResult, any Error>, Int)?
    }
    let state = CallbackState()
    let liveDetector = try ObjectDetector(
      options: ObjectDetectorOptions(
        baseOptions: BaseOptions(modelAssetPath: modelPath),
        runningMode: .liveStream,
        maxResults: 3,
        liveStreamResultHandler: { outcome, timestamp in
          state.lock.withLock {
            if state.received == nil {
              state.received = (outcome, timestamp)
              expectation.fulfill()
            }
          }
        }
      ))
    try liveDetector.detectAsync(image: image, timestampInMilliseconds: 42)
    wait(for: [expectation], timeout: 10.0)
    let pair = try XCTUnwrap(state.lock.withLock { state.received })
    let liveResult = try pair.0.get()
    XCTAssertEqual(pair.1, 42)
    XCTAssertEqual(liveResult.timestampInMilliseconds, 42)
    XCTAssertEqual(liveResult.detections.count, 3)
    XCTAssertEqual(liveResult.detections[0].categories.first?.categoryName, "cat")
  }
}
