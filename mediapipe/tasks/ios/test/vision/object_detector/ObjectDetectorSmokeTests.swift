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
import MediaPipeTasksObjectDetector
import XCTest

final class ObjectDetectorSmokeTests: XCTestCase {

  func testOptionsAndDelegateCoverage() {
    var options = ObjectDetectorOptions()
    options.runningMode = .liveStream
    options.scoreThreshold = 0.5
    options.maxResults = 3
    options.displayNamesLocale = "en"
    options.categoryAllowlist = ["cat", "dog"]
    options.categoryDenylist = ["bird"]
    options.liveStreamResultHandler = { _, _ in }

    XCTAssertNotNil(options.liveStreamResultHandler)
    XCTAssertEqual(options.runningMode, .liveStream)
    XCTAssertEqual(options.scoreThreshold, 0.5, accuracy: 1e-3)
    XCTAssertEqual(options.maxResults, 3)
    XCTAssertEqual(options.displayNamesLocale, "en")
    XCTAssertEqual(options.categoryAllowlist, ["cat", "dog"])
    XCTAssertEqual(options.categoryDenylist, ["bird"])
  }

  func testRuntimeRegistrationAndInitialization() {
    var options = ObjectDetectorOptions()
    options.baseOptions.modelAssetPath = "/nonexistent/object_detector.task"

    XCTAssertThrowsError(try ObjectDetector(options: options)) { error in
      let nsError = error as NSError
      XCTAssertEqual(nsError.domain, TasksError.errorDomain)
      XCTAssertTrue(
        nsError.code == TasksErrorCode.invalidArgument.rawValue
          || nsError.code == TasksErrorCode.notFound.rawValue
      )
      XCTAssertTrue(
        nsError.localizedDescription.contains("Unable to open file"),
        "Unexpected error: \(nsError.localizedDescription)"
      )
    }

    XCTAssertThrowsError(
      try ObjectDetector(modelPath: "/nonexistent/object_detector.task")
    ) { error in
      let nsError = error as NSError
      XCTAssertEqual(nsError.domain, TasksError.errorDomain)
      XCTAssertTrue(
        nsError.code == TasksErrorCode.invalidArgument.rawValue
          || nsError.code == TasksErrorCode.notFound.rawValue
      )
    }
  }

  func testExecutionMethodSignaturesCompile() {
    let _:
      (ObjectDetector) -> (MPImage, ImageProcessingOptions?) throws ->
        ObjectDetectorResult =
        ObjectDetector.detect(image:imageProcessingOptions:)
    let _:
      (ObjectDetector) -> (MPImage, Int, ImageProcessingOptions?) throws ->
        ObjectDetectorResult =
        ObjectDetector.detect(videoFrame:timestampInMilliseconds:imageProcessingOptions:)
    let _: (ObjectDetector) -> (MPImage, Int, ImageProcessingOptions?) throws -> Void =
      ObjectDetector.detectAsync(image:timestampInMilliseconds:imageProcessingOptions:)
  }

  func testResultContainerTypes() {
    let category = ResultCategory(
      index: 0,
      score: 0.9,
      categoryName: "cat",
      displayName: "Cat"
    )
    XCTAssertEqual(category.index, 0)
    XCTAssertEqual(category.score, 0.9, accuracy: 1e-3)
    XCTAssertEqual(category.categoryName, "cat")
    XCTAssertEqual(category.displayName, "Cat")

    let detection = Detection(
      categories: [category],
      boundingBox: CGRect(x: 0, y: 0, width: 100, height: 100),
      keypoints: nil
    )
    XCTAssertEqual(detection.categories.count, 1)
    XCTAssertEqual(detection.boundingBox.origin.x, 0)
    XCTAssertEqual(detection.boundingBox.origin.y, 0)
    XCTAssertEqual(detection.boundingBox.size.width, 100)
    XCTAssertEqual(detection.boundingBox.size.height, 100)
    XCTAssertNil(detection.keypoints)

    let result = ObjectDetectorResult(
      detections: [detection],
      timestampInMilliseconds: 1000
    )
    XCTAssertEqual(result.detections.count, 1)
    XCTAssertEqual(result.timestampInMilliseconds, 1000)
  }
}
