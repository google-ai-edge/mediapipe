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
import MPPObjectDetector
import XCTest

extension TasksErrorCode {
  fileprivate static let errorDomain = "com.google.mediapipe.tasks"
}

final class ObjectDetectorSmokeTests: XCTestCase {

  private final class MockLiveStreamDelegate: NSObject,
    ObjectDetectorLiveStreamDelegate
  {
    func objectDetector(
      _ objectDetector: ObjectDetector,
      didFinishDetection result: ObjectDetectorResult?,
      timestampInMilliseconds: Int,
      error: (any Error)?
    ) {}
  }

  func testOptionsAndDelegateCoverage() {
    let options = ObjectDetectorOptions()
    options.runningMode = .liveStream
    options.scoreThreshold = 0.5
    options.maxResults = 3
    options.displayNamesLocale = "en"
    options.categoryAllowlist = ["cat", "dog"]
    options.categoryDenylist = ["bird"]

    let delegate = MockLiveStreamDelegate()
    options.objectDetectorLiveStreamDelegate = delegate
    XCTAssertNotNil(options.objectDetectorLiveStreamDelegate)
    XCTAssertEqual(options.runningMode, .liveStream)
    XCTAssertEqual(options.scoreThreshold, 0.5, accuracy: 1e-3)
    XCTAssertEqual(options.maxResults, 3)
    XCTAssertEqual(options.displayNamesLocale, "en")
    XCTAssertEqual(options.categoryAllowlist, ["cat", "dog"])
    XCTAssertEqual(options.categoryDenylist, ["bird"])
  }

  func testRuntimeRegistrationAndInitialization() {
    let options = ObjectDetectorOptions()
    options.baseOptions.modelAssetPath = "/nonexistent/object_detector.task"

    XCTAssertThrowsError(try ObjectDetector(options: options)) { error in
      let nsError = error as NSError
      XCTAssertEqual(nsError.domain, TasksErrorCode.errorDomain)
      XCTAssertTrue(
        nsError.code == TasksErrorCode.invalidArgumentError.rawValue
          || nsError.code == TasksErrorCode.notFoundError.rawValue
      )
      XCTAssertFalse(
        nsError.localizedDescription.contains("No registered object with name"),
        "ObjectDetector subgraph not linked"
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
      XCTAssertEqual(nsError.domain, TasksErrorCode.errorDomain)
      XCTAssertTrue(
        nsError.code == TasksErrorCode.invalidArgumentError.rawValue
          || nsError.code == TasksErrorCode.notFoundError.rawValue
      )
    }
  }

  func testExecutionMethodSignaturesCompile() {
    let _: (ObjectDetector) -> (MPImage) throws -> ObjectDetectorResult =
      ObjectDetector.detect(image:)
    let _: (ObjectDetector) -> (MPImage, Int) throws -> ObjectDetectorResult =
      ObjectDetector.detect(videoFrame:timestampInMilliseconds:)
    let _: (ObjectDetector) -> (MPImage, Int) throws -> Void =
      ObjectDetector.detectAsync(image:timestampInMilliseconds:)
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
