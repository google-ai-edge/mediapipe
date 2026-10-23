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
import MPPImageSegmenter
import XCTest

extension TasksErrorCode {
  fileprivate static let errorDomain = "com.google.mediapipe.tasks"
}

final class ImageSegmenterSmokeTests: XCTestCase {

  private final class MockLiveStreamDelegate: NSObject,
    ImageSegmenterLiveStreamDelegate
  {
    func imageSegmenter(
      _ imageSegmenter: ImageSegmenter,
      didFinishSegmentation result: ImageSegmenterResult?,
      timestampInMilliseconds: Int,
      error: (any Error)?
    ) {}
  }

  func testOptionsAndDelegateCoverage() {
    let options = ImageSegmenterOptions()
    options.runningMode = .liveStream
    options.displayNamesLocale = "en"
    options.shouldOutputConfidenceMasks = true
    options.shouldOutputCategoryMask = true

    let delegate = MockLiveStreamDelegate()
    options.imageSegmenterLiveStreamDelegate = delegate
    XCTAssertNotNil(options.imageSegmenterLiveStreamDelegate)
    XCTAssertEqual(options.runningMode, .liveStream)
    XCTAssertEqual(options.displayNamesLocale, "en")
    XCTAssertTrue(options.shouldOutputConfidenceMasks)
    XCTAssertTrue(options.shouldOutputCategoryMask)
  }

  func testRuntimeRegistrationAndInitialization() {
    let options = ImageSegmenterOptions()
    options.baseOptions.modelAssetPath =
      "/nonexistent/image_segmenter.task"

    XCTAssertThrowsError(try ImageSegmenter(options: options)) { error in
      let nsError = error as NSError
      XCTAssertEqual(nsError.domain, TasksErrorCode.errorDomain)
      XCTAssertTrue(
        nsError.code == TasksErrorCode.invalidArgumentError.rawValue
          || nsError.code == TasksErrorCode.notFoundError.rawValue
      )
      XCTAssertFalse(
        nsError.localizedDescription.contains("No registered object with name"),
        "ImageSegmenter subgraph not linked"
      )
      XCTAssertTrue(
        nsError.localizedDescription.contains("Unable to open file"),
        "Unexpected error: \(nsError.localizedDescription)"
      )
    }

    XCTAssertThrowsError(
      try ImageSegmenter(modelPath: "/nonexistent/image_segmenter.task")
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
    let _: (ImageSegmenter) -> (MPImage) throws -> ImageSegmenterResult =
      ImageSegmenter.segment(image:)
    let _:
      (ImageSegmenter) -> (
        MPImage,
        @escaping (ImageSegmenterResult?, (any Error)?) -> Void
      ) -> Void =
        ImageSegmenter.segment(image:completion:)
    let _: (ImageSegmenter) -> (MPImage, Int) throws -> ImageSegmenterResult =
      ImageSegmenter.segment(videoFrame:timestampInMilliseconds:)
    let _:
      (ImageSegmenter) -> (
        MPImage,
        Int,
        @escaping (ImageSegmenterResult?, (any Error)?) -> Void
      ) -> Void =
        ImageSegmenter.segment(videoFrame:timestampInMilliseconds:completion:)
    let _: (ImageSegmenter) -> (MPImage, Int) throws -> Void =
      ImageSegmenter.segmentAsync(image:timestampInMilliseconds:)
    let _: KeyPath<ImageSegmenter, [String]> = \ImageSegmenter.labels
  }

  func testResultContainerTypes() throws {
    var uint8Pixel: UInt8 = 128
    let mask = Mask(
      uInt8Data: &uint8Pixel,
      width: 1,
      height: 1,
      shouldCopy: true
    )
    let unwrappedMask = try XCTUnwrap(mask)
    XCTAssertEqual(unwrappedMask.width, 1)
    XCTAssertEqual(unwrappedMask.height, 1)
    XCTAssertEqual(unwrappedMask.dataType, .uInt8)

    var float32Pixel: Float = 0.5
    let floatMask = Mask(
      float32Data: &float32Pixel,
      width: 1,
      height: 1,
      shouldCopy: true
    )
    let unwrappedFloatMask = try XCTUnwrap(floatMask)
    XCTAssertEqual(unwrappedFloatMask.width, 1)
    XCTAssertEqual(unwrappedFloatMask.height, 1)
    XCTAssertEqual(unwrappedFloatMask.dataType, .float32)

    let qualityScores: [NSNumber] = [NSNumber(value: 0.95)]
    let result = ImageSegmenterResult(
      confidenceMasks: [unwrappedFloatMask],
      categoryMask: unwrappedMask,
      qualityScores: qualityScores,
      timestampInMilliseconds: 1000
    )
    let unwrappedConfidenceMasks = try XCTUnwrap(result.confidenceMasks)
    XCTAssertEqual(unwrappedConfidenceMasks.count, 1)
    XCTAssertNotNil(result.categoryMask)
    let scores = try XCTUnwrap(result.qualityScores)
    XCTAssertEqual(scores.count, 1)
    let firstScore = try XCTUnwrap(scores.first)
    XCTAssertEqual(firstScore.floatValue, 0.95, accuracy: 1e-3)
    XCTAssertEqual(result.timestampInMilliseconds, 1000)
  }
}
