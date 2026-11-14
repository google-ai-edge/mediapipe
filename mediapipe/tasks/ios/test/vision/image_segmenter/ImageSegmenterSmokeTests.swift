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
import MediaPipeTasksImageSegmenter
import XCTest

final class ImageSegmenterSmokeTests: XCTestCase {

  func testOptionsAndDelegateCoverage() {
    var options = ImageSegmenterOptions()
    XCTAssertTrue(options.outputConfidenceMasks)
    XCTAssertFalse(options.outputCategoryMask)

    options.runningMode = .liveStream
    options.displayNamesLocale = "en"
    options.outputConfidenceMasks = false
    options.outputCategoryMask = true
    options.liveStreamResultHandler = { _, _ in }

    XCTAssertNotNil(options.liveStreamResultHandler)
    XCTAssertEqual(options.runningMode, .liveStream)
    XCTAssertEqual(options.displayNamesLocale, "en")
    XCTAssertFalse(options.outputConfidenceMasks)
    XCTAssertTrue(options.outputCategoryMask)
  }

  func testRuntimeRegistrationAndInitialization() {
    var options = ImageSegmenterOptions()
    options.baseOptions.modelAssetPath =
      "/nonexistent/image_segmenter.task"

    XCTAssertThrowsError(try ImageSegmenter(options: options)) { error in
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
      try ImageSegmenter(modelPath: "/nonexistent/image_segmenter.task")
    ) { error in
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
  }

  func testExecutionMethodSignaturesCompile() {
    let _:
      (ImageSegmenter) -> (MPImage, ImageProcessingOptions?) throws ->
        ImageSegmenterResult =
        ImageSegmenter.segment(image:imageProcessingOptions:)
    let _:
      (ImageSegmenter) -> (MPImage, Int, ImageProcessingOptions?) throws ->
        ImageSegmenterResult =
        ImageSegmenter.segment(videoFrame:timestampInMilliseconds:imageProcessingOptions:)
    let _: (ImageSegmenter) -> (MPImage, Int, ImageProcessingOptions?) throws -> Void =
      ImageSegmenter.segmentAsync(image:timestampInMilliseconds:imageProcessingOptions:)
    let _: KeyPath<ImageSegmenter, [String]> = \ImageSegmenter.labels
  }

  func testResultContainerTypes() throws {
    var uint8Pixel: UInt8 = 128
    let mask = Mask(
      uint8Data: &uint8Pixel,
      width: 1,
      height: 1
    )
    XCTAssertEqual(mask.width, 1)
    XCTAssertEqual(mask.height, 1)
    XCTAssertEqual(mask.dataType, .uInt8)

    var float32Pixel: Float = 0.5
    let floatMask = Mask(
      float32Data: &float32Pixel,
      width: 1,
      height: 1
    )
    XCTAssertEqual(floatMask.width, 1)
    XCTAssertEqual(floatMask.height, 1)
    XCTAssertEqual(floatMask.dataType, .float32)

    let result = ImageSegmenterResult(
      confidenceMasks: [floatMask],
      categoryMask: mask,
      qualityScores: [0.95],
      timestampInMilliseconds: 1000
    )
    let unwrappedConfidenceMasks = try XCTUnwrap(result.confidenceMasks)
    XCTAssertEqual(unwrappedConfidenceMasks.count, 1)
    XCTAssertNotNil(result.categoryMask)
    XCTAssertEqual(result.qualityScores.count, 1)
    let firstScore = try XCTUnwrap(result.qualityScores.first)
    XCTAssertEqual(firstScore, 0.95, accuracy: 1e-3)
    XCTAssertEqual(result.timestampInMilliseconds, 1000)
  }
}
