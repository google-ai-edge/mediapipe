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
import MediaPipeTasksInteractiveSegmenterLegacy
import XCTest

final class InteractiveSegmenterLegacySmokeTests: XCTestCase {

  func testOptionsCoverage() {
    var options = InteractiveSegmenterLegacyOptions()
    XCTAssertTrue(options.outputConfidenceMasks)
    XCTAssertFalse(options.outputCategoryMask)
    XCTAssertEqual(options.displayNamesLocale, "en")

    options.outputConfidenceMasks = false
    options.outputCategoryMask = true
    options.displayNamesLocale = "fr"

    XCTAssertFalse(options.outputConfidenceMasks)
    XCTAssertTrue(options.outputCategoryMask)
    XCTAssertEqual(options.displayNamesLocale, "fr")
  }

  func testRuntimeRegistrationAndInitialization() {
    var options = InteractiveSegmenterLegacyOptions()
    options.baseOptions.modelAssetPath =
      "/nonexistent/interactive_segmenter.task"

    XCTAssertThrowsError(try InteractiveSegmenterLegacy(options: options)) { error in
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
      try InteractiveSegmenterLegacy(modelPath: "/nonexistent/interactive_segmenter.task")
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
      (InteractiveSegmenterLegacy) -> (
        MPImage,
        RegionOfInterest,
        ImageProcessingOptions?
      ) throws -> InteractiveSegmenterLegacyResult =
        InteractiveSegmenterLegacy.segment(image:regionOfInterest:imageProcessingOptions:)
    let _: KeyPath<InteractiveSegmenterLegacy, [String]> = \InteractiveSegmenterLegacy.labels
  }

  func testRegionOfInterestAndResultContainer() throws {
    let keypoint = NormalizedKeypoint(
      location: CGPoint(x: 0.5, y: 0.5),
      label: "point",
      score: 1.0
    )
    let roiPoint = RegionOfInterest(keypoint: keypoint)
    XCTAssertNotNil(roiPoint.keypoint)
    XCTAssertNil(roiPoint.scribbles)

    let roiScribbles = RegionOfInterest(scribbles: [keypoint])
    XCTAssertNotNil(roiScribbles.scribbles)
    XCTAssertNil(roiScribbles.keypoint)

    var uint8Pixel: UInt8 = 128
    let mask = Mask(
      uint8Data: &uint8Pixel,
      width: 1,
      height: 1
    )

    let result = InteractiveSegmenterLegacyResult(
      confidenceMasks: nil,
      categoryMask: mask,
      qualityScores: [0.9],
      timestampInMilliseconds: 1000
    )
    XCTAssertNil(result.confidenceMasks)
    XCTAssertNotNil(result.categoryMask)
    XCTAssertEqual(result.qualityScores.count, 1)
    let firstScore = try XCTUnwrap(result.qualityScores.first)
    XCTAssertEqual(firstScore, 0.9, accuracy: 1e-3)
    XCTAssertEqual(result.timestampInMilliseconds, 1000)
  }
}
