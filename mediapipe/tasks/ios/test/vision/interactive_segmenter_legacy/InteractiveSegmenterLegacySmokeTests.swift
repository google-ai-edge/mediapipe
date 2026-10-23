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
import MPPInteractiveSegmenterLegacy
import XCTest

extension TasksErrorCode {
  fileprivate static let errorDomain = "com.google.mediapipe.tasks"
}

final class InteractiveSegmenterLegacySmokeTests: XCTestCase {

  func testOptionsCoverage() {
    let options = InteractiveSegmenterLegacyOptions()
    options.shouldOutputConfidenceMasks = true
    options.shouldOutputCategoryMask = true
    options.displayNamesLocale = "en"

    XCTAssertTrue(options.shouldOutputConfidenceMasks)
    XCTAssertTrue(options.shouldOutputCategoryMask)
    XCTAssertEqual(options.displayNamesLocale, "en")
  }

  func testRuntimeRegistrationAndInitialization() {
    let options = InteractiveSegmenterLegacyOptions()
    options.baseOptions.modelAssetPath =
      "/nonexistent/interactive_segmenter.task"

    XCTAssertThrowsError(try InteractiveSegmenterLegacy(options: options)) { error in
      let nsError = error as NSError
      XCTAssertEqual(nsError.domain, TasksErrorCode.errorDomain)
      XCTAssertTrue(
        nsError.code == TasksErrorCode.invalidArgumentError.rawValue
          || nsError.code == TasksErrorCode.notFoundError.rawValue
      )
      XCTAssertFalse(
        nsError.localizedDescription.contains("No registered object with name"),
        "InteractiveSegmenterLegacy subgraph not linked"
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
      XCTAssertEqual(nsError.domain, TasksErrorCode.errorDomain)
      XCTAssertTrue(
        nsError.code == TasksErrorCode.invalidArgumentError.rawValue
          || nsError.code == TasksErrorCode.notFoundError.rawValue
      )
    }
  }

  func testExecutionMethodSignaturesCompile() {
    let _:
      (InteractiveSegmenterLegacy) -> (MPImage, RegionOfInterest) throws ->
        InteractiveSegmenterLegacyResult =
        InteractiveSegmenterLegacy.segment(image:regionOfInterest:)
    let _:
      (InteractiveSegmenterLegacy) -> (
        MPImage,
        RegionOfInterest,
        @escaping (InteractiveSegmenterLegacyResult?, (any Error)?) -> Void
      ) -> Void =
        InteractiveSegmenterLegacy.segment(image:regionOfInterest:completion:)
    let _: KeyPath<InteractiveSegmenterLegacy, [String]> = \InteractiveSegmenterLegacy.labels
  }

  func testRegionOfInterestAndResultContainer() throws {
    let keypoint = NormalizedKeypoint(
      location: CGPoint(x: 0.5, y: 0.5),
      label: "point",
      score: 1.0
    )
    let roiPoint = RegionOfInterest(normalizedKeyPoint: keypoint)
    XCTAssertNotNil(roiPoint.keypoint)
    XCTAssertNil(roiPoint.scribbles)

    let roiScribble = RegionOfInterest(scribbles: [keypoint])
    XCTAssertNil(roiScribble.keypoint)
    XCTAssertNotNil(roiScribble.scribbles)

    var uint8Pixel: UInt8 = 128
    let mask = Mask(
      uInt8Data: &uint8Pixel,
      width: 1,
      height: 1,
      shouldCopy: true
    )
    let unwrappedMask = try XCTUnwrap(mask)

    let result = InteractiveSegmenterLegacyResult(
      confidenceMasks: nil,
      categoryMask: unwrappedMask,
      qualityScores: [NSNumber(value: 0.9)],
      timestampInMilliseconds: 1000
    )
    XCTAssertNil(result.confidenceMasks)
    XCTAssertNotNil(result.categoryMask)
    let scores = try XCTUnwrap(result.qualityScores)
    XCTAssertEqual(scores.count, 1)
    let firstScore = try XCTUnwrap(scores.first)
    XCTAssertEqual(firstScore.floatValue, 0.9, accuracy: 1e-3)
    XCTAssertEqual(result.timestampInMilliseconds, 1000)
  }
}
