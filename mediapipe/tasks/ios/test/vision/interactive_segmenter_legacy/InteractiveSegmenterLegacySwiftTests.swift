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
import UIKit
import XCTest
import third_party_mediapipe_tasks_c_vision_interactive_segmenter_legacy_interactive_segmenter_legacy_lib

final class InteractiveSegmenterLegacySwiftTests: XCTestCase {
  private enum TestFixture {
    static let modelName = "ptm_512_hdt_ptm_woid"
    static let imageName = "cats_and_dogs"
    static let dogPoint = CGPoint(x: 0.44, y: 0.7)
    static let backgroundPoint = CGPoint(x: 0.02, y: 0.02)
    static let dogScribbles = [0.7, 0.71, 0.72].map {
      NormalizedKeypoint(location: CGPoint(x: 0.44, y: $0))
    }
  }

  private func path(
    _ name: String, _ ext: String, file: StaticString = #filePath, line: UInt = #line
  ) throws -> String {
    try XCTUnwrap(
      Bundle(for: Self.self).path(forResource: name, ofType: ext), file: file, line: line)
  }

  private func makeSegmenter(
    categoryMask: Bool = false,
    displayNamesLocale: String = "en",
    file: StaticString = #filePath,
    line: UInt = #line
  ) throws -> InteractiveSegmenterLegacy {
    try InteractiveSegmenterLegacy(
      options: InteractiveSegmenterLegacyOptions(
        baseOptions: BaseOptions(
          modelAssetPath: path(TestFixture.modelName, "tflite", file: file, line: line)),
        outputConfidenceMasks: !categoryMask,
        outputCategoryMask: categoryMask,
        displayNamesLocale: displayNamesLocale))
  }

  private func makeImage(
    orientation: UIImage.Orientation = .up, file: StaticString = #filePath, line: UInt = #line
  ) throws -> MPImage {
    let imagePath = try path(TestFixture.imageName, "jpg", file: file, line: line)
    let uiImage = try XCTUnwrap(UIImage(contentsOfFile: imagePath), file: file, line: line)
    return try MPImage(uiImage: uiImage, orientation: orientation)
  }

  private func pixelIndex(of point: CGPoint, in mask: MPMask) -> Int {
    Int(point.y * CGFloat(mask.height)) * mask.width + Int(point.x * CGFloat(mask.width))
  }

  func testOptionsDisplayNamesLocaleAndLabels() throws {
    let defaultOptions = InteractiveSegmenterLegacyOptions()
    XCTAssertTrue(defaultOptions.outputConfidenceMasks)
    XCTAssertFalse(defaultOptions.outputCategoryMask)
    XCTAssertEqual(defaultOptions.displayNamesLocale, "en")
    defaultOptions.withCOptions { cOptions in
      XCTAssertEqual(cOptions.display_names_locale.map { String(cString: $0) }, "en")
      XCTAssertTrue(cOptions.output_confidence_masks)
      XCTAssertFalse(cOptions.output_category_mask)
    }

    let customOptions = InteractiveSegmenterLegacyOptions(displayNamesLocale: "fr")
    XCTAssertEqual(customOptions.displayNamesLocale, "fr")
    XCTAssertNotEqual(defaultOptions, customOptions)
    customOptions.withCOptions { cOptions in
      XCTAssertEqual(cOptions.display_names_locale.map { String(cString: $0) }, "fr")
    }

    let segmenter = try makeSegmenter(displayNamesLocale: "fr")
    XCTAssertEqual(segmenter.labels, [])
  }

  func testRegionOfInterestBridgesKeypointAndScribblesToC() throws {
    let keypoint = RegionOfInterest(keypoint: NormalizedKeypoint(location: TestFixture.dogPoint))
    try keypoint.withCRegionOfInterest { cRegion in
      XCTAssertEqual(cRegion.pointee.format, MP_REGION_OF_INTEREST_FORMAT_KEYPOINT)
      XCTAssertEqual(cRegion.pointee.keypoint.x, 0.44, accuracy: 1e-6)
      XCTAssertEqual(cRegion.pointee.keypoint.y, 0.7, accuracy: 1e-6)
      XCTAssertNil(cRegion.pointee.scribble)
    }
    try RegionOfInterest(scribbles: TestFixture.dogScribbles).withCRegionOfInterest { cRegion in
      XCTAssertEqual(cRegion.pointee.format, MP_REGION_OF_INTEREST_FORMAT_SCRIBBLE)
      XCTAssertEqual(cRegion.pointee.scribble_count, 3)
      let scribble = try XCTUnwrap(cRegion.pointee.scribble)
      XCTAssertEqual(scribble[2].y, 0.72, accuracy: 1e-6)
    }
    XCTAssertThrowsError(try RegionOfInterest(scribbles: []).withCRegionOfInterest { _ in }) {
      XCTAssertEqual(($0 as? TasksError)?.code, .invalidArgument)
    }
    XCTAssertEqual(
      Set([keypoint, keypoint, RegionOfInterest(scribbles: TestFixture.dogScribbles)]).count, 2)
  }

  func testInitValidationErrors() {
    XCTAssertThrowsError(try InteractiveSegmenterLegacy(options: .init())) {
      XCTAssertEqual(($0 as? TasksError)?.code, .invalidArgument)
    }
    XCTAssertThrowsError(try InteractiveSegmenterLegacy(modelPath: "/nonexistent/model.tflite"))
  }

  func testSegmentWithKeypointProducesCategoryAndConfidenceMasks() throws {
    let image = try makeImage()
    let region = RegionOfInterest(keypoint: NormalizedKeypoint(location: TestFixture.dogPoint))

    let category = try makeSegmenter(categoryMask: true).segment(
      image: image, regionOfInterest: region)
    XCTAssertNil(category.confidenceMasks)
    XCTAssertEqual(category.timestampInMilliseconds, 0)
    let categoryMask = try XCTUnwrap(category.categoryMask)
    XCTAssertEqual(categoryMask.dataType, .uInt8)
    XCTAssertEqual(categoryMask.width, Int(image.width))
    XCTAssertEqual(categoryMask.height, Int(image.height))
    let categories = categoryMask.uint8Elements
    XCTAssertEqual(categories[pixelIndex(of: TestFixture.dogPoint, in: categoryMask)], 1)
    XCTAssertEqual(categories[pixelIndex(of: TestFixture.backgroundPoint, in: categoryMask)], 0)

    let confidence = try makeSegmenter().segment(image: image, regionOfInterest: region)
    XCTAssertNil(confidence.categoryMask)
    let masks = try XCTUnwrap(confidence.confidenceMasks)
    XCTAssertEqual(masks.count, 2)
    let foreground = masks[1].float32Elements
    XCTAssertGreaterThan(foreground[pixelIndex(of: TestFixture.dogPoint, in: masks[1])], 0.5)
    XCTAssertLessThan(foreground[pixelIndex(of: TestFixture.backgroundPoint, in: masks[1])], 0.5)
  }

  func testSegmentWithScribblesAndRotatedImage() throws {
    let segmenter = try makeSegmenter(categoryMask: true)
    let result = try segmenter.segment(
      image: makeImage(), regionOfInterest: RegionOfInterest(scribbles: TestFixture.dogScribbles))
    let categoryMask = try XCTUnwrap(result.categoryMask)
    XCTAssertEqual(
      categoryMask.uint8Elements[pixelIndex(of: TestFixture.dogPoint, in: categoryMask)], 1)

    let rotated = try makeSegmenter().segment(
      image: makeImage(orientation: .left),
      regionOfInterest: RegionOfInterest(
        keypoint: NormalizedKeypoint(location: TestFixture.dogPoint))
    )
    XCTAssertEqual(rotated.confidenceMasks?.count, 2)
    XCTAssertNil(rotated.categoryMask)
  }
}
