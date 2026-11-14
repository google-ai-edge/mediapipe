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
import UIKit
import XCTest
import third_party_mediapipe_tasks_c_core_mp_status
import third_party_mediapipe_tasks_c_vision_core_image
import third_party_mediapipe_tasks_c_vision_image_segmenter_image_segmenter_result

final class ImageSegmenterSwiftTests: XCTestCase {
  private enum TestFixture {
    static let deeplabModel = "deeplabv3"
    static let selfieModel = "selfie_segm_128_128_3"
    static let deeplabLabelCount = 21
  }

  private func path(
    _ name: String, _ ext: String, file: StaticString = #filePath, line: UInt = #line
  ) throws -> String {
    try XCTUnwrap(
      Bundle(for: Self.self).path(forResource: name, ofType: ext), file: file, line: line)
  }

  private func image(
    _ name: String, file: StaticString = #filePath, line: UInt = #line
  ) throws -> MPImage {
    let uiImage = UIImage(contentsOfFile: try path(name, "jpg", file: file, line: line))
    return try MPImage(uiImage: XCTUnwrap(uiImage, file: file, line: line))
  }

  private func options(
    _ model: String, mode: VisionRunningMode = .image
  ) throws -> ImageSegmenterOptions {
    ImageSegmenterOptions(
      baseOptions: BaseOptions(modelAssetPath: try path(model, "tflite")), runningMode: mode)
  }

  func testResultDeepCopiesCMasksAndScores() throws {
    let floats: [Float] = [0.0, 0.25, 0.5, 1.0]
    let bytes: [UInt8] = [0, 1, 2, 3]
    var cFloatMask: MpImagePtr?
    var cByteMask: MpImagePtr?
    try MpStatus.check {
      MpImageCreateFromFloatData(kMpImageFormatVec32F1, 2, 2, floats, 4, &cFloatMask, &$0)
    }
    try MpStatus.check {
      MpImageCreateFromUint8Data(kMpImageFormatGray8, 2, 2, bytes, 4, &cByteMask, &$0)
    }
    defer {
      for mask in [cFloatMask, cByteMask].compactMap({ $0 }) {
        MpImageFree(mask)
      }
    }
    var cMasks: [MpImagePtr?] = [cFloatMask]
    var scores: [Float] = [0.9]
    let result = try cMasks.withUnsafeMutableBufferPointer { masks in
      try scores.withUnsafeMutableBufferPointer { scoreBuffer in
        var cResult = MpImageSegmenterResult()
        cResult.confidence_masks = masks.baseAddress
        cResult.confidence_masks_count = 1
        cResult.category_mask = cByteMask
        cResult.has_category_mask = 1
        cResult.quality_scores = scoreBuffer.baseAddress
        cResult.quality_scores_count = 1
        return try ImageSegmenterResult(cResult: cResult, timestampInMilliseconds: 7)
      }
    }
    XCTAssertEqual(result.confidenceMasks?.first?.float32Elements, floats)
    XCTAssertEqual(result.categoryMask?.uint8Elements, bytes)
    XCTAssertEqual(result.qualityScores, [0.9])
    XCTAssertEqual(result.timestampInMilliseconds, 7)
  }

  func testEmptyCResultProducesNilMasksAndEmptyScores() throws {
    let empty = try ImageSegmenterResult(
      cResult: MpImageSegmenterResult(), timestampInMilliseconds: 0)
    XCTAssertNil(empty.confidenceMasks)
    XCTAssertNil(empty.categoryMask)
    XCTAssertEqual(empty.qualityScores, [])
  }

  func testInitFailsWithMissingModel() {
    XCTAssertThrowsError(try ImageSegmenter(options: ImageSegmenterOptions())) { error in
      XCTAssertEqual((error as? TasksError)?.code, .invalidArgument)
    }
  }

  func testInitFailsWithNonexistentModelPath() {
    XCTAssertThrowsError(try ImageSegmenter(modelPath: "/nonexistent/model.tflite"))
  }

  func testInitFailsInLiveStreamModeWithoutResultHandler() throws {
    let liveStreamWithoutHandler = try options(TestFixture.selfieModel, mode: .liveStream)
    XCTAssertThrowsError(try ImageSegmenter(options: liveStreamWithoutHandler)) { error in
      XCTAssertEqual((error as? TasksError)?.code, .invalidArgument)
    }
  }

  func testInitFailsInImageModeWithResultHandler() throws {
    var imageWithHandler = try options(TestFixture.selfieModel)
    imageWithHandler.liveStreamResultHandler = { _, _ in }
    XCTAssertThrowsError(try ImageSegmenter(options: imageWithHandler)) { error in
      XCTAssertEqual((error as? TasksError)?.code, .invalidArgument)
    }
  }

  func testSegmentImageOutputsConfidenceMasksAndLabels() throws {
    let catImage = try image("cat")
    let segmenter = try ImageSegmenter(options: options(TestFixture.deeplabModel))
    XCTAssertEqual(segmenter.labels.count, TestFixture.deeplabLabelCount)
    XCTAssertEqual(segmenter.labels.first, "background")
    XCTAssertEqual(segmenter.labels[8], "cat")

    let confidence = try segmenter.segment(image: catImage)
    XCTAssertNil(confidence.categoryMask)
    let masks = try XCTUnwrap(confidence.confidenceMasks)
    XCTAssertEqual(masks.count, TestFixture.deeplabLabelCount)
    XCTAssertEqual(masks[8].dataType, .float32)
    XCTAssertEqual(masks[8].width, Int(catImage.width))
    XCTAssertEqual(masks[8].height, Int(catImage.height))
    XCTAssertTrue(masks[8].float32Elements.allSatisfy { (0.0...1.0).contains($0) })
  }

  func testSegmentImageOutputsCategoryMask() throws {
    let catImage = try image("cat")
    let categoryOptions = ImageSegmenterOptions(
      baseOptions: BaseOptions(modelAssetPath: try path(TestFixture.deeplabModel, "tflite")),
      outputConfidenceMasks: false,
      outputCategoryMask: true
    )
    let category = try ImageSegmenter(options: categoryOptions).segment(image: catImage)
    XCTAssertNil(category.confidenceMasks)
    let categoryMask = try XCTUnwrap(category.categoryMask)
    XCTAssertEqual(categoryMask.dataType, .uInt8)
    XCTAssertTrue(categoryMask.uint8Elements.contains(8))
    XCTAssertTrue(categoryMask.uint8Elements.allSatisfy { $0 < TestFixture.deeplabLabelCount })
  }

  func testSegmentRejectsUnsupportedAndOutOfBoundsRegionOfInterest() throws {
    let mozart = try image("mozart_square")
    let segmenter = try ImageSegmenter(options: options(TestFixture.selfieModel))

    XCTAssertThrowsError(
      try segmenter.segment(
        image: mozart,
        imageProcessingOptions: ImageProcessingOptions(
          regionOfInterest: CGRect(x: 0.1, y: 0.1, width: 0.5, height: 0.5)))
    ) { error in
      XCTAssertEqual((error as? TasksError)?.code, .invalidArgument)
    }

    XCTAssertThrowsError(
      try segmenter.segment(
        image: mozart,
        imageProcessingOptions: ImageProcessingOptions(
          regionOfInterest: CGRect(x: -0.1, y: 0.0, width: 0.5, height: 0.5)))
    ) { error in
      XCTAssertEqual((error as? TasksError)?.code, .invalidArgument)
    }
  }

  func testSegmentRejectsMismatchedRunningModes() throws {
    let mozart = try image("mozart_square")

    let imageSegmenter = try ImageSegmenter(options: options(TestFixture.selfieModel, mode: .image))
    XCTAssertThrowsError(
      try imageSegmenter.segment(videoFrame: mozart, timestampInMilliseconds: 0)
    ) {
      XCTAssertEqual(($0 as? TasksError)?.code, .invalidArgument)
    }
    XCTAssertThrowsError(
      try imageSegmenter.segmentAsync(image: mozart, timestampInMilliseconds: 0)
    ) {
      XCTAssertEqual(($0 as? TasksError)?.code, .invalidArgument)
    }

    let videoSegmenter = try ImageSegmenter(options: options(TestFixture.selfieModel, mode: .video))
    XCTAssertThrowsError(try videoSegmenter.segment(image: mozart)) {
      XCTAssertEqual(($0 as? TasksError)?.code, .invalidArgument)
    }
    XCTAssertThrowsError(
      try videoSegmenter.segmentAsync(image: mozart, timestampInMilliseconds: 0)
    ) {
      XCTAssertEqual(($0 as? TasksError)?.code, .invalidArgument)
    }

    var liveOptions = try options(TestFixture.selfieModel, mode: .liveStream)
    liveOptions.liveStreamResultHandler = { _, _ in }
    let liveSegmenter = try ImageSegmenter(options: liveOptions)
    XCTAssertThrowsError(try liveSegmenter.segment(image: mozart)) {
      XCTAssertEqual(($0 as? TasksError)?.code, .invalidArgument)
    }
    XCTAssertThrowsError(
      try liveSegmenter.segment(videoFrame: mozart, timestampInMilliseconds: 0)
    ) {
      XCTAssertEqual(($0 as? TasksError)?.code, .invalidArgument)
    }
  }

  func testSegmentVideoFramesAtMonotonicTimestamps() throws {
    let mozart = try image("mozart_square")
    let segmenter = try ImageSegmenter(options: options(TestFixture.selfieModel, mode: .video))

    let first = try segmenter.segment(videoFrame: mozart, timestampInMilliseconds: 0)
    XCTAssertEqual(first.timestampInMilliseconds, 0)
    XCTAssertEqual(first.confidenceMasks?.count, 2)

    let second = try segmenter.segment(videoFrame: mozart, timestampInMilliseconds: 33)
    XCTAssertEqual(second.timestampInMilliseconds, 33)
    XCTAssertEqual(second.confidenceMasks?.count, 2)
  }

  func testSegmentVideoRejectsNonMonotonicAndOutOfRangeTimestamps() throws {
    let mozart = try image("mozart_square")
    let segmenter = try ImageSegmenter(options: options(TestFixture.selfieModel, mode: .video))

    _ = try segmenter.segment(videoFrame: mozart, timestampInMilliseconds: 33)
    XCTAssertThrowsError(try segmenter.segment(videoFrame: mozart, timestampInMilliseconds: 10))
    XCTAssertThrowsError(try segmenter.segment(videoFrame: mozart, timestampInMilliseconds: -1)) {
      XCTAssertEqual(($0 as? TasksError)?.code, .invalidArgument)
    }
    XCTAssertThrowsError(try segmenter.segment(videoFrame: mozart, timestampInMilliseconds: .max)) {
      XCTAssertEqual(($0 as? TasksError)?.code, .invalidArgument)
    }
  }

  func testLiveStreamDeliversResultsToHandler() throws {
    final class CallbackBox: @unchecked Sendable {
      let lock = NSLock()
      var captured: (Result<ImageSegmenterResult, any Error>, Int)?
    }
    let box = CallbackBox()
    let mozart = try image("mozart_square")
    let delivered = expectation(description: "live stream result")
    var liveOptions = try options(TestFixture.selfieModel, mode: .liveStream)
    liveOptions.liveStreamResultHandler = { result, timestamp in
      box.lock.withLock {
        if box.captured == nil {
          box.captured = (result, timestamp)
          delivered.fulfill()
        }
      }
    }
    var segmenter: ImageSegmenter? = try ImageSegmenter(options: liveOptions)
    try segmenter?.segmentAsync(image: mozart, timestampInMilliseconds: 42)
    wait(for: [delivered], timeout: 10)
    let captured = try XCTUnwrap(box.lock.withLock { box.captured })
    XCTAssertEqual(captured.1, 42)
    XCTAssertEqual(try captured.0.get().confidenceMasks?.count, 2)
    segmenter = nil
  }
}
