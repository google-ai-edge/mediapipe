/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

 http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
 ==============================================================================*/
// import GMLImageUtils
import XCTest

// @testable import TFLImageSegmenter

class TextClassifierTests: XCTestCase {

  func testExample() throws {
       XCTAssertEqual(1, 1)
    }

  // static let bundle = Bundle(for: TextClassifierTests.self)
  // static let modelPath = bundle.path(
  //   forResource: "deeplabv3",
  //   ofType: "tflite")

  // // The maximum fraction of pixels in the candidate mask that can have a
  // // different class than the golden mask for the test to pass.
  // let kGoldenMaskTolerance: Float = 1e-2

  // // Magnification factor used when creating the golden category masks to make
  // // them more human-friendly. Each pixel in the golden masks has its value
  // // multiplied by this factor, i.e. a value of 10 means class index 1, a value of
  // // 20 means class index 2, etc.
  // let kGoldenMaskMagnificationFactor: UInt8 = 10

  // let deepLabV3SegmentationWidth = 257

  // let deepLabV3SegmentationHeight = 257

  // func verifyDeeplabV3PartialSegmentationResult(_ coloredLabels: [ColoredLabel]) {

  //   self.verifyColoredLabel(
  //     coloredLabels[0],
  //     expectedR: 0,
  //     expectedG: 0,
  //     expectedB: 0,
  //     expectedLabel: "background")

  //   self.verifyColoredLabel(
  //     coloredLabels[1],
  //     expectedR: 128,
  //     expectedG: 0,
  //     expectedB: 0,
  //     expectedLabel: "aeroplane")

  //   self.verifyColoredLabel(
  //     coloredLabels[2],
  //     expectedR: 0,
  //     expectedG: 128,
  //     expectedB: 0,
  //     expectedLabel: "bicycle")

  //   self.verifyColoredLabel(
  //     coloredLabels[3],
  //     expectedR: 128,
  //     expectedG: 128,
  //     expectedB: 0,
  //     expectedLabel: "bird")

  //   self.verifyColoredLabel(
  //     coloredLabels[4],
  //     expectedR: 0,
  //     expectedG: 0,
  //     expectedB: 128,
  //     expectedLabel: "boat")

  //   self.verifyColoredLabel(
  //     coloredLabels[5],
  //     expectedR: 128,
  //     expectedG: 0,
  //     expectedB: 128,
  //     expectedLabel: "bottle")

  //   self.verifyColoredLabel(
  //     coloredLabels[6],
  //     expectedR: 0,
  //     expectedG: 128,
  //     expectedB: 128,
  //     expectedLabel: "bus")

  //   self.verifyColoredLabel(
  //     coloredLabels[7],
  //     expectedR: 128,
  //     expectedG: 128,
  //     expectedB: 128,
  //     expectedLabel: "car")

  //   self.verifyColoredLabel(
  //     coloredLabels[8],
  //     expectedR: 64,
  //     expectedG: 0,
  //     expectedB: 0,
  //     expectedLabel: "cat")

  //   self.verifyColoredLabel(
  //     coloredLabels[9],
  //     expectedR: 192,
  //     expectedG: 0,
  //     expectedB: 0,
  //     expectedLabel: "chair")

  //   self.verifyColoredLabel(
  //     coloredLabels[10],
  //     expectedR: 64,
  //     expectedG: 128,
  //     expectedB: 0,
  //     expectedLabel: "cow")

  //   self.verifyColoredLabel(
  //     coloredLabels[11],
  //     expectedR: 192,
  //     expectedG: 128,
  //     expectedB: 0,
  //     expectedLabel: "dining table")

  //   self.verifyColoredLabel(
  //     coloredLabels[12],
  //     expectedR: 64,
  //     expectedG: 0,
  //     expectedB: 128,
  //     expectedLabel: "dog")

  //   self.verifyColoredLabel(
  //     coloredLabels[13],
  //     expectedR: 192,
  //     expectedG: 0,
  //     expectedB: 128,
  //     expectedLabel: "horse")

  //   self.verifyColoredLabel(
  //     coloredLabels[14],
  //     expectedR: 64,
  //     expectedG: 128,
  //     expectedB: 128,
  //     expectedLabel: "motorbike")

  //   self.verifyColoredLabel(
  //     coloredLabels[15],
  //     expectedR: 192,
  //     expectedG: 128,
  //     expectedB: 128,
  //     expectedLabel: "person")

  //   self.verifyColoredLabel(
  //     coloredLabels[16],
  //     expectedR: 0,
  //     expectedG: 64,
  //     expectedB: 0,
  //     expectedLabel: "potted plant")

  //   self.verifyColoredLabel(
  //     coloredLabels[17],
  //     expectedR: 128,
  //     expectedG: 64,
  //     expectedB: 0,
  //     expectedLabel: "sheep")

  //   self.verifyColoredLabel(
  //     coloredLabels[18],
  //     expectedR: 0,
  //     expectedG: 192,
  //     expectedB: 0,
  //     expectedLabel: "sofa")

  //   self.verifyColoredLabel(
  //     coloredLabels[19],
  //     expectedR: 128,
  //     expectedG: 192,
  //     expectedB: 0,
  //     expectedLabel: "train")

  //   self.verifyColoredLabel(
  //     coloredLabels[20],
  //     expectedR: 0,
  //     expectedG: 64,
  //     expectedB: 128,
  //     expectedLabel: "tv")
  // }

  // func verifyColoredLabel(
  //   _ coloredLabel: ColoredLabel,
  //   expectedR: UInt,
  //   expectedG: UInt,
  //   expectedB: UInt,
  //   expectedLabel: String
  // ) {
  //   XCTAssertEqual(
  //     coloredLabel.r,
  //     expectedR)
  //   XCTAssertEqual(
  //     coloredLabel.g,
  //     expectedG)
  //   XCTAssertEqual(
  //     coloredLabel.b,
  //     expectedB)
  //   XCTAssertEqual(
  //     coloredLabel.label,
  //     expectedLabel)
  // }

  // func testSuccessfullInferenceOnMLImageWithUIImage() throws {

  //   let modelPath = try XCTUnwrap(ImageSegmenterTests.modelPath)

  //   let imageSegmenterOptions = ImageSegmenterOptions(modelPath: modelPath)

  //   let imageSegmenter =
  //     try ImageSegmenter.segmenter(options: imageSegmenterOptions)

  //   let gmlImage = try XCTUnwrap(
  //     MLImage.imageFromBundle(
  //       class: type(of: self),
  //       filename: "segmentation_input_rotation0",
  //       type: "jpg"))
  //   let segmentationResult: SegmentationResult =
  //     try XCTUnwrap(imageSegmenter.segment(mlImage: gmlImage))

  //   XCTAssertEqual(segmentationResult.segmentations.count, 1)

  //   let coloredLabels = try XCTUnwrap(segmentationResult.segmentations[0].coloredLabels)
  //   verifyDeeplabV3PartialSegmentationResult(coloredLabels)

  //   let categoryMask = try XCTUnwrap(segmentationResult.segmentations[0].categoryMask)
  //   XCTAssertEqual(deepLabV3SegmentationWidth, categoryMask.width)
  //   XCTAssertEqual(deepLabV3SegmentationHeight, categoryMask.height)

  //   let goldenMaskImage = try XCTUnwrap(
  //     MLImage.imageFromBundle(
  //       class: type(of: self),
  //       filename: "segmentation_golden_rotation0",
  //       type: "png"))

  //   let pixelBuffer = goldenMaskImage.grayScalePixelBuffer().takeRetainedValue()

  //   CVPixelBufferLockBaseAddress(pixelBuffer, CVPixelBufferLockFlags.readOnly)

  //   let pixelBufferBaseAddress = (try XCTUnwrap(CVPixelBufferGetBaseAddress(pixelBuffer)))
  //     .assumingMemoryBound(to: UInt8.self)

  //   let numPixels = deepLabV3SegmentationWidth * deepLabV3SegmentationHeight

  //   let mask = try XCTUnwrap(categoryMask.mask)

  //   var inconsistentPixels: Float = 0.0

  //   for i in 0..<numPixels {
  //     if mask[i] * kGoldenMaskMagnificationFactor != pixelBufferBaseAddress[i] {
  //       inconsistentPixels += 1
  //     }
  //   }

  //   CVPixelBufferUnlockBaseAddress(pixelBuffer, CVPixelBufferLockFlags.readOnly)

  //   XCTAssertLessThan(inconsistentPixels / Float(numPixels), kGoldenMaskTolerance)
  // }

}
