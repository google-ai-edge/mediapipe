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
import CoreMedia
import CoreVideo
import MediaPipeTasksCore
import MediaPipeTasksVisionCore
import Testing
import UIKit
import third_party_mediapipe_tasks_c_core_mp_status
import third_party_mediapipe_tasks_c_vision_core_image

struct MPImageTests {
  @Test
  func `MPImage bridges RGBA pixel data to MpImagePtr and round-trips`() throws {
    let rgbaBytes: [UInt8] = [255, 0, 0, 255, 0, 255, 0, 255, 0, 0, 255, 255, 255, 255, 255, 255]
    let image = try MPImage(rgbaData: rgbaBytes, width: 2, height: 2)
    #expect(image.width == 2 && image.height == 2 && image.imageSourceType == .image)
    #expect(image.rgbaData == rgbaBytes && image.floatData == nil)

    try image.withCImage { cImage in
      #expect(MpImageGetWidth(cImage) == 2 && MpImageGetHeight(cImage) == 2)
      #expect(MpImageGetChannels(cImage) == 4 && MpImageGetFormat(cImage) == kMpImageFormatSrgba)
      #expect(!MpImageUsesGpu(cImage))

      var dataPtr: UnsafePointer<UInt8>?
      try MpStatus.check(defaultMessage: "Failed to read RGBA data.") { errorMessage in
        MpImageDataUint8(cImage, &dataPtr, &errorMessage)
      }
      let extracted = Array(UnsafeBufferPointer(start: try #require(dataPtr), count: 16))
      #expect(extracted == rgbaBytes)
    }
  }

  @Test
  func `MPImage bridges Float32 pixel data to MpImagePtr and round-trips`() throws {
    let floats: [Float] = [0.1, 0.25, 0.75, 0.9]
    let image = try MPImage(floatData: floats, width: 2, height: 2)
    #expect(image.width == 2 && image.height == 2 && image.floatData == floats)
    #expect(image.rgbaData == nil)

    try image.withCImage { cImage in
      #expect(MpImageGetWidth(cImage) == 2 && MpImageGetHeight(cImage) == 2)
      #expect(MpImageGetChannels(cImage) == 1 && MpImageGetFormat(cImage) == kMpImageFormatVec32F1)

      var dataPtr: UnsafePointer<Float>?
      try MpStatus.check(defaultMessage: "Failed to read Float32 data.") { errorMessage in
        MpImageDataFloat32(cImage, &dataPtr, &errorMessage)
      }
      let extracted = Array(UnsafeBufferPointer(start: try #require(dataPtr), count: 4))
      #expect(extracted == floats)
    }
  }

  @Test
  func `MPImage rejects mismatched buffer counts and integer overflow dimensions`() {
    #expect(throws: TasksError.self) { _ = try MPImage(rgbaData: [0], width: 2, height: 2) }
    #expect(throws: TasksError.self) { _ = try MPImage(rgbaData: [], width: Int.max, height: 2) }
    #expect(throws: TasksError.self) { _ = try MPImage(floatData: [1.0], width: 2, height: 2) }
    #expect(throws: TasksError.self) { _ = try MPImage(floatData: [], width: Int.max, height: 2) }
  }

  @Test
  func `MPImage converts UIImage and BGRA CVPixelBuffer to MpImagePtr`() throws {
    let rendererFormat = UIGraphicsImageRendererFormat()
    rendererFormat.scale = 1.0
    let renderer = UIGraphicsImageRenderer(
      size: CGSize(width: 4, height: 3), format: rendererFormat)
    let uiImage = renderer.image { context in
      UIColor.red.setFill()
      context.fill(CGRect(x: 0, y: 0, width: 4, height: 3))
    }

    let mpImage = try MPImage(uiImage: uiImage, orientation: .right)
    #expect(mpImage.width == 4 && mpImage.height == 3 && mpImage.orientation == .right)
    #expect(mpImage.imageSourceType == .image && mpImage.image != nil)

    try mpImage.withCImage { cImage in
      #expect(MpImageGetWidth(cImage) == 4 && MpImageGetHeight(cImage) == 3)
      #expect(MpImageGetFormat(cImage) == kMpImageFormatSrgba)
    }

    var pixelBufferOut: CVPixelBuffer?
    let status = CVPixelBufferCreate(
      kCFAllocatorDefault, 4, 2, kCVPixelFormatType_32BGRA, nil, &pixelBufferOut)
    #expect(status == kCVReturnSuccess)
    let pbImage = try MPImage(pixelBuffer: try #require(pixelBufferOut))
    #expect(pbImage.width == 4 && pbImage.height == 2 && pbImage.imageSourceType == .pixelBuffer)
    #expect(pbImage.pixelBuffer != nil && pbImage.image == nil && pbImage.sampleBuffer == nil)

    try pbImage.withCImage { cImage in
      #expect(MpImageGetWidth(cImage) == 4 && MpImageGetHeight(cImage) == 2)
      #expect(MpImageGetFormat(cImage) == kMpImageFormatSrgba)
    }

    var floatBuffer: CVPixelBuffer?
    #expect(
      CVPixelBufferCreate(
        kCFAllocatorDefault, 2, 2, kCVPixelFormatType_OneComponent32Float, nil, &floatBuffer)
        == kCVReturnSuccess)
    try MPImage(pixelBuffer: try #require(floatBuffer)).withCImage { cImage in
      #expect(MpImageGetWidth(cImage) == 2 && MpImageGetHeight(cImage) == 2)
      #expect(MpImageGetFormat(cImage) == kMpImageFormatVec32F1)
    }

    let validPB = try #require(pixelBufferOut)

    var invalidBuffer: CVPixelBuffer?
    #expect(
      CVPixelBufferCreate(
        kCFAllocatorDefault, 2, 2, kCVPixelFormatType_16Gray, nil, &invalidBuffer) == 0)
    #expect(throws: TasksError.self) { _ = try MPImage(pixelBuffer: try #require(invalidBuffer)) }
    #expect(throws: TasksError.self) { _ = try MPImage(uiImage: UIImage()) }
    var formatDesc: CMVideoFormatDescription?
    #expect(
      CMVideoFormatDescriptionCreateForImageBuffer(
        allocator: kCFAllocatorDefault, imageBuffer: validPB, formatDescriptionOut: &formatDesc)
        == noErr)
    var timing = CMSampleTimingInfo(
      duration: .invalid, presentationTimeStamp: .zero, decodeTimeStamp: .invalid)
    var sampleBufferOut: CMSampleBuffer?
    #expect(
      CMSampleBufferCreateReadyWithImageBuffer(
        allocator: kCFAllocatorDefault, imageBuffer: validPB,
        formatDescription: try #require(formatDesc), sampleTiming: &timing,
        sampleBufferOut: &sampleBufferOut) == noErr)
    let sbImage = try MPImage(sampleBuffer: try #require(sampleBufferOut))
    #expect(sbImage.imageSourceType == .sampleBuffer && sbImage.sampleBuffer != nil)
    try sbImage.withCImage { cImage in
      #expect(MpImageGetWidth(cImage) == 4 && MpImageGetHeight(cImage) == 2)
    }
  }
}
