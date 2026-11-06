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
import MediaPipeTasksCore
import MediaPipeTasksVisionCore
import Testing
import third_party_mediapipe_tasks_c_components_containers_rect
import third_party_mediapipe_tasks_c_core_mp_status
import third_party_mediapipe_tasks_c_vision_core_image
import third_party_mediapipe_tasks_c_vision_core_image_processing_options

struct VisionCoreTests {
  @Test
  func `VisionRunningMode bridges to MpRunningMode and displayName`() {
    #expect(VisionRunningMode.image.cRunningMode == MP_RUNNING_MODE_IMAGE)
    #expect(VisionRunningMode.video.cRunningMode == MP_RUNNING_MODE_VIDEO)
    #expect(VisionRunningMode.liveStream.cRunningMode == MP_RUNNING_MODE_LIVE_STREAM)
    #expect(VisionRunningMode.image.displayName == "Image")
    #expect(VisionRunningMode.video.displayName == "Video")
    #expect(VisionRunningMode.liveStream.displayName == "Live Stream")
    #expect(VisionRunningMode.allCases.count == 3)
  }

  @Test
  func `ImageProcessingOptions validates regionOfInterest and rotationDegrees`() throws {
    let roi = CGRect(x: 0.1, y: 0.2, width: 0.5, height: 0.6)
    let options = ImageProcessingOptions(regionOfInterest: roi, rotationDegrees: 90)
    let standardOptions = ImageProcessingOptions(
      regionOfInterest: CGRect(x: 0.25, y: 0.25, width: 0.5, height: 0.5),
      rotationDegrees: 90
    )
    let flippedOptions = ImageProcessingOptions(
      regionOfInterest: CGRect(x: 0.75, y: 0.75, width: -0.5, height: -0.5),
      rotationDegrees: 90
    )
    #expect(
      standardOptions == flippedOptions && standardOptions.hashValue == flippedOptions.hashValue)

    try options.withCOptions { cOptions in
      #expect(cOptions.has_region_of_interest == 1)
      #expect(abs(cOptions.region_of_interest.left - 0.1) < 1e-5)
      #expect(abs(cOptions.region_of_interest.top - 0.2) < 1e-5)
      #expect(abs(cOptions.region_of_interest.right - 0.6) < 1e-5)
      #expect(abs(cOptions.region_of_interest.bottom - 0.8) < 1e-5)
      #expect(cOptions.rotation_degrees == 90)
    }

    #expect(throws: TasksError.self) { try ImageProcessingOptions(rotationDegrees: 45).validate() }
    #expect(throws: TasksError.self) {
      try ImageProcessingOptions(regionOfInterest: CGRect(x: 0.5, y: 0.2, width: 0.6, height: 0.3))
        .withCOptions { _ in }
    }
  }

  @Test
  func `MPMask converts UInt8 lazily to Float32 and copies independently`() throws {
    let rawValues: [UInt8] = [0, 51, 128, 255]
    try rawValues.withUnsafeBufferPointer { buffer in
      let base = try #require(buffer.baseAddress)
      let mask = MPMask(uint8Data: base, width: 2, height: 2)
      #expect(
        mask.width == 2 && mask.height == 2 && mask.pixelCount == 4 && mask.dataType == .uInt8)
      #expect(mask.uint8Elements == rawValues)
      #expect(mask.withUnsafeUInt8BufferPointer { Array($0) } == rawValues)

      let floats = mask.float32Elements
      #expect(floats.count == 4 && abs(floats[0] - 0.0) < 1e-5 && abs(floats[3] - 1.0) < 1e-5)
      #expect(abs(floats[1] - (51.0 / 255.0)) < 1e-5 && abs(floats[2] - (128.0 / 255.0)) < 1e-5)

      var grayCImage: MpImagePtr?
      try MpStatus.check(defaultMessage: "Failed to create Gray8 MpImage.") { errorMessage in
        MpImageCreateFromUint8Data(kMpImageFormatGray8, 2, 2, base, 4, &grayCImage, &errorMessage)
      }
      let validGrayImage = try #require(grayCImage)
      defer { MpImageFree(validGrayImage) }
      #expect(try MPMask(copyingCImage: validGrayImage) == mask)

      var srgbaImage: MpImagePtr?
      try MpStatus.check(defaultMessage: "Failed to create SRGBA MpImage.") { errorMessage in
        MpImageCreateFromUint8Data(kMpImageFormatSrgba, 1, 1, base, 4, &srgbaImage, &errorMessage)
      }
      let validSrgba = try #require(srgbaImage)
      defer { MpImageFree(validSrgba) }
      #expect(throws: TasksError.self) { _ = try MPMask(copyingCImage: validSrgba) }

      let negativeMask = MPMask(uint8Data: base, width: -2, height: -2)
      #expect(
        negativeMask.pixelCount == 0 && negativeMask.uint8Elements.isEmpty && negativeMask != mask)
    }
  }

  @Test
  func `MPMask converts Float32 lazily to UInt8 with clamping and NaN handling`() throws {
    let rawFloats: [Float] = [-0.5, .nan, 0.5, 1.5]
    let mask = try rawFloats.withUnsafeBufferPointer { buffer in
      MPMask(float32Data: try #require(buffer.baseAddress), width: 2, height: 2)
    }
    #expect(mask.dataType == .float32 && mask.uint8Elements == [0, 0, 127, 255])

    let sourceFloats: [Float] = [0.25, 0.75, 0.5, 1.0]
    try sourceFloats.withUnsafeBufferPointer { buffer in
      let base = try #require(buffer.baseAddress)
      let floatMask = MPMask(float32Data: base, width: 2, height: 2)
      #expect(floatMask.withUnsafeFloat32BufferPointer { Array($0) } == sourceFloats)
      #expect(floatMask.float32Elements == sourceFloats)

      var cImageOut: MpImagePtr?
      try MpStatus.check(defaultMessage: "Failed to create C MpImage.") { errorMessage in
        MpImageCreateFromFloatData(kMpImageFormatVec32F1, 2, 2, base, 4, &cImageOut, &errorMessage)
      }
      let cImage = try #require(cImageOut)
      defer { MpImageFree(cImage) }
      let copiedMask = try MPMask(copyingCImage: cImage)
      #expect(copiedMask.width == 2 && copiedMask.height == 2 && copiedMask.dataType == .float32)
      #expect(copiedMask.float32Elements == sourceFloats)
    }
  }
}
