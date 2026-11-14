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
import CoreVideo
import Foundation
import MediaPipeTasksCore
import MediaPipeTasksVisionCore
import Testing
import UIKit
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

    #expect(standardOptions == flippedOptions)
    #expect(standardOptions.hashValue == flippedOptions.hashValue)

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

      #expect(mask.width == 2)
      #expect(mask.height == 2)
      #expect(mask.pixelCount == 4)
      #expect(mask.dataType == .uInt8)
      #expect(mask.uint8Elements == rawValues)
      #expect(mask.withUnsafeUInt8BufferPointer { Array($0) } == rawValues)

      let floats = mask.float32Elements

      #expect(floats.count == 4)
      #expect(abs(floats[0] - 0.0) < 1e-5)
      #expect(abs(floats[1] - (51.0 / 255.0)) < 1e-5)
      #expect(abs(floats[2] - (128.0 / 255.0)) < 1e-5)
      #expect(abs(floats[3] - 1.0) < 1e-5)

      var grayCImage: MpImagePtr?
      try MpStatus.check(defaultMessage: "Failed to create Gray8 MpImage.") { errorMessage in
        MpImageCreateFromUint8Data(kMpImageFormatGray8, 2, 2, base, 4, &grayCImage, &errorMessage)
      }
      let validGrayImage = try #require(grayCImage)
      defer { MpImageFree(validGrayImage) }

      let copiedGrayMask = try MPMask(copyingCImage: validGrayImage)

      #expect(copiedGrayMask == mask)

      var srgbaImage: MpImagePtr?
      try MpStatus.check(defaultMessage: "Failed to create SRGBA MpImage.") { errorMessage in
        MpImageCreateFromUint8Data(kMpImageFormatSrgba, 1, 1, base, 4, &srgbaImage, &errorMessage)
      }
      let validSrgba = try #require(srgbaImage)
      defer { MpImageFree(validSrgba) }

      #expect(throws: TasksError.self) { _ = try MPMask(copyingCImage: validSrgba) }

      let negativeMask = MPMask(uint8Data: base, width: -2, height: -2)

      #expect(negativeMask.pixelCount == 0)
      #expect(negativeMask.uint8Elements.isEmpty)
      #expect(negativeMask != mask)
    }
  }

  @Test
  func `MPMask converts Float32 lazily to UInt8 with clamping and NaN handling`() throws {
    let rawFloats: [Float] = [-0.5, .nan, 0.5, 1.5]

    let mask = try rawFloats.withUnsafeBufferPointer { buffer in
      MPMask(float32Data: try #require(buffer.baseAddress), width: 2, height: 2)
    }

    #expect(mask.dataType == .float32)
    #expect(mask.uint8Elements == [0, 0, 127, 255])

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

      #expect(copiedMask.width == 2)
      #expect(copiedMask.height == 2)
      #expect(copiedMask.dataType == .float32)
      #expect(copiedMask.float32Elements == sourceFloats)
    }
  }

  @Test
  func `VisionTaskRunner validates baseOptions, handler consistency, and runningMode`() throws {
    let modelOptions = BaseOptions(modelAssetPath: "/tmp/model.task")
    let bufferOptions = BaseOptions(modelAssetBuffer: Data([0x01, 0x02]))

    #expect(throws: TasksError.self) {
      _ = try VisionTaskRunner(baseOptions: BaseOptions(), runningMode: .image)
    }
    #expect(throws: TasksError.self) {
      _ = try VisionTaskRunner(
        baseOptions: modelOptions, runningMode: .image, hasLiveStreamHandler: true)
    }
    #expect(throws: TasksError.self) {
      _ = try VisionTaskRunner(
        baseOptions: modelOptions, runningMode: .liveStream, hasLiveStreamHandler: false)
    }

    let runner = try VisionTaskRunner(
      baseOptions: bufferOptions, runningMode: .video, roiAllowed: true)

    try runner.validate(runningMode: .video)
    #expect(throws: TasksError.self) { try runner.validate(runningMode: .image) }
    #expect(try runner.cTimestamp(Int(Int64.max / 1000)) == Int64.max / 1000)
    #expect(throws: TasksError.self) { try runner.cTimestamp(-1) }
    #expect(throws: TasksError.self) { try runner.cTimestamp(Int(Int64.max / 1000) + 1) }
  }

  @Test
  func `VisionTaskRunner resolves orientations, ROI, and capability restrictions`() throws {
    let modelOptions = BaseOptions(modelAssetPath: "/tmp/model.task")
    let roiRunner = try VisionTaskRunner(
      baseOptions: modelOptions, runningMode: .image, roiAllowed: true)
    let roi = CGRect(x: 0.1, y: 0.2, width: 0.5, height: 0.4)

    #expect(try roiRunner.resolveImageProcessingOptions(for: .up).rotation_degrees == 0)
    #expect(try roiRunner.resolveImageProcessingOptions(for: .right).rotation_degrees == -270)
    #expect(try roiRunner.resolveImageProcessingOptions(for: .down).rotation_degrees == -180)
    #expect(try roiRunner.resolveImageProcessingOptions(for: .left).rotation_degrees == -90)

    let composed = try roiRunner.resolveImageProcessingOptions(
      ImageProcessingOptions(regionOfInterest: roi, rotationDegrees: 90), for: .right)

    #expect(composed.rotation_degrees == -180)
    #expect(composed.has_region_of_interest == 1)
    #expect(abs(composed.region_of_interest.left - 0.1) < 1e-5)
    #expect(abs(composed.region_of_interest.bottom - 0.6) < 1e-5)

    let minInt32MultipleOf90 = Int(Int32.min / 90) * 90
    let overflowError = try #require(throws: TasksError.self) {
      _ = try roiRunner.resolveImageProcessingOptions(
        ImageProcessingOptions(rotationDegrees: minInt32MultipleOf90), for: .right)
    }
    #expect(overflowError.code == .invalidArgument)

    for mirrored: UIImage.Orientation in [
      .upMirrored, .downMirrored, .leftMirrored, .rightMirrored,
    ] {
      #expect(throws: TasksError.self) {
        _ = try roiRunner.resolveImageProcessingOptions(for: mirrored)
      }
    }

    let noRoiRunner = try VisionTaskRunner(
      baseOptions: modelOptions, runningMode: .image, roiAllowed: false)

    #expect(throws: TasksError.self) {
      _ = try noRoiRunner.resolveImageProcessingOptions(
        ImageProcessingOptions(regionOfInterest: roi), for: .up)
    }

    let noNormRectRunner = try VisionTaskRunner(
      baseOptions: modelOptions, runningMode: .image, roiAllowed: false, normRectAllowed: false)

    #expect(throws: TasksError.self) {
      _ = try noNormRectRunner.resolveImageProcessingOptions(for: .right)
    }
    #expect(throws: TasksError.self) {
      _ = try noNormRectRunner.resolveImageProcessingOptions(
        ImageProcessingOptions(rotationDegrees: 90), for: .up)
    }

    let image = try MPImage(rgbaData: [UInt8](repeating: 255, count: 16), width: 2, height: 2)

    let widthAndRotation = try roiRunner.withPreparedInput(
      image: image,
      imageProcessingOptions: ImageProcessingOptions(rotationDegrees: 180),
      expectedMode: .image
    ) { cImage, cOptionsPtr in
      (MpImageGetWidth(cImage), cOptionsPtr.pointee.rotation_degrees)
    }

    #expect(widthAndRotation.0 == 2)
    #expect(widthAndRotation.1 == 180)
  }

  @Test
  func `VisionLiveStreamRegistration dispatches results, forwards errors, and recycles slots`()
    async throws
  {
    typealias TestCCallback =
      @convention(c) (MpStatus, UnsafePointer<MpRectF>?, MpImagePtr?, Int64) -> Void

    let (deliveredOutcome, deliveredTimestamp): (Result<Float, any Error>, Int) =
      try await withCheckedThrowingContinuation { continuation in
        do {
          let registered = try VisionLiveStreamRegistration.register(
            queueLabelSuffix: "test",
            resultType: MpRectF.self,
            as: TestCCallback.self,
            decode: { rectPtr, _, _ in rectPtr.pointee.right - rectPtr.pointee.left },
            handler: { outcome, ts in continuation.resume(returning: (outcome, ts)) }
          )
          defer { registered.registration.unregister() }
          var rect = MpRectF(left: 0.25, top: 0.0, bottom: 1.0, right: 0.75)
          withUnsafePointer(to: &rect) { rectPtr in
            registered.cCallback(kMpOk, rectPtr, nil, 123)
          }
        } catch {
          continuation.resume(throwing: error)
        }
      }

    let deliveredValue = try deliveredOutcome.get()
    #expect(abs(deliveredValue - 0.5) < 1e-5)
    #expect(deliveredTimestamp == 123)

    let (errorOutcome, errorTimestamp): (Result<Float, any Error>, Int) =
      try await withCheckedThrowingContinuation { continuation in
        do {
          let registered = try VisionLiveStreamRegistration.register(
            queueLabelSuffix: "test_err",
            as: TestCCallback.self,
            decode: { (rectPtr: UnsafePointer<MpRectF>, _, _) in rectPtr.pointee.left },
            handler: { outcome, ts in continuation.resume(returning: (outcome, ts)) }
          )
          defer { registered.registration.unregister() }
          registered.cCallback(kMpInvalidArgument, nil, nil, 456)
        } catch {
          continuation.resume(throwing: error)
        }
      }

    #expect(errorTimestamp == 456)
    #expect(throws: TasksError.self) { _ = try errorOutcome.get() }

    var active: [VisionLiveStreamRegistration] = []
    defer { active.forEach { $0.unregister() } }
    for _ in 0..<VisionLiveStreamRegistration.capacity {
      let reg = try VisionLiveStreamRegistration.register(
        queueLabelSuffix: "pool",
        as: TestCCallback.self,
        decode: { (ptr: UnsafePointer<MpRectF>, _, _) in ptr.pointee.left },
        handler: { _, _ in }
      )
      active.append(reg.registration)
    }

    let error = try #require(throws: TasksError.self) {
      try VisionLiveStreamRegistration.register(
        queueLabelSuffix: "overflow",
        as: TestCCallback.self,
        decode: { (ptr: UnsafePointer<MpRectF>, _, _) in ptr.pointee.left },
        handler: { _, _ in }
      )
    }

    #expect(error.code == .resourceExhausted)

    let released = active.removeLast()
    released.unregister()
    released.unregister()

    let recycled = try VisionLiveStreamRegistration.register(
      queueLabelSuffix: "recycled",
      as: TestCCallback.self,
      decode: { (ptr: UnsafePointer<MpRectF>, _, _) in ptr.pointee.left },
      handler: { _, _ in }
    )
    recycled.registration.unregister()
  }

  @Test
  func `MPImage cMaskImage converts VEC32F1 and SRGBA with stride and orientation`() throws {
    let vecFloats: [Float] = [0.0, 0.25, 0.5, 0.75, 0.125, 1.0]

    let vecMaskImage: MPImage = try vecFloats.withUnsafeBufferPointer { buffer in
      var cImageOut: MpImagePtr?
      try MpStatus.check(defaultMessage: "Failed to create VEC32F1 MpImage.") { errorMessage in
        MpImageCreateFromFloatData(
          kMpImageFormatVec32F1, 3, 2, buffer.baseAddress, 6, &cImageOut, &errorMessage)
      }
      let cImage = try #require(cImageOut)
      defer { MpImageFree(cImage) }
      return try MPImage(cMaskImage: cImage, orientation: .right)
    }

    #expect(vecMaskImage.imageSourceType == .pixelBuffer)
    #expect(vecMaskImage.orientation == .right)
    #expect(vecMaskImage.width == 3)
    #expect(vecMaskImage.height == 2)
    let vecPixelBuffer = try #require(vecMaskImage.pixelBuffer)
    #expect(
      CVPixelBufferGetPixelFormatType(vecPixelBuffer) == kCVPixelFormatType_OneComponent32Float)
    #expect(try Self.readPlanarFloats(from: vecPixelBuffer, width: 3, height: 2) == vecFloats)

    let channel0Values: [UInt8] = [0, 51, 102, 153, 204, 255]
    var srgbaBytes: [UInt8] = []
    for r in channel0Values {
      srgbaBytes.append(contentsOf: [r, 255 &- r, 128, 200])
    }

    let srgbaMaskImage: MPImage = try srgbaBytes.withUnsafeBufferPointer { buffer in
      var cImageOut: MpImagePtr?
      try MpStatus.check(defaultMessage: "Failed to create SRGBA MpImage.") { errorMessage in
        MpImageCreateFromUint8Data(
          kMpImageFormatSrgba, 3, 2, buffer.baseAddress, Int32(buffer.count), &cImageOut,
          &errorMessage)
      }
      let cImage = try #require(cImageOut)
      defer { MpImageFree(cImage) }
      return try MPImage(cMaskImage: cImage, orientation: .leftMirrored)
    }

    #expect(srgbaMaskImage.imageSourceType == .pixelBuffer)
    #expect(srgbaMaskImage.orientation == .leftMirrored)
    #expect(srgbaMaskImage.width == 3)
    #expect(srgbaMaskImage.height == 2)
    let srgbaPixelBuffer = try #require(srgbaMaskImage.pixelBuffer)
    #expect(
      CVPixelBufferGetPixelFormatType(srgbaPixelBuffer) == kCVPixelFormatType_OneComponent32Float)
    let actualFloats = try Self.readPlanarFloats(from: srgbaPixelBuffer, width: 3, height: 2)
    let expectedFloats: [Float] = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    #expect(actualFloats.count == expectedFloats.count)
    for (actual, expected) in zip(actualFloats, expectedFloats) {
      #expect(abs(actual - expected) < 1e-5)
    }

    let grayBytes: [UInt8] = [0, 64, 128, 255]

    try grayBytes.withUnsafeBufferPointer { buffer in
      var grayImageOut: MpImagePtr?
      try MpStatus.check(defaultMessage: "Failed to create Gray8 MpImage.") { errorMessage in
        MpImageCreateFromUint8Data(
          kMpImageFormatGray8, 2, 2, buffer.baseAddress, 4, &grayImageOut, &errorMessage)
      }
      let grayImage = try #require(grayImageOut)
      defer { MpImageFree(grayImage) }

      let error = try #require(throws: TasksError.self) {
        _ = try MPImage(cMaskImage: grayImage, orientation: .up)
      }

      #expect(error.code == .internalError)
    }
  }

  private static func readPlanarFloats(
    from pixelBuffer: CVPixelBuffer,
    width: Int,
    height: Int,
    sourceLocation: SourceLocation = #_sourceLocation
  ) throws -> [Float] {
    #expect(
      CVPixelBufferLockBaseAddress(pixelBuffer, .readOnly) == kCVReturnSuccess,
      sourceLocation: sourceLocation
    )
    defer { CVPixelBufferUnlockBaseAddress(pixelBuffer, .readOnly) }
    let baseAddress = try #require(
      CVPixelBufferGetBaseAddress(pixelBuffer),
      sourceLocation: sourceLocation
    )
    let bytesPerRow = CVPixelBufferGetBytesPerRow(pixelBuffer)
    var values: [Float] = []
    values.reserveCapacity(width * height)
    for row in 0..<height {
      let rowPtr = baseAddress.advanced(by: row * bytesPerRow)
        .assumingMemoryBound(to: Float.self)
      for col in 0..<width {
        values.append(rowPtr[col])
      }
    }
    return values
  }
}
