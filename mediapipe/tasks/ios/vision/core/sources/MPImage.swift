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

import Accelerate
import CoreGraphics
import CoreMedia
import CoreVideo
import Foundation
import UIKit
package import third_party_mediapipe_tasks_c_core_mp_status
package import third_party_mediapipe_tasks_c_vision_core_image

#if !MEDIAPIPE_SINGLE_MODULE
  import MediaPipeTasksCore
#endif

/// Underlying source representation of an `MPImage`.
@frozen public enum MPImageSourceType: Int, Sendable, Equatable, Hashable, CaseIterable {
  /// Backed by a `UIImage` or raw RGBA/float pixel buffer.
  case image = 0

  /// Backed by a `CVPixelBuffer`.
  case pixelBuffer = 1

  /// Backed by a `CMSampleBuffer`.
  case sampleBuffer = 2
}

/// Input image container for on-device MediaPipe Vision Tasks inference.
///
/// `MPImage` is `@unchecked Sendable` because all stored properties are `let` constants and its
/// wrapped `UIImage`, `CVPixelBuffer`, and `CMSampleBuffer` instances are accessed strictly as
/// read-only pixel sources during C `MpImage` conversion.
public final class MPImage: @unchecked Sendable {
  private enum PixelStorage {
    case rgba([UInt8])
    case float32([Float])
    case uiImage(UIImage)
    case pixelBuffer(CVPixelBuffer)
    case sampleBuffer(CMSampleBuffer)
  }

  private let storage: PixelStorage

  /// Width of the image in pixels.
  public let width: CGFloat

  /// Height of the image in pixels.
  public let height: CGFloat

  /// Underlying pixel storage representation.
  public let imageSourceType: MPImageSourceType

  /// The 4-channel RGBA pixel buffer when initialized via `init(rgbaData:width:height:)`, or `nil`
  /// otherwise.
  public var rgbaData: [UInt8]? {
    guard case .rgba(let bytes) = storage else { return nil }
    return bytes
  }

  /// The single-channel `VEC32F1` pixel buffer when initialized via
  /// `init(floatData:width:height:)`, or `nil` otherwise.
  public var floatData: [Float]? {
    guard case .float32(let floats) = storage else { return nil }
    return floats
  }

  /// Display orientation of the image.
  public let orientation: UIImage.Orientation

  /// The underlying `UIImage` when `imageSourceType == .image`, or `nil` otherwise.
  public var image: UIImage? {
    guard case .uiImage(let uiImage) = storage else { return nil }
    return uiImage
  }

  /// The underlying `CVPixelBuffer` when `imageSourceType == .pixelBuffer`, or `nil` otherwise.
  public var pixelBuffer: CVPixelBuffer? {
    guard case .pixelBuffer(let buffer) = storage else { return nil }
    return buffer
  }

  /// The underlying `CMSampleBuffer` when `imageSourceType == .sampleBuffer`, or `nil`
  /// otherwise.
  public var sampleBuffer: CMSampleBuffer? {
    guard case .sampleBuffer(let buffer) = storage else { return nil }
    return buffer
  }

  /// Creates an image from `uiImage`, defaulting `orientation` to `uiImage.imageOrientation`.
  public init(uiImage: UIImage, orientation: UIImage.Orientation? = nil) throws {
    guard uiImage.cgImage != nil else {
      throw TasksError.invalidArgument("UIImage must have a valid underlying CGImage.")
    }
    self.storage = .uiImage(uiImage)
    self.imageSourceType = .image
    self.orientation = orientation ?? uiImage.imageOrientation
    self.width = uiImage.size.width * uiImage.scale
    self.height = uiImage.size.height * uiImage.scale
  }

  /// Creates an image from a `32BGRA` or `OneComponent32Float` pixel buffer.
  public init(pixelBuffer: CVPixelBuffer, orientation: UIImage.Orientation = .up) throws {
    let format = CVPixelBufferGetPixelFormatType(pixelBuffer)
    guard
      format == kCVPixelFormatType_32BGRA
        || format == kCVPixelFormatType_OneComponent32Float
    else {
      throw TasksError.invalidArgument(
        "Unsupported CVPixelBuffer format (\(format)); expected 32BGRA or OneComponent32Float."
      )
    }
    self.storage = .pixelBuffer(pixelBuffer)
    self.imageSourceType = .pixelBuffer
    self.orientation = orientation
    self.width = CGFloat(CVPixelBufferGetWidth(pixelBuffer))
    self.height = CGFloat(CVPixelBufferGetHeight(pixelBuffer))
  }

  /// Creates an image from a sample buffer containing a `CVImageBuffer`.
  public init(sampleBuffer: CMSampleBuffer, orientation: UIImage.Orientation = .up) throws {
    guard CMSampleBufferIsValid(sampleBuffer),
      let imageBuffer = CMSampleBufferGetImageBuffer(sampleBuffer)
    else {
      throw TasksError.invalidArgument(
        "Sample buffer is not valid or does not contain a CVImageBuffer."
      )
    }
    self.storage = .sampleBuffer(sampleBuffer)
    self.imageSourceType = .sampleBuffer
    self.orientation = orientation
    self.width = CGFloat(CVPixelBufferGetWidth(imageBuffer))
    self.height = CGFloat(CVPixelBufferGetHeight(imageBuffer))
  }

  /// Creates an image from 4-channel RGBA pixel data.
  public init(rgbaData: [UInt8], width: Int, height: Int) throws {
    let (pixelCount, pixelOverflow) = width.multipliedReportingOverflow(by: height)
    let (byteCount, byteOverflow) = pixelCount.multipliedReportingOverflow(by: 4)
    guard
      width > 0, height > 0, !pixelOverflow, !byteOverflow,
      Int32(exactly: byteCount) != nil, rgbaData.count == byteCount
    else {
      throw TasksError.invalidArgument(
        "RGBA byte count (\(rgbaData.count)) must equal width (\(width)) * height (\(height)) * 4."
      )
    }
    self.storage = .rgba(rgbaData)
    self.imageSourceType = .image
    self.width = CGFloat(width)
    self.height = CGFloat(height)
    self.orientation = .up
  }

  /// Creates an image from single-channel `VEC32F1` pixel data.
  public init(floatData: [Float], width: Int, height: Int) throws {
    let (pixelCount, pixelOverflow) = width.multipliedReportingOverflow(by: height)
    guard
      width > 0, height > 0, !pixelOverflow,
      Int32(exactly: pixelCount) != nil, floatData.count == pixelCount
    else {
      throw TasksError.invalidArgument(
        "Float element count (\(floatData.count)) must equal width (\(width)) * height (\(height))."
      )
    }
    self.storage = .float32(floatData)
    self.imageSourceType = .image
    self.width = CGFloat(width)
    self.height = CGFloat(height)
    self.orientation = .up
  }

  /// Creates a C `MpImagePtr` for this `MPImage`, invokes `body`, and guarantees `MpImageFree`
  /// on scope exit.
  package func withCImage<Result>(_ body: (MpImagePtr) throws -> Result) throws -> Result {
    let cImage = try createCImage()
    defer { MpImageFree(cImage) }
    return try body(cImage)
  }

  private func createCImage() throws -> MpImagePtr {
    switch storage {
    case .rgba(let bytes):
      return try Self.createCImageFromRGBA(bytes: bytes, width: Int(width), height: Int(height))
    case .float32(let floats):
      return try Self.createCImageFromFloats(floats: floats, width: Int(width), height: Int(height))
    case .uiImage(let uiImage):
      guard let cgImage = uiImage.cgImage else {
        throw TasksError.invalidArgument("UIImage does not contain a convertible CGImage.")
      }
      return try Self.createCImage(from: cgImage)
    case .pixelBuffer(let pixelBuffer):
      return try Self.createCImage(from: pixelBuffer)
    case .sampleBuffer(let sampleBuffer):
      guard let imageBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else {
        throw TasksError.invalidArgument("CMSampleBuffer does not contain a CVImageBuffer.")
      }
      return try Self.createCImage(from: imageBuffer)
    }
  }

  private static func createCImageFromRGBA(
    bytes: [UInt8], width: Int, height: Int
  ) throws -> MpImagePtr {
    let (pixelCount, pixelOverflow) = width.multipliedReportingOverflow(by: height)
    let (expectedByteCount, byteOverflow) = pixelCount.multipliedReportingOverflow(by: 4)
    guard width > 0, height > 0, !pixelOverflow, !byteOverflow, bytes.count == expectedByteCount
    else {
      throw TasksError.invalidArgument(
        "RGBA buffer byte count (\(bytes.count)) does not match width * height * 4 (\(expectedByteCount))."
      )
    }
    var cImage: MpImagePtr?
    try bytes.withUnsafeBufferPointer { buffer in
      try MpStatus.check(defaultMessage: "Failed to create C MpImage from RGBA data.") { errorMsg in
        MpImageCreateFromUint8Data(
          kMpImageFormatSrgba, Int32(width), Int32(height), buffer.baseAddress,
          Int32(buffer.count), &cImage, &errorMsg)
      }
    }
    guard let validImage = cImage else {
      throw TasksError.internalError("Failed to create C MpImage from RGBA data.")
    }
    return validImage
  }

  private static func createCImageFromFloats(
    floats: [Float], width: Int, height: Int
  ) throws -> MpImagePtr {
    let (expectedCount, overflow) = width.multipliedReportingOverflow(by: height)
    guard width > 0, height > 0, !overflow, floats.count == expectedCount else {
      throw TasksError.invalidArgument(
        "Float32 buffer element count (\(floats.count)) does not match width * height (\(expectedCount))."
      )
    }
    var cImage: MpImagePtr?
    try floats.withUnsafeBufferPointer { buffer in
      try MpStatus.check(defaultMessage: "Failed to create C MpImage from Float32 data.") {
        errorMsg in
        MpImageCreateFromFloatData(
          kMpImageFormatVec32F1, Int32(width), Int32(height), buffer.baseAddress,
          Int32(buffer.count), &cImage, &errorMsg)
      }
    }
    guard let validImage = cImage else {
      throw TasksError.internalError("Failed to create C MpImage from Float32 data.")
    }
    return validImage
  }

  private static func createCImage(from cgImage: CGImage) throws -> MpImagePtr {
    let width = cgImage.width
    let height = cgImage.height
    let bytesPerRow = width * 4
    var rgbaBytes = [UInt8](repeating: 0, count: height * bytesPerRow)
    let bitmapInfo =
      CGBitmapInfo.byteOrder32Big.rawValue | CGImageAlphaInfo.premultipliedLast.rawValue

    try rgbaBytes.withUnsafeMutableBytes { rawBuffer in
      guard
        let baseAddress = rawBuffer.baseAddress,
        let context = CGContext(
          data: baseAddress, width: width, height: height, bitsPerComponent: 8,
          bytesPerRow: bytesPerRow, space: CGColorSpaceCreateDeviceRGB(), bitmapInfo: bitmapInfo)
      else {
        throw TasksError.internalError("Failed to create CGContext for CGImage conversion.")
      }
      context.draw(cgImage, in: CGRect(x: 0, y: 0, width: width, height: height))

      var buffer = vImage_Buffer(
        data: baseAddress,
        height: vImagePixelCount(height),
        width: vImagePixelCount(width),
        rowBytes: bytesPerRow
      )
      let status = vImageUnpremultiplyData_RGBA8888(&buffer, &buffer, vImage_Flags(kvImageNoFlags))
      guard status == kvImageNoError else {
        throw TasksError.internalError("Failed to unpremultiply RGBA image data.")
      }
    }
    return try createCImageFromRGBA(bytes: rgbaBytes, width: width, height: height)
  }

  private static func createCImage(from pixelBuffer: CVPixelBuffer) throws -> MpImagePtr {
    let width = CVPixelBufferGetWidth(pixelBuffer)
    let height = CVPixelBufferGetHeight(pixelBuffer)
    let stride = CVPixelBufferGetBytesPerRow(pixelBuffer)
    let format = CVPixelBufferGetPixelFormatType(pixelBuffer)

    guard CVPixelBufferLockBaseAddress(pixelBuffer, .readOnly) == kCVReturnSuccess else {
      throw TasksError.internalError("Failed to lock CVPixelBuffer base address.")
    }
    defer { CVPixelBufferUnlockBaseAddress(pixelBuffer, .readOnly) }

    guard let baseAddress = CVPixelBufferGetBaseAddress(pixelBuffer) else {
      throw TasksError.invalidArgument("CVPixelBuffer baseAddress is nil.")
    }

    switch format {
    case kCVPixelFormatType_32BGRA:
      let rgbaRowBytes = width * 4
      var rgbaBytes = [UInt8](repeating: 0, count: height * rgbaRowBytes)
      let convertError: vImage_Error = rgbaBytes.withUnsafeMutableBytes { dstRaw in
        var src = vImage_Buffer(
          data: baseAddress, height: vImagePixelCount(height), width: vImagePixelCount(width),
          rowBytes: stride)
        var dst = vImage_Buffer(
          data: dstRaw.baseAddress, height: vImagePixelCount(height),
          width: vImagePixelCount(width), rowBytes: rgbaRowBytes)
        let permuteStatus = vImagePermuteChannels_ARGB8888(
          &src, &dst, [2, 1, 0, 3], vImage_Flags(kvImageNoFlags))
        guard permuteStatus == kvImageNoError else { return permuteStatus }
        return vImageUnpremultiplyData_RGBA8888(&dst, &dst, vImage_Flags(kvImageNoFlags))
      }
      guard convertError == kvImageNoError else {
        throw TasksError.internalError("Failed to convert BGRA CVPixelBuffer to RGBA.")
      }
      return try createCImageFromRGBA(bytes: rgbaBytes, width: width, height: height)

    case kCVPixelFormatType_OneComponent32Float:
      var floats = [Float](repeating: 0.0, count: width * height)
      let rowByteCount = width * MemoryLayout<Float>.stride
      floats.withUnsafeMutableBytes { dstRaw in
        guard let dstBase = dstRaw.baseAddress else { return }
        if stride == rowByteCount {
          dstBase.copyMemory(from: baseAddress, byteCount: height * rowByteCount)
        } else {
          for row in 0..<height {
            dstBase.advanced(by: row * rowByteCount).copyMemory(
              from: baseAddress.advanced(by: row * stride), byteCount: rowByteCount)
          }
        }
      }
      return try createCImageFromFloats(floats: floats, width: width, height: height)

    default:
      throw TasksError.invalidArgument(
        "Unsupported pixel format for CVPixelBuffer. Expected 32BGRA or OneComponent32Float."
      )
    }
  }
}
