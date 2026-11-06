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

package import third_party_mediapipe_tasks_c_core_mp_status
package import third_party_mediapipe_tasks_c_vision_core_image

#if !MEDIAPIPE_SINGLE_MODULE
  import MediaPipeTasksCore
#endif

/// Underlying element type of a MediaPipe segmentation mask.
@frozen public enum MaskDataType: Int, Sendable, Equatable, Hashable, CaseIterable {
  /// Single-channel 8-bit unsigned integer values in `0...255`.
  case uInt8 = 0

  /// Single-channel 32-bit floating-point values in `0.0...1.0`.
  case float32 = 1
}

/// Single-channel MediaPipe segmentation mask supporting conversion between `0...255` and
/// `0.0...1.0` representations.
public final class MPMask: Sendable {
  private enum Storage: Sendable {
    case uInt8([UInt8])
    case float32([Float])
  }

  private let storage: Storage

  /// Width of the segmentation mask in pixels.
  public let width: Int

  /// Height of the segmentation mask in pixels.
  public let height: Int

  /// Primary native data type of the segmentation mask.
  public let dataType: MaskDataType

  /// Total number of pixels in the mask (`width * height`).
  public let pixelCount: Int

  /// Mask values in `0...255`.
  ///
  /// When `dataType` is `.float32`, each element is clamped to `0.0...1.0` (mapping `NaN` to `0`)
  /// and scaled by `255.0`.
  public var uint8Elements: [UInt8] {
    switch storage {
    case .uInt8(let elements):
      return elements
    case .float32(let elements):
      return elements.map { value in
        let sanitized = value.isNaN ? 0.0 : min(max(value, 0.0), 1.0)
        return UInt8(sanitized * 255.0)
      }
    }
  }

  /// Mask values in `0.0...1.0`.
  ///
  /// When `dataType` is `.uInt8`, each element is scaled by `1.0 / 255.0`.
  public var float32Elements: [Float] {
    switch storage {
    case .float32(let elements):
      return elements
    case .uInt8(let elements):
      return elements.map { Float($0) / 255.0 }
    }
  }

  private init(
    width: Int,
    height: Int,
    pixelCount: Int,
    dataType: MaskDataType,
    storage: Storage
  ) {
    self.width = width
    self.height = height
    self.pixelCount = pixelCount
    self.dataType = dataType
    self.storage = storage
  }

  /// Creates a `.uInt8` mask by copying `width * height` elements from `uint8Data`.
  public convenience init(
    uint8Data: UnsafePointer<UInt8>,
    width: Int,
    height: Int
  ) {
    let (w, h, count) = Self.clampDimensions(width: width, height: height)
    self.init(
      width: w,
      height: h,
      pixelCount: count,
      dataType: .uInt8,
      storage: .uInt8(Array(UnsafeBufferPointer(start: uint8Data, count: count)))
    )
  }

  /// Creates a `.float32` mask by copying `width * height` elements from `float32Data`.
  public convenience init(
    float32Data: UnsafePointer<Float>,
    width: Int,
    height: Int
  ) {
    let (w, h, count) = Self.clampDimensions(width: width, height: height)
    self.init(
      width: w,
      height: h,
      pixelCount: count,
      dataType: .float32,
      storage: .float32(Array(UnsafeBufferPointer(start: float32Data, count: count)))
    )
  }

  /// Creates a mask by copying a C `MpImagePtr` (`kMpImageFormatVec32F1` or
  /// `kMpImageFormatGray8`).
  package convenience init(copyingCImage cImage: MpImagePtr) throws {
    let width = Int(MpImageGetWidth(cImage))
    let height = Int(MpImageGetHeight(cImage))
    let format = MpImageGetFormat(cImage)

    switch format {
    case kMpImageFormatVec32F1:
      var floatDataPtr: UnsafePointer<Float>?
      try MpStatus.check(defaultMessage: "Failed to read float32 mask data.") { errorMessage in
        MpImageDataFloat32(cImage, &floatDataPtr, &errorMessage)
      }
      guard let floatDataPtr else {
        throw TasksError.internalError("Failed to read float32 mask data.")
      }
      self.init(float32Data: floatDataPtr, width: width, height: height)

    case kMpImageFormatGray8:
      var uint8DataPtr: UnsafePointer<UInt8>?
      try MpStatus.check(defaultMessage: "Failed to read uint8 mask data.") { errorMessage in
        MpImageDataUint8(cImage, &uint8DataPtr, &errorMessage)
      }
      guard let uint8DataPtr else {
        throw TasksError.internalError("Failed to read uint8 mask data.")
      }
      self.init(uint8Data: uint8DataPtr, width: width, height: height)

    default:
      throw TasksError.internalError(
        "Unsupported C MpImage format (\(format.rawValue)) for MPMask."
      )
    }
  }

  /// Invokes `body` with an `UnsafeBufferPointer<UInt8>` covering all `width * height` pixels.
  public func withUnsafeUInt8BufferPointer<Result>(
    _ body: (UnsafeBufferPointer<UInt8>) throws -> Result
  ) rethrows -> Result {
    try uint8Elements.withUnsafeBufferPointer(body)
  }

  /// Invokes `body` with an `UnsafeBufferPointer<Float>` covering all `width * height` pixels.
  public func withUnsafeFloat32BufferPointer<Result>(
    _ body: (UnsafeBufferPointer<Float>) throws -> Result
  ) rethrows -> Result {
    try float32Elements.withUnsafeBufferPointer(body)
  }

  private static func clampDimensions(width: Int, height: Int) -> (Int, Int, Int) {
    let clampedWidth = max(0, width)
    let clampedHeight = max(0, height)
    let (product, overflow) = clampedWidth.multipliedReportingOverflow(by: clampedHeight)
    return overflow ? (0, 0, 0) : (clampedWidth, clampedHeight, product)
  }
}

extension MPMask: Equatable {
  public static func == (lhs: MPMask, rhs: MPMask) -> Bool {
    guard lhs.width == rhs.width, lhs.height == rhs.height, lhs.dataType == rhs.dataType else {
      return false
    }
    switch lhs.dataType {
    case .uInt8:
      return lhs.uint8Elements == rhs.uint8Elements
    case .float32:
      return lhs.float32Elements == rhs.float32Elements
    }
  }
}

/// Alias matching the Objective-C `NS_SWIFT_NAME(Mask)` bridging name.
public typealias Mask = MPMask
