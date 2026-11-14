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
import CoreVideo
import UIKit
package import third_party_mediapipe_tasks_c_core_mp_status
package import third_party_mediapipe_tasks_c_vision_core_image

#if !MEDIAPIPE_SINGLE_MODULE
  import MediaPipeTasksCore
#endif

extension MPImage {
  /// Creates a `OneComponent32Float` `CVPixelBuffer`-backed `MPImage` from a C `MpImagePtr` mask
  /// in `VEC32F1` or `SRGBA` format (extracting channel 0 normalized to `0.0...1.0`).
  package convenience init(cMaskImage: MpImagePtr, orientation: UIImage.Orientation) throws {
    let width = Int(MpImageGetWidth(cMaskImage))
    let height = Int(MpImageGetHeight(cMaskImage))
    let (pixelCount, pixelOverflow) = width.multipliedReportingOverflow(by: height)
    guard width > 0, height > 0, !pixelOverflow else {
      throw TasksError.internalError("Invalid output mask dimensions (\(width)x\(height)).")
    }
    let format = MpImageGetFormat(cMaskImage)
    guard format == kMpImageFormatVec32F1 || format == kMpImageFormatSrgba else {
      throw TasksError.internalError("Unsupported output mask MpImageFormat (\(format.rawValue)).")
    }

    var pixelBufferOut: CVPixelBuffer?
    let cvStatus = CVPixelBufferCreate(
      kCFAllocatorDefault,
      width,
      height,
      kCVPixelFormatType_OneComponent32Float,
      nil,
      &pixelBufferOut
    )
    guard cvStatus == kCVReturnSuccess, let pixelBuffer = pixelBufferOut else {
      throw TasksError.internalError(
        "Failed to allocate OneComponent32Float CVPixelBuffer for segmentation mask."
      )
    }

    do {
      guard CVPixelBufferLockBaseAddress(pixelBuffer, []) == kCVReturnSuccess else {
        throw TasksError.internalError("Failed to lock output CVPixelBuffer base address.")
      }
      defer {
        CVPixelBufferUnlockBaseAddress(pixelBuffer, [])
      }
      guard let dstBaseAddress = CVPixelBufferGetBaseAddress(pixelBuffer) else {
        throw TasksError.internalError("Allocated CVPixelBuffer has a nil baseAddress.")
      }

      let dstBytesPerRow = CVPixelBufferGetBytesPerRow(pixelBuffer)
      let rowFloatBytes = width * MemoryLayout<Float>.stride

      switch format {
      case kMpImageFormatVec32F1:
        var floatPtr: UnsafePointer<Float>?
        try MpStatus.check(defaultMessage: "Failed to read Float32 mask data.") { errorMessage in
          MpImageDataFloat32(cMaskImage, &floatPtr, &errorMessage)
        }
        guard let validFloatPtr = floatPtr else {
          throw TasksError.internalError("MpImageDataFloat32 returned a null pointer.")
        }
        let srcBase = UnsafeRawPointer(validFloatPtr)
        if dstBytesPerRow == rowFloatBytes {
          dstBaseAddress.copyMemory(from: srcBase, byteCount: height * rowFloatBytes)
        } else {
          for row in 0..<height {
            dstBaseAddress.advanced(by: row * dstBytesPerRow).copyMemory(
              from: srcBase.advanced(by: row * rowFloatBytes),
              byteCount: rowFloatBytes
            )
          }
        }

      case kMpImageFormatSrgba:
        let (_, rgbaRowOverflow) = width.multipliedReportingOverflow(by: 4)
        guard !rgbaRowOverflow else {
          throw TasksError.internalError("RGBA row byte count overflow for width (\(width)).")
        }
        var uint8Ptr: UnsafePointer<UInt8>?
        try MpStatus.check(defaultMessage: "Failed to read RGBA mask data.") { errorMessage in
          MpImageDataUint8(cMaskImage, &uint8Ptr, &errorMessage)
        }
        guard let validUint8Ptr = uint8Ptr else {
          throw TasksError.internalError("MpImageDataUint8 returned a null pointer.")
        }

        var planar8Bytes = [UInt8](repeating: 0, count: pixelCount)
        try planar8Bytes.withUnsafeMutableBytes { planar8Raw in
          var srcRgbaBuffer = vImage_Buffer(
            data: UnsafeMutableRawPointer(mutating: validUint8Ptr),
            height: vImagePixelCount(height),
            width: vImagePixelCount(width),
            rowBytes: width * 4
          )
          var planar8Buffer = vImage_Buffer(
            data: planar8Raw.baseAddress,
            height: vImagePixelCount(height),
            width: vImagePixelCount(width),
            rowBytes: width
          )
          let extractStatus = vImageExtractChannel_ARGB8888(
            &srcRgbaBuffer,
            &planar8Buffer,
            0,
            vImage_Flags(kvImageNoFlags)
          )
          guard extractStatus == kvImageNoError else {
            throw TasksError.internalError(
              "Failed to extract channel 0 from RGBA mask (\(extractStatus))."
            )
          }

          var dstFloatBuffer = vImage_Buffer(
            data: dstBaseAddress,
            height: vImagePixelCount(height),
            width: vImagePixelCount(width),
            rowBytes: dstBytesPerRow
          )
          let convertStatus = vImageConvert_Planar8toPlanarF(
            &planar8Buffer,
            &dstFloatBuffer,
            255.0,
            0.0,
            vImage_Flags(kvImageNoFlags)
          )
          guard convertStatus == kvImageNoError else {
            throw TasksError.internalError(
              "Failed to convert 8-bit mask plane to Float32 (\(convertStatus))."
            )
          }
        }

        var divisor: Float = 255.0
        if dstBytesPerRow == rowFloatBytes {
          let floatBase = dstBaseAddress.bindMemory(to: Float.self, capacity: pixelCount)
          vDSP_vsdiv(floatBase, 1, &divisor, floatBase, 1, vDSP_Length(pixelCount))
        } else {
          for row in 0..<height {
            let rowPtr = dstBaseAddress.advanced(by: row * dstBytesPerRow)
              .bindMemory(to: Float.self, capacity: width)
            vDSP_vsdiv(rowPtr, 1, &divisor, rowPtr, 1, vDSP_Length(width))
          }
        }

      default:
        throw TasksError.internalError(
          "Unsupported output mask MpImageFormat (\(format.rawValue))."
        )
      }
    }

    try self.init(pixelBuffer: pixelBuffer, orientation: orientation)
  }
}
