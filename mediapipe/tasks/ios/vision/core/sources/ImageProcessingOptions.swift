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
package import third_party_mediapipe_tasks_c_components_containers_rect
package import third_party_mediapipe_tasks_c_vision_core_image_processing_options

#if !MEDIAPIPE_SINGLE_MODULE
  import MediaPipeTasksCore
#endif

/// Options for cropping and rotating input images prior to MediaPipe Vision Task inference.
public struct ImageProcessingOptions: Sendable, Equatable {
  /// Optional normalized region of interest in `[0.0, 1.0]` with `minX < maxX` and `minY < maxY`.
  public var regionOfInterest: CGRect?

  /// Clockwise rotation in degrees to apply to the image or cropped region. Must be a multiple of
  /// `90`.
  public var rotationDegrees: Int

  public init(regionOfInterest: CGRect? = nil, rotationDegrees: Int = 0) {
    self.regionOfInterest = regionOfInterest
    self.rotationDegrees = rotationDegrees
  }

  /// Validates that `rotationDegrees` is a multiple of `90` and `regionOfInterest` (if present)
  /// lies within normalized `[0.0, 1.0]` bounds with positive width and height.
  public func validate() throws {
    guard Int32(exactly: rotationDegrees) != nil, rotationDegrees.isMultiple(of: 90) else {
      throw TasksError.invalidArgument(
        "rotationDegrees (\(rotationDegrees)) must be a 32-bit integer multiple of 90."
      )
    }
    if let roi = regionOfInterest {
      let minX = Float(roi.origin.x)
      let minY = Float(roi.origin.y)
      let maxX = Float(roi.origin.x + roi.size.width)
      let maxY = Float(roi.origin.y + roi.size.height)
      guard minX < maxX, minY < maxY, minX >= 0.0, minY >= 0.0, maxX <= 1.0, maxY <= 1.0 else {
        throw TasksError.invalidArgument(
          "regionOfInterest must have normalized coordinates in [0.0, 1.0] with left < right "
            + "and top < bottom."
        )
      }
    }
  }

  /// Validates options and invokes `body` with a C `MpImageProcessingOptions` struct.
  package func withCOptions<Result>(
    _ body: (MpImageProcessingOptions) throws -> Result
  ) throws -> Result {
    try validate()
    var cOptions = MpImageProcessingOptions()
    cOptions.rotation_degrees = Int32(clamping: rotationDegrees)
    if let roi = regionOfInterest {
      let left = Float(roi.origin.x)
      let top = Float(roi.origin.y)
      let right = Float(roi.origin.x + roi.size.width)
      let bottom = Float(roi.origin.y + roi.size.height)
      cOptions.has_region_of_interest = 1
      cOptions.region_of_interest = MpRectF(
        left: left,
        top: top,
        bottom: bottom,
        right: right
      )
    } else {
      cOptions.has_region_of_interest = 0
    }
    return try body(cOptions)
  }
}

extension ImageProcessingOptions: Hashable {
  public func hash(into hasher: inout Hasher) {
    if let roi = regionOfInterest {
      hasher.combine(true)
      let standardized = roi.standardized
      hasher.combine(standardized.isNull)
      if !standardized.isNull {
        hasher.combine(standardized.origin.x)
        hasher.combine(standardized.origin.y)
        hasher.combine(standardized.size.width)
        hasher.combine(standardized.size.height)
      }
    } else {
      hasher.combine(false)
    }
    hasher.combine(rotationDegrees)
  }
}
