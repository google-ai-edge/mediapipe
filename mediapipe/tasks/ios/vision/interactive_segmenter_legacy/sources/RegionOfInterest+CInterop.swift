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
package import third_party_mediapipe_tasks_c_components_containers_keypoint
package import third_party_mediapipe_tasks_c_vision_interactive_segmenter_legacy_interactive_segmenter_legacy_lib

#if !MEDIAPIPE_SINGLE_MODULE
  @_exported import MediaPipeTasksComponents
  @_exported import MediaPipeTasksCore
#endif

extension RegionOfInterest {
  /// Invokes `body` with a C `MpRegionOfInterest` whose `scribble` pointer is valid only for the
  /// duration of the closure call.
  ///
  /// - Throws: `TasksError` if `scribbles` is empty or exceeds `UInt32.max` points.
  package func withCRegionOfInterest<Result>(
    _ body: (UnsafePointer<MpRegionOfInterest>) throws -> Result
  ) throws -> Result {
    var cRegion = MpRegionOfInterest()
    switch kind {
    case .keypoint(let keypoint):
      cRegion.format = MP_REGION_OF_INTEREST_FORMAT_KEYPOINT
      cRegion.keypoint = Self.cKeypoint(keypoint)
      return try withUnsafePointer(to: cRegion, body)
    case .scribbles(let keypoints):
      guard !keypoints.isEmpty, let count = UInt32(exactly: keypoints.count) else {
        throw TasksError.invalidArgument("Scribbles must contain between 1 and UInt32.max points.")
      }
      var cKeypoints = keypoints.map(Self.cKeypoint)
      return try cKeypoints.withUnsafeMutableBufferPointer { buffer in
        cRegion.format = MP_REGION_OF_INTEREST_FORMAT_SCRIBBLE
        cRegion.scribble = buffer.baseAddress
        cRegion.scribble_count = count
        return try withUnsafePointer(to: cRegion, body)
      }
    }
  }

  private static func cKeypoint(_ keypoint: NormalizedKeypoint) -> MpNormalizedKeypoint {
    var cKeypoint = MpNormalizedKeypoint()
    cKeypoint.x = Float(keypoint.location.x)
    cKeypoint.y = Float(keypoint.location.y)
    return cKeypoint
  }
}
