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
package import third_party_mediapipe_tasks_c_vision_pose_landmarker_pose_landmarker_result

#if !MEDIAPIPE_SINGLE_MODULE
  @_exported import MediaPipeTasksComponents
  @_exported import MediaPipeTasksCore
  @_exported import MediaPipeTasksVisionCore
#endif

/// Pose landmarks, world landmarks, and optional segmentation masks produced by `PoseLandmarker`.
public struct PoseLandmarkerResult: Sendable, Equatable {
  /// Normalized `[0.0, 1.0]` image-space landmarks for each detected pose.
  public let landmarks: [[NormalizedLandmark]]
  /// Metric 3D world-space landmarks (in meters) for each detected pose.
  public let worldLandmarks: [[Landmark]]
  /// Per-pose segmentation masks, or `nil` when segmentation mask output is disabled.
  public let segmentationMasks: [MPMask]?
  /// Timestamp in milliseconds associated with this result (`0` in `.image` mode).
  public let timestampInMilliseconds: Int

  /// Creates a `PoseLandmarkerResult` with landmarks, world landmarks, optional masks, and timestamp.
  public init(
    landmarks: [[NormalizedLandmark]],
    worldLandmarks: [[Landmark]],
    segmentationMasks: [MPMask]? = nil,
    timestampInMilliseconds: Int
  ) {
    self.landmarks = landmarks
    self.worldLandmarks = worldLandmarks
    self.segmentationMasks = segmentationMasks
    self.timestampInMilliseconds = timestampInMilliseconds
  }

  /// Deep-copies landmarks and optional segmentation masks before `MpPoseLandmarkerCloseResult`.
  package init(
    cResult: MpPoseLandmarkerResult,
    outputSegmentationMasks: Bool,
    timestampInMilliseconds: Int
  ) throws {
    let masks: [MPMask]?
    if let cMasks = cResult.segmentation_masks, cResult.segmentation_masks_count > 0 {
      masks = try (0..<Int(cResult.segmentation_masks_count)).map { index in
        guard let cMask = cMasks[index] else {
          throw TasksError.internalError("Null pose segmentation mask at index \(index).")
        }
        return try MPMask(copyingCImage: cMask)
      }
    } else {
      masks = outputSegmentationMasks ? [] : nil
    }
    self.init(
      landmarks: Array(
        cArray: cResult.pose_landmarks, count: cResult.pose_landmarks_count,
        transform: NormalizedLandmark.landmarks(from:)),
      worldLandmarks: Array(
        cArray: cResult.pose_world_landmarks, count: cResult.pose_world_landmarks_count,
        transform: Landmark.landmarks(from:)),
      segmentationMasks: masks,
      timestampInMilliseconds: timestampInMilliseconds
    )
  }
}
