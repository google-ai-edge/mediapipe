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
package import third_party_mediapipe_tasks_c_vision_holistic_landmarker_holistic_landmarker_result

#if !MEDIAPIPE_SINGLE_MODULE
  @_exported import MediaPipeTasksComponents
  @_exported import MediaPipeTasksCore
  @_exported import MediaPipeTasksVisionCore
#endif

/// Holistic face, pose, and hand landmarks detected by `HolisticLandmarker` (not `Hashable`
/// because `MPMask` conforms only to `Equatable`).
public struct HolisticLandmarkerResult: Sendable, Equatable {
  /// Detected face landmarks in normalized `[0.0, 1.0]` image coordinates.
  public let faceLandmarks: [NormalizedLandmark]
  /// Optional face blendshape classifications when `outputFaceBlendshapes` is enabled.
  public let faceBlendshapes: Classifications?
  /// Detected pose landmarks in normalized `[0.0, 1.0]` image coordinates.
  public let poseLandmarks: [NormalizedLandmark]
  /// Detected 3D pose landmarks in world coordinates (meters).
  public let poseWorldLandmarks: [Landmark]
  /// Optional `Float32` pose segmentation mask when `outputPoseSegmentationMasks` is enabled.
  public let poseSegmentationMask: MPMask?
  /// Detected left-hand landmarks in normalized `[0.0, 1.0]` image coordinates.
  public let leftHandLandmarks: [NormalizedLandmark]
  /// Detected 3D left-hand landmarks in world coordinates (meters).
  public let leftHandWorldLandmarks: [Landmark]
  /// Detected right-hand landmarks in normalized `[0.0, 1.0]` image coordinates.
  public let rightHandLandmarks: [NormalizedLandmark]
  /// Detected 3D right-hand landmarks in world coordinates (meters).
  public let rightHandWorldLandmarks: [Landmark]
  /// Timestamp in milliseconds of the input frame, or `0` in `.image` mode.
  public let timestampInMilliseconds: Int

  /// Creates a holistic landmarker result for a single frame.
  public init(
    faceLandmarks: [NormalizedLandmark], faceBlendshapes: Classifications? = nil,
    poseLandmarks: [NormalizedLandmark], poseWorldLandmarks: [Landmark],
    poseSegmentationMask: MPMask? = nil, leftHandLandmarks: [NormalizedLandmark],
    leftHandWorldLandmarks: [Landmark], rightHandLandmarks: [NormalizedLandmark],
    rightHandWorldLandmarks: [Landmark], timestampInMilliseconds: Int = 0
  ) {
    self.faceLandmarks = faceLandmarks
    self.faceBlendshapes = faceBlendshapes
    self.poseLandmarks = poseLandmarks
    self.poseWorldLandmarks = poseWorldLandmarks
    self.poseSegmentationMask = poseSegmentationMask
    self.leftHandLandmarks = leftHandLandmarks
    self.leftHandWorldLandmarks = leftHandWorldLandmarks
    self.rightHandLandmarks = rightHandLandmarks
    self.rightHandWorldLandmarks = rightHandWorldLandmarks
    self.timestampInMilliseconds = timestampInMilliseconds
  }

  package init(cResult: MpHolisticLandmarkerResult, timestampInMilliseconds: Int) throws {
    let blendshapes = Array(
      cArray: cResult.face_blendshapes.categories, count: cResult.face_blendshapes.categories_count,
      transform: ResultCategory.init)
    self.init(
      faceLandmarks: NormalizedLandmark.landmarks(from: cResult.face_landmarks),
      faceBlendshapes: blendshapes.isEmpty
        ? nil : Classifications(headIndex: 0, categories: blendshapes),
      poseLandmarks: NormalizedLandmark.landmarks(from: cResult.pose_landmarks),
      poseWorldLandmarks: Landmark.landmarks(from: cResult.pose_world_landmarks),
      poseSegmentationMask: try cResult.pose_segmentation_mask.map {
        try MPMask(copyingCImage: $0)
      },
      leftHandLandmarks: NormalizedLandmark.landmarks(from: cResult.left_hand_landmarks),
      leftHandWorldLandmarks: Landmark.landmarks(from: cResult.left_hand_world_landmarks),
      rightHandLandmarks: NormalizedLandmark.landmarks(from: cResult.right_hand_landmarks),
      rightHandWorldLandmarks: Landmark.landmarks(from: cResult.right_hand_world_landmarks),
      timestampInMilliseconds: timestampInMilliseconds)
  }
}
