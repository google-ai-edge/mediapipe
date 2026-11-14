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
package import third_party_mediapipe_tasks_c_components_containers_category
package import third_party_mediapipe_tasks_c_vision_hand_landmarker_hand_landmarker_result

#if !MEDIAPIPE_SINGLE_MODULE
  @_exported import MediaPipeTasksComponents
#endif

/// Indices (`0...20`) for the 21 anatomical hand keypoints emitted by `HandLandmarker` (`CMC` =
/// carpometacarpal, `MCP` = metacarpophalangeal, `IP` = interphalangeal, `PIP` = proximal
/// interphalangeal, `DIP` = distal interphalangeal).
public enum HandLandmark: Int, Sendable, Equatable, Hashable, CaseIterable {
  case wrist = 0
  case thumbCMC = 1
  case thumbMCP = 2
  case thumbIP = 3
  case thumbTip = 4
  case indexFingerMCP = 5
  case indexFingerPIP = 6
  case indexFingerDIP = 7
  case indexFingerTip = 8
  case middleFingerMCP = 9
  case middleFingerPIP = 10
  case middleFingerDIP = 11
  case middleFingerTip = 12
  case ringFingerMCP = 13
  case ringFingerPIP = 14
  case ringFingerDIP = 15
  case ringFingerTip = 16
  case pinkyMCP = 17
  case pinkyPIP = 18
  case pinkyDIP = 19
  case pinkyTip = 20
}

/// Hand landmarks, 3D world landmarks, and handedness classifications produced by `HandLandmarker`.
public struct HandLandmarkerResult: Sendable, Equatable, Hashable {
  /// Normalized image-space 21-point landmark lists, one per detected hand.
  public let landmarks: [[NormalizedLandmark]]
  /// Metric 3D world-space landmark lists (in meters), one per detected hand.
  public let worldLandmarks: [[Landmark]]
  /// Subject-perspective handedness classifications (`"Left"` / `"Right"`), one list per detected
  /// hand. Horizontally mirrored input images invert these labels.
  public let handedness: [[ResultCategory]]
  /// Frame timestamp in milliseconds (`0` in `.image` mode).
  public let timestampInMilliseconds: Int

  /// Creates a `HandLandmarkerResult` from per-hand landmarks, world landmarks, handedness, and
  /// timestamp.
  public init(
    landmarks: [[NormalizedLandmark]],
    worldLandmarks: [[Landmark]],
    handedness: [[ResultCategory]],
    timestampInMilliseconds: Int
  ) {
    self.landmarks = landmarks
    self.worldLandmarks = worldLandmarks
    self.handedness = handedness
    self.timestampInMilliseconds = timestampInMilliseconds
  }

  /// Creates a `HandLandmarkerResult` copied from a C `MpHandLandmarkerResult` struct.
  package init(cResult c: MpHandLandmarkerResult, timestampInMilliseconds: Int) {
    self.landmarks = Array(
      cArray: c.hand_landmarks, count: c.hand_landmarks_count,
      transform: NormalizedLandmark.landmarks(from:))
    self.worldLandmarks = Array(
      cArray: c.hand_world_landmarks, count: c.hand_world_landmarks_count,
      transform: Landmark.landmarks(from:))
    self.handedness = Array(cArray: c.handedness, count: c.handedness_count) { list in
      Array(cArray: list.categories, count: list.categories_count, transform: ResultCategory.init)
    }
    self.timestampInMilliseconds = timestampInMilliseconds
  }
}
