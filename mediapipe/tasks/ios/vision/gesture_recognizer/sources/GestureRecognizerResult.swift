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
package import third_party_mediapipe_tasks_c_components_containers_landmark
package import third_party_mediapipe_tasks_c_vision_gesture_recognizer_gesture_recognizer_result

#if !MEDIAPIPE_SINGLE_MODULE
  @_exported import MediaPipeTasksComponents
  @_exported import MediaPipeTasksCore
  @_exported import MediaPipeTasksVisionCore
#endif

/// Hand gesture, handedness, and 3D landmark predictions produced by `GestureRecognizer`.
public struct GestureRecognizerResult: Sendable, Equatable, Hashable {
  /// Recognized gestures per detected hand, sorted by descending score (`index` is always `-1`).
  public let gestures: [[ResultCategory]]
  /// Subject-relative handedness (`"Left"` / `"Right"`) per detected hand.
  public let handedness: [[ResultCategory]]
  /// 21 normalized image-space hand landmarks per detected hand.
  public let landmarks: [[NormalizedLandmark]]
  /// 21 metric world-space hand landmarks (in meters) per detected hand.
  public let worldLandmarks: [[Landmark]]
  /// Frame timestamp in milliseconds, or `0` in `.image` mode.
  public let timestampInMilliseconds: Int

  /// Creates a gesture recognition result for a single frame.
  public init(
    gestures: [[ResultCategory]], handedness: [[ResultCategory]],
    landmarks: [[NormalizedLandmark]], worldLandmarks: [[Landmark]], timestampInMilliseconds: Int
  ) {
    self.gestures = gestures
    self.handedness = handedness
    self.landmarks = landmarks
    self.worldLandmarks = worldLandmarks
    self.timestampInMilliseconds = timestampInMilliseconds
  }

  package init(cResult: MpGestureRecognizerResult, timestampInMilliseconds: Int) {
    self.gestures = Array(cArray: cResult.gestures, count: cResult.gestures_count) { cList in
      Array(cArray: cList.categories, count: cList.categories_count) { c in
        ResultCategory(
          index: -1, score: c.score, categoryName: c.category_name.map { String(cString: $0) },
          displayName: c.display_name.map { String(cString: $0) })
      }
    }
    self.handedness = Array(cArray: cResult.handedness, count: cResult.handedness_count) {
      Array(cArray: $0.categories, count: $0.categories_count, transform: ResultCategory.init)
    }
    self.landmarks = Array(
      cArray: cResult.hand_landmarks, count: cResult.hand_landmarks_count,
      transform: NormalizedLandmark.landmarks)
    self.worldLandmarks = Array(
      cArray: cResult.hand_world_landmarks, count: cResult.hand_world_landmarks_count,
      transform: Landmark.landmarks)
    self.timestampInMilliseconds = timestampInMilliseconds
  }
}
