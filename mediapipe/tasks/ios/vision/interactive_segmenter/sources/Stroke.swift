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

#if !MEDIAPIPE_SINGLE_MODULE
  @_exported import MediaPipeTasksComponents
#endif

/// A single user-drawn stroke for interactive image segmentation.
public struct Stroke: Sendable, Equatable, Hashable {
  /// The sequence of normalized keypoints forming the stroke path.
  public var points: [NormalizedKeypoint]

  /// The brush mode indicating how this stroke affects the segmentation mask.
  public var brushMode: BrushMode

  /// Whether the user has finished drawing this stroke.
  public var isCompleted: Bool

  /// Creates an interactive segmentation stroke from normalized `[0, 1]` image-space `points`,
  /// `brushMode`, and completion state (`true` once the gesture ends).
  public init(
    points: [NormalizedKeypoint],
    brushMode: BrushMode,
    isCompleted: Bool = true
  ) {
    self.points = points
    self.brushMode = brushMode
    self.isCompleted = isCompleted
  }
}
