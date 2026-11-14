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

package import third_party_mediapipe_tasks_c_vision_interactive_segmenter_interactive_segmenter

/// The brush mode representing the intent of a user-drawn segmentation stroke.
public enum BrushMode: Sendable, Equatable, Hashable, CaseIterable {
  /// Unspecified brush mode (no-op).
  case unspecified

  /// Adds regions or objects of interest to the segmentation mask.
  case positive

  /// Excludes regions or objects from the segmentation mask.
  case negative

  /// Draws a boundary around an object of interest to add to the segmentation mask.
  case lasso

  /// The underlying C `MpBrushMode` representation.
  package var cBrushMode: MpBrushMode {
    switch self {
    case .unspecified: return MP_BRUSH_MODE_UNSPECIFIED
    case .positive: return MP_BRUSH_MODE_POSITIVE
    case .negative: return MP_BRUSH_MODE_NEGATIVE
    case .lasso: return MP_BRUSH_MODE_LASSO
    }
  }
}
