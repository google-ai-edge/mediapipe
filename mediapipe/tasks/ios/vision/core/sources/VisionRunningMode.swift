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

package import third_party_mediapipe_tasks_c_vision_core_image

/// Execution modes for MediaPipe Vision Tasks.
@frozen public enum VisionRunningMode: Int, Sendable, Equatable, Hashable, CaseIterable {
  /// Runs inference on single, standalone image inputs.
  case image = 1

  /// Runs inference on sequential decoded frames of a video.
  case video = 2

  /// Runs inference on a live stream of camera frames with asynchronous result delivery.
  case liveStream = 3

  /// Human-readable display name for this running mode.
  public var displayName: String {
    switch self {
    case .image:
      return "Image"
    case .video:
      return "Video"
    case .liveStream:
      return "Live Stream"
    }
  }

  /// The underlying C running mode representation.
  package var cRunningMode: MpRunningMode {
    switch self {
    case .image:
      return MP_RUNNING_MODE_IMAGE
    case .video:
      return MP_RUNNING_MODE_VIDEO
    case .liveStream:
      return MP_RUNNING_MODE_LIVE_STREAM
    }
  }
}
