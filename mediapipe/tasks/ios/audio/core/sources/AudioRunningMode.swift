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
package import third_party_mediapipe_tasks_c_audio_core_common

/// Execution modes for MediaPipe Audio Tasks.
@frozen public enum AudioRunningMode: Int, Sendable, Equatable, Hashable, CaseIterable {
  /// Runs inference on independent, standalone audio clips.
  case audioClips = 1

  /// Runs inference on a continuous audio stream such as microphone input.
  case audioStream = 2

  /// Human-readable display name for this running mode.
  public var displayName: String {
    switch self {
    case .audioClips:
      return "Audio Clips"
    case .audioStream:
      return "Audio Stream"
    }
  }

  /// The underlying C running mode representation.
  package var cRunningMode: MpAudioRunningMode {
    switch self {
    case .audioClips:
      return kMpAudioRunningModeAudioClips
    case .audioStream:
      return kMpAudioRunningModeAudioStream
    }
  }
}
