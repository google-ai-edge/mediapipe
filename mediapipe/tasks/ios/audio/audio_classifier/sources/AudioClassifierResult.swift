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
package import third_party_mediapipe_tasks_c_audio_audio_classifier_audio_classifier_lib

#if !MEDIAPIPE_SINGLE_MODULE
  import MediaPipeTasksComponents
#endif

/// Predictions produced across one or more audio windows.
public struct AudioClassifierResult: Sendable, Equatable, Hashable {
  /// Per-window predictions ordered by start time.
  public let classificationResults: [ClassificationResult]

  /// Packet start time in milliseconds.
  public let timestampInMilliseconds: Int

  /// Creates a result from per-window predictions and packet start time.
  public init(classificationResults: [ClassificationResult], timestampInMilliseconds: Int) {
    self.classificationResults = classificationResults
    self.timestampInMilliseconds = timestampInMilliseconds
  }

  package init(cResult: MpAudioClassifierResult, timestampInMilliseconds: Int? = nil) {
    let results = Array(
      cArray: cResult.results, count: cResult.results_count, transform: ClassificationResult.init
    )
    self.classificationResults = results
    self.timestampInMilliseconds =
      timestampInMilliseconds ?? results.first?.timestampInMilliseconds ?? 0
  }
}
