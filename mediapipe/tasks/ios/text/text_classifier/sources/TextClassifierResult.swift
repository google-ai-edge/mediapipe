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
package import third_party_mediapipe_tasks_c_components_containers_classification_result
package import third_party_mediapipe_tasks_c_text_text_classifier_text_classifier_lib

#if !MEDIAPIPE_SINGLE_MODULE
  @_exported import MediaPipeTasksComponents
  @_exported import MediaPipeTasksCore
#endif

/// Classification output produced by `TextClassifier` for an input string.
public struct TextClassifierResult: Sendable, Equatable, Hashable {
  /// Per-head category predictions produced by the classification model.
  public let classificationResult: ClassificationResult

  /// Timestamp in milliseconds associated with this result.
  public let timestampInMilliseconds: Int

  /// Creates a result from the per-head classification predictions and associated timestamp.
  public init(
    classificationResult: ClassificationResult,
    timestampInMilliseconds: Int = 0
  ) {
    self.classificationResult = classificationResult
    self.timestampInMilliseconds = timestampInMilliseconds
  }

  /// Creates a `TextClassifierResult` copied from a C `MpTextClassifierResult` struct.
  package init(cResult: MpTextClassifierResult) {
    self.classificationResult = ClassificationResult(cResult: cResult)
    self.timestampInMilliseconds = cResult.has_timestamp_ms ? Int(cResult.timestamp_ms) : 0
  }
}
