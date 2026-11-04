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

/// Predicted categories for a single classifier output head.
public struct Classifications: Sendable, Equatable, Hashable {
  /// The index of the classifier head these categories refer to.
  public let headIndex: Int

  /// The optional tensor metadata name of the classifier head.
  public let headName: String?

  /// The predicted categories for this classifier head, typically sorted by descending score.
  public let categories: [ResultCategory]

  /// Creates `Classifications` for the specified classifier head index, optional name, and
  /// predicted categories.
  public init(headIndex: Int, headName: String? = nil, categories: [ResultCategory]) {
    self.headIndex = headIndex
    self.headName = headName
    self.categories = categories
  }

  /// Creates `Classifications` copied from a C `MpClassifications` struct.
  package init(cClassifications: MpClassifications) {
    self.headIndex = Int(cClassifications.head_index)
    self.headName = cClassifications.head_name.map { String(cString: $0) }
    self.categories = Array(
      cArray: cClassifications.categories,
      count: cClassifications.categories_count,
      transform: ResultCategory.init
    )
  }
}

/// Classification results produced across all output heads of a classification model.
public struct ClassificationResult: Sendable, Equatable, Hashable {
  /// Classification predictions for each head of the model.
  public let classifications: [Classifications]

  /// Timestamp in milliseconds of the start of the data chunk corresponding to these results, or
  /// `nil` when no timestamp is associated with the result.
  public let timestampInMilliseconds: Int?

  /// Creates a `ClassificationResult` with the given head classifications and optional timestamp.
  public init(classifications: [Classifications], timestampInMilliseconds: Int? = nil) {
    self.classifications = classifications
    self.timestampInMilliseconds = timestampInMilliseconds
  }

  /// Creates a `ClassificationResult` copied from a C `MpClassificationResult` struct.
  package init(cResult: MpClassificationResult) {
    self.classifications = Array(
      cArray: cResult.classifications,
      count: cResult.classifications_count,
      transform: Classifications.init
    )
    self.timestampInMilliseconds = cResult.has_timestamp_ms ? Int(cResult.timestamp_ms) : nil
  }
}
