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

/// A single classification category containing a label, optional display name, confidence score,
/// and label index.
public struct ResultCategory: Sendable, Equatable, Hashable {
  /// The index of the label in the corresponding label file, or `-1` if unset.
  public let index: Int

  /// Confidence score for this category.
  public let score: Float

  /// The optional category label read from the model metadata.
  public let categoryName: String?

  /// The optional localized display name for the label read from the model metadata.
  public let displayName: String?

  /// Creates a `ResultCategory` with the given index, score, category name, and display name.
  public init(
    index: Int, score: Float, categoryName: String? = nil, displayName: String? = nil
  ) {
    self.index = index
    self.score = score
    self.categoryName = categoryName
    self.displayName = displayName
  }

  /// Creates a `ResultCategory` copied from a C `MpCategory` struct.
  package init(cCategory: MpCategory) {
    self.index = Int(cCategory.index)
    self.score = cCategory.score
    self.categoryName = String(nonEmptyCString: cCategory.category_name)
    self.displayName = String(nonEmptyCString: cCategory.display_name)
  }
}

/// Alias matching the canonical MediaPipe `Category` container name.
public typealias Category = ResultCategory
