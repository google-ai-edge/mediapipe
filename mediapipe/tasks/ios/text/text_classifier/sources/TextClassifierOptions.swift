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
package import third_party_mediapipe_tasks_c_components_processors_classifier_options
package import third_party_mediapipe_tasks_c_core_base_options
package import third_party_mediapipe_tasks_c_text_text_classifier_text_classifier_lib

#if !MEDIAPIPE_SINGLE_MODULE
  @_exported import MediaPipeTasksComponents
  @_exported import MediaPipeTasksCore
#endif

/// Configuration options for setting up a `TextClassifier` task.
public struct TextClassifierOptions: Sendable, Equatable, Hashable {
  /// Base options for model asset loading and hardware acceleration delegation.
  public var baseOptions: BaseOptions

  /// BCP-47 locale for localized category display names in model metadata.
  public var displayNamesLocale: String

  /// Maximum number of top-scored categories to return (`< 0` returns all; `0` is invalid).
  public var maxResults: Int

  /// Minimum confidence score threshold overriding model metadata.
  public var scoreThreshold: Float

  /// Category names to include exclusively (mutually exclusive with `categoryDenylist`).
  public var categoryAllowlist: [String]

  /// Category names to exclude (mutually exclusive with `categoryAllowlist`).
  public var categoryDenylist: [String]

  /// Creates options with default model loading and classification filtering settings.
  public init(
    baseOptions: BaseOptions = BaseOptions(),
    displayNamesLocale: String = "en",
    maxResults: Int = -1,
    scoreThreshold: Float = 0.0,
    categoryAllowlist: [String] = [],
    categoryDenylist: [String] = []
  ) {
    self.baseOptions = baseOptions
    self.displayNamesLocale = displayNamesLocale
    self.maxResults = maxResults
    self.scoreThreshold = scoreThreshold
    self.categoryAllowlist = categoryAllowlist
    self.categoryDenylist = categoryDenylist
  }

  /// Invokes `body` with a C `MpTextClassifierOptions` struct whose pointer fields remain valid
  /// for the duration of the closure call.
  package func withCOptions<Result>(
    _ body: (MpTextClassifierOptions) throws -> Result
  ) rethrows -> Result {
    let classifierOptions = ClassifierOptions(
      displayNamesLocale: displayNamesLocale,
      maxResults: maxResults,
      scoreThreshold: scoreThreshold,
      categoryAllowlist: categoryAllowlist,
      categoryDenylist: categoryDenylist
    )
    return try baseOptions.withCOptions { cBaseOptions in
      try classifierOptions.withCOptions { cClassifierOptions in
        let cOptions = MpTextClassifierOptions(
          base_options: cBaseOptions,
          classifier_options: cClassifierOptions
        )
        return try body(cOptions)
      }
    }
  }
}
