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
package import third_party_mediapipe_tasks_c_text_language_detector_language_detector_lib

#if !MEDIAPIPE_SINGLE_MODULE
  @_exported import MediaPipeTasksComponents
  @_exported import MediaPipeTasksCore
#endif

/// Configuration for initializing and filtering a `LanguageDetector` task.
public struct LanguageDetectorOptions: Sendable, Equatable, Hashable {
  /// Model asset source and hardware acceleration settings.
  public var baseOptions: BaseOptions

  /// BCP-47 locale for localized display names in model metadata.
  public var displayNamesLocale: String

  /// Maximum number of top-scored predictions to return (`< 0` returns all; `0` is invalid).
  public var maxResults: Int

  /// Minimum probability threshold overriding model metadata.
  public var scoreThreshold: Float

  /// Language codes to include exclusively (mutually exclusive with `categoryDenylist`).
  public var categoryAllowlist: [String]

  /// Language codes to exclude (mutually exclusive with `categoryAllowlist`).
  public var categoryDenylist: [String]

  /// Creates configuration options with the specified model and filtering settings.
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

  /// Invokes `body` with a C `MpLanguageDetectorOptions` struct whose pointer fields remain valid
  /// for the duration of the closure call.
  package func withCOptions<Result>(
    _ body: (MpLanguageDetectorOptions) throws -> Result
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
        let cOptions = MpLanguageDetectorOptions(
          base_options: cBaseOptions,
          classifier_options: cClassifierOptions
        )
        return try body(cOptions)
      }
    }
  }
}
