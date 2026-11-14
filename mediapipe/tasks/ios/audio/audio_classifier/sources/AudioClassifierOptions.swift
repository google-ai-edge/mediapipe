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
  import MediaPipeTasksAudioCore
  import MediaPipeTasksCore
#endif

/// Configuration for an `AudioClassifier` instance.
///
/// Not `Equatable` or `Hashable` because `audioStreamResultHandler` is a closure.
public struct AudioClassifierOptions: Sendable {
  /// Asynchronous handler receiving stream classification results or runtime errors.
  public typealias AudioStreamResultHandler =
    @Sendable (Result<AudioClassifierResult, any Error>) -> Void

  /// Model asset and hardware accelerator settings.
  public var baseOptions: BaseOptions
  /// Execution mode selecting standalone clips or a continuous stream.
  public var runningMode: AudioRunningMode
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
  /// Asynchronous handler required for `.audioStream` and disallowed for `.audioClips`.
  public var audioStreamResultHandler: AudioStreamResultHandler?

  /// Creates configuration with model, mode, filtering, and optional stream handler.
  public init(
    baseOptions: BaseOptions = BaseOptions(),
    runningMode: AudioRunningMode = .audioClips,
    displayNamesLocale: String = "en",
    maxResults: Int = -1,
    scoreThreshold: Float = 0.0,
    categoryAllowlist: [String] = [],
    categoryDenylist: [String] = [],
    audioStreamResultHandler: AudioStreamResultHandler? = nil
  ) {
    self.baseOptions = baseOptions
    self.runningMode = runningMode
    self.displayNamesLocale = displayNamesLocale
    self.maxResults = maxResults
    self.scoreThreshold = scoreThreshold
    self.categoryAllowlist = categoryAllowlist
    self.categoryDenylist = categoryDenylist
    self.audioStreamResultHandler = audioStreamResultHandler
  }

  package func withCOptions<R>(
    cCallback: MpAudioClassifierResultCallbackFn?,
    _ body: (inout MpAudioClassifierOptions) throws -> R
  ) rethrows -> R {
    let classifierOptions = ClassifierOptions(
      displayNamesLocale: displayNamesLocale,
      maxResults: maxResults,
      scoreThreshold: scoreThreshold,
      categoryAllowlist: categoryAllowlist,
      categoryDenylist: categoryDenylist
    )
    return try baseOptions.withCOptions { cBase in
      try classifierOptions.withCOptions { cClassifier in
        var cOptions = MpAudioClassifierOptions(
          base_options: cBase, classifier_options: cClassifier,
          running_mode: runningMode.cRunningMode, result_callback: cCallback)
        return try body(&cOptions)
      }
    }
  }
}
