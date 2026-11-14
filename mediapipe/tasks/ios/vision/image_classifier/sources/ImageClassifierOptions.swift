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
package import third_party_mediapipe_tasks_c_vision_core_image
package import third_party_mediapipe_tasks_c_vision_image_classifier_image_classifier_lib

#if !MEDIAPIPE_SINGLE_MODULE
  @_exported import MediaPipeTasksComponents
  @_exported import MediaPipeTasksCore
  @_exported import MediaPipeTasksVisionCore
#endif

/// Configuration for initializing an `ImageClassifier`.
///
/// Not `Equatable` or `Hashable` because `liveStreamResultHandler` is a closure.
public struct ImageClassifierOptions: Sendable {
  /// Asynchronous callback receiving live-stream output and the frame timestamp in milliseconds.
  public typealias LiveStreamResultHandler =
    @Sendable (Result<ImageClassifierResult, any Error>, _ timestampInMilliseconds: Int) -> Void

  /// Model asset loading and hardware acceleration settings.
  public var baseOptions: BaseOptions
  /// Execution mode (`.image`, `.video`, or `.liveStream`).
  public var runningMode: VisionRunningMode
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
  /// Callback required when `runningMode == .liveStream` and disallowed in other modes.
  public var liveStreamResultHandler: LiveStreamResultHandler?

  /// Creates configuration options defaulting to `.image` mode.
  public init(
    baseOptions: BaseOptions = BaseOptions(),
    runningMode: VisionRunningMode = .image,
    displayNamesLocale: String = "en",
    maxResults: Int = -1,
    scoreThreshold: Float = 0.0,
    categoryAllowlist: [String] = [],
    categoryDenylist: [String] = [],
    liveStreamResultHandler: LiveStreamResultHandler? = nil
  ) {
    self.baseOptions = baseOptions
    self.runningMode = runningMode
    self.displayNamesLocale = displayNamesLocale
    self.maxResults = maxResults
    self.scoreThreshold = scoreThreshold
    self.categoryAllowlist = categoryAllowlist
    self.categoryDenylist = categoryDenylist
    self.liveStreamResultHandler = liveStreamResultHandler
  }

  package func withCOptions<Result>(
    cCallback: MpImageClassifierResultCallback?,
    _ body: (inout MpImageClassifierOptions) throws -> Result
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
        var cOptions = MpImageClassifierOptions(
          base_options: cBaseOptions,
          running_mode: runningMode.cRunningMode,
          classifier_options: cClassifierOptions,
          result_callback: cCallback
        )
        return try body(&cOptions)
      }
    }
  }
}
