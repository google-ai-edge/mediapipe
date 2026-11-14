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
package import third_party_mediapipe_tasks_c_vision_object_detector_object_detector_lib

#if !MEDIAPIPE_SINGLE_MODULE
  @_exported import MediaPipeTasksCore
  @_exported import MediaPipeTasksVisionCore
#endif

/// Configuration options for the MediaPipe `ObjectDetector` task.
///
/// Conforms to `Sendable` (`Equatable` and `Hashable` are omitted because
/// `liveStreamResultHandler` is an escaping closure).
public struct ObjectDetectorOptions: Sendable {
  /// Asynchronous callback invoked on a serial queue with live-stream detection results.
  public typealias LiveStreamResultHandler =
    @Sendable (Result<ObjectDetectorResult, any Error>, _ timestampInMilliseconds: Int) -> Void

  /// Model asset and hardware delegate configuration.
  public var baseOptions: BaseOptions

  /// Execution mode (`image`, `video`, or `liveStream`).
  public var runningMode: VisionRunningMode

  /// BCP-47 locale for localized category display names in model metadata.
  public var displayNamesLocale: String

  /// Maximum number of top-scored detections to return (`< 0` returns all; `0` is invalid).
  public var maxResults: Int

  /// Minimum confidence score threshold overriding model metadata.
  public var scoreThreshold: Float

  /// Category names to include exclusively (mutually exclusive with `categoryDenylist`).
  public var categoryAllowlist: [String]

  /// Category names to exclude (mutually exclusive with `categoryAllowlist`).
  public var categoryDenylist: [String]

  /// Asynchronous delivery closure required when `runningMode` is `.liveStream`.
  public var liveStreamResultHandler: LiveStreamResultHandler?

  /// Creates `ObjectDetectorOptions` with the specified configuration.
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

  /// Invokes `body` with a mutable C `MpObjectDetectorOptions` pointer whose nested C buffers
  /// remain valid for the duration of the closure call.
  package func withCOptions<Result>(
    cCallback: MpObjectDetectorResultCallback?,
    _ body: (UnsafeMutablePointer<MpObjectDetectorOptions>) throws -> Result
  ) throws -> Result {
    guard let cMaxResults = Int32(exactly: maxResults) else {
      throw TasksError.invalidArgument("maxResults (\(maxResults)) exceeds Int32 range.")
    }
    return try baseOptions.withCOptions { cBaseOptions in
      try displayNamesLocale.withCString { localePtr in
        try categoryAllowlist.withCStringArray { allowlistPtr, allowlistCount in
          try categoryDenylist.withCStringArray { denylistPtr, denylistCount in
            var cOptions = MpObjectDetectorOptions()
            cOptions.base_options = cBaseOptions
            cOptions.running_mode = runningMode.cRunningMode
            cOptions.display_names_locale = localePtr
            cOptions.max_results = cMaxResults
            cOptions.score_threshold = scoreThreshold
            cOptions.category_allowlist = allowlistPtr
            cOptions.category_allowlist_count = allowlistCount
            cOptions.category_denylist = denylistPtr
            cOptions.category_denylist_count = denylistCount
            cOptions.result_callback = cCallback
            return try withUnsafeMutablePointer(to: &cOptions, body)
          }
        }
      }
    }
  }
}
