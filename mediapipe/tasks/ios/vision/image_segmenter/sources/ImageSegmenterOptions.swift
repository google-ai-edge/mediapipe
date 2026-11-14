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
package import third_party_mediapipe_tasks_c_vision_core_image
package import third_party_mediapipe_tasks_c_vision_image_segmenter_image_segmenter_lib

#if !MEDIAPIPE_SINGLE_MODULE
  @_exported import MediaPipeTasksCore
  @_exported import MediaPipeTasksVisionCore
#endif

/// Options for configuring an `ImageSegmenter` task.
///
/// Not `Equatable` or `Hashable` because `liveStreamResultHandler` is a closure.
public struct ImageSegmenterOptions: Sendable {
  /// Receives asynchronous `.liveStream` segmentation results (or errors) with their input
  /// timestamp on a private serial queue.
  public typealias LiveStreamResultHandler =
    @Sendable (Result<ImageSegmenterResult, any Error>, _ timestampInMilliseconds: Int) -> Void

  /// Base options for model loading and hardware acceleration delegation.
  public var baseOptions: BaseOptions

  /// Execution mode of the segmenter. `.liveStream` requires `liveStreamResultHandler`.
  public var runningMode: VisionRunningMode

  /// Locale used for display names from the TFLite Model Metadata.
  public var displayNamesLocale: String

  /// Whether to output per-category `Float32` confidence masks.
  public var outputConfidenceMasks: Bool

  /// Whether to output a `UInt8` category mask.
  public var outputCategoryMask: Bool

  /// Handler for `.liveStream` results. Must be `nil` in `.image` and `.video` modes.
  public var liveStreamResultHandler: LiveStreamResultHandler?

  /// Creates options for single-image segmentation that output confidence masks only.
  public init(
    baseOptions: BaseOptions = BaseOptions(),
    runningMode: VisionRunningMode = .image,
    displayNamesLocale: String = "en",
    outputConfidenceMasks: Bool = true,
    outputCategoryMask: Bool = false,
    liveStreamResultHandler: LiveStreamResultHandler? = nil
  ) {
    self.baseOptions = baseOptions
    self.runningMode = runningMode
    self.displayNamesLocale = displayNamesLocale
    self.outputConfidenceMasks = outputConfidenceMasks
    self.outputCategoryMask = outputCategoryMask
    self.liveStreamResultHandler = liveStreamResultHandler
  }

  /// Invokes `body` with a C `MpImageSegmenterOptions` struct whose pointer fields remain valid
  /// for the duration of the closure call.
  package func withCOptions<Result>(
    resultCallback: MpImageSegmenterResultCallback?,
    _ body: (inout MpImageSegmenterOptions) throws -> Result
  ) rethrows -> Result {
    try baseOptions.withCOptions { cBaseOptions in
      try displayNamesLocale.withCString { localePtr in
        var cOptions = MpImageSegmenterOptions()
        cOptions.base_options = cBaseOptions
        cOptions.running_mode = runningMode.cRunningMode
        cOptions.display_names_locale = localePtr
        cOptions.output_confidence_masks = outputConfidenceMasks
        cOptions.output_category_mask = outputCategoryMask
        cOptions.result_callback = resultCallback
        return try body(&cOptions)
      }
    }
  }
}
