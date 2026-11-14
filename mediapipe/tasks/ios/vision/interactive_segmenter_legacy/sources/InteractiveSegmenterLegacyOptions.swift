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
package import third_party_mediapipe_tasks_c_vision_interactive_segmenter_legacy_interactive_segmenter_legacy_lib

#if !MEDIAPIPE_SINGLE_MODULE
  @_exported import MediaPipeTasksCore
#endif

/// Options for configuring an `InteractiveSegmenterLegacy` task.
public struct InteractiveSegmenterLegacyOptions: Sendable, Equatable, Hashable {
  /// Base options for model loading and hardware acceleration delegation.
  public var baseOptions: BaseOptions

  /// Whether to output per-category `Float32` confidence masks.
  public var outputConfidenceMasks: Bool

  /// Whether to output a `UInt8` category mask.
  public var outputCategoryMask: Bool

  /// Locale used for display names from the TFLite Model Metadata.
  public var displayNamesLocale: String

  /// Creates options that output confidence masks only.
  public init(
    baseOptions: BaseOptions = BaseOptions(),
    outputConfidenceMasks: Bool = true,
    outputCategoryMask: Bool = false,
    displayNamesLocale: String = "en"
  ) {
    self.baseOptions = baseOptions
    self.outputConfidenceMasks = outputConfidenceMasks
    self.outputCategoryMask = outputCategoryMask
    self.displayNamesLocale = displayNamesLocale
  }

  /// Invokes `body` with a C `MpInteractiveSegmenterLegacyOptions` struct whose pointer fields
  /// remain valid for the duration of the closure call.
  package func withCOptions<Result>(
    _ body: (inout MpInteractiveSegmenterLegacyOptions) throws -> Result
  ) rethrows -> Result {
    try baseOptions.withCOptions { cBaseOptions in
      try displayNamesLocale.withCString { cLocale in
        var cOptions = MpInteractiveSegmenterLegacyOptions()
        cOptions.base_options = cBaseOptions
        cOptions.display_names_locale = cLocale
        cOptions.output_confidence_masks = outputConfidenceMasks
        cOptions.output_category_mask = outputCategoryMask
        return try body(&cOptions)
      }
    }
  }
}
