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
package import third_party_mediapipe_tasks_c_core_common
package import third_party_mediapipe_tasks_c_core_mp_status
package import third_party_mediapipe_tasks_c_vision_image_segmenter_image_segmenter_result
package import third_party_mediapipe_tasks_c_vision_interactive_segmenter_legacy_interactive_segmenter_legacy_lib

#if !MEDIAPIPE_SINGLE_MODULE
  @_exported import MediaPipeTasksCore
  @_exported import MediaPipeTasksImageSegmenter
  @_exported import MediaPipeTasksVisionCore
#endif

/// Segments the object indicated by a user-provided `RegionOfInterest` in single images, using a
/// TFLite model with mandatory TFLite Model Metadata.
///
/// - Note: This class is not thread-safe. Concurrent callers should serialize access (for example,
///   by isolating an instance inside a Swift `actor`).
public final class InteractiveSegmenterLegacy {
  /// Category labels from the model metadata, indexed by category mask value and confidence mask
  /// position, or empty if the model has no label map.
  public let labels: [String]

  private let cSegmenter: MpInteractiveSegmenterLegacyPtr
  private let runner: VisionTaskRunner

  /// Creates an interactive segmenter and initializes the underlying MediaPipe task graph.
  ///
  /// - Throws: `TasksError` if the model asset is missing or graph initialization fails.
  public init(options: InteractiveSegmenterLegacyOptions) throws {
    let runner = try VisionTaskRunner(
      baseOptions: options.baseOptions,
      runningMode: .image,
      roiAllowed: false,
      normRectAllowed: true
    )
    let cSegmenter: MpInteractiveSegmenterLegacyPtr = try options.withCOptions { cOptions in
      var segmenterPtr: MpInteractiveSegmenterLegacyPtr?
      try MpStatus.check(defaultMessage: "Failed to create InteractiveSegmenterLegacy.") {
        errorMessage in
        MpInteractiveSegmenterLegacyCreate(&cOptions, &segmenterPtr, &errorMessage)
      }
      guard let segmenterPtr else {
        throw TasksError.internalError(
          "MpInteractiveSegmenterLegacyCreate returned a null pointer.")
      }
      return segmenterPtr
    }
    let labels: [String]
    do {
      labels = try Self.labels(of: cSegmenter)
    } catch {
      _ = MpInteractiveSegmenterLegacyClose(cSegmenter, nil)
      throw error
    }
    self.cSegmenter = cSegmenter
    self.labels = labels
    self.runner = runner
  }

  /// Creates an interactive segmenter from a local model file path using default options.
  ///
  /// - Throws: `TasksError` if the model cannot be loaded or initialized.
  public convenience init(modelPath: String) throws {
    let baseOptions = BaseOptions(modelAssetPath: modelPath)
    try self.init(options: InteractiveSegmenterLegacyOptions(baseOptions: baseOptions))
  }

  deinit {
    _ = MpInteractiveSegmenterLegacyClose(cSegmenter, nil)
  }

  /// Segments the object in `image` indicated by `regionOfInterest`, honoring `image.orientation`
  /// and the optional rotation in `imageProcessingOptions` (region-of-interest cropping is not
  /// supported).
  ///
  /// - Throws: `TasksError` if the inputs are invalid or inference fails.
  public func segment(
    image: MPImage,
    regionOfInterest: RegionOfInterest,
    imageProcessingOptions: ImageProcessingOptions? = nil
  ) throws -> InteractiveSegmenterLegacyResult {
    var cResult = MpImageSegmenterResult()
    defer { MpInteractiveSegmenterLegacyCloseResult(&cResult) }
    try regionOfInterest.withCRegionOfInterest { cRegion in
      try runner.withPreparedInput(
        image: image, imageProcessingOptions: imageProcessingOptions, expectedMode: .image
      ) { cImage, cOptions in
        try MpStatus.check(defaultMessage: "Failed to segment image.") { errorMessage in
          MpInteractiveSegmenterLegacySegmentImage(
            cSegmenter, cImage, cRegion, cOptions, &cResult, &errorMessage)
        }
      }
    }
    return try InteractiveSegmenterLegacyResult(cResult: cResult, timestampInMilliseconds: 0)
  }

  private static func labels(of cSegmenter: MpInteractiveSegmenterLegacyPtr) throws -> [String] {
    var cLabels = MpStringList()
    defer { MpStringListFree(&cLabels) }
    try MpStatus.check(defaultMessage: "Failed to get InteractiveSegmenterLegacy labels.") {
      errorMessage in
      MpInteractiveSegmenterLegacyGetLabels(cSegmenter, &cLabels, &errorMessage)
    }
    guard let cStrings = cLabels.strings else { return [] }
    return (0..<Int(cLabels.num_strings)).map { cStrings[$0].map { String(cString: $0) } ?? "" }
  }
}
