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
package import third_party_mediapipe_tasks_c_vision_image_segmenter_image_segmenter_result

#if !MEDIAPIPE_SINGLE_MODULE
  @_exported import MediaPipeTasksCore
  @_exported import MediaPipeTasksVisionCore
#endif

/// Segmentation masks and quality scores produced by `ImageSegmenter`.
///
/// Not `Hashable` because `MPMask` is only `Equatable`.
public struct ImageSegmenterResult: Sendable, Equatable {
  /// Per-category `Float32` confidence masks, or `nil` if confidence masks were not requested.
  public let confidenceMasks: [MPMask]?

  /// `UInt8` mask of per-pixel category indices, or `nil` if a category mask was not requested.
  public let categoryMask: MPMask?

  /// Per-category mask quality scores in `0.0...1.0` (`1.0` when the model has no score output).
  public let qualityScores: [Float]

  /// Timestamp in milliseconds of the input frame, or `0` in `.image` mode.
  public let timestampInMilliseconds: Int

  /// Creates a segmentation result from masks, quality scores, and timestamp.
  public init(
    confidenceMasks: [MPMask]?,
    categoryMask: MPMask?,
    qualityScores: [Float],
    timestampInMilliseconds: Int
  ) {
    self.confidenceMasks = confidenceMasks
    self.categoryMask = categoryMask
    self.qualityScores = qualityScores
    self.timestampInMilliseconds = timestampInMilliseconds
  }

  /// Creates a result by deep-copying all masks and scores out of a C `MpImageSegmenterResult`,
  /// so the result stays valid after `MpImageSegmenterCloseResult`.
  package init(cResult: MpImageSegmenterResult, timestampInMilliseconds: Int) throws {
    var confidenceMasks: [MPMask]?
    if let cMasks = cResult.confidence_masks, cResult.confidence_masks_count > 0 {
      let count = Int(cResult.confidence_masks_count)
      confidenceMasks = try UnsafeBufferPointer(start: cMasks, count: count).map { cMask in
        guard let cMask else { throw TasksError.internalError("Null confidence mask.") }
        return try MPMask(copyingCImage: cMask)
      }
    }
    // `category_mask` is uninitialized unless `has_category_mask` is set.
    var categoryMask: MPMask?
    if cResult.has_category_mask != 0, let cMask = cResult.category_mask {
      categoryMask = try MPMask(copyingCImage: cMask)
    }
    let qualityScores =
      cResult.quality_scores.map {
        Array(UnsafeBufferPointer(start: $0, count: Int(cResult.quality_scores_count)))
      } ?? []
    self.init(
      confidenceMasks: confidenceMasks,
      categoryMask: categoryMask,
      qualityScores: qualityScores,
      timestampInMilliseconds: timestampInMilliseconds
    )
  }
}
