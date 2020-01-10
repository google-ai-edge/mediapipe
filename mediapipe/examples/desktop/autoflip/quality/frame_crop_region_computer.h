// Copyright 2019 The MediaPipe Authors.
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

#ifndef MEDIAPIPE_EXAMPLES_DESKTOP_AUTOFLIP_QUALITY_FRAME_CROP_REGION_COMPUTER_H_
#define MEDIAPIPE_EXAMPLES_DESKTOP_AUTOFLIP_QUALITY_FRAME_CROP_REGION_COMPUTER_H_

#include <vector>

#include "mediapipe/examples/desktop/autoflip/autoflip_messages.pb.h"
#include "mediapipe/examples/desktop/autoflip/quality/cropping.pb.h"
#include "mediapipe/framework/port/status.h"

namespace mediapipe {
namespace autoflip {

// This class computes per-frame crop regions based on crop frame options.
// It aggregates required regions and then tries to fit in non-required regions
// with best effort. It does not make use of static features.
class FrameCropRegionComputer {
 public:
  FrameCropRegionComputer() = delete;

  explicit FrameCropRegionComputer(
      const KeyFrameCropOptions& crop_frame_options)
      : options_(crop_frame_options) {}

  ~FrameCropRegionComputer() {}

  // Computes the crop region for the key frame using the crop options. The crop
  // region covers all the required regions, and attempts to cover the
  // non-required regions with best effort. Note: this function does not
  // consider static features, and simply tries to fit the detected features
  // within the target frame size. The score of the crop region is aggregated
  // from individual feature scores given the score aggregation type.
  ::mediapipe::Status ComputeFrameCropRegion(
      const KeyFrameInfo& frame_info, KeyFrameCropResult* crop_result) const;

 protected:
  // A segment is a 1-d object defined by its left and right point.
  using LeftPoint = int;
  using RightPoint = int;
  using Segment = std::pair<LeftPoint, RightPoint>;
  // How much a segment is covered in the combined segment.
  enum CoverType {
    FULLY_COVERED = 1,
    PARTIALLY_COVERED = 2,
    NOT_COVERED = 3,
  };
  // Expands a base segment to cover a segment to be added given maximum length
  // constraint. The operation is best-effort. The resulting enlarged segment is
  // set in the returned combined segment. Returns a CoverType to indicate the
  // coverage of the segment to be added in the combined segment.
  // There are 3 cases:
  //   case 1: the length of the union of the two segments is not larger than
  //           the maximum length.
  //           In this case the combined segment is simply the union, and cover
  //           type is FULLY_COVERED.
  //   case 2: the union of the two segments exceeds the maximum length, but the
  //           union of the base segment and required minimum centered fraction
  //           of the new segment fits in the maximum length.
  //           In this case the combined segment is this latter union, and cover
  //           type is PARTIALLY_COVERED.
  //   case 3: the union of the base segment and required minimum centered
  //           fraction of the new segment exceeds the maximum length.
  //           In this case the combined segment is the base segment, and cover
  //           type is NOT_COVERED.
  ::mediapipe::Status ExpandSegmentUnderConstraint(
      const Segment& segment_to_add, const Segment& base_segment,
      const int max_length, Segment* combined_segment,
      CoverType* cover_type) const;

  // Expands a base rectangle to cover a new rectangle to be added under width
  // and height constraints. The operation is best-effort. It considers
  // horizontal and vertical directions separately, using the
  // ExpandSegmentUnderConstraint function for each direction. The cover type is
  // FULLY_COVERED if the new rectangle is fully covered in both directions,
  // PARTIALLY_COVERED if it is at least partially covered in both directions,
  // and NOT_COVERED if it is not covered in either direction.
  ::mediapipe::Status ExpandRectUnderConstraints(const Rect& rect_to_add,
                                                 const int max_width,
                                                 const int max_height,
                                                 Rect* base_rect,
                                                 CoverType* cover_type) const;

  // Updates crop region score given current feature score, whether the feature
  // is required, and the score aggregation type. Ignores negative scores.
  static void UpdateCropRegionScore(
      const KeyFrameCropOptions::ScoreAggregationType score_aggregation_type,
      const float feature_score, const bool is_required,
      float* crop_region_score);

 private:
  // Crop frame options.
  KeyFrameCropOptions options_;
};
}  // namespace autoflip
}  // namespace mediapipe

#endif  // MEDIAPIPE_EXAMPLES_DESKTOP_AUTOFLIP_QUALITY_FRAME_CROP_REGION_COMPUTER_H_
