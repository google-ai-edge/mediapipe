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

#include "mediapipe/examples/desktop/autoflip/quality/frame_crop_region_computer.h"

#include <cmath>

#include "mediapipe/examples/desktop/autoflip/quality/utils.h"
#include "mediapipe/framework/port/ret_check.h"

namespace mediapipe {
namespace autoflip {

::mediapipe::Status FrameCropRegionComputer::ExpandSegmentUnderConstraint(
    const Segment& segment_to_add, const Segment& base_segment,
    const int max_length, Segment* combined_segment,
    CoverType* cover_type) const {
  RET_CHECK(combined_segment != nullptr) << "Combined segment is null.";
  RET_CHECK(cover_type != nullptr) << "Cover type is null.";

  const LeftPoint segment_to_add_left = segment_to_add.first;
  const RightPoint segment_to_add_right = segment_to_add.second;
  RET_CHECK(segment_to_add_right >= segment_to_add_left)
      << "Invalid segment to add.";
  const LeftPoint base_segment_left = base_segment.first;
  const RightPoint base_segment_right = base_segment.second;
  RET_CHECK(base_segment_right >= base_segment_left) << "Invalid base segment.";
  const int base_length = base_segment_right - base_segment_left;
  RET_CHECK(base_length <= max_length)
      << "Base segment length exceeds max length.";

  const int segment_to_add_length = segment_to_add_right - segment_to_add_left;
  const int max_leftout_amount =
      std::ceil((1.0 - options_.non_required_region_min_coverage_fraction()) *
                segment_to_add_length / 2);
  const LeftPoint min_coverage_segment_to_add_left =
      segment_to_add_left + max_leftout_amount;
  const LeftPoint min_coverage_segment_to_add_right =
      segment_to_add_right - max_leftout_amount;

  LeftPoint combined_segment_left =
      std::min(segment_to_add_left, base_segment_left);
  RightPoint combined_segment_right =
      std::max(segment_to_add_right, base_segment_right);

  LeftPoint min_coverage_combined_segment_left =
      std::min(min_coverage_segment_to_add_left, base_segment_left);
  RightPoint min_coverage_combined_segment_right =
      std::max(min_coverage_segment_to_add_right, base_segment_right);

  if ((combined_segment_right - combined_segment_left) <= max_length) {
    *cover_type = FULLY_COVERED;
  } else if (min_coverage_combined_segment_right -
                 min_coverage_combined_segment_left <=
             max_length) {
    *cover_type = PARTIALLY_COVERED;
    combined_segment_left = min_coverage_combined_segment_left;
    combined_segment_right = min_coverage_combined_segment_right;
  } else {
    *cover_type = NOT_COVERED;
    combined_segment_left = base_segment_left;
    combined_segment_right = base_segment_right;
  }

  *combined_segment =
      std::make_pair(combined_segment_left, combined_segment_right);
  return ::mediapipe::OkStatus();
}

::mediapipe::Status FrameCropRegionComputer::ExpandRectUnderConstraints(
    const Rect& rect_to_add, const int max_width, const int max_height,
    Rect* base_rect, CoverType* cover_type) const {
  RET_CHECK(base_rect != nullptr) << "Base rect is null.";
  RET_CHECK(cover_type != nullptr) << "Cover type is null.";
  RET_CHECK(base_rect->width() <= max_width &&
            base_rect->height() <= max_height)
      << "Base rect already exceeds target size.";

  const LeftPoint rect_to_add_left = rect_to_add.x();
  const RightPoint rect_to_add_right = rect_to_add.x() + rect_to_add.width();
  const LeftPoint rect_to_add_top = rect_to_add.y();
  const RightPoint rect_to_add_bottom = rect_to_add.y() + rect_to_add.height();
  const LeftPoint base_rect_left = base_rect->x();
  const RightPoint base_rect_right = base_rect->x() + base_rect->width();
  const LeftPoint base_rect_top = base_rect->y();
  const RightPoint base_rect_bottom = base_rect->y() + base_rect->height();

  Segment horizontal_combined_segment, vertical_combined_segment;
  CoverType horizontal_cover_type, vertical_cover_type;
  const auto horizontal_status = ExpandSegmentUnderConstraint(
      std::make_pair(rect_to_add_left, rect_to_add_right),
      std::make_pair(base_rect_left, base_rect_right), max_width,
      &horizontal_combined_segment, &horizontal_cover_type);
  MP_RETURN_IF_ERROR(horizontal_status);
  const auto vertical_status = ExpandSegmentUnderConstraint(
      std::make_pair(rect_to_add_top, rect_to_add_bottom),
      std::make_pair(base_rect_top, base_rect_bottom), max_height,
      &vertical_combined_segment, &vertical_cover_type);
  MP_RETURN_IF_ERROR(vertical_status);

  if (horizontal_cover_type == NOT_COVERED ||
      vertical_cover_type == NOT_COVERED) {
    // Gives up if the segment is not covered in either direction.
    *cover_type = NOT_COVERED;
  } else {
    // Tries to (partially) cover the new rect to be added.
    base_rect->set_x(horizontal_combined_segment.first);
    base_rect->set_y(vertical_combined_segment.first);
    base_rect->set_width(horizontal_combined_segment.second -
                         horizontal_combined_segment.first);
    base_rect->set_height(vertical_combined_segment.second -
                          vertical_combined_segment.first);
    if (horizontal_cover_type == FULLY_COVERED &&
        vertical_cover_type == FULLY_COVERED) {
      *cover_type = FULLY_COVERED;
    } else {
      *cover_type = PARTIALLY_COVERED;
    }
  }

  return ::mediapipe::OkStatus();
}

void FrameCropRegionComputer::UpdateCropRegionScore(
    const KeyFrameCropOptions::ScoreAggregationType score_aggregation_type,
    const float feature_score, const bool is_required,
    float* crop_region_score) {
  if (feature_score < 0.0) {
    LOG(WARNING) << "Ignoring negative score";
    return;
  }

  switch (score_aggregation_type) {
    case KeyFrameCropOptions::MAXIMUM: {
      *crop_region_score = std::max(feature_score, *crop_region_score);
      break;
    }
    case KeyFrameCropOptions::SUM_REQUIRED: {
      if (is_required) {
        *crop_region_score += feature_score;
      }
      break;
    }
    case KeyFrameCropOptions::SUM_ALL: {
      *crop_region_score += feature_score;
      break;
    }
    case KeyFrameCropOptions::CONSTANT: {
      *crop_region_score = 1.0;
      break;
    }
    default: {
      LOG(WARNING) << "Unknown CropRegionScoreType " << score_aggregation_type;
      break;
    }
  }
}

::mediapipe::Status FrameCropRegionComputer::ComputeFrameCropRegion(
    const KeyFrameInfo& frame_info, KeyFrameCropResult* crop_result) const {
  RET_CHECK(crop_result != nullptr) << "KeyFrameCropResult is null.";

  // Sorts required and non-required regions.
  std::vector<SalientRegion> required_regions, non_required_regions;
  const auto sort_status = SortDetections(
      frame_info.detections(), &required_regions, &non_required_regions);
  MP_RETURN_IF_ERROR(sort_status);

  int target_width = options_.target_width();
  int target_height = options_.target_height();
  auto* region = crop_result->mutable_region();
  RET_CHECK(region != nullptr) << "Crop region is null.";

  bool crop_region_is_empty = true;
  float crop_region_score = 0.0;

  // Gets union of all required regions.
  for (int i = 0; i < required_regions.size(); ++i) {
    const Rect& required_region = required_regions[i].location();
    if (crop_region_is_empty) {
      *region = required_region;
      crop_region_is_empty = false;
    } else {
      RectUnion(required_region, region);
    }
    UpdateCropRegionScore(options_.score_aggregation_type(),
                          required_regions[i].score(), true,
                          &crop_region_score);
  }
  crop_result->set_required_region_is_empty(crop_region_is_empty);
  if (!crop_region_is_empty) {
    *crop_result->mutable_required_region() = *region;
    crop_result->set_are_required_regions_covered_in_target_size(
        region->width() <= target_width && region->height() <= target_height);
    target_width = std::max(target_width, region->width());
    target_height = std::max(target_height, region->height());
  } else {
    crop_result->set_are_required_regions_covered_in_target_size(true);
  }

  // Tries to fit non-required regions.
  int num_covered = 0;
  for (int i = 0; i < non_required_regions.size(); ++i) {
    const Rect& non_required_region = non_required_regions[i].location();
    CoverType cover_type = NOT_COVERED;
    if (crop_region_is_empty) {
      // If the crop region is empty, tries to expand an empty base region
      // at the center of this region to include itself.
      region->set_x(non_required_region.x() + non_required_region.width() / 2);
      region->set_y(non_required_region.y() + non_required_region.height() / 2);
      region->set_width(0);
      region->set_height(0);
      MP_RETURN_IF_ERROR(ExpandRectUnderConstraints(non_required_region,
                                                    target_width, target_height,
                                                    region, &cover_type));
      if (cover_type != NOT_COVERED) {
        crop_region_is_empty = false;
      }
    } else {
      // Otherwise tries to expand the crop region to cover the non-required
      // region under target size constraint.
      MP_RETURN_IF_ERROR(ExpandRectUnderConstraints(non_required_region,
                                                    target_width, target_height,
                                                    region, &cover_type));
    }

    // Updates number of covered non-required regions and score.
    if (cover_type == FULLY_COVERED) {
      num_covered++;
      UpdateCropRegionScore(options_.score_aggregation_type(),
                            non_required_regions[i].score(), false,
                            &crop_region_score);
    }
  }

  const float fraction_covered =
      non_required_regions.empty()
          ? 0.0
          : static_cast<float>(num_covered) / non_required_regions.size();
  crop_result->set_fraction_non_required_covered(fraction_covered);

  crop_result->set_region_is_empty(crop_region_is_empty);
  crop_result->set_region_score(crop_region_score);
  return ::mediapipe::OkStatus();
}

}  // namespace autoflip
}  // namespace mediapipe
