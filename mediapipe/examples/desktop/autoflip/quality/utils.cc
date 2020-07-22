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

#include "mediapipe/examples/desktop/autoflip/quality/utils.h"

#include <math.h>

#include <algorithm>
#include <utility>

#include "absl/memory/memory.h"
#include "mediapipe/examples/desktop/autoflip/quality/math_utils.h"
#include "mediapipe/framework/port/opencv_imgproc_inc.h"
#include "mediapipe/framework/port/ret_check.h"

namespace mediapipe {
namespace autoflip {
namespace {

// Returns true if the first pair should be considered greater than the second.
// This is used to sort detections by scores (from high to low).
bool PairCompare(const std::pair<float, int>& pair1,
                 const std::pair<float, int>& pair2) {
  return pair1.first > pair2.first;
}

}  // namespace

template <typename T>
void ScaleRect(const T& original_location, const double scale_x,
               const double scale_y, Rect* scaled_location) {
  scaled_location->set_x(round(original_location.x() * scale_x));
  scaled_location->set_y(round(original_location.y() * scale_y));
  scaled_location->set_width(round(original_location.width() * scale_x));
  scaled_location->set_height(round(original_location.height() * scale_y));
}
template void ScaleRect<Rect>(const Rect&, const double, const double, Rect*);
template void ScaleRect<RectF>(const RectF&, const double, const double, Rect*);

void NormalizedRectToRect(const RectF& normalized_location, const int width,
                          const int height, Rect* location) {
  ScaleRect(normalized_location, width, height, location);
}

::mediapipe::Status ClampRect(const int width, const int height,
                              Rect* location) {
  return ClampRect(0, 0, width, height, location);
}

::mediapipe::Status ClampRect(const int x0, const int y0, const int x1,
                              const int y1, Rect* location) {
  RET_CHECK(!(location->x() >= x1 || location->x() + location->width() <= x0 ||
              location->y() >= y1 || location->y() + location->height() <= y0));

  int clamped_left, clamped_right, clamped_top, clamped_bottom;
  RET_CHECK(MathUtil::Clamp(x0, x1, location->x(), &clamped_left));
  RET_CHECK(MathUtil::Clamp(x0, x1, location->x() + location->width(),
                            &clamped_right));
  RET_CHECK(MathUtil::Clamp(y0, y1, location->y(), &clamped_top));
  RET_CHECK(MathUtil::Clamp(y0, y1, location->y() + location->height(),
                            &clamped_bottom));
  location->set_x(clamped_left);
  location->set_y(clamped_top);
  location->set_width(std::max(0, clamped_right - clamped_left));
  location->set_height(std::max(0, clamped_bottom - clamped_top));
  return ::mediapipe::OkStatus();
}

void RectUnion(const Rect& rect_to_add, Rect* rect) {
  const int x1 = std::min(rect->x(), rect_to_add.x());
  const int y1 = std::min(rect->y(), rect_to_add.y());
  const int x2 = std::max(rect->x() + rect->width(),
                          rect_to_add.x() + rect_to_add.width());
  const int y2 = std::max(rect->y() + rect->height(),
                          rect_to_add.y() + rect_to_add.height());
  rect->set_x(x1);
  rect->set_y(y1);
  rect->set_width(x2 - x1);
  rect->set_height(y2 - y1);
}

::mediapipe::Status PackKeyFrameInfo(const int64 frame_timestamp_ms,
                                     const DetectionSet& detections,
                                     const int original_frame_width,
                                     const int original_frame_height,
                                     const int feature_frame_width,
                                     const int feature_frame_height,
                                     KeyFrameInfo* key_frame_info) {
  RET_CHECK(key_frame_info != nullptr) << "KeyFrameInfo is null";
  RET_CHECK(original_frame_width > 0 && original_frame_height > 0 &&
            feature_frame_width > 0 && feature_frame_height > 0)
      << "Invalid frame size.";

  const double scale_x =
      static_cast<double>(original_frame_width) / feature_frame_width;
  const double scale_y =
      static_cast<double>(original_frame_height) / feature_frame_height;

  key_frame_info->set_timestamp_ms(frame_timestamp_ms);

  // Scales detections and filter out the ones with no bounding boxes.
  auto* processed_detections = key_frame_info->mutable_detections();
  for (const auto& original_detection : detections.detections()) {
    bool has_valid_location = true;
    Rect location;
    if (original_detection.has_location_normalized()) {
      NormalizedRectToRect(original_detection.location_normalized(),
                           original_frame_width, original_frame_height,
                           &location);
    } else if (original_detection.has_location()) {
      ScaleRect(original_detection.location(), scale_x, scale_y, &location);
    } else {
      has_valid_location = false;
      LOG(ERROR) << "Detection missing a bounding box, skipped.";
    }
    if (has_valid_location) {
      if (!ClampRect(original_frame_width, original_frame_height, &location)
               .ok()) {
        LOG(ERROR) << "Invalid detection bounding box, skipped.";
        continue;
      }
      auto* detection = processed_detections->add_detections();
      *detection = original_detection;
      *(detection->mutable_location()) = location;
    }
  }

  return ::mediapipe::OkStatus();
}

::mediapipe::Status SortDetections(
    const DetectionSet& detections,
    std::vector<SalientRegion>* required_regions,
    std::vector<SalientRegion>* non_required_regions) {
  required_regions->clear();
  non_required_regions->clear();

  // Makes pairs of score and index.
  std::vector<std::pair<float, int>> required_score_idx_pairs;
  std::vector<std::pair<float, int>> non_required_score_idx_pairs;
  for (int i = 0; i < detections.detections_size(); ++i) {
    const auto& detection = detections.detections(i);
    const auto pair = std::make_pair(detection.score(), i);
    if (detection.is_required()) {
      required_score_idx_pairs.push_back(pair);
    } else {
      non_required_score_idx_pairs.push_back(pair);
    }
  }

  // Sorts required regions by score.
  std::stable_sort(required_score_idx_pairs.begin(),
                   required_score_idx_pairs.end(), PairCompare);
  for (int i = 0; i < required_score_idx_pairs.size(); ++i) {
    const int original_idx = required_score_idx_pairs[i].second;
    required_regions->push_back(detections.detections(original_idx));
  }

  // Sorts non-required regions by score.
  std::stable_sort(non_required_score_idx_pairs.begin(),
                   non_required_score_idx_pairs.end(), PairCompare);
  for (int i = 0; i < non_required_score_idx_pairs.size(); ++i) {
    const int original_idx = non_required_score_idx_pairs[i].second;
    non_required_regions->push_back(detections.detections(original_idx));
  }

  return ::mediapipe::OkStatus();
}

::mediapipe::Status SetKeyFrameCropTarget(const int frame_width,
                                          const int frame_height,
                                          const double target_aspect_ratio,
                                          KeyFrameCropOptions* crop_options) {
  RET_CHECK_NE(crop_options, nullptr) << "KeyFrameCropOptions is null.";
  RET_CHECK_GT(frame_width, 0) << "Frame width is non-positive.";
  RET_CHECK_GT(frame_height, 0) << "Frame height is non-positive.";
  RET_CHECK_GT(target_aspect_ratio, 0)
      << "Target aspect ratio is non-positive.";
  const double input_aspect_ratio =
      static_cast<double>(frame_width) / frame_height;
  const int crop_target_width =
      target_aspect_ratio < input_aspect_ratio
          ? std::round(frame_height * target_aspect_ratio)
          : frame_width;
  const int crop_target_height =
      target_aspect_ratio < input_aspect_ratio
          ? frame_height
          : std::round(frame_width / target_aspect_ratio);
  crop_options->set_target_width(crop_target_width);
  crop_options->set_target_height(crop_target_height);
  return ::mediapipe::OkStatus();
}

::mediapipe::Status AggregateKeyFrameResults(
    const KeyFrameCropOptions& key_frame_crop_options,
    const std::vector<KeyFrameCropResult>& key_frame_crop_results,
    const int scene_frame_width, const int scene_frame_height,
    SceneKeyFrameCropSummary* scene_summary) {
  RET_CHECK_NE(scene_summary, nullptr)
      << "Output SceneKeyFrameCropSummary is null.";

  const int num_key_frames = key_frame_crop_results.size();

  RET_CHECK_GT(scene_frame_width, 0) << "Non-positive frame width.";
  RET_CHECK_GT(scene_frame_height, 0) << "Non-positive frame height.";

  const int target_width = key_frame_crop_options.target_width();
  const int target_height = key_frame_crop_options.target_height();
  RET_CHECK_GT(target_width, 0) << "Non-positive target width.";
  RET_CHECK_GT(target_height, 0) << "Non-positive target height.";
  RET_CHECK_LE(target_width, scene_frame_width)
      << "Target width exceeds frame width.";
  RET_CHECK_LE(target_height, scene_frame_height)
      << "Target height exceeds frame height.";

  scene_summary->set_scene_frame_width(scene_frame_width);
  scene_summary->set_scene_frame_height(scene_frame_height);
  scene_summary->set_crop_window_width(target_width);
  scene_summary->set_crop_window_height(target_height);

  // Handles the corner case of no key frames.
  if (num_key_frames == 0) {
    scene_summary->set_has_salient_region(false);
    return ::mediapipe::OkStatus();
  }

  scene_summary->set_num_key_frames(num_key_frames);
  scene_summary->set_key_frame_center_min_x(scene_frame_width);
  scene_summary->set_key_frame_center_max_x(0);
  scene_summary->set_key_frame_center_min_y(scene_frame_height);
  scene_summary->set_key_frame_center_max_y(0);
  scene_summary->set_key_frame_min_score(std::numeric_limits<float>::max());
  scene_summary->set_key_frame_max_score(0.0);

  const float half_height = target_height / 2.0f;
  const float half_width = target_width / 2.0f;
  bool has_salient_region = false;
  int num_success_frames = 0;
  std::unique_ptr<Rect> required_crop_region_union = nullptr;
  for (int i = 0; i < num_key_frames; ++i) {
    auto* key_frame_compact_info = scene_summary->add_key_frame_compact_infos();
    const auto& result = key_frame_crop_results[i];
    key_frame_compact_info->set_timestamp_ms(result.timestamp_ms());
    if (result.are_required_regions_covered_in_target_size()) {
      num_success_frames++;
    }
    if (result.region_is_empty()) {
      key_frame_compact_info->set_center_x(-1.0);
      key_frame_compact_info->set_center_y(-1.0);
      key_frame_compact_info->set_score(-1.0);
      continue;
    }

    has_salient_region = true;
    if (!result.required_region_is_empty()) {
      if (required_crop_region_union == nullptr) {
        required_crop_region_union =
            absl::make_unique<Rect>(result.required_region());
      } else {
        RectUnion(result.required_region(), required_crop_region_union.get());
      }
    }

    const auto& region = result.region();
    float original_center_x = region.x() + region.width() / 2.0f;
    float original_center_y = region.y() + region.height() / 2.0f;
    RET_CHECK_GE(original_center_x, 0) << "Negative horizontal center.";
    RET_CHECK_GE(original_center_y, 0) << "Negative vertical center.";
    // Ensure that centered region of target size does not exceed frame size.
    float center_x, center_y;
    RET_CHECK(MathUtil::Clamp(half_width, scene_frame_width - half_width,
                              original_center_x, &center_x));
    RET_CHECK(MathUtil::Clamp(half_height, scene_frame_height - half_height,
                              original_center_y, &center_y));
    key_frame_compact_info->set_center_x(center_x);
    key_frame_compact_info->set_center_y(center_y);
    scene_summary->set_key_frame_center_min_x(
        std::min(scene_summary->key_frame_center_min_x(), center_x));
    scene_summary->set_key_frame_center_max_x(
        std::max(scene_summary->key_frame_center_max_x(), center_x));
    scene_summary->set_key_frame_center_min_y(
        std::min(scene_summary->key_frame_center_min_y(), center_y));
    scene_summary->set_key_frame_center_max_y(
        std::max(scene_summary->key_frame_center_max_y(), center_y));

    scene_summary->set_crop_window_width(
        std::max(scene_summary->crop_window_width(), region.width()));
    scene_summary->set_crop_window_height(
        std::max(scene_summary->crop_window_height(), region.height()));

    const float score = result.region_score();
    RET_CHECK_GE(score, 0.0) << "Negative score.";
    key_frame_compact_info->set_score(result.region_score());
    scene_summary->set_key_frame_min_score(
        std::min(scene_summary->key_frame_min_score(), score));
    scene_summary->set_key_frame_max_score(
        std::max(scene_summary->key_frame_max_score(), score));
  }

  scene_summary->set_has_salient_region(has_salient_region);
  scene_summary->set_has_required_salient_region(required_crop_region_union !=
                                                 nullptr);
  if (required_crop_region_union) {
    *(scene_summary->mutable_key_frame_required_crop_region_union()) =
        *required_crop_region_union;
  }
  const float success_rate =
      static_cast<float>(num_success_frames) / num_key_frames;
  scene_summary->set_frame_success_rate(success_rate);
  const float motion_x =
      static_cast<float>(scene_summary->key_frame_center_max_x() -
                         scene_summary->key_frame_center_min_x()) /
      scene_frame_width;
  scene_summary->set_horizontal_motion_amount(motion_x);
  const float motion_y =
      static_cast<float>(scene_summary->key_frame_center_max_y() -
                         scene_summary->key_frame_center_min_y()) /
      scene_frame_height;
  scene_summary->set_vertical_motion_amount(motion_y);
  return ::mediapipe::OkStatus();
}

::mediapipe::Status ComputeSceneStaticBordersSize(
    const std::vector<StaticFeatures>& static_features, int* top_border_size,
    int* bottom_border_size) {
  RET_CHECK(top_border_size) << "Output top border size is null.";
  RET_CHECK(bottom_border_size) << "Output bottom border size is null.";

  *top_border_size = -1;
  for (int i = 0; i < static_features.size(); ++i) {
    bool has_static_top_border = false;
    for (const auto& feature : static_features[i].border()) {
      if (feature.relative_position() == Border::TOP) {
        has_static_top_border = true;
        const int static_size = feature.border_position().height();
        *top_border_size = (*top_border_size > 0)
                               ? std::min(*top_border_size, static_size)
                               : static_size;
      }
    }
    if (!has_static_top_border) {
      *top_border_size = 0;
      break;
    }
  }

  *bottom_border_size = -1;
  for (int i = 0; i < static_features.size(); ++i) {
    bool has_static_bottom_border = false;
    for (const auto& feature : static_features[i].border()) {
      if (feature.relative_position() == Border::BOTTOM) {
        has_static_bottom_border = true;
        const int static_size = feature.border_position().height();
        *bottom_border_size = (*bottom_border_size > 0)
                                  ? std::min(*bottom_border_size, static_size)
                                  : static_size;
      }
    }
    if (!has_static_bottom_border) {
      *bottom_border_size = 0;
      break;
    }
  }

  *top_border_size = std::max(0, *top_border_size);
  *bottom_border_size = std::max(0, *bottom_border_size);
  return ::mediapipe::OkStatus();
}

::mediapipe::Status FindSolidBackgroundColor(
    const std::vector<StaticFeatures>& static_features,
    const std::vector<int64>& static_features_timestamps,
    const double min_fraction_solid_background_color,
    bool* has_solid_background,
    PiecewiseLinearFunction* background_color_l_function,
    PiecewiseLinearFunction* background_color_a_function,
    PiecewiseLinearFunction* background_color_b_function) {
  RET_CHECK(has_solid_background) << "Output boolean is null.";
  RET_CHECK(background_color_l_function) << "Output color l function is null.";
  RET_CHECK(background_color_a_function) << "Output color a function is null.";
  RET_CHECK(background_color_b_function) << "Output color b function is null.";

  *has_solid_background = false;
  int solid_background_frames = 0;
  for (int i = 0; i < static_features.size(); ++i) {
    if (static_features[i].has_solid_background()) {
      solid_background_frames++;
      const auto& color = static_features[i].solid_background();
      const int64 timestamp = static_features_timestamps[i];
      // BorderDetectionCalculator sets color assuming the input frame is
      // BGR, but in reality we have RGB, so we need to revert it here.
      // TODO remove this custom logic in BorderDetectionCalculator,
      // original CroppingCalculator, and this calculator.
      cv::Mat3f rgb_mat(1, 1, cv::Vec3b(color.b(), color.g(), color.r()));
      // Necessary scaling of the RGB values from [0, 255] to [0, 1] based on:
      // https://docs.opencv.org/2.4/modules/imgproc/doc/miscellaneous_transformations.html#cvtcolor
      rgb_mat *= 1.0 / 255;
      cv::Mat3f lab_mat(1, 1);
      cv::cvtColor(rgb_mat, lab_mat, cv::COLOR_RGB2Lab);
      // TODO change to piecewise constant interpolation if there is
      // visual artifact. We can simply add one more point right before the
      // next point with same value to mimic piecewise constant behavior.
      const auto lab = lab_mat.at<cv::Vec3f>(0, 0);
      background_color_l_function->AddPoint(timestamp, lab[0]);
      background_color_a_function->AddPoint(timestamp, lab[1]);
      background_color_b_function->AddPoint(timestamp, lab[2]);
    }
  }

  if (!static_features.empty() &&
      static_cast<float>(solid_background_frames) / static_features.size() >=
          min_fraction_solid_background_color) {
    *has_solid_background = true;
  }
  return ::mediapipe::OkStatus();
}

::mediapipe::Status AffineRetarget(
    const cv::Size& output_size, const std::vector<cv::Mat>& frames,
    const std::vector<cv::Mat>& affine_projection,
    std::vector<cv::Mat>* cropped_frames) {
  RET_CHECK(frames.size() == affine_projection.size())
      << "number of frames and retarget offsets must be the same.";
  RET_CHECK(cropped_frames->size() == frames.size())
      << "Output vector cropped_frames must be populated with output images of "
         "the same type, size and count.";
  for (int i = 0; i < frames.size(); i++) {
    RET_CHECK(frames[i].type() == (*cropped_frames)[i].type())
        << "input and output images must be the same type.";
    const auto affine = affine_projection[i];
    RET_CHECK(affine.cols == 3) << "Affine matrix must be 2x3";
    RET_CHECK(affine.rows == 2) << "Affine matrix must be 2x3";
    cv::warpAffine(frames[i], (*cropped_frames)[i], affine, output_size);
  }
  return ::mediapipe::OkStatus();
}
}  // namespace autoflip
}  // namespace mediapipe
