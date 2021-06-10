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

#include "mediapipe/examples/desktop/autoflip/quality/scene_camera_motion_analyzer.h"

#include <limits>

#include "absl/memory/memory.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "mediapipe/examples/desktop/autoflip/quality/math_utils.h"
#include "mediapipe/examples/desktop/autoflip/quality/piecewise_linear_function.h"
#include "mediapipe/examples/desktop/autoflip/quality/utils.h"
#include "mediapipe/framework/port/integral_types.h"
#include "mediapipe/framework/port/ret_check.h"
#include "mediapipe/framework/port/status.h"
#include "mediapipe/framework/timestamp.h"

namespace mediapipe {
namespace autoflip {

absl::Status SceneCameraMotionAnalyzer::AnalyzeSceneAndPopulateFocusPointFrames(
    const KeyFrameCropOptions& key_frame_crop_options,
    const std::vector<KeyFrameCropResult>& key_frame_crop_results,
    const int scene_frame_width, const int scene_frame_height,
    const std::vector<int64>& scene_frame_timestamps,
    const bool has_solid_color_background,
    SceneKeyFrameCropSummary* scene_summary,
    std::vector<FocusPointFrame>* focus_point_frames,
    SceneCameraMotion* scene_camera_motion) {
  has_solid_color_background_ = has_solid_color_background;
  total_scene_frames_ = scene_frame_timestamps.size();
  MP_RETURN_IF_ERROR(AggregateKeyFrameResults(
      key_frame_crop_options, key_frame_crop_results, scene_frame_width,
      scene_frame_height, scene_summary));

  const int64 scene_span_ms =
      scene_frame_timestamps.empty()
          ? 0
          : scene_frame_timestamps.back() - scene_frame_timestamps.front();
  const double scene_span_sec = TimestampDiff(scene_span_ms).Seconds();
  SceneCameraMotion camera_motion;
  MP_RETURN_IF_ERROR(DecideCameraMotionType(
      key_frame_crop_options, scene_span_sec, scene_frame_timestamps.back(),
      scene_summary, &camera_motion));
  if (scene_summary->has_salient_region()) {
    last_scene_with_salient_region_ = camera_motion;
    time_since_last_salient_region_us_ = scene_frame_timestamps.back();
  }
  if (scene_camera_motion != nullptr) {
    *scene_camera_motion = camera_motion;
  }

  return PopulateFocusPointFrames(*scene_summary, camera_motion,
                                  scene_frame_timestamps, focus_point_frames);
}

absl::Status SceneCameraMotionAnalyzer::ToUseSteadyMotion(
    const float look_at_center_x, const float look_at_center_y,
    const int crop_window_width, const int crop_window_height,
    SceneKeyFrameCropSummary* scene_summary,
    SceneCameraMotion* scene_camera_motion) const {
  scene_summary->set_crop_window_width(crop_window_width);
  scene_summary->set_crop_window_height(crop_window_height);
  auto* steady_motion = scene_camera_motion->mutable_steady_motion();
  steady_motion->set_steady_look_at_center_x(look_at_center_x);
  steady_motion->set_steady_look_at_center_y(look_at_center_y);
  return absl::OkStatus();
}

absl::Status SceneCameraMotionAnalyzer::ToUseSweepingMotion(
    const float start_x, const float start_y, const float end_x,
    const float end_y, const int crop_window_width,
    const int crop_window_height, const double time_duration_in_sec,
    SceneKeyFrameCropSummary* scene_summary,
    SceneCameraMotion* scene_camera_motion) const {
  auto* sweeping_motion = scene_camera_motion->mutable_sweeping_motion();
  sweeping_motion->set_sweep_start_center_x(start_x);
  sweeping_motion->set_sweep_start_center_y(start_y);
  sweeping_motion->set_sweep_end_center_x(end_x);
  sweeping_motion->set_sweep_end_center_y(end_y);
  scene_summary->set_crop_window_width(crop_window_width);
  scene_summary->set_crop_window_height(crop_window_height);
  const auto sweeping_log = absl::StrFormat(
      "Success rate %.2f is low - Camera is sweeping from (%.1f, %.1f) to "
      "(%.1f, %.1f) in %.2f seconds.",
      scene_summary->frame_success_rate(), start_x, start_y, end_x, end_y,
      time_duration_in_sec);
  VLOG(1) << sweeping_log;
  return absl::OkStatus();
}

absl::Status SceneCameraMotionAnalyzer::DecideCameraMotionType(
    const KeyFrameCropOptions& key_frame_crop_options,
    const double scene_span_sec, const int64 end_time_us,
    SceneKeyFrameCropSummary* scene_summary,
    SceneCameraMotion* scene_camera_motion) const {
  RET_CHECK_GE(scene_span_sec, 0.0) << "Scene time span is negative.";
  RET_CHECK_NE(scene_summary, nullptr) << "Scene summary is null.";
  RET_CHECK_NE(scene_camera_motion, nullptr) << "Scene camera motion is null.";
  const float scene_frame_center_x = scene_summary->scene_frame_width() / 2.0f;
  const float scene_frame_center_y = scene_summary->scene_frame_height() / 2.0f;

  // If no frame has any focus region, that is, the scene has no focus
  // regions, then default to look at the center.
  if (!scene_summary->has_salient_region()) {
    VLOG(1) << "No focus regions - camera is set to be steady on center.";
    float no_salient_position_x = scene_frame_center_x;
    float no_salient_position_y = scene_frame_center_y;
    if (end_time_us - time_since_last_salient_region_us_ <
            options_.duration_before_centering_us() &&
        last_scene_with_salient_region_.has_steady_motion()) {
      no_salient_position_x = last_scene_with_salient_region_.steady_motion()
                                  .steady_look_at_center_x();
      no_salient_position_y = last_scene_with_salient_region_.steady_motion()
                                  .steady_look_at_center_y();
    }
    MP_RETURN_IF_ERROR(ToUseSteadyMotion(
        no_salient_position_x, no_salient_position_y,
        scene_summary->crop_window_width(), scene_summary->crop_window_height(),
        scene_summary, scene_camera_motion));
    return absl::OkStatus();
  }

  // Sweep across the scene when 1) success rate is too low, AND 2) the current
  // scene is long enough.
  if (options_.allow_sweeping() && !has_solid_color_background_ &&
      scene_summary->frame_success_rate() <
          options_.minimum_success_rate_for_sweeping() &&
      scene_span_sec >= options_.minimum_scene_span_sec_for_sweeping()) {
    float start_x = -1.0, start_y = -1.0, end_x = -1.0, end_y = -1.0;
    if (options_.sweep_entire_frame()) {
      if (scene_summary->crop_window_width() >
          key_frame_crop_options.target_width()) {  // horizontal sweeping
        start_x = 0.0f;
        start_y = scene_frame_center_y;
        end_x = scene_summary->scene_frame_width();
        end_y = scene_frame_center_y;
      } else {  // vertical sweeping
        start_x = scene_frame_center_x;
        start_y = 0.0f;
        end_x = scene_frame_center_x;
        end_y = scene_summary->scene_frame_height();
      }
    } else {
      start_x = scene_summary->key_frame_center_min_x();
      start_y = scene_summary->key_frame_center_min_y();
      end_x = scene_summary->key_frame_center_max_x();
      end_y = scene_summary->key_frame_center_max_y();
    }
    MP_RETURN_IF_ERROR(ToUseSweepingMotion(
        start_x, start_y, end_x, end_y, key_frame_crop_options.target_width(),
        key_frame_crop_options.target_height(), scene_span_sec, scene_summary,
        scene_camera_motion));
    return absl::OkStatus();
  }

  // If scene motion is small, then look at a steady point in the scene.
  if ((scene_summary->horizontal_motion_amount() <
           options_.motion_stabilization_threshold_percent() &&
       scene_summary->vertical_motion_amount() <
           options_.motion_stabilization_threshold_percent()) ||
      total_scene_frames_ == 1) {
    return DecideSteadyLookAtRegion(key_frame_crop_options, scene_summary,
                                    scene_camera_motion);
  }

  // Otherwise, tracks the focus regions.
  scene_camera_motion->mutable_tracking_motion();
  return absl::OkStatus();
}

// If there is no required focus region, looks at the middle of the center
// range, and snaps to the scene center if close. Otherwise, look at the center
// of the union of the required focus regions, and ensures the crop region
// covers this union.
absl::Status SceneCameraMotionAnalyzer::DecideSteadyLookAtRegion(
    const KeyFrameCropOptions& key_frame_crop_options,
    SceneKeyFrameCropSummary* scene_summary,
    SceneCameraMotion* scene_camera_motion) const {
  const float scene_frame_width = scene_summary->scene_frame_width();
  const float scene_frame_height = scene_summary->scene_frame_height();
  const int target_width = key_frame_crop_options.target_width();
  const int target_height = key_frame_crop_options.target_height();
  float center_x = -1, center_y = -1;
  float crop_width = -1, crop_height = -1;

  if (scene_summary->has_required_salient_region()) {
    // Set look-at position to be the center of the union of required focus
    // regions and the crop window size to be the maximum of this union size
    // and the target size.
    const auto& required_region_union =
        scene_summary->key_frame_required_crop_region_union();
    center_x = required_region_union.x() + required_region_union.width() / 2.0f;
    center_y =
        required_region_union.y() + required_region_union.height() / 2.0f;
    crop_width = std::max(target_width, required_region_union.width());
    crop_height = std::max(target_height, required_region_union.height());
  } else {
    // Set look-at position to be the middle of the center range, and the crop
    // window size to be the target size.
    center_x = (scene_summary->key_frame_center_min_x() +
                scene_summary->key_frame_center_max_x()) /
               2.0f;
    center_y = (scene_summary->key_frame_center_min_y() +
                scene_summary->key_frame_center_max_y()) /
               2.0f;
    crop_width = target_width;
    crop_height = target_height;

    // Optionally snap the look-at position to the scene frame center.
    const float center_x_distance =
        std::fabs(center_x - scene_frame_width / 2.0f);
    const float center_y_distance =
        std::fabs(center_y - scene_frame_height / 2.0f);
    if (center_x_distance / scene_frame_width <
        options_.snap_center_max_distance_percent()) {
      center_x = scene_frame_width / 2.0f;
    }
    if (center_y_distance / scene_frame_height <
        options_.snap_center_max_distance_percent()) {
      center_y = scene_frame_height / 2.0f;
    }
  }

  // Clamp the region to be inside the frame.
  // TODO: this may not be necessary.
  float clamped_center_x, clamped_center_y;
  RET_CHECK(MathUtil::Clamp(crop_width / 2.0f,
                            scene_frame_width - crop_width / 2.0f, center_x,
                            &clamped_center_x));
  center_x = clamped_center_x;
  RET_CHECK(MathUtil::Clamp(crop_height / 2.0f,
                            scene_frame_height - crop_height / 2.0f, center_y,
                            &clamped_center_y));
  center_y = clamped_center_y;

  VLOG(1) << "Motion is small - camera is set to be steady at " << center_x
          << ", " << center_y;
  MP_RETURN_IF_ERROR(ToUseSteadyMotion(center_x, center_y, crop_width,
                                       crop_height, scene_summary,
                                       scene_camera_motion));
  return absl::OkStatus();
}

absl::Status SceneCameraMotionAnalyzer::AddFocusPointsFromCenterTypeAndWeight(
    const float center_x, const float center_y, const int frame_width,
    const int frame_height, const FocusPointFrameType type, const float weight,
    const float bound, FocusPointFrame* focus_point_frame) const {
  RET_CHECK_NE(focus_point_frame, nullptr) << "Focus point frame is null.";
  const float norm_x = center_x / frame_width;
  const float norm_y = center_y / frame_height;
  const std::vector<float> extremal_values = {0, 1};
  if (type == TOPMOST_AND_BOTTOMMOST) {
    for (const float extremal_value : extremal_values) {
      auto* focus_point = focus_point_frame->add_point();
      focus_point->set_norm_point_x(norm_x);
      focus_point->set_norm_point_y(extremal_value);
      focus_point->set_weight(weight);
      focus_point->set_left(bound);
      focus_point->set_right(bound);
    }
  } else if (type == LEFTMOST_AND_RIGHTMOST) {
    for (const float extremal_value : extremal_values) {
      auto* focus_point = focus_point_frame->add_point();
      focus_point->set_norm_point_x(extremal_value);
      focus_point->set_norm_point_y(norm_y);
      focus_point->set_weight(weight);
      focus_point->set_top(bound);
      focus_point->set_bottom(bound);
    }
  } else if (type == CENTER) {
    auto* focus_point = focus_point_frame->add_point();
    focus_point->set_norm_point_x(norm_x);
    focus_point->set_norm_point_y(norm_y);
    focus_point->set_weight(weight);
    focus_point->set_left(bound);
    focus_point->set_right(bound);
    focus_point->set_top(bound);
    focus_point->set_bottom(bound);
  } else {
    RET_CHECK_FAIL() << absl::StrCat("Invalid FocusPointFrameType ", type);
  }
  return absl::OkStatus();
}

absl::Status SceneCameraMotionAnalyzer::PopulateFocusPointFrames(
    const SceneKeyFrameCropSummary& scene_summary,
    const SceneCameraMotion& scene_camera_motion,
    const std::vector<int64>& scene_frame_timestamps,
    std::vector<FocusPointFrame>* focus_point_frames) const {
  RET_CHECK_NE(focus_point_frames, nullptr)
      << "Output vector of FocusPointFrame is null.";

  const int num_scene_frames = scene_frame_timestamps.size();
  RET_CHECK_GT(num_scene_frames, 0) << "No scene frames.";
  RET_CHECK_EQ(scene_summary.num_key_frames(),
               scene_summary.key_frame_compact_infos_size())
      << "Key frame compact infos has wrong size:"
      << " num_key_frames = " << scene_summary.num_key_frames()
      << " key_frame_compact_infos size = "
      << scene_summary.key_frame_compact_infos_size();
  const int scene_frame_width = scene_summary.scene_frame_width();
  const int scene_frame_height = scene_summary.scene_frame_height();
  RET_CHECK_GT(scene_frame_width, 0) << "Non-positive frame width.";
  RET_CHECK_GT(scene_frame_height, 0) << "Non-positive frame height.";

  FocusPointFrameType focus_point_frame_type =
      (scene_summary.crop_window_height() == scene_frame_height)
          ? TOPMOST_AND_BOTTOMMOST
          : (scene_summary.crop_window_width() == scene_frame_width
                 ? LEFTMOST_AND_RIGHTMOST
                 : CENTER);
  focus_point_frames->reserve(num_scene_frames);

  if (scene_camera_motion.has_steady_motion()) {
    // Camera focuses on a steady point of the scene.
    const float center_x =
        scene_camera_motion.steady_motion().steady_look_at_center_x();
    const float center_y =
        scene_camera_motion.steady_motion().steady_look_at_center_y();
    for (int i = 0; i < num_scene_frames; ++i) {
      FocusPointFrame focus_point_frame;
      MP_RETURN_IF_ERROR(AddFocusPointsFromCenterTypeAndWeight(
          center_x, center_y, scene_frame_width, scene_frame_height,
          focus_point_frame_type, options_.maximum_salient_point_weight(),
          options_.salient_point_bound(), &focus_point_frame));
      focus_point_frames->push_back(focus_point_frame);
    }
    return absl::OkStatus();
  } else if (scene_camera_motion.has_sweeping_motion()) {
    // Camera sweeps across the frame.
    const auto& sweeping_motion = scene_camera_motion.sweeping_motion();
    const float start_x = sweeping_motion.sweep_start_center_x();
    const float start_y = sweeping_motion.sweep_start_center_y();
    const float end_x = sweeping_motion.sweep_end_center_x();
    const float end_y = sweeping_motion.sweep_end_center_y();
    for (int i = 0; i < num_scene_frames; ++i) {
      const float fraction =
          num_scene_frames > 1 ? static_cast<float>(i) / (num_scene_frames - 1)
                               : 0;
      const float position_x = start_x * (1.0f - fraction) + end_x * fraction;
      const float position_y = start_y * (1.0f - fraction) + end_y * fraction;
      FocusPointFrame focus_point_frame;
      MP_RETURN_IF_ERROR(AddFocusPointsFromCenterTypeAndWeight(
          position_x, position_y, scene_frame_width, scene_frame_height,
          focus_point_frame_type, options_.maximum_salient_point_weight(),
          options_.salient_point_bound(), &focus_point_frame));
      focus_point_frames->push_back(focus_point_frame);
    }
    return absl::OkStatus();
  } else if (scene_camera_motion.has_tracking_motion()) {
    // Camera tracks crop regions.
    RET_CHECK_GT(scene_summary.num_key_frames(), 0) << "No key frames.";
    return PopulateFocusPointFramesForTracking(
        scene_summary, focus_point_frame_type, scene_frame_timestamps,
        focus_point_frames);
  } else {
    return absl::Status(StatusCode::kInvalidArgument, "Unknown motion type.");
  }
}

// Linearly interpolates between key frames based on the timestamps using
// piecewise-linear functions for the crop region centers and scores. Adds one
// focus point at the center of the interpolated crop region for each frame.
// The weight for the focus point is proportional to the interpolated score
// and scaled so that the maximum weight is equal to
// maximum_focus_point_weight in the SceneCameraMotionAnalyzerOptions.
absl::Status SceneCameraMotionAnalyzer::PopulateFocusPointFramesForTracking(
    const SceneKeyFrameCropSummary& scene_summary,
    const FocusPointFrameType focus_point_frame_type,
    const std::vector<int64>& scene_frame_timestamps,
    std::vector<FocusPointFrame>* focus_point_frames) const {
  RET_CHECK_GE(scene_summary.key_frame_max_score(), 0.0)
      << "Maximum score is negative.";

  const int num_key_frames = scene_summary.num_key_frames();
  const auto& key_frame_compact_infos = scene_summary.key_frame_compact_infos();
  const int num_scene_frames = scene_frame_timestamps.size();
  const int scene_frame_width = scene_summary.scene_frame_width();
  const int scene_frame_height = scene_summary.scene_frame_height();

  PiecewiseLinearFunction center_x_function, center_y_function, score_function;
  const int64 timestamp_offset = key_frame_compact_infos[0].timestamp_ms();
  for (int i = 0; i < num_key_frames; ++i) {
    const float center_x = key_frame_compact_infos[i].center_x();
    const float center_y = key_frame_compact_infos[i].center_y();
    const float score = key_frame_compact_infos[i].score();
    // Skips empty key frames.
    if (center_x < 0 || center_y < 0 || score < 0) {
      continue;
    }
    const double relative_timestamp =
        key_frame_compact_infos[i].timestamp_ms() - timestamp_offset;
    center_x_function.AddPoint(relative_timestamp, center_x);
    center_y_function.AddPoint(relative_timestamp, center_y);
    score_function.AddPoint(relative_timestamp, score);
  }

  double max_score = 0.0;
  const double min_score = 1e-4;  // prevent constraints with 0 weight
  for (int i = 0; i < num_scene_frames; ++i) {
    const double relative_timestamp =
        static_cast<double>(scene_frame_timestamps[i] - timestamp_offset);
    const double center_x = center_x_function.Evaluate(relative_timestamp);
    const double center_y = center_y_function.Evaluate(relative_timestamp);
    const double score =
        std::max(min_score, score_function.Evaluate(relative_timestamp));
    max_score = std::max(max_score, score);
    FocusPointFrame focus_point_frame;
    MP_RETURN_IF_ERROR(AddFocusPointsFromCenterTypeAndWeight(
        center_x, center_y, scene_frame_width, scene_frame_height,
        focus_point_frame_type, score, options_.salient_point_bound(),
        &focus_point_frame));
    focus_point_frames->push_back(focus_point_frame);
  }

  // Scales weights so that maximum weight = maximum_salient_point_weight.
  // TODO: run some experiments to find out if this is necessary.
  max_score = std::max(max_score, min_score);
  const double scale = options_.maximum_salient_point_weight() / max_score;
  for (int i = 0; i < focus_point_frames->size(); ++i) {
    for (int j = 0; j < (*focus_point_frames)[i].point_size(); ++j) {
      auto* focus_point = (*focus_point_frames)[i].mutable_point(j);
      focus_point->set_weight(scale * focus_point->weight());
    }
  }
  return absl::OkStatus();
}

}  // namespace autoflip
}  // namespace mediapipe
