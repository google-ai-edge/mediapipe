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

#include "mediapipe/examples/desktop/autoflip/calculators/scene_cropping_calculator.h"

#include <cmath>

#include "absl/memory/memory.h"
#include "absl/strings/str_format.h"
#include "mediapipe/examples/desktop/autoflip/autoflip_messages.pb.h"
#include "mediapipe/examples/desktop/autoflip/quality/scene_cropping_viz.h"
#include "mediapipe/examples/desktop/autoflip/quality/utils.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/image_frame.h"
#include "mediapipe/framework/formats/image_frame_opencv.h"
#include "mediapipe/framework/port/opencv_core_inc.h"
#include "mediapipe/framework/port/opencv_imgproc_inc.h"
#include "mediapipe/framework/port/parse_text_proto.h"
#include "mediapipe/framework/port/ret_check.h"
#include "mediapipe/framework/port/status.h"
#include "mediapipe/framework/timestamp.h"

namespace mediapipe {
namespace autoflip {

constexpr char kInputVideoFrames[] = "VIDEO_FRAMES";
constexpr char kInputKeyFrames[] = "KEY_FRAMES";
constexpr char kInputDetections[] = "DETECTION_FEATURES";
constexpr char kInputStaticFeatures[] = "STATIC_FEATURES";
constexpr char kInputShotBoundaries[] = "SHOT_BOUNDARIES";
constexpr char kInputExternalSettings[] = "EXTERNAL_SETTINGS";
// This side packet must be used in conjunction with
// TargetSizeType::MAXIMIZE_TARGET_DIMENSION
constexpr char kAspectRatio[] = "EXTERNAL_ASPECT_RATIO";

// Output the cropped frames, as well as visualization of crop regions and focus
// points. Note that, KEY_FRAME_CROP_REGION_VIZ_FRAMES and
// SALIENT_POINT_FRAME_VIZ_FRAMES can only be enabled when CROPPED_FRAMES is
// enabled.
constexpr char kOutputCroppedFrames[] = "CROPPED_FRAMES";
constexpr char kOutputKeyFrameCropViz[] = "KEY_FRAME_CROP_REGION_VIZ_FRAMES";
constexpr char kOutputFocusPointFrameViz[] = "SALIENT_POINT_FRAME_VIZ_FRAMES";
constexpr char kOutputSummary[] = "CROPPING_SUMMARY";

// External rendering outputs
constexpr char kExternalRenderingPerFrame[] = "EXTERNAL_RENDERING_PER_FRAME";
constexpr char kExternalRenderingFullVid[] = "EXTERNAL_RENDERING_FULL_VID";

::mediapipe::Status SceneCroppingCalculator::GetContract(
    ::mediapipe::CalculatorContract* cc) {
  if (cc->InputSidePackets().HasTag(kInputExternalSettings)) {
    cc->InputSidePackets().Tag(kInputExternalSettings).Set<std::string>();
  }
  if (cc->InputSidePackets().HasTag(kAspectRatio)) {
    cc->InputSidePackets().Tag(kAspectRatio).Set<std::string>();
  }
  cc->Inputs().Tag(kInputVideoFrames).Set<ImageFrame>();
  if (cc->Inputs().HasTag(kInputKeyFrames)) {
    cc->Inputs().Tag(kInputKeyFrames).Set<ImageFrame>();
  }
  cc->Inputs().Tag(kInputDetections).Set<DetectionSet>();
  if (cc->Inputs().HasTag(kInputStaticFeatures)) {
    cc->Inputs().Tag(kInputStaticFeatures).Set<StaticFeatures>();
  }
  cc->Inputs().Tag(kInputShotBoundaries).Set<bool>();

  if (cc->Outputs().HasTag(kOutputCroppedFrames)) {
    cc->Outputs().Tag(kOutputCroppedFrames).Set<ImageFrame>();
  }
  if (cc->Outputs().HasTag(kOutputKeyFrameCropViz)) {
    RET_CHECK(cc->Outputs().HasTag(kOutputCroppedFrames))
        << "KEY_FRAME_CROP_REGION_VIZ_FRAMES can only be used when "
           "CROPPED_FRAMES is specified.";
    cc->Outputs().Tag(kOutputKeyFrameCropViz).Set<ImageFrame>();
  }
  if (cc->Outputs().HasTag(kOutputFocusPointFrameViz)) {
    RET_CHECK(cc->Outputs().HasTag(kOutputCroppedFrames))
        << "SALIENT_POINT_FRAME_VIZ_FRAMES can only be used when "
           "CROPPED_FRAMES is specified.";
    cc->Outputs().Tag(kOutputFocusPointFrameViz).Set<ImageFrame>();
  }
  if (cc->Outputs().HasTag(kOutputSummary)) {
    cc->Outputs().Tag(kOutputSummary).Set<VideoCroppingSummary>();
  }
  if (cc->Outputs().HasTag(kExternalRenderingPerFrame)) {
    cc->Outputs().Tag(kExternalRenderingPerFrame).Set<ExternalRenderFrame>();
  }
  if (cc->Outputs().HasTag(kExternalRenderingFullVid)) {
    cc->Outputs()
        .Tag(kExternalRenderingFullVid)
        .Set<std::vector<ExternalRenderFrame>>();
  }
  RET_CHECK(cc->Outputs().HasTag(kExternalRenderingPerFrame) ||
            cc->Outputs().HasTag(kExternalRenderingFullVid) ||
            cc->Outputs().HasTag(kOutputCroppedFrames))
      << "At leaset one output stream must be specified";
  return ::mediapipe::OkStatus();
}

::mediapipe::Status SceneCroppingCalculator::Open(CalculatorContext* cc) {
  options_ = cc->Options<SceneCroppingCalculatorOptions>();
  RET_CHECK_GT(options_.max_scene_size(), 0)
      << "Maximum scene size is non-positive.";
  RET_CHECK_GE(options_.prior_frame_buffer_size(), 0)
      << "Prior frame buffer size is negative.";

  RET_CHECK(options_.solid_background_frames_padding_fraction() >= 0.0 &&
            options_.solid_background_frames_padding_fraction() <= 1.0)
      << "Solid background frames padding fraction is not in [0, 1].";
  const auto& padding_params = options_.padding_parameters();
  background_contrast_ = padding_params.background_contrast();
  RET_CHECK(background_contrast_ >= 0.0 && background_contrast_ <= 1.0)
      << "Background contrast " << background_contrast_ << " is not in [0, 1].";
  blur_cv_size_ = padding_params.blur_cv_size();
  RET_CHECK_GT(blur_cv_size_, 0) << "Blur cv size is non-positive.";
  overlay_opacity_ = padding_params.overlay_opacity();
  RET_CHECK(overlay_opacity_ >= 0.0 && overlay_opacity_ <= 1.0)
      << "Overlay opacity " << overlay_opacity_ << " is not in [0, 1].";

  scene_cropper_ = absl::make_unique<SceneCropper>();
  if (cc->Outputs().HasTag(kOutputSummary)) {
    summary_ = absl::make_unique<VideoCroppingSummary>();
  }
  if (cc->Outputs().HasTag(kExternalRenderingFullVid)) {
    external_render_list_ =
        absl::make_unique<std::vector<ExternalRenderFrame>>();
  }
  should_perform_frame_cropping_ = cc->Outputs().HasTag(kOutputCroppedFrames);
  return ::mediapipe::OkStatus();
}

namespace {
::mediapipe::Status ParseAspectRatioString(
    const std::string& aspect_ratio_string, double* aspect_ratio) {
  std::string error_msg =
      "Aspect ratio std::string must be in the format of 'width:height', e.g. "
      "'1:1' or '5:4', your input was " +
      aspect_ratio_string;
  auto pos = aspect_ratio_string.find(":");
  RET_CHECK(pos != std::string::npos) << error_msg;
  double width_ratio;
  RET_CHECK(absl::SimpleAtod(aspect_ratio_string.substr(0, pos), &width_ratio))
      << error_msg;
  double height_ratio;
  RET_CHECK(absl::SimpleAtod(
      aspect_ratio_string.substr(pos + 1, aspect_ratio_string.size()),
      &height_ratio))
      << error_msg;
  *aspect_ratio = width_ratio / height_ratio;
  return ::mediapipe::OkStatus();
}
void ConstructExternalRenderMessage(
    const cv::Rect& crop_from_location, const cv::Rect& render_to_location,
    const cv::Scalar& padding_color, const uint64 timestamp_us,
    ExternalRenderFrame* external_render_message) {
  auto crop_from_message =
      external_render_message->mutable_crop_from_location();
  crop_from_message->set_x(crop_from_location.x);
  crop_from_message->set_y(crop_from_location.y);
  crop_from_message->set_width(crop_from_location.width);
  crop_from_message->set_height(crop_from_location.height);
  auto render_to_message =
      external_render_message->mutable_render_to_location();
  render_to_message->set_x(render_to_location.x);
  render_to_message->set_y(render_to_location.y);
  render_to_message->set_width(render_to_location.width);
  render_to_message->set_height(render_to_location.height);
  auto padding_color_message = external_render_message->mutable_padding_color();
  padding_color_message->set_r(padding_color[0]);
  padding_color_message->set_g(padding_color[1]);
  padding_color_message->set_b(padding_color[2]);
  external_render_message->set_timestamp_us(timestamp_us);
}
}  // namespace

::mediapipe::Status SceneCroppingCalculator::Process(
    ::mediapipe::CalculatorContext* cc) {
  // Sets frame dimension and format.
  if (frame_width_ < 0 &&
      !cc->Inputs().Tag(kInputVideoFrames).Value().IsEmpty()) {
    const auto& frame = cc->Inputs().Tag(kInputVideoFrames).Get<ImageFrame>();
    frame_width_ = frame.Width();
    RET_CHECK_GT(frame_width_, 0) << "Input frame width is non-positive.";
    frame_height_ = frame.Height();
    RET_CHECK_GT(frame_height_, 0) << "Input frame height is non-positive.";
    frame_format_ = frame.Format();
    target_width_ = options_.target_width();
    target_height_ = options_.target_height();
    if (cc->InputSidePackets().HasTag(kInputExternalSettings)) {
      auto conversion_options = ParseTextProtoOrDie<ConversionOptions>(
          cc->InputSidePackets()
              .Tag(kInputExternalSettings)
              .Get<std::string>());
      target_width_ = conversion_options.target_width();
      target_height_ = conversion_options.target_height();
    }
    target_aspect_ratio_ = static_cast<double>(target_width_) / target_height_;
    RET_CHECK_NE(options_.target_size_type(),
                 SceneCroppingCalculatorOptions::UNKNOWN)
        << "TargetSizeType not set properly.";
    // Resets target size if keep original height or width.
    if (options_.target_size_type() ==
        SceneCroppingCalculatorOptions::KEEP_ORIGINAL_HEIGHT) {
      target_height_ = frame_height_;
      target_width_ = std::round(target_height_ * target_aspect_ratio_);
    } else if (options_.target_size_type() ==
               SceneCroppingCalculatorOptions::KEEP_ORIGINAL_WIDTH) {
      target_width_ = frame_width_;
      target_height_ = std::round(target_width_ / target_aspect_ratio_);
    } else if (options_.target_size_type() ==
               SceneCroppingCalculatorOptions::MAXIMIZE_TARGET_DIMENSION) {
      RET_CHECK(cc->InputSidePackets().HasTag(kAspectRatio))
          << "MAXIMIZE_TARGET_DIMENSION is set without an "
             "external_aspect_ratio";
      double requested_aspect_ratio;
      MP_RETURN_IF_ERROR(ParseAspectRatioString(
          cc->InputSidePackets().Tag(kAspectRatio).Get<std::string>(),
          &requested_aspect_ratio));
      const double original_aspect_ratio =
          static_cast<double>(frame_width_) / frame_height_;
      if (original_aspect_ratio > requested_aspect_ratio) {
        target_height_ = frame_height_;
        target_width_ = std::round(target_height_ * requested_aspect_ratio);
      } else {
        target_width_ = frame_width_;
        target_height_ = std::round(target_width_ / requested_aspect_ratio);
      }
    }
    // Makes sure that target size is even if keep original width or height.
    if (options_.target_size_type() !=
        SceneCroppingCalculatorOptions::USE_TARGET_DIMENSION) {
      if (target_width_ % 2 == 1) {
        target_width_ = std::max(2, target_width_ - 1);
      }
      if (target_height_ % 2 == 1) {
        target_height_ = std::max(2, target_height_ - 1);
      }
      target_aspect_ratio_ =
          static_cast<double>(target_width_) / target_height_;
    }
    // Set keyframe width/height for feature upscaling (overwritten by keyframe
    // input if provided).
    if (options_.has_video_features_width() &&
        options_.has_video_features_height()) {
      key_frame_width_ = options_.video_features_width();
      key_frame_height_ = options_.video_features_height();
    } else if (!cc->Inputs().HasTag(kInputKeyFrames)) {
      key_frame_width_ = frame_width_;
      key_frame_height_ = frame_height_;
    }
    // Check provided dimensions.
    RET_CHECK_GT(target_width_, 0) << "Target width is non-positive.";
    RET_CHECK_NE(target_width_ % 2, 1)
        << "Target width cannot be odd, because encoder expects dimension "
           "values to be even.";
    RET_CHECK_GT(target_height_, 0) << "Target height is non-positive.";
    RET_CHECK_NE(target_height_ % 2, 1)
        << "Target height cannot be odd, because encoder expects dimension "
           "values to be even.";
  }

  // Sets key frame dimension.
  if (cc->Inputs().HasTag(kInputKeyFrames) &&
      !cc->Inputs().Tag(kInputKeyFrames).Value().IsEmpty() &&
      key_frame_width_ < 0) {
    const auto& key_frame = cc->Inputs().Tag(kInputKeyFrames).Get<ImageFrame>();
    key_frame_width_ = key_frame.Width();
    key_frame_height_ = key_frame.Height();
  }

  // Processes a scene when shot boundary or buffer is full.
  bool is_end_of_scene = false;
  if (!cc->Inputs().Tag(kInputShotBoundaries).Value().IsEmpty()) {
    is_end_of_scene = cc->Inputs().Tag(kInputShotBoundaries).Get<bool>();
  }
  const bool force_buffer_flush =
      scene_frame_timestamps_.size() >= options_.max_scene_size();
  if (!scene_frame_timestamps_.empty() &&
      (is_end_of_scene || force_buffer_flush)) {
    MP_RETURN_IF_ERROR(ProcessScene(is_end_of_scene, cc));
  }

  // Saves frame and timestamp and whether it is a key frame.
  if (!cc->Inputs().Tag(kInputVideoFrames).Value().IsEmpty()) {
    LOG_EVERY_N(ERROR, 10)
        << "------------------------ (Breathing) Time(s): "
        << cc->Inputs().Tag(kInputVideoFrames).Value().Timestamp().Seconds();
    // Only buffer frames if |should_perform_frame_cropping_| is true.
    if (should_perform_frame_cropping_) {
      const auto& frame = cc->Inputs().Tag(kInputVideoFrames).Get<ImageFrame>();
      const cv::Mat frame_mat = formats::MatView(&frame);
      cv::Mat copy_mat;
      frame_mat.copyTo(copy_mat);
      scene_frames_or_empty_.push_back(copy_mat);
    }
    scene_frame_timestamps_.push_back(cc->InputTimestamp().Value());
    is_key_frames_.push_back(
        !cc->Inputs().Tag(kInputDetections).Value().IsEmpty());
  }

  // Packs key frame info.
  if (!cc->Inputs().Tag(kInputDetections).Value().IsEmpty()) {
    const auto& detections =
        cc->Inputs().Tag(kInputDetections).Get<DetectionSet>();
    KeyFrameInfo key_frame_info;
    MP_RETURN_IF_ERROR(PackKeyFrameInfo(
        cc->InputTimestamp().Value(), detections, frame_width_, frame_height_,
        key_frame_width_, key_frame_height_, &key_frame_info));
    key_frame_infos_.push_back(key_frame_info);
  }

  // Buffers static features.
  if (cc->Inputs().HasTag(kInputStaticFeatures) &&
      !cc->Inputs().Tag(kInputStaticFeatures).Value().IsEmpty()) {
    static_features_.push_back(
        cc->Inputs().Tag(kInputStaticFeatures).Get<StaticFeatures>());
    static_features_timestamps_.push_back(cc->InputTimestamp().Value());
  }

  return ::mediapipe::OkStatus();
}

::mediapipe::Status SceneCroppingCalculator::Close(
    ::mediapipe::CalculatorContext* cc) {
  if (!scene_frame_timestamps_.empty()) {
    MP_RETURN_IF_ERROR(ProcessScene(/* is_end_of_scene = */ true, cc));
  }
  if (cc->Outputs().HasTag(kOutputSummary)) {
    cc->Outputs()
        .Tag(kOutputSummary)
        .Add(summary_.release(), Timestamp::PostStream());
  }
  if (cc->Outputs().HasTag(kExternalRenderingFullVid)) {
    cc->Outputs()
        .Tag(kExternalRenderingFullVid)
        .Add(external_render_list_.release(), Timestamp::PostStream());
  }
  return ::mediapipe::OkStatus();
}

// TODO: split this function into two, one for calculating the border
// sizes, the other for the actual removal of borders from the frames.
::mediapipe::Status SceneCroppingCalculator::RemoveStaticBorders(
    int* top_border_size, int* bottom_border_size) {
  *top_border_size = 0;
  *bottom_border_size = 0;
  MP_RETURN_IF_ERROR(ComputeSceneStaticBordersSize(
      static_features_, top_border_size, bottom_border_size));
  const double scale = static_cast<double>(frame_height_) / key_frame_height_;
  top_border_distance_ = std::round(scale * *top_border_size);
  const int bottom_border_distance = std::round(scale * *bottom_border_size);
  effective_frame_height_ =
      frame_height_ - top_border_distance_ - bottom_border_distance;

  if (top_border_distance_ > 0 || bottom_border_distance > 0) {
    VLOG(1) << "Remove top border " << top_border_distance_ << " bottom border "
            << bottom_border_distance;
    // Remove borders from frames.
    cv::Rect roi(0, top_border_distance_, frame_width_,
                 effective_frame_height_);
    for (int i = 0; i < scene_frames_or_empty_.size(); ++i) {
      cv::Mat tmp;
      scene_frames_or_empty_[i](roi).copyTo(tmp);
      scene_frames_or_empty_[i] = tmp;
    }
    // Adjust detection bounding boxes.
    for (int i = 0; i < key_frame_infos_.size(); ++i) {
      DetectionSet adjusted_detections;
      const auto& detections = key_frame_infos_[i].detections();
      for (int j = 0; j < detections.detections_size(); ++j) {
        const auto& detection = detections.detections(j);
        SalientRegion adjusted_detection = detection;
        // Clamp the box to be within the de-bordered frame.
        if (!ClampRect(0, top_border_distance_, frame_width_,
                       top_border_distance_ + effective_frame_height_,
                       adjusted_detection.mutable_location())
                 .ok()) {
          continue;
        }
        // Offset the y position.
        adjusted_detection.mutable_location()->set_y(
            adjusted_detection.location().y() - top_border_distance_);
        *adjusted_detections.add_detections() = adjusted_detection;
      }
      *key_frame_infos_[i].mutable_detections() = adjusted_detections;
    }
  }
  return ::mediapipe::OkStatus();
}

::mediapipe::Status
SceneCroppingCalculator::InitializeFrameCropRegionComputer() {
  key_frame_crop_options_ = options_.key_frame_crop_options();
  MP_RETURN_IF_ERROR(
      SetKeyFrameCropTarget(frame_width_, effective_frame_height_,
                            target_aspect_ratio_, &key_frame_crop_options_));
  VLOG(1) << "Target width " << key_frame_crop_options_.target_width();
  VLOG(1) << "Target height " << key_frame_crop_options_.target_height();
  frame_crop_region_computer_ =
      absl::make_unique<FrameCropRegionComputer>(key_frame_crop_options_);
  return ::mediapipe::OkStatus();
}

void SceneCroppingCalculator::FilterKeyFrameInfo() {
  if (!options_.user_hint_override()) {
    return;
  }
  std::vector<KeyFrameInfo> user_hints_only;
  bool has_user_hints = false;
  for (auto key_frame : key_frame_infos_) {
    DetectionSet user_hint_only_set;
    for (const auto& detection : key_frame.detections().detections()) {
      if (detection.signal_type().has_standard() &&
          detection.signal_type().standard() == SignalType::USER_HINT) {
        *user_hint_only_set.add_detections() = detection;
        has_user_hints = true;
      }
    }
    *key_frame.mutable_detections() = user_hint_only_set;
    user_hints_only.push_back(key_frame);
  }
  if (has_user_hints) {
    key_frame_infos_ = user_hints_only;
  }
}

::mediapipe::Status SceneCroppingCalculator::ProcessScene(
    const bool is_end_of_scene, CalculatorContext* cc) {
  // Removes detections under special circumstances.
  FilterKeyFrameInfo();

  // Removes any static borders.
  int top_static_border_size, bottom_static_border_size;
  MP_RETURN_IF_ERROR(
      RemoveStaticBorders(&top_static_border_size, &bottom_static_border_size));

  // Decides if solid background color padding is possible and sets up color
  // interpolation functions in CIELAB. Uses linear interpolation by default.
  MP_RETURN_IF_ERROR(FindSolidBackgroundColor(
      static_features_, static_features_timestamps_,
      options_.solid_background_frames_padding_fraction(),
      &has_solid_background_, &background_color_l_function_,
      &background_color_a_function_, &background_color_b_function_));

  // Computes key frame crop regions.
  MP_RETURN_IF_ERROR(InitializeFrameCropRegionComputer());
  const int num_key_frames = key_frame_infos_.size();
  std::vector<KeyFrameCropResult> key_frame_crop_results(num_key_frames);
  for (int i = 0; i < num_key_frames; ++i) {
    MP_RETURN_IF_ERROR(frame_crop_region_computer_->ComputeFrameCropRegion(
        key_frame_infos_[i], &key_frame_crop_results[i]));
  }

  // Analyzes scene camera motion and generates FocusPointFrames.
  auto analyzer_options = options_.scene_camera_motion_analyzer_options();
  analyzer_options.set_allow_sweeping(analyzer_options.allow_sweeping() &&
                                      !has_solid_background_);
  scene_camera_motion_analyzer_ =
      absl::make_unique<SceneCameraMotionAnalyzer>(analyzer_options);
  SceneKeyFrameCropSummary scene_summary;
  std::vector<FocusPointFrame> focus_point_frames;
  SceneCameraMotion scene_camera_motion;
  MP_RETURN_IF_ERROR(
      scene_camera_motion_analyzer_->AnalyzeSceneAndPopulateFocusPointFrames(
          key_frame_infos_, key_frame_crop_options_, key_frame_crop_results,
          frame_width_, effective_frame_height_, scene_frame_timestamps_,
          &scene_summary, &focus_point_frames, &scene_camera_motion));

  // Crops scene frames.
  std::vector<cv::Mat> cropped_frames;
  std::vector<cv::Rect> crop_from_locations;

  auto* cropped_frames_ptr =
      should_perform_frame_cropping_ ? &cropped_frames : nullptr;

  MP_RETURN_IF_ERROR(scene_cropper_->CropFrames(
      scene_summary, scene_frame_timestamps_.size(), scene_frames_or_empty_,
      focus_point_frames, prior_focus_point_frames_, top_static_border_size,
      bottom_static_border_size, &crop_from_locations, cropped_frames_ptr));

  // Formats and outputs cropped frames.
  bool apply_padding = false;
  float vertical_fill_percent;
  std::vector<cv::Rect> render_to_locations;
  std::vector<cv::Scalar> padding_colors;
  if (should_perform_frame_cropping_) {
    MP_RETURN_IF_ERROR(FormatAndOutputCroppedFrames(
        cropped_frames, &render_to_locations, &apply_padding, &padding_colors,
        &vertical_fill_percent, cc));
  }
  // Caches prior FocusPointFrames if this was not the end of a scene.
  prior_focus_point_frames_.clear();
  if (!is_end_of_scene) {
    const int start =
        std::max(0, static_cast<int>(scene_frame_timestamps_.size()) -
                        options_.prior_frame_buffer_size());
    for (int i = start; i < num_key_frames; ++i) {
      prior_focus_point_frames_.push_back(focus_point_frames[i]);
    }
  }

  // Optionally outputs visualization frames.
  MP_RETURN_IF_ERROR(OutputVizFrames(key_frame_crop_results, focus_point_frames,
                                     scene_summary.crop_window_width(),
                                     scene_summary.crop_window_height(), cc));

  const double start_sec = Timestamp(scene_frame_timestamps_.front()).Seconds();
  const double end_sec = Timestamp(scene_frame_timestamps_.back()).Seconds();
  VLOG(1) << absl::StrFormat("Processed a scene from %.2f sec to %.2f sec",
                             start_sec, end_sec);

  // Optionally makes summary.
  if (cc->Outputs().HasTag(kOutputSummary)) {
    auto* scene_summary = summary_->add_scene_summaries();
    scene_summary->set_start_sec(start_sec);
    scene_summary->set_end_sec(end_sec);
    *(scene_summary->mutable_camera_motion()) = scene_camera_motion;
    scene_summary->set_is_end_of_scene(is_end_of_scene);
    scene_summary->set_is_padded(apply_padding);
  }

  if (cc->Outputs().HasTag(kExternalRenderingPerFrame)) {
    for (int i = 0; i < scene_frame_timestamps_.size(); i++) {
      auto external_render_message = absl::make_unique<ExternalRenderFrame>();
      ConstructExternalRenderMessage(
          crop_from_locations[i], render_to_locations[i], padding_colors[i],
          scene_frame_timestamps_[i], external_render_message.get());
      cc->Outputs()
          .Tag(kExternalRenderingPerFrame)
          .Add(external_render_message.release(),
               Timestamp(scene_frame_timestamps_[i]));
    }
  }

  if (cc->Outputs().HasTag(kExternalRenderingFullVid)) {
    for (int i = 0; i < scene_frame_timestamps_.size(); i++) {
      ExternalRenderFrame render_frame;
      ConstructExternalRenderMessage(crop_from_locations[i],
                                     render_to_locations[i], padding_colors[i],
                                     scene_frame_timestamps_[i], &render_frame);
      external_render_list_->push_back(render_frame);
    }
  }

  key_frame_infos_.clear();
  scene_frames_or_empty_.clear();
  scene_frame_timestamps_.clear();
  is_key_frames_.clear();
  static_features_.clear();
  static_features_timestamps_.clear();
  return ::mediapipe::OkStatus();
}

::mediapipe::Status SceneCroppingCalculator::FormatAndOutputCroppedFrames(
    const std::vector<cv::Mat>& cropped_frames,
    std::vector<cv::Rect>* render_to_locations, bool* apply_padding,
    std::vector<cv::Scalar>* padding_colors, float* vertical_fill_percent,
    CalculatorContext* cc) {
  RET_CHECK(apply_padding) << "Has padding boolean is null.";
  if (cropped_frames.empty()) {
    return ::mediapipe::OkStatus();
  }

  // Computes scaling factor and decides if padding is needed.
  const int crop_width = cropped_frames.front().cols;
  const int crop_height = cropped_frames.front().rows;
  VLOG(1) << "crop_width = " << crop_width << " crop_height = " << crop_height;
  const double scaling =
      std::max(static_cast<double>(target_width_) / crop_width,
               static_cast<double>(target_height_) / crop_height);
  int scaled_width = std::round(scaling * crop_width);
  int scaled_height = std::round(scaling * crop_height);
  RET_CHECK_GE(scaled_width, target_width_)
      << "Scaled width is less than target width - something is wrong.";
  RET_CHECK_GE(scaled_height, target_height_)
      << "Scaled height is less than target height - something is wrong.";
  if (scaled_width - target_width_ <= 1) scaled_width = target_width_;
  if (scaled_height - target_height_ <= 1) scaled_height = target_height_;
  *apply_padding =
      scaled_width != target_width_ || scaled_height != target_height_;
  *vertical_fill_percent = scaled_height / static_cast<float>(target_height_);
  if (*apply_padding) {
    padder_ = absl::make_unique<PaddingEffectGenerator>(
        scaled_width, scaled_height, target_aspect_ratio_);
    VLOG(1) << "Scene is padded: scaled width = " << scaled_width
            << " target width = " << target_width_
            << " scaled height = " << scaled_height
            << " target height = " << target_height_;
  }

  // Compute the "render to" location.  This is where the rect taken from the
  // input video gets pasted on the output frame.  For use with external
  // rendering solutions.
  const int num_frames = cropped_frames.size();
  for (int i = 0; i < num_frames; i++) {
    if (*apply_padding) {
      render_to_locations->push_back(padder_->ComputeOutputLocation());
    } else {
      render_to_locations->push_back(
          cv::Rect(0, 0, target_width_, target_height_));
    }
  }

  // Resizes cropped frames, pads frames, and output frames.
  cv::Scalar* background_color = nullptr;
  cv::Scalar interpolated_color;
  for (int i = 0; i < num_frames; ++i) {
    // Set default padding color to white.
    cv::Scalar padding_color_to_add = cv::Scalar(255, 255, 255);
    const int64 time_ms = scene_frame_timestamps_[i];
    const Timestamp timestamp(time_ms);
    auto scaled_frame = absl::make_unique<ImageFrame>(
        frame_format_, scaled_width, scaled_height);
    auto destination = formats::MatView(scaled_frame.get());
    if (scaled_width == crop_width && scaled_height == crop_height) {
      cropped_frames[i].copyTo(destination);
    } else {
      // cubic is better quality for upscaling and area is good for downscaling
      const int interpolation_method =
          scaling > 1 ? cv::INTER_CUBIC : cv::INTER_AREA;
      cv::resize(cropped_frames[i], destination, destination.size(), 0, 0,
                 interpolation_method);
    }
    if (*apply_padding) {
      if (has_solid_background_) {
        double lab[3];
        lab[0] = background_color_l_function_.Evaluate(time_ms);
        lab[1] = background_color_a_function_.Evaluate(time_ms);
        lab[2] = background_color_b_function_.Evaluate(time_ms);
        cv::Mat3f lab_mat(1, 1, cv::Vec3f(lab[0], lab[1], lab[2]));
        cv::Mat3f rgb_mat(1, 1);
        // Necessary scaling of the RGB values from [0, 1] to [0, 255] based on:
        // https://docs.opencv.org/2.4/modules/imgproc/doc/miscellaneous_transformations.html#cvtcolor
        cv::cvtColor(lab_mat, rgb_mat, cv::COLOR_Lab2RGB);
        rgb_mat *= 255.0;
        auto k = rgb_mat.at<cv::Vec3f>(0, 0);
        k[0] = k[0] < 0.0 ? 0.0 : k[0] > 255.0 ? 255.0 : k[0];
        k[1] = k[1] < 0.0 ? 0.0 : k[1] > 255.0 ? 255.0 : k[1];
        k[2] = k[2] < 0.0 ? 0.0 : k[2] > 255.0 ? 255.0 : k[2];
        interpolated_color =
            cv::Scalar(std::round(k[0]), std::round(k[1]), std::round(k[2]));
        background_color = &interpolated_color;
        padding_color_to_add = interpolated_color;
      }
      auto padded_frame = absl::make_unique<ImageFrame>();
      MP_RETURN_IF_ERROR(padder_->Process(
          *scaled_frame, background_contrast_,
          std::min({blur_cv_size_, scaled_width, scaled_height}),
          overlay_opacity_, padded_frame.get(), background_color));
      RET_CHECK_EQ(padded_frame->Width(), target_width_)
          << "Padded frame width is off.";
      RET_CHECK_EQ(padded_frame->Height(), target_height_)
          << "Padded frame height is off.";
      cc->Outputs()
          .Tag(kOutputCroppedFrames)
          .Add(padded_frame.release(), timestamp);
    } else {
      cc->Outputs()
          .Tag(kOutputCroppedFrames)
          .Add(scaled_frame.release(), timestamp);
    }
    padding_colors->push_back(padding_color_to_add);
  }
  return ::mediapipe::OkStatus();
}

mediapipe::Status SceneCroppingCalculator::OutputVizFrames(
    const std::vector<KeyFrameCropResult>& key_frame_crop_results,
    const std::vector<FocusPointFrame>& focus_point_frames,
    const int crop_window_width, const int crop_window_height,
    CalculatorContext* cc) const {
  if (cc->Outputs().HasTag(kOutputKeyFrameCropViz)) {
    std::vector<std::unique_ptr<ImageFrame>> viz_frames;
    MP_RETURN_IF_ERROR(DrawDetectionsAndCropRegions(
        scene_frames_or_empty_, is_key_frames_, key_frame_infos_,
        key_frame_crop_results, frame_format_, &viz_frames));
    for (int i = 0; i < scene_frames_or_empty_.size(); ++i) {
      cc->Outputs()
          .Tag(kOutputKeyFrameCropViz)
          .Add(viz_frames[i].release(), Timestamp(scene_frame_timestamps_[i]));
    }
  }
  if (cc->Outputs().HasTag(kOutputFocusPointFrameViz)) {
    std::vector<std::unique_ptr<ImageFrame>> viz_frames;
    MP_RETURN_IF_ERROR(DrawFocusPointAndCropWindow(
        scene_frames_or_empty_, focus_point_frames,
        options_.viz_overlay_opacity(), crop_window_width, crop_window_height,
        frame_format_, &viz_frames));
    for (int i = 0; i < scene_frames_or_empty_.size(); ++i) {
      cc->Outputs()
          .Tag(kOutputFocusPointFrameViz)
          .Add(viz_frames[i].release(), Timestamp(scene_frame_timestamps_[i]));
    }
  }
  return ::mediapipe::OkStatus();
}

REGISTER_CALCULATOR(SceneCroppingCalculator);

}  // namespace autoflip
}  // namespace mediapipe
