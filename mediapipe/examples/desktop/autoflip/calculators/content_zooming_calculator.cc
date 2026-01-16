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

#include <algorithm>
#include <cmath>
#include <memory>

#include "absl/status/status.h"
#include "mediapipe/examples/desktop/autoflip/autoflip_messages.pb.h"
#include "mediapipe/examples/desktop/autoflip/calculators/content_zooming_calculator.pb.h"
#include "mediapipe/examples/desktop/autoflip/calculators/content_zooming_calculator_state.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/detection.pb.h"
#include "mediapipe/framework/formats/image_frame.h"
#include "mediapipe/framework/formats/location_data.pb.h"
#include "mediapipe/framework/packet.h"
#include "mediapipe/framework/port/ret_check.h"
#include "mediapipe/framework/port/status_builder.h"

constexpr char kVideoFrame[] = "VIDEO";
constexpr char kVideoSize[] = "VIDEO_SIZE";
constexpr char kSalientRegions[] = "SALIENT_REGIONS";
constexpr char kDetections[] = "DETECTIONS";
constexpr char kDetectedBorders[] = "BORDERS";
// Crop location as abs rect discretized.
constexpr char kCropRect[] = "CROP_RECT";
// Crop location as normalized rect.
constexpr char kNormalizedCropRect[] = "NORMALIZED_CROP_RECT";
// Crop location without position smoothing.
constexpr char kFirstCropRect[] = "FIRST_CROP_RECT";
// Can be used to control whether an animated zoom should actually performed
// (configured through option us_to_first_rect). If provided, a non-zero integer
// will allow the animated zoom to be used when the first detections arrive.
// Applies to first detection only.
constexpr char kAnimateZoom[] = "ANIMATE_ZOOM";
// Can be used to control the maximum zoom; note that it is re-evaluated only
// upon change of input resolution. A value of 100 disables zooming and is the
// smallest allowed value. A value of 200 allows zooming such that a pixel of
// the input may cover up to four times its original area. Note that
// max_zoom_value_deg from options is always respected; MAX_ZOOM_PCT can only be
// used to limit zooming further.
constexpr char kMaxZoomFactorPercent[] = "MAX_ZOOM_FACTOR_PCT";
// Can be used to control the scale factor applied when zooming. Note that this
// overrides the scale_factor from options.
constexpr char kScaleFactorPercent[] = "SCALE_FACTOR_PCT";
// Field-of-view (degrees) of the camera's x-axis (width).
// TODO: Parameterize FOV based on camera specs.
constexpr float kFieldOfView = 60;
// A pointer to a ContentZoomingCalculatorStateCacheType in a side packet.
// Used to save state on Close and load state on Open in a new graph.
// Can be used to preserve state between graphs.
constexpr char kStateCache[] = "STATE_CACHE";
// Tolerance for zooming out recentering.
constexpr float kPixelTolerance = 3;
// Returns 'true' when camera is moving (pan/tilt/zoom) & 'false' for no motion.
constexpr char kCameraActive[] = "CAMERA_ACTIVE";

namespace mediapipe {
namespace autoflip {
using StateCacheType = ContentZoomingCalculatorStateCacheType;

// Content zooming calculator zooms in on content when a detection has
// "only_required" set true or any raw detection input.  It does this by
// computing the value of top/bottom borders to remove from the output and sends
// these to the SceneCroppingCalculator using BORDERS output or a full rect crop
// using CROP_RECT output.  When more than one detections are received the
// zoom box is calculated as the union of the detections.  Typical applications
// include mobile makeover and autofliplive face reframing.
class ContentZoomingCalculator : public CalculatorBase {
 public:
  ContentZoomingCalculator() : initialized_(false) {}
  ~ContentZoomingCalculator() override {}
  ContentZoomingCalculator(const ContentZoomingCalculator&) = delete;
  ContentZoomingCalculator& operator=(const ContentZoomingCalculator&) = delete;

  static absl::Status GetContract(mediapipe::CalculatorContract* cc);
  absl::Status Open(mediapipe::CalculatorContext* cc) override;
  absl::Status Process(mediapipe::CalculatorContext* cc) override;
  absl::Status Close(mediapipe::CalculatorContext* cc) override;

 private:
  // Tries to load state from a state-cache, if provided. Fallsback to
  // initializing state if no cache or no value in the cache are available.
  absl::Status MaybeLoadState(mediapipe::CalculatorContext* cc, int frame_width,
                              int frame_height);
  // Saves state to a state-cache, if provided.
  absl::Status SaveState(mediapipe::CalculatorContext* cc) const;
  // Returns the factor for maximum zoom based on options and the
  // kMaxZoomFactorPercent input (if present).
  double GetMaxZoomFactor(mediapipe::CalculatorContext* cc) const;
  // Returns the factor for scale based on options and the
  // kScaleFactorPercent input (if present).
  double GetScaleFactor(mediapipe::CalculatorContext* cc) const;
  // Initializes the calculator for the given frame size, creating path solvers
  // and resetting history like last measured values.
  absl::Status InitializeState(mediapipe::CalculatorContext* cc,
                               int frame_width, int frame_height);
  // Adjusts state to work with an updated frame size.
  absl::Status UpdateForResolutionChange(mediapipe::CalculatorContext* cc,
                                         int frame_width, int frame_height);
  // Returns true if we are animating to the first rect.
  bool IsAnimatingToFirstRect(const Timestamp& timestamp) const;
  // Builds the output rectangle when animating to the first rect.
  absl::StatusOr<mediapipe::Rect> GetAnimationRect(
      int frame_width, int frame_height, const Timestamp& timestamp) const;
  // Converts bounds to tilt offset, pan offset and height.
  absl::Status ConvertToPanTiltZoom(float xmin, float xmax, float ymin,
                                    float ymax, double scale_factor,
                                    int* tilt_offset, int* pan_offset,
                                    int* height);
  // Sets max_frame_value_ and target_aspect_
  absl::Status UpdateAspectAndMax();
  // Smooth camera path
  absl::Status SmoothAndClampPath(int target_width, int target_height,
                                  float path_width, float path_height,
                                  float* path_offset_x, float* path_offset_y);
  // Compute box containing all detections.
  absl::Status GetDetectionsBox(mediapipe::CalculatorContext* cc, float* xmin,
                                float* xmax, float* ymin, float* ymax,
                                bool* only_required_found,
                                bool* has_detections);

  ContentZoomingCalculatorOptions options_;
  // Detection frame width/height.
  int frame_height_;
  int frame_width_;
  // Path solver used to smooth top/bottom border crop values.
  std::unique_ptr<KinematicPathSolver> path_solver_zoom_;
  std::unique_ptr<KinematicPathSolver> path_solver_pan_;
  std::unique_ptr<KinematicPathSolver> path_solver_tilt_;
  // Are parameters initialized.
  bool initialized_;
  // Stores the time of the first crop rectangle. This is used to control
  // animating to it. Until a first crop rectangle was computed, it has
  // the value Timestamp::Unset(). If animating is not requested, it receives
  // the value Timestamp::Done() instead of the time.
  Timestamp first_rect_timestamp_;
  // Stores the first crop rectangle.
  mediapipe::NormalizedRect first_rect_;
  // Stores the time of the last "only_required" input.
  int64_t last_only_required_detection_;
  // Rect values of last message with detection(s).
  int last_measured_height_;
  int last_measured_x_offset_;
  int last_measured_y_offset_;
  // Target aspect ratio.
  float target_aspect_;
  // Max size of bounding box.  If input/output aspect ratios are the same,
  // will be 1.0.  Else, will be less than 1.0 to prevent exceeding the size of
  // the image in either dimension.
  float max_frame_value_;
};
REGISTER_CALCULATOR(ContentZoomingCalculator);

absl::Status ContentZoomingCalculator::GetContract(
    mediapipe::CalculatorContract* cc) {
  RET_CHECK(
      !(cc->Inputs().HasTag(kVideoFrame) && cc->Inputs().HasTag(kVideoSize)))
      << "Provide only VIDEO or VIDEO_SIZE, not both.";
  if (cc->Inputs().HasTag(kVideoFrame)) {
    cc->Inputs().Tag(kVideoFrame).Set<ImageFrame>();
  } else if (cc->Inputs().HasTag(kVideoSize)) {
    cc->Inputs().Tag(kVideoSize).Set<std::pair<int, int>>();
  } else {
    return mediapipe::UnknownErrorBuilder(MEDIAPIPE_LOC)
           << "Input VIDEO or VIDEO_SIZE must be provided.";
  }
  if (cc->Inputs().HasTag(kMaxZoomFactorPercent)) {
    cc->Inputs().Tag(kMaxZoomFactorPercent).Set<int>();
  }
  if (cc->Inputs().HasTag(kScaleFactorPercent)) {
    cc->Inputs().Tag(kScaleFactorPercent).Set<int>();
  }
  if (cc->Inputs().HasTag(kSalientRegions)) {
    cc->Inputs().Tag(kSalientRegions).Set<DetectionSet>();
  }
  if (cc->Inputs().HasTag(kDetections)) {
    cc->Inputs().Tag(kDetections).Set<std::vector<mediapipe::Detection>>();
  }
  if (cc->Inputs().HasTag(kAnimateZoom)) {
    cc->Inputs().Tag(kAnimateZoom).Set<bool>();
  }
  if (cc->Outputs().HasTag(kDetectedBorders)) {
    cc->Outputs().Tag(kDetectedBorders).Set<StaticFeatures>();
  }
  if (cc->Outputs().HasTag(kCropRect)) {
    cc->Outputs().Tag(kCropRect).Set<mediapipe::Rect>();
  }
  if (cc->Outputs().HasTag(kNormalizedCropRect)) {
    cc->Outputs().Tag(kNormalizedCropRect).Set<mediapipe::NormalizedRect>();
  }
  if (cc->Outputs().HasTag(kFirstCropRect)) {
    cc->Outputs().Tag(kFirstCropRect).Set<mediapipe::NormalizedRect>();
  }
  if (cc->InputSidePackets().HasTag(kStateCache)) {
    cc->InputSidePackets().Tag(kStateCache).Set<StateCacheType*>();
  }
  if (cc->Outputs().HasTag(kCameraActive)) {
    cc->Outputs().Tag(kCameraActive).Set<bool>();
  }
  return absl::OkStatus();
}

absl::Status ContentZoomingCalculator::Open(mediapipe::CalculatorContext* cc) {
  cc->SetOffset(mediapipe::TimestampDiff(0));
  options_ = cc->Options<ContentZoomingCalculatorOptions>();
  if (options_.has_kinematic_options()) {
    return mediapipe::UnknownErrorBuilder(MEDIAPIPE_LOC)
           << "Deprecated kinematic_options was set, please set "
              "kinematic_options_zoom and kinematic_options_tilt.";
  }
  if (options_.has_min_motion_to_reframe()) {
    return mediapipe::UnknownErrorBuilder(MEDIAPIPE_LOC)
           << "Deprecated min_motion_to_reframe was set, please set "
              "in kinematic_options_zoom and kinematic_options_tilt "
              "directly.";
  }
  return absl::OkStatus();
}

absl::Status ContentZoomingCalculator::Close(mediapipe::CalculatorContext* cc) {
  if (initialized_) {
    MP_RETURN_IF_ERROR(SaveState(cc));
  }
  return absl::OkStatus();
}

absl::Status ContentZoomingCalculator::ConvertToPanTiltZoom(
    float xmin, float xmax, float ymin, float ymax, double scale_factor,
    int* tilt_offset, int* pan_offset, int* height) {
  // Find center of the y-axis offset (for tilt control).
  float y_center = ymin + (ymax - ymin) / 2;
  // Find center of the x-axis offset (for pan control).
  float x_center = xmin + (xmax - xmin) / 2;
  // Find size and apply scale factor to y-axis.
  float fit_size_raw = fmax((ymax - ymin) / scale_factor, xmax - xmin);
  // Apply max frame for cases where the target size is different than input
  // frame size.
  float fit_size = fmin(max_frame_value_, fit_size_raw);
  // Prevent box from extending beyond the image.
  if (!options_.allow_cropping_outside_frame()) {
    float half_fit_size = fit_size / 2.0f;
    y_center = std::clamp(y_center, half_fit_size, 1 - half_fit_size);
    x_center = std::clamp(x_center, half_fit_size, 1 - half_fit_size);
  }
  // Scale to pixel coordinates.
  *tilt_offset = frame_height_ * y_center;
  *pan_offset = frame_width_ * x_center;
  *height = frame_height_ * fit_size_raw;
  return absl::OkStatus();
}

namespace {
mediapipe::LocationData::RelativeBoundingBox ShiftDetection(
    const mediapipe::LocationData::RelativeBoundingBox& relative_bounding_box,
    const float y_offset_percent, const float x_offset_percent) {
  auto shifted_bb = relative_bounding_box;
  shifted_bb.set_ymin(relative_bounding_box.ymin() +
                      relative_bounding_box.height() * y_offset_percent);
  shifted_bb.set_xmin(relative_bounding_box.xmin() +
                      relative_bounding_box.width() * x_offset_percent);
  return shifted_bb;
}
mediapipe::autoflip::RectF ShiftDetection(
    const mediapipe::autoflip::RectF& relative_bounding_box,
    const float y_offset_percent, const float x_offset_percent) {
  auto shifted_bb = relative_bounding_box;
  shifted_bb.set_y(relative_bounding_box.y() +
                   relative_bounding_box.height() * y_offset_percent);
  shifted_bb.set_x(relative_bounding_box.x() +
                   relative_bounding_box.width() * x_offset_percent);
  return shifted_bb;
}
absl::Status UpdateRanges(const SalientRegion& region,
                          const float shift_vertical,
                          const float shift_horizontal,
                          const float pad_vertical, const float pad_horizontal,
                          float* xmin, float* xmax, float* ymin, float* ymax) {
  if (!region.has_location_normalized()) {
    return mediapipe::UnknownErrorBuilder(MEDIAPIPE_LOC)
           << "SalientRegion did not have location normalized set.";
  }
  auto location = ShiftDetection(region.location_normalized(), shift_vertical,
                                 shift_horizontal);

  const float x_padding = pad_horizontal * location.width();
  const float y_padding = pad_vertical * location.height();

  *xmin = fmin(*xmin, location.x() - x_padding);
  *xmax = fmax(*xmax, location.x() + location.width() + x_padding);
  *ymin = fmin(*ymin, location.y() - y_padding);
  *ymax = fmax(*ymax, location.y() + location.height() + y_padding);

  return absl::OkStatus();
}
absl::Status UpdateRanges(const mediapipe::Detection& detection,
                          const float shift_vertical,
                          const float shift_horizontal,
                          const float pad_vertical, const float pad_horizontal,
                          float* xmin, float* xmax, float* ymin, float* ymax) {
  RET_CHECK(detection.location_data().format() ==
            mediapipe::LocationData::RELATIVE_BOUNDING_BOX)
      << "Face detection input is lacking required relative_bounding_box()";
  const auto& location =
      ShiftDetection(detection.location_data().relative_bounding_box(),
                     shift_vertical, shift_horizontal);

  const float x_padding = pad_horizontal * location.width();
  const float y_padding = pad_vertical * location.height();

  *xmin = fmin(*xmin, location.xmin() - x_padding);
  *xmax = fmax(*xmax, location.xmin() + location.width() + x_padding);
  *ymin = fmin(*ymin, location.ymin() - y_padding);
  *ymax = fmax(*ymax, location.ymin() + location.height() + y_padding);

  return absl::OkStatus();
}
void MakeStaticFeatures(const int top_border, const int bottom_border,
                        const int frame_width, const int frame_height,
                        StaticFeatures* static_feature) {
  auto border_top = static_feature->add_border();
  border_top->set_relative_position(Border::TOP);
  border_top->mutable_border_position()->set_x(0);
  border_top->mutable_border_position()->set_y(0);
  border_top->mutable_border_position()->set_width(frame_width);
  border_top->mutable_border_position()->set_height(top_border);

  auto border_bottom = static_feature->add_border();
  border_bottom->set_relative_position(Border::BOTTOM);
  border_bottom->mutable_border_position()->set_x(0);
  border_bottom->mutable_border_position()->set_y(frame_height - bottom_border);
  border_bottom->mutable_border_position()->set_width(frame_width);
  border_bottom->mutable_border_position()->set_height(bottom_border);
}
absl::Status GetVideoResolution(mediapipe::CalculatorContext* cc,
                                int* frame_width, int* frame_height) {
  if (cc->Inputs().HasTag(kVideoFrame)) {
    *frame_width = cc->Inputs().Tag(kVideoFrame).Get<ImageFrame>().Width();
    *frame_height = cc->Inputs().Tag(kVideoFrame).Get<ImageFrame>().Height();
  } else if (cc->Inputs().HasTag(kVideoSize)) {
    *frame_width =
        cc->Inputs().Tag(kVideoSize).Get<std::pair<int, int>>().first;
    *frame_height =
        cc->Inputs().Tag(kVideoSize).Get<std::pair<int, int>>().second;
  } else {
    return mediapipe::UnknownErrorBuilder(MEDIAPIPE_LOC)
           << "Input VIDEO or VIDEO_SIZE must be provided.";
  }
  return absl::OkStatus();
}
}  // namespace

absl::Status ContentZoomingCalculator::UpdateAspectAndMax() {
  max_frame_value_ = 1.0;
  target_aspect_ = frame_width_ / static_cast<float>(frame_height_);
  // If target size is set and wider than input aspect, make sure to always
  // crop the min required amount.
  if (options_.has_target_size()) {
    RET_CHECK_GT(options_.target_size().width(), 0)
        << "Provided target width not valid.";
    RET_CHECK_GT(options_.target_size().height(), 0)
        << "Provided target height not valid.";
    float input_aspect = frame_width_ / static_cast<float>(frame_height_);
    target_aspect_ = options_.target_size().width() /
                     static_cast<float>(options_.target_size().height());
    max_frame_value_ =
        std::min(input_aspect / target_aspect_, target_aspect_ / input_aspect);
  }
  return absl::OkStatus();
}

absl::Status ContentZoomingCalculator::MaybeLoadState(
    mediapipe::CalculatorContext* cc, int frame_width, int frame_height) {
  const auto* state_cache =
      cc->InputSidePackets().HasTag(kStateCache)
          ? cc->InputSidePackets().Tag(kStateCache).Get<StateCacheType*>()
          : nullptr;
  if (!state_cache || !state_cache->has_value()) {
    return InitializeState(cc, frame_width, frame_height);
  }

  const ContentZoomingCalculatorState& state = state_cache->value();
  frame_width_ = state.frame_width;
  frame_height_ = state.frame_height;
  path_solver_pan_ =
      std::make_unique<KinematicPathSolver>(state.path_solver_pan);
  path_solver_tilt_ =
      std::make_unique<KinematicPathSolver>(state.path_solver_tilt);
  path_solver_zoom_ =
      std::make_unique<KinematicPathSolver>(state.path_solver_zoom);
  first_rect_timestamp_ = state.first_rect_timestamp;
  first_rect_ = state.first_rect;
  last_only_required_detection_ = state.last_only_required_detection;
  last_measured_height_ = state.last_measured_height;
  last_measured_x_offset_ = state.last_measured_x_offset;
  last_measured_y_offset_ = state.last_measured_y_offset;
  MP_RETURN_IF_ERROR(UpdateAspectAndMax());

  return UpdateForResolutionChange(cc, frame_width, frame_height);
}

absl::Status ContentZoomingCalculator::SaveState(
    mediapipe::CalculatorContext* cc) const {
  auto* state_cache =
      cc->InputSidePackets().HasTag(kStateCache)
          ? cc->InputSidePackets().Tag(kStateCache).Get<StateCacheType*>()
          : nullptr;
  if (!state_cache) {
    return absl::OkStatus();
  }

  *state_cache = ContentZoomingCalculatorState{
      .frame_height = frame_height_,
      .frame_width = frame_width_,
      .path_solver_zoom = *path_solver_zoom_,
      .path_solver_pan = *path_solver_pan_,
      .path_solver_tilt = *path_solver_tilt_,
      .first_rect_timestamp = first_rect_timestamp_,
      .first_rect = first_rect_,
      .last_only_required_detection = last_only_required_detection_,
      .last_measured_height = last_measured_height_,
      .last_measured_x_offset = last_measured_x_offset_,
      .last_measured_y_offset = last_measured_y_offset_,
  };
  return absl::OkStatus();
}

double ContentZoomingCalculator::GetMaxZoomFactor(
    mediapipe::CalculatorContext* cc) const {
  double max_zoom_value =
      options_.max_zoom_value_deg() / static_cast<double>(kFieldOfView);
  if (cc->Inputs().HasTag(kMaxZoomFactorPercent)) {
    const double factor = std::max(
        1.0, cc->Inputs().Tag(kMaxZoomFactorPercent).Get<int>() / 100.0);
    max_zoom_value = std::max(max_zoom_value, 1.0 / factor);
  }
  return max_zoom_value;
}

double ContentZoomingCalculator::GetScaleFactor(
    mediapipe::CalculatorContext* cc) const {
  const double min_scale_factor = options_.scale_factor();
  if (cc->Inputs().HasTag(kScaleFactorPercent)) {
    const double factor =
        cc->Inputs().Tag(kScaleFactorPercent).Get<int>() / 100.0;
    if (factor > 0.0) {
      return std::min(factor, 1.0);
    }
  }
  return min_scale_factor;
}

absl::Status ContentZoomingCalculator::InitializeState(
    mediapipe::CalculatorContext* cc, int frame_width, int frame_height) {
  frame_width_ = frame_width;
  frame_height_ = frame_height;
  path_solver_pan_ = std::make_unique<KinematicPathSolver>(
      options_.kinematic_options_pan(), 0, frame_width_,
      static_cast<float>(frame_width_) / kFieldOfView);
  path_solver_tilt_ = std::make_unique<KinematicPathSolver>(
      options_.kinematic_options_tilt(), 0, frame_height_,
      static_cast<float>(frame_height_) / kFieldOfView);
  MP_RETURN_IF_ERROR(UpdateAspectAndMax());
  int min_zoom_size = frame_height_ * GetMaxZoomFactor(cc);
  path_solver_zoom_ = std::make_unique<KinematicPathSolver>(
      options_.kinematic_options_zoom(), min_zoom_size,
      max_frame_value_ * frame_height_,
      static_cast<float>(frame_height_) / kFieldOfView);
  first_rect_timestamp_ = Timestamp::Unset();
  last_only_required_detection_ = 0;
  last_measured_height_ = max_frame_value_ * frame_height_;
  last_measured_x_offset_ = frame_width_ / 2;
  last_measured_y_offset_ = frame_height_ / 2;
  return absl::OkStatus();
}

absl::Status ContentZoomingCalculator::UpdateForResolutionChange(
    mediapipe::CalculatorContext* cc, int frame_width, int frame_height) {
  // Update state for change in input resolution.
  if (frame_width_ != frame_width || frame_height_ != frame_height) {
    double width_scale = frame_width / static_cast<double>(frame_width_);
    double height_scale = frame_height / static_cast<double>(frame_height_);
    last_measured_height_ = last_measured_height_ * height_scale;
    last_measured_y_offset_ = last_measured_y_offset_ * height_scale;
    last_measured_x_offset_ = last_measured_x_offset_ * width_scale;
    frame_width_ = frame_width;
    frame_height_ = frame_height;
    MP_RETURN_IF_ERROR(UpdateAspectAndMax());
    MP_RETURN_IF_ERROR(path_solver_pan_->UpdateMinMaxLocation(0, frame_width_));
    MP_RETURN_IF_ERROR(
        path_solver_tilt_->UpdateMinMaxLocation(0, frame_height_));
    int min_zoom_size = frame_height_ * GetMaxZoomFactor(cc);
    MP_RETURN_IF_ERROR(path_solver_zoom_->UpdateMinMaxLocation(
        min_zoom_size, max_frame_value_ * frame_height_));
    MP_RETURN_IF_ERROR(path_solver_zoom_->UpdatePixelsPerDegree(
        static_cast<float>(frame_height_) / kFieldOfView));
  }
  return absl::OkStatus();
}

bool ContentZoomingCalculator::IsAnimatingToFirstRect(
    const Timestamp& timestamp) const {
  if (options_.us_to_first_rect() == 0 ||
      first_rect_timestamp_ == Timestamp::Unset() ||
      first_rect_timestamp_ == Timestamp::Done()) {
    return false;
  }

  const int64_t delta_us = (timestamp - first_rect_timestamp_).Value();
  return (0 <= delta_us && delta_us <= options_.us_to_first_rect());
}

namespace {
double easeInQuad(double t) { return t * t; }
double easeOutQuad(double t) { return -1 * t * (t - 2); }
double easeInOutQuad(double t) {
  if (t < 0.5) {
    return easeInQuad(t * 2) * 0.5;
  } else {
    return easeOutQuad(t * 2 - 1) * 0.5 + 0.5;
  }
}
double lerp(double a, double b, double i) { return a * (1 - i) + b * i; }
}  // namespace

absl::StatusOr<mediapipe::Rect> ContentZoomingCalculator::GetAnimationRect(
    int frame_width, int frame_height, const Timestamp& timestamp) const {
  RET_CHECK(IsAnimatingToFirstRect(timestamp))
      << "Must only be called if animating to first rect.";

  const int64_t delta_us = (timestamp - first_rect_timestamp_).Value();
  const int64_t delay = options_.us_to_first_rect_delay();
  const double interpolation = easeInOutQuad(std::max(
      0.0, (delta_us - delay) /
               static_cast<double>(options_.us_to_first_rect() - delay)));

  const double x_center = lerp(0.5, first_rect_.x_center(), interpolation);
  const double y_center = lerp(0.5, first_rect_.y_center(), interpolation);
  const double width = lerp(1.0, first_rect_.width(), interpolation);
  const double height = lerp(1.0, first_rect_.height(), interpolation);

  mediapipe::Rect gpu_rect;
  gpu_rect.set_x_center(x_center * frame_width);
  gpu_rect.set_width(width * frame_width);
  gpu_rect.set_y_center(y_center * frame_height);
  gpu_rect.set_height(height * frame_height);
  return gpu_rect;
}

absl::Status ContentZoomingCalculator::Process(
    mediapipe::CalculatorContext* cc) {
  // For async subgraph support, return on empty video size packets.
  if (cc->Inputs().HasTag(kVideoSize) &&
      cc->Inputs().Tag(kVideoSize).IsEmpty()) {
    return absl::OkStatus();
  }
  int frame_width, frame_height;
  MP_RETURN_IF_ERROR(GetVideoResolution(cc, &frame_width, &frame_height));

  // Init on first call or re-init always if configured to be stateless.
  if (!initialized_) {
    MP_RETURN_IF_ERROR(MaybeLoadState(cc, frame_width, frame_height));
    initialized_ = !options_.is_stateless();
  } else {
    MP_RETURN_IF_ERROR(
        UpdateForResolutionChange(cc, frame_width, frame_height));
  }

  // Compute the box that contains all "is_required" detections.
  float xmin = 1, ymin = 1, xmax = 0, ymax = 0;
  bool only_required_found = false;
  bool has_detections = true;
  MP_RETURN_IF_ERROR(GetDetectionsBox(cc, &xmin, &xmax, &ymin, &ymax,
                                      &only_required_found, &has_detections));
  if (!has_detections) return absl::OkStatus();

  const bool may_start_animation = (options_.us_to_first_rect() != 0) &&
                                   (!cc->Inputs().HasTag(kAnimateZoom) ||
                                    cc->Inputs().Tag(kAnimateZoom).Get<bool>());
  bool is_animating = IsAnimatingToFirstRect(cc->InputTimestamp());

  int offset_y, height, offset_x;
  if (!is_animating && options_.start_zoomed_out() && !may_start_animation &&
      first_rect_timestamp_ == Timestamp::Unset()) {
    // If we should start zoomed out and won't be doing an animation,
    // initialize the path solvers using the full frame, ignoring detections.
    height = max_frame_value_ * frame_height_;
    offset_x = (target_aspect_ * height) / 2;
    offset_y = frame_height_ / 2;
  } else if (!is_animating && only_required_found) {
    // Convert bounds to tilt/zoom and in pixel coordinates.
    const double scale_factor = GetScaleFactor(cc);
    RET_CHECK(scale_factor > 0.0) << "Scale factor must be positive.";
    MP_RETURN_IF_ERROR(ConvertToPanTiltZoom(
        xmin, xmax, ymin, ymax, scale_factor, &offset_y, &offset_x, &height));
    // A only required detection was found.
    last_only_required_detection_ = cc->InputTimestamp().Microseconds();
    last_measured_height_ = height;
    last_measured_x_offset_ = offset_x;
    last_measured_y_offset_ = offset_y;
  } else if (!is_animating && cc->InputTimestamp().Microseconds() -
                                      last_only_required_detection_ >=
                                  options_.us_before_zoomout()) {
    // No only_require detections found within salient regions packets
    // arriving since us_before_zoomout duration.
    height = max_frame_value_ * frame_height_ +
             (options_.kinematic_options_zoom().min_motion_to_reframe() *
              (static_cast<float>(frame_height_) / kFieldOfView));
    offset_x = (target_aspect_ * height) / 2;
    offset_y = frame_height_ / 2;
  } else {
    // Either animating to the first rectangle, or
    // no only detection found but using last detection due to
    // duration_before_zoomout_us setting.
    height = last_measured_height_;
    offset_x = last_measured_x_offset_;
    offset_y = last_measured_y_offset_;
  }

  // Check if the camera is changing in pan, tilt or zoom.  If the camera is in
  // motion disable temporal filtering.
  bool pan_state, tilt_state, zoom_state;
  MP_RETURN_IF_ERROR(path_solver_pan_->PredictMotionState(
      offset_x, cc->InputTimestamp().Microseconds(), &pan_state));
  MP_RETURN_IF_ERROR(path_solver_tilt_->PredictMotionState(
      offset_y, cc->InputTimestamp().Microseconds(), &tilt_state));
  MP_RETURN_IF_ERROR(path_solver_zoom_->PredictMotionState(
      height, cc->InputTimestamp().Microseconds(), &zoom_state));
  if (pan_state || tilt_state || zoom_state) {
    path_solver_pan_->ClearHistory();
    path_solver_tilt_->ClearHistory();
    path_solver_zoom_->ClearHistory();
  }
  const bool camera_active =
      is_animating || ((pan_state || tilt_state || zoom_state) &&
                       !options_.disable_animations());
  // Waiting for first rect before setting any value of the camera active flag
  // so we avoid setting it to false during initialization.
  if (cc->Outputs().HasTag(kCameraActive) &&
      first_rect_timestamp_ != Timestamp::Unset()) {
    cc->Outputs()
        .Tag(kCameraActive)
        .AddPacket(MakePacket<bool>(camera_active).At(cc->InputTimestamp()));
  }

  // Skip the path solvers to the final destination if not animating.
  const bool disable_animations =
      options_.disable_animations() && path_solver_zoom_->IsInitialized();
  if (disable_animations) {
    MP_RETURN_IF_ERROR(path_solver_zoom_->SetState(height));
    MP_RETURN_IF_ERROR(path_solver_tilt_->SetState(offset_y));
    MP_RETURN_IF_ERROR(path_solver_pan_->SetState(offset_x));
  }

  // Compute smoothed zoom camera path.
  MP_RETURN_IF_ERROR(path_solver_zoom_->AddObservation(
      height, cc->InputTimestamp().Microseconds()));
  float path_height;
  MP_RETURN_IF_ERROR(path_solver_zoom_->GetState(&path_height));
  const float path_width = path_height * target_aspect_;

  // Update pixel-per-degree value for pan/tilt.
  int target_height;
  MP_RETURN_IF_ERROR(path_solver_zoom_->GetTargetPosition(&target_height));
  const int target_width = target_height * target_aspect_;
  MP_RETURN_IF_ERROR(path_solver_pan_->UpdatePixelsPerDegree(
      static_cast<float>(target_width) / kFieldOfView));
  MP_RETURN_IF_ERROR(path_solver_tilt_->UpdatePixelsPerDegree(
      static_cast<float>(target_height) / kFieldOfView));

  // Compute smoothed pan/tilt paths.
  MP_RETURN_IF_ERROR(path_solver_pan_->AddObservation(
      offset_x, cc->InputTimestamp().Microseconds()));
  MP_RETURN_IF_ERROR(path_solver_tilt_->AddObservation(
      offset_y, cc->InputTimestamp().Microseconds()));
  float path_offset_x;
  MP_RETURN_IF_ERROR(path_solver_pan_->GetState(&path_offset_x));
  float path_offset_y;
  MP_RETURN_IF_ERROR(path_solver_tilt_->GetState(&path_offset_y));

  // Update path.
  MP_RETURN_IF_ERROR(SmoothAndClampPath(target_width, target_height, path_width,
                                        path_height, &path_offset_x,
                                        &path_offset_y));

  // Transmit result downstream to scenecroppingcalculator.
  if (cc->Outputs().HasTag(kDetectedBorders)) {
    // Convert to top/bottom borders to remove.
    const int path_top = path_offset_y - path_height / 2;
    const int path_bottom = frame_height_ - (path_offset_y + path_height / 2);
    std::unique_ptr<StaticFeatures> features =
        absl::make_unique<StaticFeatures>();
    MakeStaticFeatures(path_top, path_bottom, frame_width_, frame_height_,
                       features.get());
    cc->Outputs()
        .Tag(kDetectedBorders)
        .AddPacket(Adopt(features.release()).At(cc->InputTimestamp()));
  }

  // Record the first crop rectangle
  if (first_rect_timestamp_ == Timestamp::Unset()) {
    first_rect_.set_x_center(path_offset_x / static_cast<float>(frame_width_));
    first_rect_.set_width(path_height * target_aspect_ /
                          static_cast<float>(frame_width_));
    first_rect_.set_y_center(path_offset_y / static_cast<float>(frame_height_));
    first_rect_.set_height(path_height / static_cast<float>(frame_height_));

    // Record the time to serve as departure point for the animation.
    // If we are not allowed to start the animation, set Timestamp::Done.
    first_rect_timestamp_ =
        may_start_animation ? cc->InputTimestamp() : Timestamp::Done();
    // After setting the first rectangle, check whether we should animate to it.
    is_animating = IsAnimatingToFirstRect(cc->InputTimestamp());
  }

  // Transmit downstream to glcroppingcalculator in discrete int values.
  if (cc->Outputs().HasTag(kCropRect)) {
    std::unique_ptr<mediapipe::Rect> gpu_rect;
    if (is_animating) {
      auto rect =
          GetAnimationRect(frame_width, frame_height, cc->InputTimestamp());
      MP_RETURN_IF_ERROR(rect.status());
      gpu_rect = absl::make_unique<mediapipe::Rect>(*rect);
    } else {
      gpu_rect = absl::make_unique<mediapipe::Rect>();
      gpu_rect->set_x_center(path_offset_x);
      gpu_rect->set_width(path_width);
      gpu_rect->set_y_center(path_offset_y);
      gpu_rect->set_height(path_height);
    }
    cc->Outputs().Tag(kCropRect).Add(gpu_rect.release(),
                                     Timestamp(cc->InputTimestamp()));
  }
  if (cc->Outputs().HasTag(kNormalizedCropRect)) {
    std::unique_ptr<mediapipe::NormalizedRect> gpu_rect =
        absl::make_unique<mediapipe::NormalizedRect>();
    const float float_frame_width = static_cast<float>(frame_width_);
    const float float_frame_height = static_cast<float>(frame_height_);
    if (is_animating) {
      auto rect =
          GetAnimationRect(frame_width, frame_height, cc->InputTimestamp());
      MP_RETURN_IF_ERROR(rect.status());
      gpu_rect->set_x_center(rect->x_center() / float_frame_width);
      gpu_rect->set_width(rect->width() / float_frame_width);
      gpu_rect->set_y_center(rect->y_center() / float_frame_height);
      gpu_rect->set_height(rect->height() / float_frame_height);
    } else {
      gpu_rect->set_x_center(path_offset_x / float_frame_width);
      gpu_rect->set_width(path_width / float_frame_width);
      gpu_rect->set_y_center(path_offset_y / float_frame_height);
      gpu_rect->set_height(path_height / float_frame_height);
    }
    cc->Outputs()
        .Tag(kNormalizedCropRect)
        .Add(gpu_rect.release(), Timestamp(cc->InputTimestamp()));
  }

  if (cc->Outputs().HasTag(kFirstCropRect)) {
    cc->Outputs()
        .Tag(kFirstCropRect)
        .Add(new mediapipe::NormalizedRect(first_rect_),
             Timestamp(cc->InputTimestamp()));
  }

  return absl::OkStatus();
}

absl::Status ContentZoomingCalculator::SmoothAndClampPath(
    int target_width, int target_height, float path_width, float path_height,
    float* path_offset_x, float* path_offset_y) {
  if (options_.allow_cropping_outside_frame()) {
    return absl::OkStatus();
  }

  float delta_height;
  MP_RETURN_IF_ERROR(path_solver_zoom_->GetDeltaState(&delta_height));
  const int delta_width = delta_height * target_aspect_;

  // Smooth centering when zooming out.
  const float remaining_width = target_width - path_width;
  const int width_space = frame_width_ - target_width;
  if (abs(*path_offset_x - frame_width_ / 2) >
          width_space / 2 + kPixelTolerance &&
      remaining_width > kPixelTolerance) {
    const float required_width =
        abs(*path_offset_x - frame_width_ / 2) - width_space / 2;
    if (*path_offset_x < frame_width_ / 2) {
      *path_offset_x += delta_width * (required_width / remaining_width);
    } else {
      *path_offset_x -= delta_width * (required_width / remaining_width);
    }
  }

  const float remaining_height = target_height - path_height;
  const int height_space = frame_height_ - target_height;
  if (abs(*path_offset_y - frame_height_ / 2) >
          height_space / 2 + kPixelTolerance &&
      remaining_height > kPixelTolerance) {
    const float required_height =
        abs(*path_offset_y - frame_height_ / 2) - height_space / 2;
    if (*path_offset_y < frame_height_ / 2) {
      *path_offset_y += delta_height * (required_height / remaining_height);
    } else {
      *path_offset_y -= delta_height * (required_height / remaining_height);
    }
  }

  // Prevent box from extending beyond the image after camera smoothing.
  float half_path_height = ceil(path_height / 2.0);
  *path_offset_y = std::clamp(*path_offset_y, half_path_height,
                              frame_height_ - half_path_height);

  float half_path_width = ceil(path_width / 2.0);
  *path_offset_x = std::clamp(*path_offset_x, half_path_width,
                              frame_width_ - half_path_width);

  MP_RETURN_IF_ERROR(path_solver_pan_->SetState(*path_offset_x));
  MP_RETURN_IF_ERROR(path_solver_tilt_->SetState(*path_offset_y));

  return absl::OkStatus();
}

absl::Status ContentZoomingCalculator::GetDetectionsBox(
    mediapipe::CalculatorContext* cc, float* xmin, float* xmax, float* ymin,
    float* ymax, bool* only_required_found, bool* has_detections) {
  if (cc->Inputs().HasTag(kSalientRegions)) {
    auto detection_set = cc->Inputs().Tag(kSalientRegions).Get<DetectionSet>();
    for (const auto& region : detection_set.detections()) {
      if (!region.only_required()) {
        continue;
      }
      *only_required_found = true;
      MP_RETURN_IF_ERROR(UpdateRanges(
          region, options_.detection_shift_vertical(),
          options_.detection_shift_horizontal(),
          options_.extra_vertical_padding(),
          options_.extra_horizontal_padding(), xmin, xmax, ymin, ymax));
    }
  }

  if (cc->Inputs().HasTag(kDetections)) {
    if (cc->Inputs().Tag(kDetections).IsEmpty()) {
      if (last_only_required_detection_ == 0) {
        // If no detections are available and we never had any,
        // simply return the full-image rectangle as crop-rect.
        if (cc->Outputs().HasTag(kCropRect)) {
          auto default_rect = absl::make_unique<mediapipe::Rect>();
          default_rect->set_x_center(frame_width_ / 2);
          default_rect->set_y_center(frame_height_ / 2);
          default_rect->set_width(frame_width_);
          default_rect->set_height(frame_height_);
          cc->Outputs().Tag(kCropRect).Add(default_rect.release(),
                                           Timestamp(cc->InputTimestamp()));
        }
        if (cc->Outputs().HasTag(kNormalizedCropRect)) {
          auto default_rect = absl::make_unique<mediapipe::NormalizedRect>();
          default_rect->set_x_center(0.5);
          default_rect->set_y_center(0.5);
          default_rect->set_width(1.0);
          default_rect->set_height(1.0);
          cc->Outputs()
              .Tag(kNormalizedCropRect)
              .Add(default_rect.release(), Timestamp(cc->InputTimestamp()));
        }
        // Also provide a first crop rect: in this case a zero-sized one.
        if (cc->Outputs().HasTag(kFirstCropRect)) {
          cc->Outputs()
              .Tag(kFirstCropRect)
              .Add(new mediapipe::NormalizedRect(),
                   Timestamp(cc->InputTimestamp()));
        }
        *has_detections = false;
        return absl::OkStatus();
      }
    } else {
      auto raw_detections = cc->Inputs()
                                .Tag(kDetections)
                                .Get<std::vector<mediapipe::Detection>>();
      for (const auto& detection : raw_detections) {
        *only_required_found = true;
        MP_RETURN_IF_ERROR(UpdateRanges(
            detection, options_.detection_shift_vertical(),
            options_.detection_shift_horizontal(),
            options_.extra_vertical_padding(),
            options_.extra_horizontal_padding(), xmin, xmax, ymin, ymax));
      }
    }
  }
  return absl::OkStatus();
}

}  // namespace autoflip
}  // namespace mediapipe
