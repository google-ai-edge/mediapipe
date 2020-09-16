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
#include <memory>

#include "mediapipe/examples/desktop/autoflip/autoflip_messages.pb.h"
#include "mediapipe/examples/desktop/autoflip/calculators/content_zooming_calculator.pb.h"
#include "mediapipe/examples/desktop/autoflip/quality/kinematic_path_solver.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/detection.pb.h"
#include "mediapipe/framework/formats/image_frame.h"
#include "mediapipe/framework/formats/location_data.pb.h"
#include "mediapipe/framework/formats/rect.pb.h"
#include "mediapipe/framework/port/ret_check.h"
#include "mediapipe/framework/port/status.h"
#include "mediapipe/framework/port/status_builder.h"

constexpr char kVideoFrame[] = "VIDEO";
constexpr char kVideoSize[] = "VIDEO_SIZE";
constexpr char kSalientRegions[] = "SALIENT_REGIONS";
constexpr char kDetections[] = "DETECTIONS";
constexpr char kDetectedBorders[] = "BORDERS";
constexpr char kCropRect[] = "CROP_RECT";
// Field-of-view (degrees) of the camera's x-axis (width).
// TODO: Parameterize FOV based on camera specs.
constexpr float kFieldOfView = 60;

namespace mediapipe {
namespace autoflip {

// Content zooming calculator zooms in on content when a detection has
// "only_required" set true or any raw detection input.  It does this by
// computing the value of top/bottom borders to remove from the output and sends
// these to the SceneCroppingCalculator using BORDERS output or a full rect crop
// using CROP_RECT output.  When more than one detections are received the
// zoom box is calculated as the union of the detections.  Typical applications
// include mobile makeover and autofliplive face reframing.
class ContentZoomingCalculator : public CalculatorBase {
 public:
  ContentZoomingCalculator()
      : initialized_(false), last_only_required_detection_(0) {}
  ~ContentZoomingCalculator() override {}
  ContentZoomingCalculator(const ContentZoomingCalculator&) = delete;
  ContentZoomingCalculator& operator=(const ContentZoomingCalculator&) = delete;

  static ::mediapipe::Status GetContract(mediapipe::CalculatorContract* cc);
  ::mediapipe::Status Open(mediapipe::CalculatorContext* cc) override;
  ::mediapipe::Status Process(mediapipe::CalculatorContext* cc) override;

 private:
  // Converts bounds to tilt offset, pan offset and height.
  ::mediapipe::Status ConvertToPanTiltZoom(float xmin, float xmax, float ymin,
                                           float ymax, int* tilt_offset,
                                           int* pan_offset, int* height);
  ContentZoomingCalculatorOptions options_;
  // Detection frame width/height.
  int frame_height_;
  int frame_width_;
  // Path solver used to smooth top/bottom border crop values.
  std::unique_ptr<KinematicPathSolver> path_solver_height_;
  std::unique_ptr<KinematicPathSolver> path_solver_width_;
  std::unique_ptr<KinematicPathSolver> path_solver_offset_;
  // Are parameters initialized.
  bool initialized_;
  // Stores the time of the last "only_required" input.
  int64 last_only_required_detection_;
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

::mediapipe::Status ContentZoomingCalculator::GetContract(
    mediapipe::CalculatorContract* cc) {
  RET_CHECK(
      !(cc->Inputs().HasTag(kVideoFrame) && cc->Inputs().HasTag(kVideoSize)))
      << "Provide only VIDEO or VIDEO_SIZE, not both.";
  if (cc->Inputs().HasTag(kVideoFrame)) {
    cc->Inputs().Tag(kVideoFrame).Set<ImageFrame>();
  } else if (cc->Inputs().HasTag(kVideoSize)) {
    cc->Inputs().Tag(kVideoSize).Set<std::pair<int, int>>();
  } else {
    return ::mediapipe::UnknownErrorBuilder(MEDIAPIPE_LOC)
           << "Input VIDEO or VIDEO_SIZE must be provided.";
  }
  if (cc->Inputs().HasTag(kSalientRegions)) {
    cc->Inputs().Tag(kSalientRegions).Set<DetectionSet>();
  }
  if (cc->Inputs().HasTag(kDetections)) {
    cc->Inputs().Tag(kDetections).Set<std::vector<mediapipe::Detection>>();
  }
  if (cc->Outputs().HasTag(kDetectedBorders)) {
    cc->Outputs().Tag(kDetectedBorders).Set<StaticFeatures>();
  }
  if (cc->Outputs().HasTag(kCropRect)) {
    cc->Outputs().Tag(kCropRect).Set<mediapipe::Rect>();
  }
  return ::mediapipe::OkStatus();
}

::mediapipe::Status ContentZoomingCalculator::Open(
    mediapipe::CalculatorContext* cc) {
  options_ = cc->Options<ContentZoomingCalculatorOptions>();
  if (options_.has_kinematic_options()) {
    return ::mediapipe::UnknownErrorBuilder(MEDIAPIPE_LOC)
           << "Deprecated kinematic_options was set, please set "
              "kinematic_options_zoom and kinematic_options_tilt.";
  }
  if (options_.has_min_motion_to_reframe()) {
    return ::mediapipe::UnknownErrorBuilder(MEDIAPIPE_LOC)
           << "Deprecated min_motion_to_reframe was set, please set "
              "in kinematic_options_zoom and kinematic_options_tilt "
              "directly.";
  }
  return ::mediapipe::OkStatus();
}

::mediapipe::Status ContentZoomingCalculator::ConvertToPanTiltZoom(
    float xmin, float xmax, float ymin, float ymax, int* tilt_offset,
    int* pan_offset, int* height) {
  // Find center of the y-axis offset (for tilt control).
  float y_center = ymin + (ymax - ymin) / 2;
  // Find center of the x-axis offset (for pan control).
  float x_center = xmin + (xmax - xmin) / 2;
  // Find size and apply scale factor to y-axis.
  float fit_size = fmax((ymax - ymin) / options_.scale_factor(), xmax - xmin);
  // Apply max frame for cases where the target size is different than input
  // frame size.
  fit_size = fmin(max_frame_value_, fit_size);
  // Prevent box from extending beyond the image.
  if (y_center - fit_size / 2 < 0) {
    y_center = fit_size / 2;
  } else if (y_center + fit_size / 2 > 1) {
    y_center = 1 - fit_size / 2;
  }
  if (x_center - fit_size / 2 < 0) {
    x_center = fit_size / 2;
  } else if (x_center + fit_size / 2 > 1) {
    x_center = 1 - fit_size / 2;
  }
  // Scale to pixel coordinates.
  *tilt_offset = frame_height_ * y_center;
  *pan_offset = frame_width_ * x_center;
  *height = frame_height_ * fit_size;
  return ::mediapipe::OkStatus();
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
::mediapipe::Status UpdateRanges(const SalientRegion& region,
                                 const float shift_vertical,
                                 const float shift_horizontal, float* xmin,
                                 float* xmax, float* ymin, float* ymax) {
  if (!region.has_location_normalized()) {
    return ::mediapipe::UnknownErrorBuilder(MEDIAPIPE_LOC)
           << "SalientRegion did not have location normalized set.";
  }
  auto location = ShiftDetection(region.location_normalized(), shift_vertical,
                                 shift_horizontal);
  *xmin = fmin(*xmin, location.x());
  *xmax = fmax(*xmax, location.x() + location.width());
  *ymin = fmin(*ymin, location.y());
  *ymax = fmax(*ymax, location.y() + location.height());

  return ::mediapipe::OkStatus();
}
::mediapipe::Status UpdateRanges(const mediapipe::Detection& detection,
                                 const float shift_vertical,
                                 const float shift_horizontal, float* xmin,
                                 float* xmax, float* ymin, float* ymax) {
  RET_CHECK(detection.location_data().format() ==
            mediapipe::LocationData::RELATIVE_BOUNDING_BOX)
      << "Face detection input is lacking required relative_bounding_box()";
  const auto& location =
      ShiftDetection(detection.location_data().relative_bounding_box(),
                     shift_vertical, shift_horizontal);
  *xmin = fmin(*xmin, location.xmin());
  *xmax = fmax(*xmax, location.xmin() + location.width());
  *ymin = fmin(*ymin, location.ymin());
  *ymax = fmax(*ymax, location.ymin() + location.height());

  return ::mediapipe::OkStatus();
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
}  // namespace

::mediapipe::Status ContentZoomingCalculator::Process(
    mediapipe::CalculatorContext* cc) {
  if (cc->Inputs().HasTag(kVideoFrame)) {
    frame_width_ = cc->Inputs().Tag(kVideoFrame).Get<ImageFrame>().Width();
    frame_height_ = cc->Inputs().Tag(kVideoFrame).Get<ImageFrame>().Height();
  } else if (cc->Inputs().HasTag(kVideoSize)) {
    frame_width_ =
        cc->Inputs().Tag(kVideoSize).Get<std::pair<int, int>>().first;
    frame_height_ =
        cc->Inputs().Tag(kVideoSize).Get<std::pair<int, int>>().second;
  } else {
    return ::mediapipe::UnknownErrorBuilder(MEDIAPIPE_LOC)
           << "Input VIDEO or VIDEO_SIZE must be provided.";
  }

  if (!initialized_) {
    path_solver_height_ = std::make_unique<KinematicPathSolver>(
        options_.kinematic_options_zoom(), 0, frame_height_,
        static_cast<float>(frame_height_) / kFieldOfView);
    path_solver_width_ = std::make_unique<KinematicPathSolver>(
        options_.kinematic_options_pan(), 0, frame_width_,
        static_cast<float>(frame_width_) / kFieldOfView);
    path_solver_offset_ = std::make_unique<KinematicPathSolver>(
        options_.kinematic_options_tilt(), 0, frame_height_,
        static_cast<float>(frame_height_) / kFieldOfView);
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
      max_frame_value_ = std::min(input_aspect / target_aspect_,
                                  target_aspect_ / input_aspect);
    }
    last_measured_height_ = max_frame_value_ * frame_height_;
    last_measured_x_offset_ = target_aspect_ * frame_width_;
    last_measured_y_offset_ = frame_width_ / 2;
    initialized_ = true;
  }

  bool only_required_found = false;

  // Compute the box that contains all "is_required" detections.
  float xmin = 1, ymin = 1, xmax = 0, ymax = 0;
  if (cc->Inputs().HasTag(kSalientRegions)) {
    auto detection_set = cc->Inputs().Tag(kSalientRegions).Get<DetectionSet>();
    for (const auto& region : detection_set.detections()) {
      if (!region.only_required()) {
        continue;
      }
      only_required_found = true;
      MP_RETURN_IF_ERROR(UpdateRanges(
          region, options_.detection_shift_vertical(),
          options_.detection_shift_horizontal(), &xmin, &xmax, &ymin, &ymax));
    }
  }

  if (cc->Inputs().HasTag(kDetections)) {
    auto raw_detections =
        cc->Inputs().Tag(kDetections).Get<std::vector<mediapipe::Detection>>();
    for (const auto& detection : raw_detections) {
      only_required_found = true;
      MP_RETURN_IF_ERROR(UpdateRanges(
          detection, options_.detection_shift_vertical(),
          options_.detection_shift_horizontal(), &xmin, &xmax, &ymin, &ymax));
    }
  }

  // Convert bounds to tilt/zoom and in pixel coordinates.
  int offset_y, height, offset_x;
  MP_RETURN_IF_ERROR(ConvertToPanTiltZoom(xmin, xmax, ymin, ymax, &offset_y,
                                          &offset_x, &height));

  if (only_required_found) {
    // A only required detection was found.
    last_only_required_detection_ = cc->InputTimestamp().Microseconds();
    last_measured_height_ = height;
    last_measured_x_offset_ = offset_x;
    last_measured_y_offset_ = offset_y;
  } else if (cc->InputTimestamp().Microseconds() -
                 last_only_required_detection_ >=
             options_.us_before_zoomout()) {
    // No only_require detections found within salient regions packets
    // arriving since us_before_zoomout duration.
    height = max_frame_value_ * frame_height_;
    offset_x = (target_aspect_ * height) / 2;
    offset_y = frame_height_ / 2;
  } else {
    // No only detection found but using last detection due to
    // duration_before_zoomout_us setting.
    height = last_measured_height_;
    offset_x = last_measured_x_offset_;
    offset_y = last_measured_y_offset_;
  }

  // Compute smoothed zoom camera path.
  MP_RETURN_IF_ERROR(path_solver_height_->AddObservation(
      height, cc->InputTimestamp().Microseconds()));
  int path_height;
  MP_RETURN_IF_ERROR(path_solver_height_->GetState(&path_height));
  int path_width = path_height * target_aspect_;

  // Update pixel-per-degree value for pan/tilt.
  MP_RETURN_IF_ERROR(path_solver_width_->UpdatePixelsPerDegree(
      static_cast<float>(path_width) / kFieldOfView));
  MP_RETURN_IF_ERROR(path_solver_offset_->UpdatePixelsPerDegree(
      static_cast<float>(path_height) / kFieldOfView));

  // Compute smoothed pan/tilt paths.
  MP_RETURN_IF_ERROR(path_solver_width_->AddObservation(
      offset_x, cc->InputTimestamp().Microseconds()));
  MP_RETURN_IF_ERROR(path_solver_offset_->AddObservation(
      offset_y, cc->InputTimestamp().Microseconds()));
  int path_offset_x;
  MP_RETURN_IF_ERROR(path_solver_width_->GetState(&path_offset_x));
  int path_offset_y;
  MP_RETURN_IF_ERROR(path_solver_offset_->GetState(&path_offset_y));

  // Prevent box from extending beyond the image after camera smoothing.
  if (path_offset_y - ceil(path_height / 2.0) < 0) {
    path_offset_y = ceil(path_height / 2.0);
  } else if (path_offset_y + ceil(path_height / 2.0) > frame_height_) {
    path_offset_y = frame_height_ - ceil(path_height / 2.0);
  }

  if (path_offset_x - ceil(path_width / 2.0) < 0) {
    path_offset_x = ceil(path_width / 2.0);
  } else if (path_offset_x + ceil(path_width / 2.0) > frame_width_) {
    path_offset_x = frame_width_ - ceil(path_width / 2.0);
  }

  // Convert to top/bottom borders to remove.
  int path_top = path_offset_y - path_height / 2;
  int path_bottom = frame_height_ - (path_offset_y + path_height / 2);

  // Transmit result downstream to scenecroppingcalculator.
  if (cc->Outputs().HasTag(kDetectedBorders)) {
    std::unique_ptr<StaticFeatures> features =
        absl::make_unique<StaticFeatures>();
    MakeStaticFeatures(path_top, path_bottom, frame_width_, frame_height_,
                       features.get());
    cc->Outputs()
        .Tag(kDetectedBorders)
        .AddPacket(Adopt(features.release()).At(cc->InputTimestamp()));
  }

  // Transmit downstream to glcroppingcalculator.
  if (cc->Outputs().HasTag(kCropRect)) {
    auto gpu_rect = absl::make_unique<mediapipe::Rect>();
    gpu_rect->set_x_center(path_offset_x);
    gpu_rect->set_width(path_height * target_aspect_);
    gpu_rect->set_y_center(path_offset_y);
    gpu_rect->set_height(path_height);
    cc->Outputs().Tag(kCropRect).Add(gpu_rect.release(),
                                     Timestamp(cc->InputTimestamp()));
  }

  return ::mediapipe::OkStatus();
}

}  // namespace autoflip
}  // namespace mediapipe
