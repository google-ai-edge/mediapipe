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
#include "mediapipe/framework/formats/image_frame.h"
#include "mediapipe/framework/formats/image_frame_opencv.h"
#include "mediapipe/framework/port/ret_check.h"
#include "mediapipe/framework/port/status.h"
#include "mediapipe/framework/port/status_builder.h"

constexpr char kVideoFrame[] = "VIDEO";
constexpr char kVideoSize[] = "VIDEO_SIZE";
constexpr char kDetectionSet[] = "DETECTIONS";
constexpr char kDetectedBorders[] = "BORDERS";
// Field-of-view (degrees) of the camera's x-axis (width).
// TODO: Parameterize FOV based on camera specs.
constexpr float kWidthFieldOfView = 60;

namespace mediapipe {
namespace autoflip {

// Content zooming calculator zooms in on content when a detection has
// "only_required" set true.  It does this by computing the value of top/bottom
// borders to remove from the output and sends these to the
// SceneCroppingCalculator.  When more than one detections are received the zoom
// box is calculated as the union of the detections.  Typical applications
// include mobile makeover and autofliplive face reframing.  Currently only
// supports y-dimension zooming.
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
  // Converts bounds to tilt offset and height.
  ::mediapipe::Status ConvertToTiltZoom(float xmin, float xmax, float ymin,
                                        float ymax, int* tilt_offset,
                                        int* height);
  ContentZoomingCalculatorOptions options_;
  // Detection frame width/height.
  int frame_height_;
  int frame_width_;
  // Path solver used to smooth top/bottom border crop values.
  std::unique_ptr<KinematicPathSolver> path_solver_height_;
  std::unique_ptr<KinematicPathSolver> path_solver_offset_;
  // Are parameters initialized.
  bool initialized_;
  // Stores the time of the last "only_required" input.
  int64 last_only_required_detection_;
  // Border values of last message with detection.
  int last_measured_height_;
  int last_measured_y_offset_;
  // Min border values.
  float min_height_value_;
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
  cc->Inputs().Tag(kDetectionSet).Set<DetectionSet>();
  cc->Outputs().Tag(kDetectedBorders).Set<StaticFeatures>();
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
              "in kinematic_options_zoom and kinematic_options_tilt directly.";
  }
  return ::mediapipe::OkStatus();
}

::mediapipe::Status ContentZoomingCalculator::ConvertToTiltZoom(
    float xmin, float xmax, float ymin, float ymax, int* tilt_offset,
    int* height) {
  // Find center of the y-axis offset (for tilt control).
  float y_center = ymin + (ymax - ymin) / 2;
  // Find size and apply scale factor to y-axis.
  float fit_size = fmax((ymax - ymin) / options_.scale_factor(), xmax - xmin);
  // Apply min zoom for cases where the target size is wider than input frame
  // size.
  fit_size = fmin(min_height_value_, fit_size);
  // Prevent box from extending beyond the image.
  if (y_center - fit_size / 2 < 0) {
    y_center = fit_size / 2;
  } else if (y_center + fit_size / 2 > 1) {
    y_center = 1 - fit_size / 2;
  }
  // Scale to pixel coordinates.
  *tilt_offset = frame_height_ * y_center;
  *height = frame_height_ * fit_size;
  return ::mediapipe::OkStatus();
}

namespace {
::mediapipe::Status UpdateRanges(const SalientRegion& region, float* xmin,
                                 float* xmax, float* ymin, float* ymax) {
  if (!region.has_location_normalized()) {
    return ::mediapipe::UnknownErrorBuilder(MEDIAPIPE_LOC)
           << "SalientRegion did not have location normalized set.";
  }
  *xmin = fmin(*xmin, region.location_normalized().x());
  *xmax = fmax(*xmax, region.location_normalized().x() +
                          region.location_normalized().width());
  *ymin = fmin(*ymin, region.location_normalized().y());
  *ymax = fmax(*ymax, region.location_normalized().y() +
                          region.location_normalized().height());

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
    cv::Mat frame = mediapipe::formats::MatView(
        &cc->Inputs().Tag(kVideoFrame).Get<ImageFrame>());
    frame_width_ = frame.cols;
    frame_height_ = frame.rows;
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
        static_cast<float>(frame_width_) / kWidthFieldOfView);
    path_solver_offset_ = std::make_unique<KinematicPathSolver>(
        options_.kinematic_options_tilt(), 0, frame_height_,
        static_cast<float>(frame_width_) / kWidthFieldOfView);
    min_height_value_ = 1.0;
    // If target size is set and wider than input aspect, make sure to always
    // crop the min required amount.
    if (options_.has_target_size()) {
      RET_CHECK_GT(options_.target_size().width(), 0)
          << "Provided target width not valid.";
      RET_CHECK_GT(options_.target_size().height(), 0)
          << "Provided target height not valid.";
      float input_aspect = frame_width_ / static_cast<float>(frame_height_);
      float target_aspect = options_.target_size().width() /
                            static_cast<float>(options_.target_size().height());
      min_height_value_ =
          (input_aspect < target_aspect) ? input_aspect / target_aspect : 1.0;
    }
    last_measured_height_ = min_height_value_ * frame_height_;
    last_measured_y_offset_ = frame_width_ / 2;
    initialized_ = true;
  }

  auto detection_set = cc->Inputs().Tag(kDetectionSet).Get<DetectionSet>();
  bool only_required_found = false;

  // Compute the box that contains all "is_required" detections.
  float xmin = 1, ymin = 1, xmax = 0, ymax = 0;
  for (const auto& region : detection_set.detections()) {
    if (!region.only_required()) {
      continue;
    }
    only_required_found = true;
    MP_RETURN_IF_ERROR(UpdateRanges(region, &xmin, &xmax, &ymin, &ymax));
  }

  // Convert bounds to tilt/zoom and in pixel coordinates.
  int offset, height;
  MP_RETURN_IF_ERROR(
      ConvertToTiltZoom(xmin, xmax, ymin, ymax, &offset, &height));

  if (only_required_found) {
    // A only required detection was found.
    last_only_required_detection_ = cc->InputTimestamp().Microseconds();
    last_measured_height_ = height;
    last_measured_y_offset_ = offset;
  } else if (cc->InputTimestamp().Microseconds() -
                 last_only_required_detection_ >=
             options_.us_before_zoomout()) {
    // No only_require detections found within salient regions packets arriving
    // since us_before_zoomout duration.
    height = min_height_value_ * frame_height_;
    offset = frame_height_ / 2;
  } else {
    // No only detection found but using last detection due to
    // duration_before_zoomout_us setting.
    height = last_measured_height_;
    offset = last_measured_y_offset_;
  }

  // Compute smoothed camera paths.
  MP_RETURN_IF_ERROR(path_solver_height_->AddObservation(
      height, cc->InputTimestamp().Microseconds()));
  MP_RETURN_IF_ERROR(path_solver_offset_->AddObservation(
      offset, cc->InputTimestamp().Microseconds()));
  int path_size;
  MP_RETURN_IF_ERROR(path_solver_height_->GetState(&path_size));
  int path_offset;
  MP_RETURN_IF_ERROR(path_solver_offset_->GetState(&path_offset));

  // Convert to top/bottom borders to remove.
  int path_top = path_offset - path_size / 2;
  int path_bottom = frame_height_ - (path_offset + path_size / 2);

  // Transmit result downstream.
  std::unique_ptr<StaticFeatures> features =
      absl::make_unique<StaticFeatures>();
  MakeStaticFeatures(path_top, path_bottom, frame_width_, frame_height_,
                     features.get());
  cc->Outputs()
      .Tag(kDetectedBorders)
      .AddPacket(Adopt(features.release()).At(cc->InputTimestamp()));

  return ::mediapipe::OkStatus();
}

}  // namespace autoflip
}  // namespace mediapipe
