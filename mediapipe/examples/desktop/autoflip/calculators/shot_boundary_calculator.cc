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

#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/log/absl_log.h"
#include "mediapipe/examples/desktop/autoflip/calculators/shot_boundary_calculator.pb.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/image_frame.h"
#include "mediapipe/framework/formats/image_frame_opencv.h"
#include "mediapipe/framework/port/opencv_imgproc_inc.h"
#include "mediapipe/framework/port/ret_check.h"
#include "mediapipe/framework/port/status.h"
#include "mediapipe/framework/timestamp.h"

using mediapipe::ImageFrame;
using mediapipe::PacketTypeSet;

// IO labels.
constexpr char kVideoInputTag[] = "VIDEO";
constexpr char kShotChangeTag[] = "IS_SHOT_CHANGE";
// Histogram settings.
const int kSaturationBins = 8;
const int kHistogramChannels[] = {0, 1, 2};
const int kHistogramBinNum[] = {kSaturationBins, kSaturationBins,
                                kSaturationBins};
const float kRange[] = {0, 256};
const float* kHistogramRange[] = {kRange, kRange, kRange};

namespace mediapipe {
namespace autoflip {

// This calculator computes a shot (or scene) change within a video.  It works
// by computing a 3d color histogram and comparing this frame-to-frame. Settings
// to control the shot change logic are presented in the options proto.
//
// Example:
//  node {
//    calculator: "ShotBoundaryCalculator"
//    input_stream: "VIDEO:camera_frames"
//    output_stream: "IS_SHOT_CHANGE:is_shot"
//  }
class ShotBoundaryCalculator : public mediapipe::CalculatorBase {
 public:
  ShotBoundaryCalculator() {}
  ShotBoundaryCalculator(const ShotBoundaryCalculator&) = delete;
  ShotBoundaryCalculator& operator=(const ShotBoundaryCalculator&) = delete;

  static absl::Status GetContract(mediapipe::CalculatorContract* cc);
  absl::Status Open(mediapipe::CalculatorContext* cc) override;
  absl::Status Process(mediapipe::CalculatorContext* cc) override;

 private:
  // Computes the histogram of an image.
  void ComputeHistogram(const cv::Mat& image, cv::Mat* image_histogram);
  // Transmits signal to next calculator.
  void Transmit(mediapipe::CalculatorContext* cc, bool is_shot_change);
  // Calculator options.
  ShotBoundaryCalculatorOptions options_;
  // Last time a shot was detected.
  Timestamp last_shot_timestamp_;
  // Defines if the calculator has received a frame yet.
  bool init_;
  // Histogram from the last frame.
  cv::Mat last_histogram_;
  // History of histogram motion.
  std::deque<double> motion_history_;
};
REGISTER_CALCULATOR(ShotBoundaryCalculator);

void ShotBoundaryCalculator::ComputeHistogram(const cv::Mat& image,
                                              cv::Mat* image_histogram) {
  cv::Mat equalized_image;
  cv::cvtColor(image.clone(), equalized_image, cv::COLOR_RGB2GRAY);

  double min, max;
  cv::minMaxLoc(equalized_image, &min, &max);

  if (options_.equalize_histogram()) {
    cv::equalizeHist(equalized_image, equalized_image);
  }

  cv::calcHist(&image, 1, kHistogramChannels, cv::Mat(), *image_histogram, 2,
               kHistogramBinNum, kHistogramRange, true, false);
}

absl::Status ShotBoundaryCalculator::Open(mediapipe::CalculatorContext* cc) {
  options_ = cc->Options<ShotBoundaryCalculatorOptions>();
  last_shot_timestamp_ = Timestamp(0);
  init_ = false;
  return absl::OkStatus();
}

void ShotBoundaryCalculator::Transmit(mediapipe::CalculatorContext* cc,
                                      bool is_shot_change) {
  if ((cc->InputTimestamp() - last_shot_timestamp_).Seconds() <
      options_.min_shot_span()) {
    is_shot_change = false;
  }
  if (is_shot_change) {
    ABSL_LOG(INFO) << "Shot change at: " << cc->InputTimestamp().Seconds()
                   << " seconds.";
    cc->Outputs()
        .Tag(kShotChangeTag)
        .AddPacket(Adopt(std::make_unique<bool>(true).release())
                       .At(cc->InputTimestamp()));
  } else if (!options_.output_only_on_change()) {
    cc->Outputs()
        .Tag(kShotChangeTag)
        .AddPacket(Adopt(std::make_unique<bool>(false).release())
                       .At(cc->InputTimestamp()));
  }
}

absl::Status ShotBoundaryCalculator::Process(mediapipe::CalculatorContext* cc) {
  // Connect to input frame and make a mutable copy.
  cv::Mat frame_org = mediapipe::formats::MatView(
      &cc->Inputs().Tag(kVideoInputTag).Get<ImageFrame>());
  cv::Mat frame = frame_org.clone();

  // Extract histogram from the current frame.
  cv::Mat current_histogram;
  ComputeHistogram(frame, &current_histogram);

  if (!init_) {
    last_histogram_ = current_histogram;
    init_ = true;
    Transmit(cc, false);
    return absl::OkStatus();
  }

  double current_motion_estimate =
      1 - cv::compareHist(current_histogram, last_histogram_, CV_COMP_CORREL);
  last_histogram_ = current_histogram;
  motion_history_.push_front(current_motion_estimate);

  if (motion_history_.size() != options_.window_size()) {
    Transmit(cc, false);
    return absl::OkStatus();
  }

  // Shot detection algorithm is a mixture of adaptive (controlled with
  // shot_measure) and hard thresholds. In saturation it uses hard thresholds
  // to account for black startups, shot cuts across high motion etc.
  // In the operating region it uses an adaptive threshold to tune motion vs.
  // cut boundary.
  double current_max =
      *std::max_element(motion_history_.begin(), motion_history_.end());
  double shot_measure = current_motion_estimate / current_max;

  if ((shot_measure > options_.min_shot_measure() &&
       current_motion_estimate > options_.min_motion_with_shot_measure()) ||
      current_motion_estimate > options_.min_motion()) {
    Transmit(cc, true);
    last_shot_timestamp_ = cc->InputTimestamp();
  } else {
    Transmit(cc, false);
  }

  // Store histogram for next frame.
  last_histogram_ = current_histogram;
  motion_history_.pop_back();
  return absl::OkStatus();
}

absl::Status ShotBoundaryCalculator::GetContract(
    mediapipe::CalculatorContract* cc) {
  cc->Inputs().Tag(kVideoInputTag).Set<ImageFrame>();
  cc->Outputs().Tag(kShotChangeTag).Set<bool>();
  return absl::OkStatus();
}

}  // namespace autoflip
}  // namespace mediapipe
