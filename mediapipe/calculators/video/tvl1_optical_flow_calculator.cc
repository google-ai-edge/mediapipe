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

#include "absl/base/macros.h"
#include "absl/synchronization/mutex.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/image_frame.h"
#include "mediapipe/framework/formats/image_frame_opencv.h"
#include "mediapipe/framework/formats/motion/optical_flow_field.h"
#include "mediapipe/framework/port/opencv_video_inc.h"

namespace mediapipe {
namespace {

constexpr char kBackwardFlowTag[] = "BACKWARD_FLOW";
constexpr char kForwardFlowTag[] = "FORWARD_FLOW";
constexpr char kSecondFrameTag[] = "SECOND_FRAME";
constexpr char kFirstFrameTag[] = "FIRST_FRAME";

// Checks that img1 and img2 have the same dimensions.
bool ImageSizesMatch(const ImageFrame& img1, const ImageFrame& img2) {
  return (img1.Width() == img2.Width()) && (img1.Height() == img2.Height());
}

// Converts an RGB image to grayscale.
cv::Mat ConvertToGrayscale(const cv::Mat& image) {
  if (image.channels() == 1) {
    return image;
  }
  cv::Mat gray;
  cv::cvtColor(image, gray, cv::COLOR_RGB2GRAY);
  return gray;
}

}  // namespace

// Calls OpenCV's DenseOpticalFlow to compute the optical flow between a pair of
// image frames. The calculator can output forward flow fields (optical flow
// from the first frame to the second frame), backward flow fields (optical flow
// from the second frame to the first frame), or both, depending on the tag of
// the specified output streams. Note that the timestamp of the output optical
// flow is always tied to the input timestamp. Be aware of the different
// meanings of the timestamp between the forward and the backward optical flows
// if the calculator outputs both.
//
// If the "max_in_flight" field is set to any value greater than 1, it will
// enable the calculator to process multiple inputs in parallel. The output
// packets will be automatically ordered by timestamp before they are passed
// along to downstream calculators.
//
// Inputs:
//   FIRST_FRAME: An ImageFrame in either SRGB or GRAY8 format.
//   SECOND_FRAME: An ImageFrame in either SRGB or GRAY8 format.
// Outputs:
//   FORWARD_FLOW: The OpticalFlowField from the first frame to the second
//                 frame, output at the input timestamp.
//   BACKWARD_FLOW: The OpticalFlowField from the second frame to the first
//                  frame, output at the input timestamp.
// Example config:
//   node {
//     calculator: "Tvl1OpticalFlowCalculator"
//     input_stream: "FIRST_FRAME:first_frames"
//     input_stream: "SECOND_FRAME:second_frames"
//     output_stream: "FORWARD_FLOW:forward_flow"
//     output_stream: "BACKWARD_FLOW:backward_flow"
//     max_in_flight: 10
//   }
//   num_threads: 10
class Tvl1OpticalFlowCalculator : public CalculatorBase {
 public:
  static absl::Status GetContract(CalculatorContract* cc);
  absl::Status Open(CalculatorContext* cc) override;
  absl::Status Process(CalculatorContext* cc) override;

 private:
  absl::Status CalculateOpticalFlow(const ImageFrame& current_frame,
                                    const ImageFrame& next_frame,
                                    OpticalFlowField* flow);
  bool forward_requested_ = false;
  bool backward_requested_ = false;
  // Stores the idle DenseOpticalFlow objects.
  // cv::DenseOpticalFlow is not thread-safe. Invoking multiple
  // DenseOpticalFlow::calc() in parallel may lead to memory corruption or
  // memory leak.
  std::list<cv::Ptr<cv::DenseOpticalFlow>> tvl1_computers_
      ABSL_GUARDED_BY(mutex_);
  absl::Mutex mutex_;
};

absl::Status Tvl1OpticalFlowCalculator::GetContract(CalculatorContract* cc) {
  if (!cc->Inputs().HasTag(kFirstFrameTag) ||
      !cc->Inputs().HasTag(kSecondFrameTag)) {
    return absl::InvalidArgumentError(
        "Missing required input streams. Both FIRST_FRAME and SECOND_FRAME "
        "must be specified.");
  }
  cc->Inputs().Tag(kFirstFrameTag).Set<ImageFrame>();
  cc->Inputs().Tag(kSecondFrameTag).Set<ImageFrame>();
  if (cc->Outputs().HasTag(kForwardFlowTag)) {
    cc->Outputs().Tag(kForwardFlowTag).Set<OpticalFlowField>();
  }
  if (cc->Outputs().HasTag(kBackwardFlowTag)) {
    cc->Outputs().Tag(kBackwardFlowTag).Set<OpticalFlowField>();
  }
  return absl::OkStatus();
}

absl::Status Tvl1OpticalFlowCalculator::Open(CalculatorContext* cc) {
  {
    absl::MutexLock lock(&mutex_);
    tvl1_computers_.emplace_back(cv::createOptFlow_DualTVL1());
  }
  if (cc->Outputs().HasTag(kForwardFlowTag)) {
    forward_requested_ = true;
  }
  if (cc->Outputs().HasTag(kBackwardFlowTag)) {
    backward_requested_ = true;
  }

  return absl::OkStatus();
}

absl::Status Tvl1OpticalFlowCalculator::Process(CalculatorContext* cc) {
  const ImageFrame& first_frame =
      cc->Inputs().Tag(kFirstFrameTag).Value().Get<ImageFrame>();
  const ImageFrame& second_frame =
      cc->Inputs().Tag(kSecondFrameTag).Value().Get<ImageFrame>();
  if (forward_requested_) {
    auto forward_optical_flow_field = absl::make_unique<OpticalFlowField>();
    MP_RETURN_IF_ERROR(CalculateOpticalFlow(first_frame, second_frame,
                                            forward_optical_flow_field.get()));
    cc->Outputs()
        .Tag(kForwardFlowTag)
        .Add(forward_optical_flow_field.release(), cc->InputTimestamp());
  }
  if (backward_requested_) {
    auto backward_optical_flow_field = absl::make_unique<OpticalFlowField>();
    MP_RETURN_IF_ERROR(CalculateOpticalFlow(second_frame, first_frame,
                                            backward_optical_flow_field.get()));
    cc->Outputs()
        .Tag(kBackwardFlowTag)
        .Add(backward_optical_flow_field.release(), cc->InputTimestamp());
  }
  return absl::OkStatus();
}

absl::Status Tvl1OpticalFlowCalculator::CalculateOpticalFlow(
    const ImageFrame& current_frame, const ImageFrame& next_frame,
    OpticalFlowField* flow) {
  CHECK(flow);
  if (!ImageSizesMatch(current_frame, next_frame)) {
    return tool::StatusInvalid("Images are different sizes.");
  }
  const cv::Mat& first = ConvertToGrayscale(formats::MatView(&current_frame));
  const cv::Mat& second = ConvertToGrayscale(formats::MatView(&next_frame));

  // Tries getting an idle DenseOpticalFlow object from the cache. If not,
  // creates a new DenseOpticalFlow.
  cv::Ptr<cv::DenseOpticalFlow> tvl1_computer;
  {
    absl::MutexLock lock(&mutex_);
    if (!tvl1_computers_.empty()) {
      std::swap(tvl1_computer, tvl1_computers_.front());
      tvl1_computers_.pop_front();
    }
  }
  if (tvl1_computer.empty()) {
    tvl1_computer = cv::createOptFlow_DualTVL1();
  }

  flow->Allocate(first.cols, first.rows);
  cv::Mat cv_flow(flow->mutable_flow_data());
  tvl1_computer->calc(first, second, cv_flow);
  CHECK_EQ(flow->mutable_flow_data().data, cv_flow.data);
  // Inserts the idle DenseOpticalFlow object back to the cache for reuse.
  {
    absl::MutexLock lock(&mutex_);
    tvl1_computers_.push_back(tvl1_computer);
  }
  return absl::OkStatus();
}

REGISTER_CALCULATOR(Tvl1OpticalFlowCalculator);

}  // namespace mediapipe
