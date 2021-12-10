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

#include "mediapipe/calculators/video/video_pre_stream_calculator.pb.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/image_frame.h"
#include "mediapipe/framework/formats/video_stream_header.h"

namespace mediapipe {

constexpr char kVideoPrestreamTag[] = "VIDEO_PRESTREAM";
constexpr char kFrameTag[] = "FRAME";

// Sets up VideoHeader based on the 1st ImageFrame and emits it with timestamp
// PreStream. Note that this calculator only fills in format, width, and height,
// i.e. frame_rate and duration will not be filled, unless:
// 1) an existing VideoHeader is provided at PreStream(). In such case, the
//    frame_rate and duration, if they exist, will be copied from the existing
//    VideoHeader.
// 2) you specify frame_rate and duration through the options. In this case, the
//    options will overwrite the existing VideoHeader if it is available.
//
// Example config:
// node {
//   calculator: "VideoPreStreamCalculator"
//   input_stream: "FRAME:cropped_frames"
//   input_stream: "VIDEO_PRESTREAM:original_video_header"
//   output_stream: "cropped_frames_video_header"
// }
//
// or
//
// node {
//   calculator: "VideoPreStreamCalculator"
//   input_stream: "cropped_frames"
//   output_stream: "video_header"
// }
class VideoPreStreamCalculator : public CalculatorBase {
 public:
  static absl::Status GetContract(CalculatorContract* cc);
  absl::Status Open(CalculatorContext* cc) override;
  absl::Status Process(CalculatorContext* cc) override;

 private:
  absl::Status ProcessWithFrameRateInPreStream(CalculatorContext* cc);
  absl::Status ProcessWithFrameRateInOptions(CalculatorContext* cc);

  std::unique_ptr<VideoHeader> header_;
  bool frame_rate_in_prestream_ = false;
  bool emitted_ = false;
};

REGISTER_CALCULATOR(VideoPreStreamCalculator);

absl::Status VideoPreStreamCalculator::GetContract(CalculatorContract* cc) {
  if (!cc->Inputs().UsesTags()) {
    cc->Inputs().Index(0).Set<ImageFrame>();
  } else {
    cc->Inputs().Tag(kFrameTag).Set<ImageFrame>();
    cc->Inputs().Tag(kVideoPrestreamTag).Set<VideoHeader>();
  }
  cc->Outputs().Index(0).Set<VideoHeader>();
  return absl::OkStatus();
}

absl::Status VideoPreStreamCalculator::Open(CalculatorContext* cc) {
  frame_rate_in_prestream_ = cc->Inputs().UsesTags() &&
                             cc->Inputs().HasTag(kFrameTag) &&
                             cc->Inputs().HasTag(kVideoPrestreamTag);
  header_ = absl::make_unique<VideoHeader>();
  return absl::OkStatus();
}
absl::Status VideoPreStreamCalculator::ProcessWithFrameRateInPreStream(
    CalculatorContext* cc) {
  cc->GetCounter("ProcessWithFrameRateInPreStream")->Increment();
  if (cc->InputTimestamp() == Timestamp::PreStream()) {
    RET_CHECK(cc->Inputs().Tag(kFrameTag).IsEmpty());
    RET_CHECK(!cc->Inputs().Tag(kVideoPrestreamTag).IsEmpty());
    *header_ = cc->Inputs().Tag(kVideoPrestreamTag).Get<VideoHeader>();
    RET_CHECK_NE(header_->frame_rate, 0.0) << "frame rate should be non-zero";
  } else {
    RET_CHECK(cc->Inputs().Tag(kVideoPrestreamTag).IsEmpty())
        << "Packet on VIDEO_PRESTREAM must come in at Timestamp::PreStream().";
    RET_CHECK(!cc->Inputs().Tag(kFrameTag).IsEmpty());
    const auto& frame = cc->Inputs().Tag(kFrameTag).Get<ImageFrame>();
    header_->format = frame.Format();
    header_->width = frame.Width();
    header_->height = frame.Height();
    RET_CHECK_NE(header_->frame_rate, 0.0) << "frame rate should be non-zero";
    cc->Outputs().Index(0).Add(header_.release(), Timestamp::PreStream());
    emitted_ = true;
  }
  return absl::OkStatus();
}

absl::Status VideoPreStreamCalculator::Process(CalculatorContext* cc) {
  cc->GetCounter("Process")->Increment();
  if (emitted_) {
    return absl::OkStatus();
  }
  if (frame_rate_in_prestream_) {
    return ProcessWithFrameRateInPreStream(cc);
  } else {
    return ProcessWithFrameRateInOptions(cc);
  }
}

absl::Status VideoPreStreamCalculator::ProcessWithFrameRateInOptions(
    CalculatorContext* cc) {
  cc->GetCounter("ProcessWithFrameRateInOptions")->Increment();
  RET_CHECK_NE(cc->InputTimestamp(), Timestamp::PreStream());
  const auto& frame = cc->Inputs().Index(0).Get<ImageFrame>();
  header_->format = frame.Format();
  header_->width = frame.Width();
  header_->height = frame.Height();
  const auto& options = cc->Options<VideoPreStreamCalculatorOptions>();
  if (options.fps().has_value()) {
    header_->frame_rate = options.fps().value();
  } else if (options.fps().has_ratio()) {
    const VideoPreStreamCalculatorOptions::Fps::Rational32& ratio =
        options.fps().ratio();
    if (ratio.numerator() > 0 && ratio.denominator() > 0) {
      header_->frame_rate =
          static_cast<double>(ratio.numerator()) / ratio.denominator();
    }
  }
  RET_CHECK_NE(header_->frame_rate, 0.0) << "frame rate should be non-zero";
  cc->Outputs().Index(0).Add(header_.release(), Timestamp::PreStream());
  emitted_ = true;
  return absl::OkStatus();
}

}  // namespace mediapipe
