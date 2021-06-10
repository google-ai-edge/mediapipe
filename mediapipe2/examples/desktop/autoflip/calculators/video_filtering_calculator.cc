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

#include <string>

#include "absl/strings/string_view.h"
#include "absl/strings/substitute.h"
#include "mediapipe/examples/desktop/autoflip/calculators/video_filtering_calculator.pb.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/image_frame.h"
#include "mediapipe/framework/formats/image_frame_opencv.h"
#include "mediapipe/framework/port/status_builder.h"

namespace mediapipe {
namespace autoflip {
namespace {
constexpr char kInputFrameTag[] = "INPUT_FRAMES";
constexpr char kOutputFrameTag[] = "OUTPUT_FRAMES";
}  // namespace

// This calculator filters out frames based on criteria specified in the
// options. One use case is to filter based on the aspect ratio. Future work
// can implement more filter types.
//
// Input: Video frames.
// Output: Video frames that pass all filters.
//
// Example config:
// node {
//   calculator: "VideoFilteringCalculator"
//   input_stream: "INPUT_FRAMES:frames"
//   output_stream: "OUTPUT_FRAMES:output_frames"
//   options: {
//     [mediapipe.autoflip.VideoFilteringCalculatorOptions.ext]: {
//       fail_if_any: true
//       aspect_ratio_filter {
//         target_width: 400
//         target_height: 600
//         filter_type: UPPER_ASPECT_RATIO_THRESHOLD
//       }
//     }
//   }
// }
class VideoFilteringCalculator : public CalculatorBase {
 public:
  VideoFilteringCalculator() = default;
  ~VideoFilteringCalculator() override = default;

  static absl::Status GetContract(CalculatorContract* cc);

  absl::Status Process(CalculatorContext* cc) override;
};
REGISTER_CALCULATOR(VideoFilteringCalculator);

absl::Status VideoFilteringCalculator::GetContract(CalculatorContract* cc) {
  cc->Inputs().Tag(kInputFrameTag).Set<ImageFrame>();
  cc->Outputs().Tag(kOutputFrameTag).Set<ImageFrame>();
  return absl::OkStatus();
}

absl::Status VideoFilteringCalculator::Process(CalculatorContext* cc) {
  const auto& options = cc->Options<VideoFilteringCalculatorOptions>();

  const Packet& input_packet = cc->Inputs().Tag(kInputFrameTag).Value();
  const ImageFrame& frame = input_packet.Get<ImageFrame>();

  RET_CHECK(options.has_aspect_ratio_filter());
  const auto filter_type = options.aspect_ratio_filter().filter_type();
  RET_CHECK_NE(
      filter_type,
      VideoFilteringCalculatorOptions::AspectRatioFilter::UNKNOWN_FILTER_TYPE);
  if (filter_type ==
      VideoFilteringCalculatorOptions::AspectRatioFilter::NO_FILTERING) {
    cc->Outputs().Tag(kOutputFrameTag).AddPacket(input_packet);
    return absl::OkStatus();
  }
  const int target_width = options.aspect_ratio_filter().target_width();
  const int target_height = options.aspect_ratio_filter().target_height();
  RET_CHECK_GT(target_width, 0);
  RET_CHECK_GT(target_height, 0);

  bool should_pass = false;
  cv::Mat frame_mat = mediapipe::formats::MatView(&frame);
  const double ratio = static_cast<double>(frame_mat.cols) / frame_mat.rows;
  const double target_ratio = static_cast<double>(target_width) / target_height;
  if (filter_type == VideoFilteringCalculatorOptions::AspectRatioFilter::
                         UPPER_ASPECT_RATIO_THRESHOLD &&
      ratio <= target_ratio) {
    should_pass = true;
  } else if (filter_type == VideoFilteringCalculatorOptions::AspectRatioFilter::
                                LOWER_ASPECT_RATIO_THRESHOLD &&
             ratio >= target_ratio) {
    should_pass = true;
  }
  if (should_pass) {
    cc->Outputs().Tag(kOutputFrameTag).AddPacket(input_packet);
    return absl::OkStatus();
  }
  if (options.fail_if_any()) {
    return mediapipe::UnknownErrorBuilder(MEDIAPIPE_LOC) << absl::Substitute(
               "Failing due to aspect ratio. Target aspect ratio: $0. Frame "
               "width: $1, height: $2.",
               target_ratio, frame.Width(), frame.Height());
  }

  return absl::OkStatus();
}
}  // namespace autoflip
}  // namespace mediapipe
