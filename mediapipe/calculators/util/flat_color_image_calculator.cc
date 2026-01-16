// Copyright 2023 The MediaPipe Authors.
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

#include <memory>

#include "absl/status/status.h"
#include "mediapipe/calculators/util/flat_color_image_calculator.pb.h"
#include "mediapipe/framework/api2/node.h"
#include "mediapipe/framework/calculator_contract.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/image.h"
#include "mediapipe/framework/formats/image_frame.h"
#include "mediapipe/framework/formats/image_frame_opencv.h"
#include "mediapipe/util/color.pb.h"

namespace mediapipe {

namespace {

using ::mediapipe::api2::Input;
using ::mediapipe::api2::Node;
using ::mediapipe::api2::Output;
using ::mediapipe::api2::SideOutput;
}  // namespace

// A calculator for generating an image filled with a single color.
//
// Inputs:
//   IMAGE (Image, optional)
//     If provided, the output will have the same size
//   COLOR (Color proto, optional)
//     Color to paint the output with. Takes precedence over the equivalent
//     calculator options.
//
// Outputs:
//   IMAGE (Image)
//     Image filled with the requested color. Can be either an output_stream
//     or an output_side_packet.
//
// Example useage:
// node {
//   calculator: "FlatColorImageCalculator"
//   input_stream: "IMAGE:image"
//   input_stream: "COLOR:color"
//   output_stream: "IMAGE:blank_image"
//   options {
//     [mediapipe.FlatColorImageCalculatorOptions.ext] {
//       color: {
//         r: 255
//         g: 255
//         b: 255
//       }
//     }
//   }
// }

class FlatColorImageCalculator : public Node {
 public:
  static constexpr Input<Image>::Optional kInImage{"IMAGE"};
  static constexpr Input<Color>::Optional kInColor{"COLOR"};
  static constexpr Output<Image>::Optional kOutImage{"IMAGE"};
  static constexpr SideOutput<Image>::Optional kOutSideImage{"IMAGE"};

  MEDIAPIPE_NODE_CONTRACT(kInImage, kInColor, kOutImage, kOutSideImage);

  static absl::Status UpdateContract(CalculatorContract* cc) {
    const auto& options = cc->Options<FlatColorImageCalculatorOptions>();

    RET_CHECK(kInImage(cc).IsConnected() ^
              (options.has_output_height() || options.has_output_width()))
        << "Either set IMAGE input stream, or set through options";
    RET_CHECK(kInColor(cc).IsConnected() ^ options.has_color())
        << "Either set COLOR input stream, or set through options";

    RET_CHECK(kOutImage(cc).IsConnected() ^ kOutSideImage(cc).IsConnected())
        << "Set IMAGE either as output stream, or as output side packet";

    RET_CHECK(!kOutSideImage(cc).IsConnected() ||
              (options.has_output_height() && options.has_output_width()))
        << "Set size through options, when setting IMAGE as output side packet";

    return absl::OkStatus();
  }

  absl::Status Open(CalculatorContext* cc) override;
  absl::Status Process(CalculatorContext* cc) override;

 private:
  std::optional<std::shared_ptr<ImageFrame>> CreateOutputFrame(
      CalculatorContext* cc);

  bool use_dimension_from_option_ = false;
  bool use_color_from_option_ = false;
};
MEDIAPIPE_REGISTER_NODE(FlatColorImageCalculator);

absl::Status FlatColorImageCalculator::Open(CalculatorContext* cc) {
  use_dimension_from_option_ = !kInImage(cc).IsConnected();
  use_color_from_option_ = !kInColor(cc).IsConnected();

  if (!kOutImage(cc).IsConnected()) {
    std::optional<std::shared_ptr<ImageFrame>> output_frame =
        CreateOutputFrame(cc);
    if (output_frame.has_value()) {
      kOutSideImage(cc).Set(Image(output_frame.value()));
    }
  }
  return absl::OkStatus();
}

absl::Status FlatColorImageCalculator::Process(CalculatorContext* cc) {
  if (kOutImage(cc).IsConnected()) {
    std::optional<std::shared_ptr<ImageFrame>> output_frame =
        CreateOutputFrame(cc);
    if (output_frame.has_value()) {
      kOutImage(cc).Send(Image(output_frame.value()));
    }
  }

  return absl::OkStatus();
}

std::optional<std::shared_ptr<ImageFrame>>
FlatColorImageCalculator::CreateOutputFrame(CalculatorContext* cc) {
  const auto& options = cc->Options<FlatColorImageCalculatorOptions>();

  int output_height = -1;
  int output_width = -1;
  if (use_dimension_from_option_) {
    output_height = options.output_height();
    output_width = options.output_width();
  } else if (!kInImage(cc).IsEmpty()) {
    const Image& input_image = kInImage(cc).Get();
    output_height = input_image.height();
    output_width = input_image.width();
  } else {
    return std::nullopt;
  }

  Color color;
  if (use_color_from_option_) {
    color = options.color();
  } else if (!kInColor(cc).IsEmpty()) {
    color = kInColor(cc).Get();
  } else {
    return std::nullopt;
  }

  auto output_frame = std::make_shared<ImageFrame>(ImageFormat::SRGB,
                                                   output_width, output_height);
  cv::Mat output_mat = mediapipe::formats::MatView(output_frame.get());

  output_mat.setTo(cv::Scalar(color.r(), color.g(), color.b()));

  return output_frame;
}

}  // namespace mediapipe
