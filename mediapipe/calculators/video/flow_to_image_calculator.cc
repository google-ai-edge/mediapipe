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

// MediaPipe calculator to take a flow field as input, and outputs a normalized
// RGB image where the B channel is forced to zero.
// TODO: Add video stream header for visualization

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "mediapipe/calculators/video/flow_to_image_calculator.pb.h"
#include "mediapipe/calculators/video/tool/flow_quantizer_model.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/image_format.pb.h"
#include "mediapipe/framework/formats/image_frame.h"
#include "mediapipe/framework/formats/image_frame_opencv.h"
#include "mediapipe/framework/formats/motion/optical_flow_field.h"
#include "mediapipe/framework/port/opencv_core_inc.h"
#include "mediapipe/framework/port/parse_text_proto.h"

namespace mediapipe {

// Reads optical flow fields defined in
// mediapipe/framework/formats/motion/optical_flow_field.h,
// returns a VideoFrame with 2 channels (v_x and v_y), each channel is quantized
// to 0-255.
//
// Example config:
// node {
//   calculator: "FlowToImageCalculator"
//   input_stream: "flow_fields"
//   output_stream: "frames"
//   options:  {
//     [type.googleapis.com/mediapipe.FlowToImageCalculatorOptions]:{
//       min_value: -40.0
//       max_value: 40.0
//     }
//   }
// }
class FlowToImageCalculator : public CalculatorBase {
 public:
  FlowToImageCalculator() {}
  ~FlowToImageCalculator() override {}
  static ::mediapipe::Status GetContract(CalculatorContract* cc);
  ::mediapipe::Status Open(CalculatorContext* cc) override;
  ::mediapipe::Status Process(CalculatorContext* cc) override;

 private:
  FlowQuantizerModel model_;
};

::mediapipe::Status FlowToImageCalculator::GetContract(CalculatorContract* cc) {
  cc->Inputs().Index(0).Set<OpticalFlowField>();
  cc->Outputs().Index(0).Set<ImageFrame>();

  // Model sanity check
  const auto& options = cc->Options<FlowToImageCalculatorOptions>();
  if (options.min_value() >= options.max_value()) {
    return ::mediapipe::InvalidArgumentError("Invalid quantizer model.");
  }
  return ::mediapipe::OkStatus();
}

::mediapipe::Status FlowToImageCalculator::Open(CalculatorContext* cc) {
  const auto& options = cc->Options<FlowToImageCalculatorOptions>();
  // Fill the the model_data, ideally we want to train the model, but we omit
  // the step for now, and takes the (min, max) range from protobuf.
  const QuantizerModelData& model_data =
      ParseTextProtoOrDie<QuantizerModelData>(
          absl::StrFormat("min_value:%f min_value:%f max_value:%f max_value:%f",
                          options.min_value(), options.min_value(),
                          options.max_value(), options.max_value()));
  model_.LoadFromProto(model_data);
  return ::mediapipe::OkStatus();
}

::mediapipe::Status FlowToImageCalculator::Process(CalculatorContext* cc) {
  const auto& input = cc->Inputs().Index(0).Get<OpticalFlowField>();
  // Input flow is 2-channel with x-dim flow and y-dim flow.
  // Convert it to a ImageFrame in SRGB space, the 3rd channel is not used (0).
  const cv::Mat_<cv::Point2f>& flow = input.flow_data();
  std::unique_ptr<ImageFrame> output(
      new ImageFrame(ImageFormat::SRGB, input.width(), input.height()));
  cv::Mat image = ::mediapipe::formats::MatView(output.get());

  for (int j = 0; j != input.height(); ++j) {
    for (int i = 0; i != input.width(); ++i) {
      image.at<cv::Vec3b>(j, i) =
          cv::Vec3b(model_.Apply(flow.at<cv::Point2f>(j, i).x, 0),
                    model_.Apply(flow.at<cv::Point2f>(j, i).y, 1), 0);
    }
  }
  cc->Outputs().Index(0).Add(output.release(), cc->InputTimestamp());
  return ::mediapipe::OkStatus();
}

REGISTER_CALCULATOR(FlowToImageCalculator);

}  // namespace mediapipe
