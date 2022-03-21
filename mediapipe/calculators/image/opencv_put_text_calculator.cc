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

#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/image_frame_opencv.h"
#include "mediapipe/framework/port/opencv_imgproc_inc.h"
#include "mediapipe/framework/port/status.h"
#include "mediapipe/framework/port/status_builder.h"

namespace mediapipe {

// Takes in a string, draws the text string by cv::putText(), and outputs an
// ImageFrame.
//
// Example config:
// node {
//   calculator: "OpenCvPutTextCalculator"
//   input_stream: "text_to_put"
//   output_stream: "out_image_frames"
// }
// TODO: Generalize the calculator for other text use cases.
class OpenCvPutTextCalculator : public CalculatorBase {
 public:
  static absl::Status GetContract(CalculatorContract* cc);
  absl::Status Process(CalculatorContext* cc) override;
};

absl::Status OpenCvPutTextCalculator::GetContract(CalculatorContract* cc) {
  cc->Inputs().Index(0).Set<std::string>();
  cc->Outputs().Index(0).Set<ImageFrame>();
  return absl::OkStatus();
}

absl::Status OpenCvPutTextCalculator::Process(CalculatorContext* cc) {
  const std::string& text_content = cc->Inputs().Index(0).Get<std::string>();
  cv::Mat mat = cv::Mat::zeros(640, 640, CV_8UC4);
  cv::putText(mat, text_content, cv::Point(15, 70), cv::FONT_HERSHEY_PLAIN, 3,
              cv::Scalar(255, 255, 0, 255), 4);
  std::unique_ptr<ImageFrame> output_frame = absl::make_unique<ImageFrame>(
      ImageFormat::SRGBA, mat.size().width, mat.size().height);
  mat.copyTo(formats::MatView(output_frame.get()));
  cc->Outputs().Index(0).Add(output_frame.release(), cc->InputTimestamp());
  return absl::OkStatus();
}

REGISTER_CALCULATOR(OpenCvPutTextCalculator);

}  // namespace mediapipe
