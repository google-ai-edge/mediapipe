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

#include "mediapipe/calculators/image/opencv_encoded_image_to_image_frame_calculator.pb.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/image_frame_opencv.h"
#include "mediapipe/framework/port/opencv_imgcodecs_inc.h"
#include "mediapipe/framework/port/opencv_imgproc_inc.h"
#include "mediapipe/framework/port/status.h"
#include "mediapipe/framework/port/status_builder.h"

namespace mediapipe {

// Takes in an encoded image string, decodes it by OpenCV, and converts to an
// ImageFrame. Note that this calculator only supports grayscale and RGB images
// for now.
//
// Example config:
// node {
//   calculator: "OpenCvEncodedImageToImageFrameCalculator"
//   input_stream: "encoded_image"
//   output_stream: "image_frame"
// }
class OpenCvEncodedImageToImageFrameCalculator : public CalculatorBase {
 public:
  static absl::Status GetContract(CalculatorContract* cc);
  absl::Status Open(CalculatorContext* cc) override;
  absl::Status Process(CalculatorContext* cc) override;

 private:
  mediapipe::OpenCvEncodedImageToImageFrameCalculatorOptions options_;
};

absl::Status OpenCvEncodedImageToImageFrameCalculator::GetContract(
    CalculatorContract* cc) {
  cc->Inputs().Index(0).Set<std::string>();
  cc->Outputs().Index(0).Set<ImageFrame>();
  return absl::OkStatus();
}

absl::Status OpenCvEncodedImageToImageFrameCalculator::Open(
    CalculatorContext* cc) {
  options_ =
      cc->Options<mediapipe::OpenCvEncodedImageToImageFrameCalculatorOptions>();
  return absl::OkStatus();
}

absl::Status OpenCvEncodedImageToImageFrameCalculator::Process(
    CalculatorContext* cc) {
  const std::string& contents = cc->Inputs().Index(0).Get<std::string>();
  const std::vector<char> contents_vector(contents.begin(), contents.end());
  cv::Mat decoded_mat;
  if (options_.apply_orientation_from_exif_data()) {
    // We want to respect the orientation from the EXIF data, which
    // IMREAD_UNCHANGED ignores, but otherwise we want to be as permissive as
    // possible with our reading flags. Therefore, we use IMREAD_ANYCOLOR and
    // IMREAD_ANYDEPTH.
    decoded_mat = cv::imdecode(contents_vector,
                               cv::IMREAD_ANYCOLOR | cv::IMREAD_ANYDEPTH);
  } else {
    // Return the loaded image as-is
    decoded_mat = cv::imdecode(contents_vector, cv::IMREAD_UNCHANGED);
  }
  ImageFormat::Format image_format = ImageFormat::UNKNOWN;
  cv::Mat output_mat;
  switch (decoded_mat.channels()) {
    case 1:
      image_format = ImageFormat::GRAY8;
      output_mat = decoded_mat;
      break;
    case 3:
      image_format = ImageFormat::SRGB;
      cv::cvtColor(decoded_mat, output_mat, cv::COLOR_BGR2RGB);
      break;
    case 4:
      image_format = ImageFormat::SRGBA;
      cv::cvtColor(decoded_mat, output_mat, cv::COLOR_BGR2RGBA);
      break;
    default:
      return mediapipe::FailedPreconditionErrorBuilder(MEDIAPIPE_LOC)
             << "Unsupported number of channels: " << decoded_mat.channels();
  }
  std::unique_ptr<ImageFrame> output_frame = absl::make_unique<ImageFrame>(
      image_format, decoded_mat.size().width, decoded_mat.size().height,
      ImageFrame::kGlDefaultAlignmentBoundary);
  output_mat.copyTo(formats::MatView(output_frame.get()));
  cc->Outputs().Index(0).Add(output_frame.release(), cc->InputTimestamp());
  return absl::OkStatus();
}

REGISTER_CALCULATOR(OpenCvEncodedImageToImageFrameCalculator);

}  // namespace mediapipe
