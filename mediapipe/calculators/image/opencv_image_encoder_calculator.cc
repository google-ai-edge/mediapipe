// Copyright 2018 The MediaPipe Authors.
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

#include "absl/log/absl_check.h"
#include "mediapipe/calculators/image/opencv_image_encoder_calculator.pb.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/image_frame_opencv.h"
#include "mediapipe/framework/port/opencv_imgcodecs_inc.h"
#include "mediapipe/framework/port/opencv_imgproc_inc.h"
#include "mediapipe/framework/port/status.h"
#include "mediapipe/framework/port/status_builder.h"

namespace mediapipe {

// Calculator to encode raw image frames. This will result in considerable space
// savings if the frames need to be stored on disk.
//
// Example config:
// node {
//   calculator: "OpenCvImageEncoderCalculator"
//   input_stream: "image"
//   output_stream: "encoded_image"
//   node_options {
//     [type.googleapis.com/mediapipe.OpenCvImageEncoderCalculatorOptions]: {
//       quality: 80
//     }
//   }
// }
class OpenCvImageEncoderCalculator : public CalculatorBase {
 public:
  static absl::Status GetContract(CalculatorContract* cc);
  absl::Status Open(CalculatorContext* cc) override;
  absl::Status Process(CalculatorContext* cc) override;
  absl::Status Close(CalculatorContext* cc) override;

 private:
  int encoding_quality_;
};

absl::Status OpenCvImageEncoderCalculator::GetContract(CalculatorContract* cc) {
  cc->Inputs().Index(0).Set<ImageFrame>();
  cc->Outputs().Index(0).Set<OpenCvImageEncoderCalculatorResults>();
  return absl::OkStatus();
}

absl::Status OpenCvImageEncoderCalculator::Open(CalculatorContext* cc) {
  auto options = cc->Options<OpenCvImageEncoderCalculatorOptions>();
  encoding_quality_ = options.quality();
  return absl::OkStatus();
}

absl::Status OpenCvImageEncoderCalculator::Process(CalculatorContext* cc) {
  const ImageFrame& image_frame = cc->Inputs().Index(0).Get<ImageFrame>();
  ABSL_CHECK_EQ(1, image_frame.ByteDepth());

  std::unique_ptr<OpenCvImageEncoderCalculatorResults> encoded_result =
      absl::make_unique<OpenCvImageEncoderCalculatorResults>();
  encoded_result->set_width(image_frame.Width());
  encoded_result->set_height(image_frame.Height());

  cv::Mat original_mat = formats::MatView(&image_frame);
  cv::Mat input_mat;
  switch (original_mat.channels()) {
    case 1:
      input_mat = original_mat;
      encoded_result->set_colorspace(
          OpenCvImageEncoderCalculatorResults::GRAYSCALE);
      break;
    case 3:
      // OpenCV assumes the image to be BGR order. To use imencode(), do color
      // conversion first.
      cv::cvtColor(original_mat, input_mat, cv::COLOR_RGB2BGR);
      encoded_result->set_colorspace(OpenCvImageEncoderCalculatorResults::RGB);
      break;
    case 4:
      return mediapipe::UnimplementedErrorBuilder(MEDIAPIPE_LOC)
             << "4-channel image isn't supported yet";
    default:
      return mediapipe::FailedPreconditionErrorBuilder(MEDIAPIPE_LOC)
             << "Unsupported number of channels: " << original_mat.channels();
  }

  std::vector<int> parameters;
  parameters.push_back(cv::IMWRITE_JPEG_QUALITY);
  parameters.push_back(encoding_quality_);

  std::vector<uchar> encode_buffer;
  // Note that imencode() will store the data in RGB order.
  // Check its JpegEncoder::write() in "imgcodecs/src/grfmt_jpeg.cpp" for more
  // info.
  if (!cv::imencode(".jpg", input_mat, encode_buffer, parameters)) {
    return mediapipe::InternalErrorBuilder(MEDIAPIPE_LOC)
           << "Fail to encode the image to be jpeg format.";
  }

  encoded_result->set_encoded_image(&encode_buffer[0], encode_buffer.size());

  cc->Outputs().Index(0).Add(encoded_result.release(), cc->InputTimestamp());
  return absl::OkStatus();
}

absl::Status OpenCvImageEncoderCalculator::Close(CalculatorContext* cc) {
  return absl::OkStatus();
}

REGISTER_CALCULATOR(OpenCvImageEncoderCalculator);

}  // namespace mediapipe
