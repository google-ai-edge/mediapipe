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

#include <cstdint>

#include "absl/log/absl_check.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/image_frame.h"
#include "mediapipe/framework/formats/image_frame_opencv.h"
#include "mediapipe/framework/port/logging.h"
#include "mediapipe/framework/port/opencv_imgproc_inc.h"
#include "mediapipe/framework/port/ret_check.h"
#include "mediapipe/framework/port/source_location.h"
#include "mediapipe/framework/port/status_builder.h"

namespace mediapipe {
namespace {
void SetColorChannel(int channel, uint8_t value, cv::Mat* mat) {
  ABSL_CHECK(mat->depth() == CV_8U);
  ABSL_CHECK(channel < mat->channels());
  const int step = mat->channels();
  for (int r = 0; r < mat->rows; ++r) {
    uint8_t* row_ptr = mat->ptr<uint8_t>(r);
    for (int offset = channel; offset < mat->cols * step; offset += step) {
      row_ptr[offset] = value;
    }
  }
}

constexpr char kRgbaInTag[] = "RGBA_IN";
constexpr char kRgbInTag[] = "RGB_IN";
constexpr char kBgrInTag[] = "BGR_IN";
constexpr char kBgraInTag[] = "BGRA_IN";
constexpr char kGrayInTag[] = "GRAY_IN";
constexpr char kRgbaOutTag[] = "RGBA_OUT";
constexpr char kRgbOutTag[] = "RGB_OUT";
constexpr char kBgraOutTag[] = "BGRA_OUT";
constexpr char kGrayOutTag[] = "GRAY_OUT";
}  // namespace

// A portable color conversion calculator calculator.
//
// The following conversions are currently supported, but it's fairly easy to
// add new ones if this doesn't meet your needs--Don't forget to add a test to
// color_convert_calculator_test.cc if you do!
//   RGBA -> RGB
//   GRAY -> RGB
//   RGB  -> GRAY
//   RGB  -> RGBA
//   RGBA -> BGRA
//   BGRA -> RGBA
//   BGR  -> RGB
//
// This calculator only supports a single input stream and output stream at a
// time. If more than one input stream or output stream is present, the
// calculator will fail at FillExpectations.
// TODO: Remove this requirement by replacing the typed input streams
// with a single generic input and allow multiple simultaneous outputs.
//
// Input streams:
//   RGBA_IN:       The input video stream (ImageFrame, SRGBA).
//   RGB_IN:        The input video stream (ImageFrame, SRGB).
//   BGRA_IN:       The input video stream (ImageFrame, SBGRA).
//   GRAY_IN:       The input video stream (ImageFrame, GRAY8).
//   BGR_IN:        The input video stream (ImageFrame, SBGR).
//
// Output streams:
//   RGBA_OUT:      The output video stream (ImageFrame, SRGBA).
//   RGB_OUT:       The output video stream (ImageFrame, SRGB).
//   BGRA_OUT:      The output video stream (ImageFrame, SBGRA).
//   GRAY_OUT:      The output video stream (ImageFrame, GRAY8).
class ColorConvertCalculator : public CalculatorBase {
 public:
  ~ColorConvertCalculator() override = default;
  static absl::Status GetContract(CalculatorContract* cc);
  absl::Status Process(CalculatorContext* cc) override;

  absl::Status Open(CalculatorContext* cc) override {
    cc->SetOffset(TimestampDiff(0));
    return absl::OkStatus();
  }

 private:
  // Wrangles the appropriate inputs and outputs to perform the color
  // conversion. The ImageFrame on input_tag is converted using the
  // open_cv_convert_code provided and then output on the output_tag stream.
  // Note that the output_format must match the destination conversion code.
  absl::Status ConvertAndOutput(const std::string& input_tag,
                                const std::string& output_tag,
                                ImageFormat::Format output_format,
                                int open_cv_convert_code,
                                CalculatorContext* cc);
};

REGISTER_CALCULATOR(ColorConvertCalculator);

absl::Status ColorConvertCalculator::GetContract(CalculatorContract* cc) {
  RET_CHECK_EQ(cc->Inputs().NumEntries(), 1)
      << "Only one input stream is allowed.";
  RET_CHECK_EQ(cc->Outputs().NumEntries(), 1)
      << "Only one output stream is allowed.";

  if (cc->Inputs().HasTag(kRgbaInTag)) {
    cc->Inputs().Tag(kRgbaInTag).Set<ImageFrame>();
  }

  if (cc->Inputs().HasTag(kGrayInTag)) {
    cc->Inputs().Tag(kGrayInTag).Set<ImageFrame>();
  }

  if (cc->Inputs().HasTag(kRgbInTag)) {
    cc->Inputs().Tag(kRgbInTag).Set<ImageFrame>();
  }

  if (cc->Inputs().HasTag(kBgraInTag)) {
    cc->Inputs().Tag(kBgraInTag).Set<ImageFrame>();
  }

  if (cc->Inputs().HasTag(kBgrInTag)) {
    cc->Inputs().Tag(kBgrInTag).Set<ImageFrame>();
  }

  if (cc->Outputs().HasTag(kRgbOutTag)) {
    cc->Outputs().Tag(kRgbOutTag).Set<ImageFrame>();
  }

  if (cc->Outputs().HasTag(kGrayOutTag)) {
    cc->Outputs().Tag(kGrayOutTag).Set<ImageFrame>();
  }

  if (cc->Outputs().HasTag(kRgbaOutTag)) {
    cc->Outputs().Tag(kRgbaOutTag).Set<ImageFrame>();
  }

  if (cc->Outputs().HasTag(kBgraOutTag)) {
    cc->Outputs().Tag(kBgraOutTag).Set<ImageFrame>();
  }

  return absl::OkStatus();
}

absl::Status ColorConvertCalculator::ConvertAndOutput(
    const std::string& input_tag, const std::string& output_tag,
    ImageFormat::Format output_format, int open_cv_convert_code,
    CalculatorContext* cc) {
  const cv::Mat& input_mat =
      formats::MatView(&cc->Inputs().Tag(input_tag).Get<ImageFrame>());
  std::unique_ptr<ImageFrame> output_frame(
      new ImageFrame(output_format, input_mat.cols, input_mat.rows));
  cv::Mat output_mat = formats::MatView(output_frame.get());
  cv::cvtColor(input_mat, output_mat, open_cv_convert_code);

  // cv::cvtColor will leave the alpha channel set to 0, which is a bizarre
  // design choice. Instead, let's set alpha to 255.
  if (open_cv_convert_code == cv::COLOR_RGB2RGBA) {
    SetColorChannel(3, 255, &output_mat);
  }
  cc->Outputs()
      .Tag(output_tag)
      .Add(output_frame.release(), cc->InputTimestamp());
  return absl::OkStatus();
}

absl::Status ColorConvertCalculator::Process(CalculatorContext* cc) {
  // RGBA -> RGB
  if (cc->Inputs().HasTag(kRgbaInTag) && cc->Outputs().HasTag(kRgbOutTag)) {
    return ConvertAndOutput(kRgbaInTag, kRgbOutTag, ImageFormat::SRGB,
                            cv::COLOR_RGBA2RGB, cc);
  }
  // GRAY -> RGB
  if (cc->Inputs().HasTag(kGrayInTag) && cc->Outputs().HasTag(kRgbOutTag)) {
    return ConvertAndOutput(kGrayInTag, kRgbOutTag, ImageFormat::SRGB,
                            cv::COLOR_GRAY2RGB, cc);
  }
  // RGB -> GRAY
  if (cc->Inputs().HasTag(kRgbInTag) && cc->Outputs().HasTag(kGrayOutTag)) {
    return ConvertAndOutput(kRgbInTag, kGrayOutTag, ImageFormat::GRAY8,
                            cv::COLOR_RGB2GRAY, cc);
  }
  // RGB -> RGBA
  if (cc->Inputs().HasTag(kRgbInTag) && cc->Outputs().HasTag(kRgbaOutTag)) {
    return ConvertAndOutput(kRgbInTag, kRgbaOutTag, ImageFormat::SRGBA,
                            cv::COLOR_RGB2RGBA, cc);
  }
  // BGRA -> RGBA
  if (cc->Inputs().HasTag(kBgraInTag) && cc->Outputs().HasTag(kRgbaOutTag)) {
    return ConvertAndOutput(kBgraInTag, kRgbaOutTag, ImageFormat::SRGBA,
                            cv::COLOR_BGRA2RGBA, cc);
  }
  // RGBA -> BGRA
  if (cc->Inputs().HasTag(kRgbaInTag) && cc->Outputs().HasTag(kBgraOutTag)) {
    return ConvertAndOutput(kRgbaInTag, kBgraOutTag, ImageFormat::SBGRA,
                            cv::COLOR_RGBA2BGRA, cc);
  }
  // BGR -> RGB
  if (cc->Inputs().HasTag(kBgrInTag) && cc->Outputs().HasTag(kRgbOutTag)) {
    return ConvertAndOutput(kBgrInTag, kRgbOutTag, ImageFormat::SRGB,
                            cv::COLOR_BGR2RGB, cc);
  }

  return mediapipe::InvalidArgumentErrorBuilder(MEDIAPIPE_LOC)
         << "Unsupported image format conversion.";
}

}  // namespace mediapipe
