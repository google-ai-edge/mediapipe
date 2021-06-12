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
#include "mediapipe/framework/calculator_runner.h"
#include "mediapipe/framework/deps/file_path.h"
#include "mediapipe/framework/formats/image_frame_opencv.h"
#include "mediapipe/framework/port/file_helpers.h"
#include "mediapipe/framework/port/gmock.h"
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/framework/port/opencv_imgcodecs_inc.h"
#include "mediapipe/framework/port/opencv_imgproc_inc.h"
#include "mediapipe/framework/port/parse_text_proto.h"
#include "mediapipe/framework/port/status_matchers.h"

namespace mediapipe {

namespace {

TEST(OpenCvEncodedImageToImageFrameCalculatorTest, TestRgbJpeg) {
  std::string contents;
  MP_ASSERT_OK(file::GetContents(
      file::JoinPath("./", "/mediapipe/calculators/image/testdata/dino.jpg"),
      &contents));
  Packet input_packet = MakePacket<std::string>(contents);

  CalculatorGraphConfig::Node node_config =
      ParseTextProtoOrDie<CalculatorGraphConfig::Node>(R"pb(
        calculator: "OpenCvEncodedImageToImageFrameCalculator"
        input_stream: "encoded_image"
        output_stream: "image_frame"
      )pb");
  CalculatorRunner runner(node_config);
  runner.MutableInputs()->Index(0).packets.push_back(
      input_packet.At(Timestamp(0)));
  MP_ASSERT_OK(runner.Run());
  const auto& outputs = runner.Outputs();
  ASSERT_EQ(1, outputs.NumEntries());
  const std::vector<Packet>& packets = outputs.Index(0).packets;
  ASSERT_EQ(1, packets.size());
  const ImageFrame& output_frame = packets[0].Get<ImageFrame>();

  cv::Mat input_mat = cv::imread(
      file::JoinPath("./", "/mediapipe/calculators/image/testdata/dino.jpg"));
  cv::Mat output_mat;
  cv::cvtColor(formats::MatView(&output_frame), output_mat, cv::COLOR_RGB2BGR);
  cv::Mat diff;
  cv::absdiff(input_mat, output_mat, diff);
  double max_val;
  cv::minMaxLoc(diff, nullptr, &max_val);
  // Expects that the maximum absolute pixel-by-pixel difference is less
  // than 10.
  EXPECT_LE(max_val, 10);
}

TEST(OpenCvEncodedImageToImageFrameCalculatorTest, TestGrayscaleJpeg) {
  cv::Mat input_mat;
  cv::cvtColor(cv::imread(file::JoinPath("./",
                                         "/mediapipe/calculators/"
                                         "image/testdata/dino.jpg")),
               input_mat, cv::COLOR_RGB2GRAY);
  std::vector<uchar> encode_buffer;
  std::vector<int> parameters;
  parameters.push_back(cv::IMWRITE_JPEG_QUALITY);
  parameters.push_back(100);
  cv::imencode(".jpg", input_mat, encode_buffer, parameters);
  Packet input_packet = MakePacket<std::string>(std::string(absl::string_view(
      reinterpret_cast<const char*>(&encode_buffer[0]), encode_buffer.size())));

  CalculatorGraphConfig::Node node_config =
      ParseTextProtoOrDie<CalculatorGraphConfig::Node>(R"pb(
        calculator: "OpenCvEncodedImageToImageFrameCalculator"
        input_stream: "encoded_image"
        output_stream: "image_frame"
      )pb");
  CalculatorRunner runner(node_config);
  runner.MutableInputs()->Index(0).packets.push_back(
      input_packet.At(Timestamp(0)));
  MP_ASSERT_OK(runner.Run());
  const auto& outputs = runner.Outputs();
  ASSERT_EQ(1, outputs.NumEntries());
  const std::vector<Packet>& packets = outputs.Index(0).packets;
  ASSERT_EQ(1, packets.size());
  const ImageFrame& output_frame = packets[0].Get<ImageFrame>();
  cv::Mat diff;
  cv::absdiff(input_mat, formats::MatView(&output_frame), diff);
  double max_val;
  cv::minMaxLoc(diff, nullptr, &max_val);
  // Expects that the maximum absolute pixel-by-pixel difference is less
  // than 10.
  EXPECT_LE(max_val, 10);
}

}  // namespace
}  // namespace mediapipe
