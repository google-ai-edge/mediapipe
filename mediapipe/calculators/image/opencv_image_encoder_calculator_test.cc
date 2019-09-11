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

#include "mediapipe/calculators/image/opencv_image_encoder_calculator.pb.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/calculator_runner.h"
#include "mediapipe/framework/deps/file_path.h"
#include "mediapipe/framework/formats/image_frame_opencv.h"
#include "mediapipe/framework/port/gmock.h"
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/framework/port/opencv_imgcodecs_inc.h"
#include "mediapipe/framework/port/opencv_imgproc_inc.h"
#include "mediapipe/framework/port/parse_text_proto.h"
#include "mediapipe/framework/port/status_matchers.h"

namespace mediapipe {

namespace {

TEST(OpenCvImageEncoderCalculatorTest, TestJpegWithQualities) {
  cv::Mat input_mat;
  cv::cvtColor(cv::imread(file::JoinPath("./",
                                         "/mediapipe/calculators/"
                                         "image/testdata/dino.jpg")),
               input_mat, cv::COLOR_BGR2RGB);
  Packet input_packet = MakePacket<ImageFrame>(
      ImageFormat::SRGB, input_mat.size().width, input_mat.size().height);
  input_mat.copyTo(formats::MatView(&(input_packet.Get<ImageFrame>())));

  std::vector<int> qualities = {50, 80};
  for (int quality : qualities) {
    CalculatorGraphConfig::Node node_config =
        ParseTextProtoOrDie<CalculatorGraphConfig::Node>(
            absl::Substitute(R"(
        calculator: "OpenCvImageEncoderCalculator"
        input_stream: "image_frames"
        output_stream: "encoded_images"
        node_options {
          [type.googleapis.com/mediapipe.OpenCvImageEncoderCalculatorOptions]: {
            quality: $0
          }
        })",
                             quality));
    CalculatorRunner runner(node_config);
    runner.MutableInputs()->Index(0).packets.push_back(
        input_packet.At(Timestamp(0)));
    MP_ASSERT_OK(runner.Run());
    const auto& outputs = runner.Outputs();
    ASSERT_EQ(1, outputs.NumEntries());
    const std::vector<Packet>& packets = outputs.Index(0).packets;
    ASSERT_EQ(1, packets.size());
    const auto& result = packets[0].Get<OpenCvImageEncoderCalculatorResults>();
    ASSERT_EQ(input_mat.size().height, result.height());
    ASSERT_EQ(input_mat.size().width, result.width());
    ASSERT_EQ(OpenCvImageEncoderCalculatorResults::RGB, result.colorspace());

    cv::Mat expected_output = cv::imread(
        file::JoinPath("./", absl::Substitute("/mediapipe/calculators/image/"
                                              "testdata/dino_quality_$0.jpg",
                                              quality)));
    const std::vector<char> contents_vector(result.encoded_image().begin(),
                                            result.encoded_image().end());
    cv::Mat decoded_output =
        cv::imdecode(contents_vector, -1 /* return the loaded image as-is */);
    cv::Mat diff;
    cv::absdiff(expected_output, decoded_output, diff);
    double max_val;
    cv::minMaxLoc(diff, nullptr, &max_val);
    // Expects that the maximum absolute pixel-by-pixel difference is less
    // than 10.
    EXPECT_LE(max_val, 10);
  }
}

}  // namespace
}  // namespace mediapipe
