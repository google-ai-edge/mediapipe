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

#include <math.h>

#include <cmath>
#include <limits>

#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/calculator_runner.h"
#include "mediapipe/framework/deps/file_path.h"
#include "mediapipe/framework/formats/image_file_properties.pb.h"
#include "mediapipe/framework/port/file_helpers.h"
#include "mediapipe/framework/port/gmock.h"
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/framework/port/parse_text_proto.h"
#include "mediapipe/framework/port/status_matchers.h"

namespace mediapipe {

namespace {

constexpr char kImageFilePath[] =
    "/mediapipe/calculators/image/testdata/"
    "front_camera_pixel2.jpg";
constexpr int kExpectedWidth = 2448;
constexpr int kExpectedHeight = 3264;
constexpr double kExpectedFocalLengthMm = 3.38;
constexpr double kExpectedFocalLengthIn35Mm = 25;
constexpr double kExpectedFocalLengthPixels = 2357.48;

double RoundToNDecimals(double value, int n) {
  return std::round(value * pow(10.0, n)) / pow(10.0, n);
}

TEST(ImageFilePropertiesCalculatorTest, ReadsFocalLengthFromJpegInStreams) {
  std::string image_filepath = file::JoinPath("./", kImageFilePath);
  std::string image_contents;
  MP_ASSERT_OK(file::GetContents(image_filepath, &image_contents));

  CalculatorGraphConfig::Node node_config =
      ParseTextProtoOrDie<CalculatorGraphConfig::Node>(R"pb(
        calculator: "ImageFilePropertiesCalculator"
        input_stream: "image_bytes"
        output_stream: "properties"
      )pb");

  CalculatorRunner runner(node_config);
  runner.MutableInputs()->Index(0).packets.push_back(
      MakePacket<std::string>(image_contents).At(Timestamp(0)));
  MP_ASSERT_OK(runner.Run());
  const auto& outputs = runner.Outputs();
  ASSERT_EQ(1, outputs.NumEntries());
  const std::vector<Packet>& packets = outputs.Index(0).packets;
  ASSERT_EQ(1, packets.size());
  const auto& result = packets[0].Get<::mediapipe::ImageFileProperties>();
  EXPECT_EQ(kExpectedWidth, result.image_width());
  EXPECT_EQ(kExpectedHeight, result.image_height());
  EXPECT_DOUBLE_EQ(kExpectedFocalLengthMm, result.focal_length_mm());
  EXPECT_DOUBLE_EQ(kExpectedFocalLengthIn35Mm, result.focal_length_35mm());
  EXPECT_DOUBLE_EQ(kExpectedFocalLengthPixels,
                   RoundToNDecimals(result.focal_length_pixels(), /*n=*/2));
}

TEST(ImageFilePropertiesCalculatorTest, ReadsFocalLengthFromJpegInSidePackets) {
  std::string image_filepath = file::JoinPath("./", kImageFilePath);
  std::string image_contents;
  MP_ASSERT_OK(file::GetContents(image_filepath, &image_contents));

  CalculatorGraphConfig::Node node_config =
      ParseTextProtoOrDie<CalculatorGraphConfig::Node>(R"pb(
        calculator: "ImageFilePropertiesCalculator"
        input_side_packet: "image_bytes"
        output_side_packet: "properties"
      )pb");

  CalculatorRunner runner(node_config);
  runner.MutableSidePackets()->Index(0) =
      MakePacket<std::string>(image_contents).At(Timestamp(0));
  MP_ASSERT_OK(runner.Run());
  const auto& outputs = runner.OutputSidePackets();
  EXPECT_EQ(1, outputs.NumEntries());
  const auto& packet = outputs.Index(0);
  const auto& result = packet.Get<::mediapipe::ImageFileProperties>();
  EXPECT_EQ(kExpectedWidth, result.image_width());
  EXPECT_EQ(kExpectedHeight, result.image_height());
  EXPECT_DOUBLE_EQ(kExpectedFocalLengthMm, result.focal_length_mm());
  EXPECT_DOUBLE_EQ(kExpectedFocalLengthIn35Mm, result.focal_length_35mm());
  EXPECT_DOUBLE_EQ(kExpectedFocalLengthPixels,
                   RoundToNDecimals(result.focal_length_pixels(), /*n=*/2));
}

TEST(ImageFilePropertiesCalculatorTest,
     ReadsFocalLengthFromJpegStreamToSidePacket) {
  std::string image_filepath = file::JoinPath("./", kImageFilePath);
  std::string image_contents;
  MP_ASSERT_OK(file::GetContents(image_filepath, &image_contents));

  CalculatorGraphConfig::Node node_config =
      ParseTextProtoOrDie<CalculatorGraphConfig::Node>(R"pb(
        calculator: "ImageFilePropertiesCalculator"
        input_stream: "image_bytes"
        output_side_packet: "properties"
      )pb");

  CalculatorRunner runner(node_config);
  runner.MutableInputs()->Index(0).packets.push_back(
      MakePacket<std::string>(image_contents).At(Timestamp(0)));
  MP_ASSERT_OK(runner.Run());
  const auto& outputs = runner.OutputSidePackets();
  EXPECT_EQ(1, outputs.NumEntries());
  const auto& packet = outputs.Index(0);
  const auto& result = packet.Get<::mediapipe::ImageFileProperties>();
  EXPECT_EQ(kExpectedWidth, result.image_width());
  EXPECT_EQ(kExpectedHeight, result.image_height());
  EXPECT_DOUBLE_EQ(kExpectedFocalLengthMm, result.focal_length_mm());
  EXPECT_DOUBLE_EQ(kExpectedFocalLengthIn35Mm, result.focal_length_35mm());
  EXPECT_DOUBLE_EQ(kExpectedFocalLengthPixels,
                   RoundToNDecimals(result.focal_length_pixels(), /*n=*/2));
}

}  // namespace
}  // namespace mediapipe
