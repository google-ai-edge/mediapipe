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

#include "absl/strings/string_view.h"
#include "mediapipe/examples/desktop/autoflip/autoflip_messages.pb.h"
#include "mediapipe/examples/desktop/autoflip/calculators/border_detection_calculator.pb.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/calculator_runner.h"
#include "mediapipe/framework/formats/image_frame.h"
#include "mediapipe/framework/formats/image_frame_opencv.h"
#include "mediapipe/framework/port/benchmark.h"
#include "mediapipe/framework/port/gmock.h"
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/framework/port/opencv_core_inc.h"
#include "mediapipe/framework/port/parse_text_proto.h"
#include "mediapipe/framework/port/ret_check.h"
#include "mediapipe/framework/port/status.h"
#include "mediapipe/framework/port/status_matchers.h"

using mediapipe::CalculatorGraphConfig;
using mediapipe::CalculatorRunner;
using mediapipe::ImageFormat;
using mediapipe::ImageFrame;
using mediapipe::Packet;
using mediapipe::PacketTypeSet;

namespace mediapipe {
namespace autoflip {
namespace {

constexpr char kDetectedBordersTag[] = "DETECTED_BORDERS";
constexpr char kVideoTag[] = "VIDEO";

const char kConfig[] = R"(
    calculator: "BorderDetectionCalculator"
    input_stream: "VIDEO:camera_frames"
    output_stream: "DETECTED_BORDERS:regions"
    options:{
    [mediapipe.autoflip.BorderDetectionCalculatorOptions.ext]:{
      border_object_padding_px: 0
    }
    })";

const char kConfigPad[] = R"(
    calculator: "BorderDetectionCalculator"
    input_stream: "VIDEO:camera_frames"
    output_stream: "DETECTED_BORDERS:regions"
    options:{
    [mediapipe.autoflip.BorderDetectionCalculatorOptions.ext]:{
      default_padding_px: 10
      border_object_padding_px: 0
    }
    })";

const int kTestFrameWidth = 640;
const int kTestFrameHeight = 480;

const int kTestFrameLargeWidth = 1920;
const int kTestFrameLargeHeight = 1080;

const int kTestFrameWidthTall = 1200;
const int kTestFrameHeightTall = 2001;

TEST(BorderDetectionCalculatorTest, NoBorderTest) {
  auto runner = ::absl::make_unique<CalculatorRunner>(
      ParseTextProtoOrDie<CalculatorGraphConfig::Node>(kConfig));

  auto input_frame = ::absl::make_unique<ImageFrame>(
      ImageFormat::SRGB, kTestFrameWidth, kTestFrameHeight);
  cv::Mat input_mat = mediapipe::formats::MatView(input_frame.get());
  input_mat.setTo(cv::Scalar(0, 0, 0));
  runner->MutableInputs()->Tag(kVideoTag).packets.push_back(
      Adopt(input_frame.release()).At(Timestamp::PostStream()));

  // Run the calculator.
  MP_ASSERT_OK(runner->Run());

  const std::vector<Packet>& output_packets =
      runner->Outputs().Tag(kDetectedBordersTag).packets;
  ASSERT_EQ(1, output_packets.size());
  const auto& static_features = output_packets[0].Get<StaticFeatures>();
  ASSERT_EQ(0, static_features.border().size());
  EXPECT_EQ(0, static_features.non_static_area().x());
  EXPECT_EQ(0, static_features.non_static_area().y());
  EXPECT_EQ(kTestFrameWidth, static_features.non_static_area().width());
  EXPECT_EQ(kTestFrameHeight, static_features.non_static_area().height());
  EXPECT_TRUE(static_features.has_solid_background());
  EXPECT_EQ(0, static_features.solid_background().r());
  EXPECT_EQ(0, static_features.solid_background().g());
  EXPECT_EQ(0, static_features.solid_background().b());
}

TEST(BorderDetectionCalculatorTest, TopBorderTest) {
  auto runner = ::absl::make_unique<CalculatorRunner>(
      ParseTextProtoOrDie<CalculatorGraphConfig::Node>(kConfig));

  const int kTopBorderHeight = 50;

  auto input_frame = ::absl::make_unique<ImageFrame>(
      ImageFormat::SRGB, kTestFrameWidth, kTestFrameHeight);
  cv::Mat input_mat = mediapipe::formats::MatView(input_frame.get());
  input_mat.setTo(cv::Scalar(0, 0, 0));
  cv::Mat sub_image =
      input_mat(cv::Rect(0, 0, kTestFrameWidth, kTopBorderHeight));
  sub_image.setTo(cv::Scalar(255, 0, 0));
  runner->MutableInputs()->Tag(kVideoTag).packets.push_back(
      Adopt(input_frame.release()).At(Timestamp::PostStream()));

  // Run the calculator.
  MP_ASSERT_OK(runner->Run());

  const std::vector<Packet>& output_packets =
      runner->Outputs().Tag(kDetectedBordersTag).packets;
  ASSERT_EQ(1, output_packets.size());
  const auto& static_features = output_packets[0].Get<StaticFeatures>();
  ASSERT_EQ(1, static_features.border().size());
  const auto& part = static_features.border(0);
  EXPECT_EQ(part.border_position().x(), 0);
  EXPECT_EQ(part.border_position().y(), 0);
  EXPECT_EQ(part.border_position().width(), kTestFrameWidth);
  EXPECT_LT(std::abs(part.border_position().height() - kTopBorderHeight), 2);
  EXPECT_TRUE(static_features.has_solid_background());
  EXPECT_EQ(0, static_features.solid_background().r());
  EXPECT_EQ(0, static_features.solid_background().g());
  EXPECT_EQ(0, static_features.solid_background().b());
  EXPECT_EQ(0, static_features.non_static_area().x());
  EXPECT_EQ(kTopBorderHeight - 1, static_features.non_static_area().y());
  EXPECT_EQ(kTestFrameWidth, static_features.non_static_area().width());
  EXPECT_EQ(kTestFrameHeight - kTopBorderHeight + 1,
            static_features.non_static_area().height());
}

TEST(BorderDetectionCalculatorTest, TopBorderPadTest) {
  auto runner = ::absl::make_unique<CalculatorRunner>(
      ParseTextProtoOrDie<CalculatorGraphConfig::Node>(kConfigPad));

  const int kTopBorderHeight = 50;

  auto input_frame = ::absl::make_unique<ImageFrame>(
      ImageFormat::SRGB, kTestFrameWidth, kTestFrameHeight);
  cv::Mat input_mat = mediapipe::formats::MatView(input_frame.get());
  input_mat.setTo(cv::Scalar(0, 0, 0));
  cv::Mat sub_image =
      input_mat(cv::Rect(0, 0, kTestFrameWidth, kTopBorderHeight));
  sub_image.setTo(cv::Scalar(255, 0, 0));
  runner->MutableInputs()->Tag(kVideoTag).packets.push_back(
      Adopt(input_frame.release()).At(Timestamp::PostStream()));

  // Run the calculator.
  MP_ASSERT_OK(runner->Run());

  const std::vector<Packet>& output_packets =
      runner->Outputs().Tag(kDetectedBordersTag).packets;
  ASSERT_EQ(1, output_packets.size());
  const auto& static_features = output_packets[0].Get<StaticFeatures>();
  ASSERT_EQ(1, static_features.border().size());
  const auto& part = static_features.border(0);
  EXPECT_EQ(part.border_position().x(), 0);
  EXPECT_EQ(part.border_position().y(), 0);
  EXPECT_EQ(part.border_position().width(), kTestFrameWidth);
  EXPECT_LT(std::abs(part.border_position().height() - kTopBorderHeight), 2);
  EXPECT_TRUE(static_features.has_solid_background());
  EXPECT_EQ(0, static_features.solid_background().r());
  EXPECT_EQ(0, static_features.solid_background().g());
  EXPECT_EQ(0, static_features.solid_background().b());
  EXPECT_EQ(Border::TOP, part.relative_position());
  EXPECT_EQ(0, static_features.non_static_area().x());
  EXPECT_EQ(9 + kTopBorderHeight, static_features.non_static_area().y());
  EXPECT_EQ(kTestFrameWidth, static_features.non_static_area().width());
  EXPECT_EQ(kTestFrameHeight - 19 - kTopBorderHeight,
            static_features.non_static_area().height());
}

TEST(BorderDetectionCalculatorTest, BottomBorderTest) {
  auto runner = ::absl::make_unique<CalculatorRunner>(
      ParseTextProtoOrDie<CalculatorGraphConfig::Node>(kConfig));

  const int kBottomBorderHeight = 50;

  auto input_frame = ::absl::make_unique<ImageFrame>(
      ImageFormat::SRGB, kTestFrameWidth, kTestFrameHeight);
  cv::Mat input_mat = mediapipe::formats::MatView(input_frame.get());
  input_mat.setTo(cv::Scalar(0, 0, 0));
  cv::Mat bottom_image =
      input_mat(cv::Rect(0, kTestFrameHeight - kBottomBorderHeight,
                         kTestFrameWidth, kBottomBorderHeight));
  bottom_image.setTo(cv::Scalar(255, 0, 0));
  runner->MutableInputs()->Tag(kVideoTag).packets.push_back(
      Adopt(input_frame.release()).At(Timestamp::PostStream()));

  // Run the calculator.
  MP_ASSERT_OK(runner->Run());

  const std::vector<Packet>& output_packets =
      runner->Outputs().Tag(kDetectedBordersTag).packets;
  ASSERT_EQ(1, output_packets.size());
  const auto& static_features = output_packets[0].Get<StaticFeatures>();
  ASSERT_EQ(1, static_features.border().size());
  const auto& part = static_features.border(0);
  EXPECT_EQ(part.border_position().x(), 0);
  EXPECT_EQ(part.border_position().y(), kTestFrameHeight - kBottomBorderHeight);
  EXPECT_EQ(part.border_position().width(), kTestFrameWidth);
  EXPECT_LT(std::abs(part.border_position().height() - kBottomBorderHeight), 2);
  EXPECT_TRUE(static_features.has_solid_background());
  EXPECT_EQ(0, static_features.solid_background().r());
  EXPECT_EQ(0, static_features.solid_background().g());
  EXPECT_EQ(0, static_features.solid_background().b());
  EXPECT_EQ(Border::BOTTOM, part.relative_position());
}

TEST(BorderDetectionCalculatorTest, TopBottomBorderTest) {
  auto runner = ::absl::make_unique<CalculatorRunner>(
      ParseTextProtoOrDie<CalculatorGraphConfig::Node>(kConfig));

  const int kBottomBorderHeight = 50;
  const int kTopBorderHeight = 25;

  auto input_frame = ::absl::make_unique<ImageFrame>(
      ImageFormat::SRGB, kTestFrameWidth, kTestFrameHeight);
  cv::Mat input_mat = mediapipe::formats::MatView(input_frame.get());
  input_mat.setTo(cv::Scalar(0, 0, 0));
  cv::Mat top_image =
      input_mat(cv::Rect(0, 0, kTestFrameWidth, kTopBorderHeight));
  top_image.setTo(cv::Scalar(0, 255, 0));
  cv::Mat bottom_image =
      input_mat(cv::Rect(0, kTestFrameHeight - kBottomBorderHeight,
                         kTestFrameWidth, kBottomBorderHeight));
  bottom_image.setTo(cv::Scalar(255, 0, 0));
  runner->MutableInputs()->Tag(kVideoTag).packets.push_back(
      Adopt(input_frame.release()).At(Timestamp::PostStream()));

  // Run the calculator.
  MP_ASSERT_OK(runner->Run());

  const std::vector<Packet>& output_packets =
      runner->Outputs().Tag(kDetectedBordersTag).packets;
  ASSERT_EQ(1, output_packets.size());
  const auto& static_features = output_packets[0].Get<StaticFeatures>();
  ASSERT_EQ(2, static_features.border().size());
  auto part = static_features.border(0);
  EXPECT_EQ(part.border_position().x(), 0);
  EXPECT_EQ(part.border_position().y(), 0);
  EXPECT_EQ(part.border_position().width(), kTestFrameWidth);
  EXPECT_LT(std::abs(part.border_position().height() - kTopBorderHeight), 2);
  EXPECT_TRUE(static_features.has_solid_background());
  EXPECT_EQ(0, static_features.solid_background().r());
  EXPECT_EQ(0, static_features.solid_background().g());
  EXPECT_EQ(0, static_features.solid_background().b());
  EXPECT_EQ(0, static_features.non_static_area().x());
  EXPECT_EQ(kTopBorderHeight - 1, static_features.non_static_area().y());
  EXPECT_EQ(kTestFrameWidth, static_features.non_static_area().width());
  EXPECT_EQ(kTestFrameHeight - kTopBorderHeight - kBottomBorderHeight + 2,
            static_features.non_static_area().height());
  EXPECT_EQ(Border::TOP, part.relative_position());

  part = static_features.border(1);
  EXPECT_EQ(part.border_position().x(), 0);
  EXPECT_EQ(part.border_position().y(), kTestFrameHeight - kBottomBorderHeight);
  EXPECT_EQ(part.border_position().width(), kTestFrameWidth);
  EXPECT_LT(std::abs(part.border_position().height() - kBottomBorderHeight), 2);
  EXPECT_EQ(Border::BOTTOM, part.relative_position());
}

TEST(BorderDetectionCalculatorTest, TopBottomBorderTestAspect2) {
  auto runner = ::absl::make_unique<CalculatorRunner>(
      ParseTextProtoOrDie<CalculatorGraphConfig::Node>(kConfig));

  const int kBottomBorderHeight = 50;
  const int kTopBorderHeight = 25;

  auto input_frame = ::absl::make_unique<ImageFrame>(
      ImageFormat::SRGB, kTestFrameWidthTall, kTestFrameHeightTall);
  cv::Mat input_mat = mediapipe::formats::MatView(input_frame.get());
  input_mat.setTo(cv::Scalar(0, 0, 0));
  cv::Mat top_image =
      input_mat(cv::Rect(0, 0, kTestFrameWidthTall, kTopBorderHeight));
  top_image.setTo(cv::Scalar(0, 255, 0));
  cv::Mat bottom_image =
      input_mat(cv::Rect(0, kTestFrameHeightTall - kBottomBorderHeight,
                         kTestFrameWidthTall, kBottomBorderHeight));
  bottom_image.setTo(cv::Scalar(255, 0, 0));
  runner->MutableInputs()->Tag(kVideoTag).packets.push_back(
      Adopt(input_frame.release()).At(Timestamp::PostStream()));

  // Run the calculator.
  MP_ASSERT_OK(runner->Run());

  const std::vector<Packet>& output_packets =
      runner->Outputs().Tag(kDetectedBordersTag).packets;
  ASSERT_EQ(1, output_packets.size());
  const auto& static_features = output_packets[0].Get<StaticFeatures>();
  ASSERT_EQ(2, static_features.border().size());
  auto part = static_features.border(0);
  EXPECT_EQ(part.border_position().x(), 0);
  EXPECT_EQ(part.border_position().y(), 0);
  EXPECT_EQ(part.border_position().width(), kTestFrameWidthTall);
  EXPECT_LT(std::abs(part.border_position().height() - kTopBorderHeight), 2);
  EXPECT_TRUE(static_features.has_solid_background());
  EXPECT_EQ(0, static_features.solid_background().r());
  EXPECT_EQ(0, static_features.solid_background().g());
  EXPECT_EQ(0, static_features.solid_background().b());
  EXPECT_EQ(Border::TOP, part.relative_position());

  part = static_features.border(1);
  EXPECT_EQ(part.border_position().x(), 0);
  EXPECT_EQ(part.border_position().y(),
            kTestFrameHeightTall - kBottomBorderHeight);
  EXPECT_EQ(part.border_position().width(), kTestFrameWidthTall);
  EXPECT_LT(std::abs(part.border_position().height() - kBottomBorderHeight), 2);
  EXPECT_TRUE(static_features.has_solid_background());
  EXPECT_EQ(0, static_features.solid_background().r());
  EXPECT_EQ(0, static_features.solid_background().g());
  EXPECT_EQ(0, static_features.solid_background().b());
  EXPECT_EQ(Border::BOTTOM, part.relative_position());
}

TEST(BorderDetectionCalculatorTest, DominantColor) {
  CalculatorGraphConfig::Node node =
      ParseTextProtoOrDie<CalculatorGraphConfig::Node>(kConfigPad);
  node.mutable_options()
      ->MutableExtension(BorderDetectionCalculatorOptions::ext)
      ->set_solid_background_tol_perc(.25);

  auto runner = ::absl::make_unique<CalculatorRunner>(node);

  auto input_frame = ::absl::make_unique<ImageFrame>(
      ImageFormat::SRGB, kTestFrameWidth, kTestFrameHeight);
  cv::Mat input_mat = mediapipe::formats::MatView(input_frame.get());
  input_mat.setTo(cv::Scalar(0, 0, 0));

  cv::Mat sub_image = input_mat(cv::Rect(
      kTestFrameWidth / 2, 0, kTestFrameWidth / 2, kTestFrameHeight / 2));
  sub_image.setTo(cv::Scalar(0, 255, 0));

  sub_image = input_mat(cv::Rect(0, kTestFrameHeight / 2, kTestFrameWidth / 2,
                                 kTestFrameHeight / 2));
  sub_image.setTo(cv::Scalar(0, 0, 255));

  sub_image =
      input_mat(cv::Rect(0, 0, kTestFrameWidth / 2 + 50, kTestFrameHeight / 2));
  sub_image.setTo(cv::Scalar(255, 0, 0));

  runner->MutableInputs()->Tag(kVideoTag).packets.push_back(
      Adopt(input_frame.release()).At(Timestamp::PostStream()));

  // Run the calculator.
  MP_ASSERT_OK(runner->Run());

  const std::vector<Packet>& output_packets =
      runner->Outputs().Tag(kDetectedBordersTag).packets;
  ASSERT_EQ(1, output_packets.size());
  const auto& static_features = output_packets[0].Get<StaticFeatures>();
  ASSERT_EQ(0, static_features.border().size());
  ASSERT_TRUE(static_features.has_solid_background());
  EXPECT_EQ(0, static_features.solid_background().r());
  EXPECT_EQ(0, static_features.solid_background().g());
  EXPECT_EQ(255, static_features.solid_background().b());
}

void BM_Large(benchmark::State& state) {
  for (auto _ : state) {
    auto runner = ::absl::make_unique<CalculatorRunner>(
        ParseTextProtoOrDie<CalculatorGraphConfig::Node>(kConfig));

    const int kTopBorderHeight = 50;

    auto input_frame = ::absl::make_unique<ImageFrame>(
        ImageFormat::SRGB, kTestFrameLargeWidth, kTestFrameLargeHeight);
    cv::Mat input_mat = mediapipe::formats::MatView(input_frame.get());
    input_mat.setTo(cv::Scalar(0, 0, 0));
    cv::Mat sub_image =
        input_mat(cv::Rect(0, 0, kTestFrameLargeWidth, kTopBorderHeight));
    sub_image.setTo(cv::Scalar(255, 0, 0));
    runner->MutableInputs()->Tag(kVideoTag).packets.push_back(
        Adopt(input_frame.release()).At(Timestamp::PostStream()));

    // Run the calculator.
    MP_ASSERT_OK(runner->Run());
  }
}
BENCHMARK(BM_Large);

}  // namespace
}  // namespace autoflip
}  // namespace mediapipe
