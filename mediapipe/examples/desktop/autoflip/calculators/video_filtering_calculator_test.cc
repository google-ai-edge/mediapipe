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

#include <memory>
#include <string>
#include <vector>

#include "absl/strings/string_view.h"
#include "absl/strings/substitute.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/calculator_runner.h"
#include "mediapipe/framework/formats/image_frame.h"
#include "mediapipe/framework/port/gmock.h"
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/framework/port/parse_text_proto.h"
#include "mediapipe/framework/port/status_builder.h"
#include "mediapipe/framework/port/status_matchers.h"

namespace mediapipe {
namespace autoflip {
namespace {

constexpr char kOutputFramesTag[] = "OUTPUT_FRAMES";
constexpr char kInputFramesTag[] = "INPUT_FRAMES";

// Default configuration of the calculator.
CalculatorGraphConfig::Node GetCalculatorNode(
    const std::string& fail_if_any, const std::string& extra_options = "") {
  return ParseTextProtoOrDie<CalculatorGraphConfig::Node>(
      absl::Substitute(R"(
        calculator: "VideoFilteringCalculator"
        input_stream: "INPUT_FRAMES:frames"
        output_stream: "OUTPUT_FRAMES:output_frames"
        options: {
          [mediapipe.autoflip.VideoFilteringCalculatorOptions.ext]: {
            fail_if_any: $0
            $1
          }
        }
    )",
                       fail_if_any, extra_options));
}

TEST(VideoFilterCalculatorTest, UpperBoundNoPass) {
  CalculatorGraphConfig::Node config = GetCalculatorNode("false", R"(
    aspect_ratio_filter {
      target_width: 2
      target_height: 1
      filter_type: UPPER_ASPECT_RATIO_THRESHOLD
    }
  )");

  auto runner = ::absl::make_unique<CalculatorRunner>(config);
  const int kFixedWidth = 1000;
  const double kAspectRatio = 5.0 / 1.0;
  auto input_frame = ::absl::make_unique<ImageFrame>(
      ImageFormat::SRGB, kFixedWidth,
      static_cast<int>(kFixedWidth / kAspectRatio), 16);
  runner->MutableInputs()
      ->Tag(kInputFramesTag)
      .packets.push_back(Adopt(input_frame.release()).At(Timestamp(1000)));
  MP_ASSERT_OK(runner->Run());
  const auto& output_packet = runner->Outputs().Tag(kOutputFramesTag).packets;
  EXPECT_TRUE(output_packet.empty());
}

TEST(VerticalFrameRemovalCalculatorTest, UpperBoundPass) {
  CalculatorGraphConfig::Node config = GetCalculatorNode("false", R"(
    aspect_ratio_filter {
      target_width: 2
      target_height: 1
      filter_type: UPPER_ASPECT_RATIO_THRESHOLD
    }
  )");

  auto runner = ::absl::make_unique<CalculatorRunner>(config);
  const int kWidth = 1000;
  const double kAspectRatio = 1.0 / 5.0;
  const double kHeight = static_cast<int>(kWidth / kAspectRatio);
  auto input_frame =
      ::absl::make_unique<ImageFrame>(ImageFormat::SRGB, kWidth, kHeight, 16);
  runner->MutableInputs()
      ->Tag(kInputFramesTag)
      .packets.push_back(Adopt(input_frame.release()).At(Timestamp(1000)));
  MP_ASSERT_OK(runner->Run());
  const auto& output_packet = runner->Outputs().Tag(kOutputFramesTag).packets;
  EXPECT_EQ(1, output_packet.size());
  auto& output_frame = output_packet[0].Get<ImageFrame>();
  EXPECT_EQ(kWidth, output_frame.Width());
  EXPECT_EQ(kHeight, output_frame.Height());
}

TEST(VideoFilterCalculatorTest, LowerBoundNoPass) {
  CalculatorGraphConfig::Node config = GetCalculatorNode("false", R"(
    aspect_ratio_filter {
      target_width: 2
      target_height: 1
      filter_type: LOWER_ASPECT_RATIO_THRESHOLD
    }
  )");

  auto runner = ::absl::make_unique<CalculatorRunner>(config);
  const int kFixedWidth = 1000;
  const double kAspectRatio = 1.0 / 1.0;
  auto input_frame = ::absl::make_unique<ImageFrame>(
      ImageFormat::SRGB, kFixedWidth,
      static_cast<int>(kFixedWidth / kAspectRatio), 16);
  runner->MutableInputs()
      ->Tag(kInputFramesTag)
      .packets.push_back(Adopt(input_frame.release()).At(Timestamp(1000)));
  MP_ASSERT_OK(runner->Run());
  const auto& output_packet = runner->Outputs().Tag(kOutputFramesTag).packets;
  EXPECT_TRUE(output_packet.empty());
}

TEST(VerticalFrameRemovalCalculatorTest, LowerBoundPass) {
  CalculatorGraphConfig::Node config = GetCalculatorNode("false", R"(
    aspect_ratio_filter {
      target_width: 2
      target_height: 1
      filter_type: LOWER_ASPECT_RATIO_THRESHOLD
    }
  )");

  auto runner = ::absl::make_unique<CalculatorRunner>(config);
  const int kWidth = 1000;
  const double kAspectRatio = 5.0 / 1.0;
  const double kHeight = static_cast<int>(kWidth / kAspectRatio);
  auto input_frame =
      ::absl::make_unique<ImageFrame>(ImageFormat::SRGB, kWidth, kHeight, 16);
  runner->MutableInputs()
      ->Tag(kInputFramesTag)
      .packets.push_back(Adopt(input_frame.release()).At(Timestamp(1000)));
  MP_ASSERT_OK(runner->Run());
  const auto& output_packet = runner->Outputs().Tag(kOutputFramesTag).packets;
  EXPECT_EQ(1, output_packet.size());
  auto& output_frame = output_packet[0].Get<ImageFrame>();
  EXPECT_EQ(kWidth, output_frame.Width());
  EXPECT_EQ(kHeight, output_frame.Height());
}

// Test that an error should be generated when fail_if_any is true.
TEST(VerticalFrameRemovalCalculatorTest, OutputError) {
  CalculatorGraphConfig::Node config = GetCalculatorNode("true", R"(
    aspect_ratio_filter {
      target_width: 2
      target_height: 1
      filter_type: LOWER_ASPECT_RATIO_THRESHOLD
    }
  )");

  auto runner = ::absl::make_unique<CalculatorRunner>(config);
  const int kFixedWidth = 1000;
  const double kAspectRatio = 1.0 / 1.0;
  auto input_frame = ::absl::make_unique<ImageFrame>(
      ImageFormat::SRGB, kFixedWidth,
      static_cast<int>(kFixedWidth / kAspectRatio), 16);
  runner->MutableInputs()
      ->Tag(kInputFramesTag)
      .packets.push_back(Adopt(input_frame.release()).At(Timestamp(1000)));
  absl::Status status = runner->Run();
  EXPECT_EQ(status.code(), absl::StatusCode::kUnknown);
  EXPECT_THAT(status.ToString(),
              ::testing::HasSubstr("Failing due to aspect ratio"));
}

}  // namespace
}  // namespace autoflip
}  // namespace mediapipe
