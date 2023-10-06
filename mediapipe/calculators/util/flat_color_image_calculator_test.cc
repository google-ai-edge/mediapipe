// Copyright 2023 The MediaPipe Authors.
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

#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/calculator_runner.h"
#include "mediapipe/framework/formats/image.h"
#include "mediapipe/framework/formats/image_frame.h"
#include "mediapipe/framework/packet.h"
#include "mediapipe/framework/port/gmock.h"
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/framework/port/status_matchers.h"
#include "mediapipe/util/color.pb.h"

namespace mediapipe {
namespace {

using ::testing::HasSubstr;

constexpr char kImageTag[] = "IMAGE";
constexpr char kColorTag[] = "COLOR";
constexpr int kImageWidth = 256;
constexpr int kImageHeight = 256;

TEST(FlatColorImageCalculatorTest, SpecifyColorThroughOptions) {
  CalculatorRunner runner(R"pb(
    calculator: "FlatColorImageCalculator"
    input_stream: "IMAGE:image"
    output_stream: "IMAGE:out_image"
    options {
      [mediapipe.FlatColorImageCalculatorOptions.ext] {
        color: {
          r: 100,
          g: 200,
          b: 255,
        }
      }
    }
  )pb");

  auto image_frame = std::make_shared<ImageFrame>(ImageFormat::SRGB,
                                                  kImageWidth, kImageHeight);

  for (int ts = 0; ts < 3; ++ts) {
    runner.MutableInputs()->Tag(kImageTag).packets.push_back(
        MakePacket<Image>(image_frame).At(Timestamp(ts)));
  }
  MP_ASSERT_OK(runner.Run());

  const auto& outputs = runner.Outputs().Tag(kImageTag).packets;
  ASSERT_EQ(outputs.size(), 3);

  for (const auto& packet : outputs) {
    const auto& image = packet.Get<Image>();
    EXPECT_EQ(image.width(), kImageWidth);
    EXPECT_EQ(image.height(), kImageHeight);
    auto image_frame = image.GetImageFrameSharedPtr();
    auto* pixel_data = image_frame->PixelData();
    EXPECT_EQ(pixel_data[0], 100);
    EXPECT_EQ(pixel_data[1], 200);
    EXPECT_EQ(pixel_data[2], 255);
  }
}

TEST(FlatColorImageCalculatorTest, SpecifyDimensionThroughOptions) {
  CalculatorRunner runner(R"pb(
    calculator: "FlatColorImageCalculator"
    input_stream: "COLOR:color"
    output_stream: "IMAGE:out_image"
    options {
      [mediapipe.FlatColorImageCalculatorOptions.ext] {
        output_width: 7,
        output_height: 13,
      }
    }
  )pb");

  Color color;
  color.set_r(0);
  color.set_g(5);
  color.set_b(0);

  for (int ts = 0; ts < 3; ++ts) {
    runner.MutableInputs()->Tag(kColorTag).packets.push_back(
        MakePacket<Color>(color).At(Timestamp(ts)));
  }
  MP_ASSERT_OK(runner.Run());

  const auto& outputs = runner.Outputs().Tag(kImageTag).packets;
  ASSERT_EQ(outputs.size(), 3);

  for (const auto& packet : outputs) {
    const auto& image = packet.Get<Image>();
    EXPECT_EQ(image.width(), 7);
    EXPECT_EQ(image.height(), 13);
    auto image_frame = image.GetImageFrameSharedPtr();
    const uint8_t* pixel_data = image_frame->PixelData();
    EXPECT_EQ(pixel_data[0], 0);
    EXPECT_EQ(pixel_data[1], 5);
    EXPECT_EQ(pixel_data[2], 0);
  }
}

TEST(FlatColorImageCalculatorTest, ProducesOutputSidePacket) {
  CalculatorRunner runner(R"pb(
    calculator: "FlatColorImageCalculator"
    output_side_packet: "IMAGE:out_packet"
    options {
      [mediapipe.FlatColorImageCalculatorOptions.ext] {
        output_width: 1
        output_height: 1
        color: {
          r: 100,
          g: 200,
          b: 255,
        }
      }
    }
  )pb");

  MP_ASSERT_OK(runner.Run());

  const auto& image = runner.OutputSidePackets().Tag(kImageTag).Get<Image>();
  EXPECT_EQ(image.width(), 1);
  EXPECT_EQ(image.height(), 1);
  auto image_frame = image.GetImageFrameSharedPtr();
  const uint8_t* pixel_data = image_frame->PixelData();
  EXPECT_EQ(pixel_data[0], 100);
  EXPECT_EQ(pixel_data[1], 200);
  EXPECT_EQ(pixel_data[2], 255);
}

TEST(FlatColorImageCalculatorTest, FailureMissingDimension) {
  CalculatorRunner runner(R"pb(
    calculator: "FlatColorImageCalculator"
    input_stream: "COLOR:color"
    output_stream: "IMAGE:out_image"
  )pb");

  Color color;
  color.set_r(0);
  color.set_g(5);
  color.set_b(0);

  for (int ts = 0; ts < 3; ++ts) {
    runner.MutableInputs()->Tag(kColorTag).packets.push_back(
        MakePacket<Color>(color).At(Timestamp(ts)));
  }
  ASSERT_THAT(runner.Run().message(),
              HasSubstr("Either set IMAGE input stream"));
}

TEST(FlatColorImageCalculatorTest, FailureMissingColor) {
  CalculatorRunner runner(R"pb(
    calculator: "FlatColorImageCalculator"
    input_stream: "IMAGE:image"
    output_stream: "IMAGE:out_image"
  )pb");

  auto image_frame = std::make_shared<ImageFrame>(ImageFormat::SRGB,
                                                  kImageWidth, kImageHeight);

  for (int ts = 0; ts < 3; ++ts) {
    runner.MutableInputs()->Tag(kImageTag).packets.push_back(
        MakePacket<Image>(image_frame).At(Timestamp(ts)));
  }
  ASSERT_THAT(runner.Run().message(),
              HasSubstr("Either set COLOR input stream"));
}

TEST(FlatColorImageCalculatorTest, FailureDuplicateDimension) {
  CalculatorRunner runner(R"pb(
    calculator: "FlatColorImageCalculator"
    input_stream: "IMAGE:image"
    input_stream: "COLOR:color"
    output_stream: "IMAGE:out_image"
    options {
      [mediapipe.FlatColorImageCalculatorOptions.ext] {
        output_width: 7,
        output_height: 13,
      }
    }
  )pb");

  auto image_frame = std::make_shared<ImageFrame>(ImageFormat::SRGB,
                                                  kImageWidth, kImageHeight);

  for (int ts = 0; ts < 3; ++ts) {
    runner.MutableInputs()->Tag(kImageTag).packets.push_back(
        MakePacket<Image>(image_frame).At(Timestamp(ts)));
  }
  ASSERT_THAT(runner.Run().message(),
              HasSubstr("Either set IMAGE input stream"));
}

TEST(FlatColorImageCalculatorTest, FailureDuplicateColor) {
  CalculatorRunner runner(R"pb(
    calculator: "FlatColorImageCalculator"
    input_stream: "IMAGE:image"
    input_stream: "COLOR:color"
    output_stream: "IMAGE:out_image"
    options {
      [mediapipe.FlatColorImageCalculatorOptions.ext] {
        color: {
          r: 100,
          g: 200,
          b: 255,
        }
      }
    }
  )pb");

  Color color;
  color.set_r(0);
  color.set_g(5);
  color.set_b(0);

  for (int ts = 0; ts < 3; ++ts) {
    runner.MutableInputs()->Tag(kColorTag).packets.push_back(
        MakePacket<Color>(color).At(Timestamp(ts)));
  }
  ASSERT_THAT(runner.Run().message(),
              HasSubstr("Either set COLOR input stream"));
}

TEST(FlatColorImageCalculatorTest, FailureDuplicateOutputs) {
  CalculatorRunner runner(R"pb(
    calculator: "FlatColorImageCalculator"
    output_stream: "IMAGE:out_image"
    output_side_packet: "IMAGE:out_packet"
    options {
      [mediapipe.FlatColorImageCalculatorOptions.ext] {
        output_width: 1
        output_height: 1
        color: {
          r: 100,
          g: 200,
          b: 255,
        }
      }
    }
  )pb");

  ASSERT_THAT(
      runner.Run().message(),
      HasSubstr("Set IMAGE either as output stream, or as output side packet"));
}

TEST(FlatColorImageCalculatorTest, FailureSettingInputImageOnOutputSidePacket) {
  CalculatorRunner runner(R"pb(
    calculator: "FlatColorImageCalculator"
    input_stream: "IMAGE:image"
    output_side_packet: "IMAGE:out_packet"
    options {
      [mediapipe.FlatColorImageCalculatorOptions.ext] {
        color: {
          r: 100,
          g: 200,
          b: 255,
        }
      }
    }
  )pb");

  auto image_frame = std::make_shared<ImageFrame>(ImageFormat::SRGB,
                                                  kImageWidth, kImageHeight);

  for (int ts = 0; ts < 3; ++ts) {
    runner.MutableInputs()->Tag(kImageTag).packets.push_back(
        MakePacket<Image>(image_frame).At(Timestamp(ts)));
  }
  ASSERT_THAT(runner.Run().message(),
              HasSubstr("Set size through options, when setting IMAGE as "
                        "output side packet"));
}

}  // namespace
}  // namespace mediapipe
