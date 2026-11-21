// Copyright 2026 The MediaPipe Authors.
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
#include <memory>
#include <utility>
#include <vector>

#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/image.h"
#include "mediapipe/framework/formats/image_format.pb.h"
#include "mediapipe/framework/formats/image_frame.h"
#include "mediapipe/framework/packet.h"
#include "mediapipe/framework/port/gmock.h"
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/framework/port/parse_text_proto.h"
#include "mediapipe/framework/port/status_matchers.h"
#include "mediapipe/framework/timestamp.h"

namespace mediapipe {
namespace {

TEST(ImageCloneCalculatorTest, ClonesSrgbOnCpuPreservingPixelsAndFormat) {
  auto graph_config = ParseTextProtoOrDie<CalculatorGraphConfig>(R"pb(
    input_stream: "input"
    output_stream: "output"
    node {
      calculator: "ImageCloneCalculator"
      input_stream: "input"
      output_stream: "output"
      options: {
        [mediapipe.ImageCloneCalculatorOptions.ext] { output_on_gpu: false }
      }
    }
  )pb");

  std::vector<Packet> outputs;
  tool::AddVectorSink("output", &graph_config, &outputs);

  CalculatorGraph graph;
  MP_ASSERT_OK(graph.Initialize(graph_config));
  MP_ASSERT_OK(graph.StartRun({}));

  auto frame = std::make_unique<ImageFrame>(
      ImageFormat::SRGB, 2, 1, ImageFrame::kDefaultAlignmentBoundary);
  uint8_t* pixels = frame->MutablePixelData();
  pixels[0] = 10;
  pixels[1] = 20;
  pixels[2] = 30;
  pixels[3] = 40;
  pixels[4] = 50;
  pixels[5] = 60;

  MP_ASSERT_OK(graph.AddPacketToInputStream(
      "input", MakePacket<Image>(std::move(frame)).At(Timestamp(0))));
  MP_ASSERT_OK(graph.CloseAllInputStreams());
  MP_ASSERT_OK(graph.WaitUntilDone());

  ASSERT_EQ(outputs.size(), 1u);
  const Image& out = outputs[0].Get<Image>();
  EXPECT_FALSE(out.UsesGpu());
  EXPECT_EQ(out.image_format(), ImageFormat::SRGB);
  const uint8_t* out_pixels = out.GetImageFrameSharedPtr()->PixelData();
  EXPECT_EQ(out_pixels[0], 10);
  EXPECT_EQ(out_pixels[1], 20);
  EXPECT_EQ(out_pixels[2], 30);
  EXPECT_EQ(out_pixels[3], 40);
  EXPECT_EQ(out_pixels[4], 50);
  EXPECT_EQ(out_pixels[5], 60);
}

#if !MEDIAPIPE_DISABLE_GPU
TEST(ImageCloneCalculatorTest, ClonesSrgbToGpuAndBackPreservingRgbValues) {
  auto graph_config = ParseTextProtoOrDie<CalculatorGraphConfig>(R"pb(
    input_stream: "input"
    output_stream: "gpu_out"
    output_stream: "cpu_out"
    node {
      calculator: "ImageCloneCalculator"
      input_stream: "input"
      output_stream: "gpu_out"
      options: {
        [mediapipe.ImageCloneCalculatorOptions.ext] { output_on_gpu: true }
      }
    }
    node {
      calculator: "ImageCloneCalculator"
      input_stream: "gpu_out"
      output_stream: "cpu_out"
      options: {
        [mediapipe.ImageCloneCalculatorOptions.ext] { output_on_gpu: false }
      }
    }
  )pb");

  std::vector<Packet> gpu_outputs;
  std::vector<Packet> cpu_outputs;
  tool::AddVectorSink("gpu_out", &graph_config, &gpu_outputs);
  tool::AddVectorSink("cpu_out", &graph_config, &cpu_outputs);

  CalculatorGraph graph;
  MP_ASSERT_OK(graph.Initialize(graph_config));
  MP_ASSERT_OK(graph.StartRun({}));

  auto frame = std::make_unique<ImageFrame>(
      ImageFormat::SRGB, 2, 1, ImageFrame::kGlDefaultAlignmentBoundary);
  uint8_t* pixels = frame->MutablePixelData();
  pixels[0] = 200;
  pixels[1] = 100;
  pixels[2] = 50;
  pixels[3] = 10;
  pixels[4] = 120;
  pixels[5] = 240;

  MP_ASSERT_OK(graph.AddPacketToInputStream(
      "input", MakePacket<Image>(std::move(frame)).At(Timestamp(0))));
  MP_ASSERT_OK(graph.CloseAllInputStreams());
  MP_ASSERT_OK(graph.WaitUntilDone());

  ASSERT_EQ(gpu_outputs.size(), 1u);
  EXPECT_TRUE(gpu_outputs[0].Get<Image>().UsesGpu());

  ASSERT_EQ(cpu_outputs.size(), 1u);
  const Image& cpu_out = cpu_outputs[0].Get<Image>();
  EXPECT_FALSE(cpu_out.UsesGpu());
  const int channels = cpu_out.channels();
#if defined(__APPLE__)
  ASSERT_EQ(channels, 4);
  EXPECT_EQ(cpu_out.image_format(), ImageFormat::SRGBA);
#else
  ASSERT_EQ(channels, 3);
  EXPECT_EQ(cpu_out.image_format(), ImageFormat::SRGB);
#endif
  const uint8_t* out_pixels = cpu_out.GetImageFrameSharedPtr()->PixelData();
  EXPECT_EQ(out_pixels[0], 200);
  EXPECT_EQ(out_pixels[1], 100);
  EXPECT_EQ(out_pixels[2], 50);
  EXPECT_EQ(out_pixels[channels + 0], 10);
  EXPECT_EQ(out_pixels[channels + 1], 120);
  EXPECT_EQ(out_pixels[channels + 2], 240);
#if defined(__APPLE__)
  EXPECT_EQ(out_pixels[3], 255);
  EXPECT_EQ(out_pixels[7], 255);
#endif
}

// Uses several rows and a width whose rows need padding bytes, to cover the row
// stride handling of the SRGB-to-SRGBA expansion on Apple platforms.
TEST(ImageCloneCalculatorTest, ClonesMultiRowSrgbToGpuAndBackPreservingRgb) {
  auto graph_config = ParseTextProtoOrDie<CalculatorGraphConfig>(R"pb(
    input_stream: "input"
    output_stream: "cpu_out"
    node {
      calculator: "ImageCloneCalculator"
      input_stream: "input"
      output_stream: "gpu_out"
      options: {
        [mediapipe.ImageCloneCalculatorOptions.ext] { output_on_gpu: true }
      }
    }
    node {
      calculator: "ImageCloneCalculator"
      input_stream: "gpu_out"
      output_stream: "cpu_out"
      options: {
        [mediapipe.ImageCloneCalculatorOptions.ext] { output_on_gpu: false }
      }
    }
  )pb");

  std::vector<Packet> cpu_outputs;
  tool::AddVectorSink("cpu_out", &graph_config, &cpu_outputs);

  CalculatorGraph graph;
  MP_ASSERT_OK(graph.Initialize(graph_config));
  MP_ASSERT_OK(graph.StartRun({}));

  constexpr int kWidth = 37;
  constexpr int kHeight = 5;
  const auto expected_value = [](int x, int y, int channel) {
    return static_cast<uint8_t>(7 * x + 31 * y + 61 * channel);
  };
  auto frame =
      std::make_unique<ImageFrame>(ImageFormat::SRGB, kWidth, kHeight,
                                   ImageFrame::kGlDefaultAlignmentBoundary);
  for (int y = 0; y < kHeight; ++y) {
    uint8_t* row = frame->MutablePixelData() + y * frame->WidthStep();
    for (int x = 0; x < kWidth; ++x) {
      for (int channel = 0; channel < 3; ++channel) {
        row[3 * x + channel] = expected_value(x, y, channel);
      }
    }
  }

  MP_ASSERT_OK(graph.AddPacketToInputStream(
      "input", MakePacket<Image>(std::move(frame)).At(Timestamp(0))));
  MP_ASSERT_OK(graph.CloseAllInputStreams());
  MP_ASSERT_OK(graph.WaitUntilDone());

  ASSERT_EQ(cpu_outputs.size(), 1u);
  const Image& cpu_out = cpu_outputs[0].Get<Image>();
  ASSERT_EQ(cpu_out.width(), kWidth);
  ASSERT_EQ(cpu_out.height(), kHeight);
  const int channels = cpu_out.channels();
#if defined(__APPLE__)
  ASSERT_EQ(channels, 4);
#else
  ASSERT_EQ(channels, 3);
#endif
  const uint8_t* out_pixels = cpu_out.GetImageFrameSharedPtr()->PixelData();
  for (int y = 0; y < kHeight; ++y) {
    const uint8_t* row = out_pixels + y * cpu_out.step();
    for (int x = 0; x < kWidth; ++x) {
      for (int channel = 0; channel < 3; ++channel) {
        ASSERT_EQ(row[channels * x + channel], expected_value(x, y, channel))
            << "x=" << x << " y=" << y << " channel=" << channel;
      }
#if defined(__APPLE__)
      ASSERT_EQ(row[channels * x + 3], 255) << "x=" << x << " y=" << y;
#endif
    }
  }
}
#endif  // !MEDIAPIPE_DISABLE_GPU

}  // namespace
}  // namespace mediapipe
