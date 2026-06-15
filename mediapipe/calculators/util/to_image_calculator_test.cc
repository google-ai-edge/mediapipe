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

#include "mediapipe/calculators/util/to_image_calculator.h"

#include <memory>

#include "mediapipe/framework/api3/function_runner.h"
#include "mediapipe/framework/api3/graph.h"
#include "mediapipe/framework/api3/packet.h"
#include "mediapipe/framework/api3/stream.h"
#include "mediapipe/framework/formats/image.h"
#include "mediapipe/framework/formats/image_format.pb.h"
#include "mediapipe/framework/formats/image_frame.h"
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/framework/port/status_matchers.h"

namespace mediapipe::api3 {
namespace {

TEST(ToImageCalculatorTest, ConvertsImageFrameToImage) {
  MP_ASSERT_OK_AND_ASSIGN(
      auto runner,
      Runner::For([](GenericGraph& graph, Stream<mediapipe::ImageFrame> image) {
        auto& node = graph.AddNode<ToImageNode>();
        node.in_image_cpu.Set(image);
        return node.out_image.Get();
      }).Create());

  MP_ASSERT_OK_AND_ASSIGN(Packet<Image> output_packet,
                          runner.Run(api3::MakePacket<mediapipe::ImageFrame>(
                              ImageFormat::SRGB, 10, 10)));

  const Image& output_image = output_packet.GetOrDie();
  EXPECT_EQ(output_image.width(), 10);
  EXPECT_EQ(output_image.height(), 10);
  EXPECT_EQ(output_image.image_format(), ImageFormat::SRGB);
}

TEST(ToImageCalculatorTest, ConvertsGenericImageToImage) {
  MP_ASSERT_OK_AND_ASSIGN(
      auto runner,
      Runner::For([](GenericGraph& graph, Stream<mediapipe::ImageFrame> image) {
        auto& node = graph.AddNode<ToImageNode>();
        node.in_image.Set(image);
        return node.out_image.Get();
      }).Create());

  MP_ASSERT_OK_AND_ASSIGN(Packet<Image> output_packet,
                          runner.Run(api3::MakePacket<mediapipe::ImageFrame>(
                              ImageFormat::SRGB, 10, 10)));

  const Image& output_image = output_packet.GetOrDie();
  EXPECT_EQ(output_image.width(), 10);
  EXPECT_EQ(output_image.height(), 10);
  EXPECT_EQ(output_image.image_format(), ImageFormat::SRGB);
}

}  // namespace
}  // namespace mediapipe::api3
