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

#include "mediapipe/calculators/util/from_image_calculator.h"

#include <memory>
#include <tuple>
#include <utility>

#include "absl/status/status.h"
#include "mediapipe/framework/api3/function_runner.h"
#include "mediapipe/framework/api3/graph.h"
#include "mediapipe/framework/api3/packet.h"
#include "mediapipe/framework/api3/stream.h"
#include "mediapipe/framework/formats/image.h"
#include "mediapipe/framework/formats/image_format.pb.h"
#include "mediapipe/framework/formats/image_frame.h"
#include "mediapipe/framework/port/gmock.h"
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/framework/port/status_matchers.h"

namespace mediapipe::api3 {
namespace {

using ::testing::HasSubstr;

TEST(FromImageCalculatorTest, ConvertsImageToImageFrame) {
  MP_ASSERT_OK_AND_ASSIGN(
      auto runner, Runner::For([](GenericGraph& graph, Stream<Image> image) {
                     auto& node = graph.AddNode<FromImageNode>();
                     node.in_image.Set(image);
                     return node.out_image_cpu.Get();
                   }).Create());

  auto image_frame = std::make_shared<ImageFrame>(ImageFormat::SRGB, 10, 10);
  Image image(image_frame);

  MP_ASSERT_OK_AND_ASSIGN(Packet<ImageFrame> output_packet,
                          runner.Run(api3::MakePacket<Image>(image)));

  const auto& output_frame = output_packet.GetOrDie();
  EXPECT_EQ(output_frame.Width(), 10);
  EXPECT_EQ(output_frame.Height(), 10);
  EXPECT_EQ(output_frame.Format(), ImageFormat::SRGB);
}

TEST(FromImageCalculatorTest, OutputsSourceOnGpu) {
  MP_ASSERT_OK_AND_ASSIGN(
      auto runner, Runner::For([](GenericGraph& graph, Stream<Image> image) {
                     auto& node = graph.AddNode<FromImageNode>();
                     node.in_image.Set(image);
                     return std::make_tuple(node.out_image_cpu.Get(),
                                            node.out_source_on_gpu.Get());
                   }).Create());

  Image image(std::make_shared<ImageFrame>(ImageFormat::SRGB, 10, 10));

  MP_ASSERT_OK_AND_ASSIGN((auto [image_cpu_packet, source_on_gpu_packet]),
                          runner.Run(api3::MakePacket<Image>(image)));

  EXPECT_FALSE(source_on_gpu_packet.GetOrDie());
}

TEST(FromImageCalculatorTest, MultipleOutputsError) {
  auto runner_or = Runner::For([](GenericGraph& graph, Stream<Image> image) {
                     auto& node = graph.AddNode<FromImageNode>();
                     node.in_image.Set(image);
                     return std::make_tuple(node.out_image_cpu.Get(),
                                            node.out_image_gpu.Get());
                   }).Create();
  EXPECT_THAT(runner_or, StatusIs(absl::StatusCode::kInternal,
                                  HasSubstr("multiple outputs")));
}

}  // namespace
}  // namespace mediapipe::api3
