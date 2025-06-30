// Copyright 2025 The MediaPipe Authors.
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
#include <utility>
#include <vector>

#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/deps/file_path.h"
#include "mediapipe/framework/formats/image.h"
#include "mediapipe/framework/formats/image_frame.h"
#include "mediapipe/framework/port/gmock.h"
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/framework/port/parse_text_proto.h"
#include "mediapipe/framework/port/status_matchers.h"
#include "mediapipe/framework/tool/test_util.h"

namespace mediapipe {

TEST(ImagePropertiesCalculatorTest, GetImageProperties) {
  auto image_frame =
      std::make_shared<ImageFrame>(ImageFormat::SRGBA, 128, 256, 4);
  int width = image_frame->Width();
  int height = image_frame->Height();
  Image image = Image(std::move(image_frame));
  auto image_packet = MakePacket<Image>(std::move(image));

  auto graph_config = ParseTextProtoOrDie<CalculatorGraphConfig>(R"pb(
    input_stream: "input_image"
    output_stream: "image_properties"
    node {
      calculator: "ImagePropertiesCalculator"
      input_stream: "IMAGE:input_image"
      output_stream: "SIZE:image_properties"
    }
  )pb");

  std::vector<Packet> output_packets;
  tool::AddVectorSink("image_properties", &graph_config, &output_packets);

  CalculatorGraph graph;
  MP_ASSERT_OK(graph.Initialize(graph_config));
  MP_ASSERT_OK(graph.StartRun({}));

  MP_ASSERT_OK(graph.AddPacketToInputStream("input_image",
                                            image_packet.At(Timestamp(0))));

  MP_ASSERT_OK(graph.WaitUntilIdle());
  ASSERT_EQ(output_packets.size(), 1);

  const auto& properties = output_packets[0].Get<std::pair<int, int>>();
  EXPECT_THAT(properties, testing::Pair(width, height));

  MP_ASSERT_OK(graph.CloseAllInputStreams());
  MP_ASSERT_OK(graph.WaitUntilDone());
}

TEST(ImagePropertiesCalculatorTest, GetImageFramePropertiesCPU) {
  std::string input_image_path =
      file::JoinPath(GetTestRootDir(),
                     "/mediapipe/calculators/"
                     "image/testdata/binary_mask.png");
  MP_ASSERT_OK_AND_ASSIGN(std::shared_ptr<ImageFrame> input_frame,
                          LoadTestImage(input_image_path, ImageFormat::SRGBA));
  int width = input_frame->Width();
  int height = input_frame->Height();
  auto image_packet = MakePacket<ImageFrame>(std::move(*input_frame));

  auto graph_config = ParseTextProtoOrDie<CalculatorGraphConfig>(R"pb(
    input_stream: "input_image_cpu"
    output_stream: "image_properties"
    node {
      calculator: "ImagePropertiesCalculator"
      input_stream: "IMAGE_CPU:input_image_cpu"
      output_stream: "SIZE:image_properties"
    }
  )pb");

  std::vector<Packet> output_packets;
  tool::AddVectorSink("image_properties", &graph_config, &output_packets);

  CalculatorGraph graph;
  MP_ASSERT_OK(graph.Initialize(graph_config));
  MP_ASSERT_OK(graph.StartRun({}));

  MP_ASSERT_OK(graph.AddPacketToInputStream("input_image_cpu",
                                            image_packet.At(Timestamp(0))));

  MP_ASSERT_OK(graph.WaitUntilIdle());
  ASSERT_EQ(output_packets.size(), 1);

  const auto& properties = output_packets[0].Get<std::pair<int, int>>();
  EXPECT_THAT(properties, testing::Pair(width, height));

  MP_ASSERT_OK(graph.CloseAllInputStreams());
  MP_ASSERT_OK(graph.WaitUntilDone());
}

TEST(ImagePropertiesCalculatorTest, GetImageFramePropertiesGPU) {
  std::string input_image_path =
      file::JoinPath(GetTestRootDir(),
                     "/mediapipe/calculators/"
                     "image/testdata/binary_mask.png");
  MP_ASSERT_OK_AND_ASSIGN(std::shared_ptr<ImageFrame> input_frame,
                          LoadTestImage(input_image_path, ImageFormat::SRGBA));
  int width = input_frame->Width();
  int height = input_frame->Height();
  auto image_packet = MakePacket<ImageFrame>(std::move(*input_frame));

  auto graph_config = ParseTextProtoOrDie<CalculatorGraphConfig>(R"pb(
    input_stream: "input_image_cpu"
    output_stream: "image_properties"

    node {
      calculator: "ImageFrameToGpuBufferCalculator"
      input_stream: "input_image_cpu"
      output_stream: "input_image_gpu"
    }

    node {
      calculator: "ImagePropertiesCalculator"
      input_stream: "IMAGE_GPU:input_image_gpu"
      output_stream: "SIZE:image_properties"
    }
  )pb");

  std::vector<Packet> output_packets;
  tool::AddVectorSink("image_properties", &graph_config, &output_packets);

  CalculatorGraph graph;
  MP_ASSERT_OK(graph.Initialize(graph_config));
  MP_ASSERT_OK(graph.StartRun({}));

  MP_ASSERT_OK(graph.AddPacketToInputStream("input_image_cpu",
                                            image_packet.At(Timestamp(0))));

  MP_ASSERT_OK(graph.WaitUntilIdle());
  ASSERT_EQ(output_packets.size(), 1);

  const auto& properties = output_packets[0].Get<std::pair<int, int>>();
  EXPECT_THAT(properties, testing::Pair(width, height));

  MP_ASSERT_OK(graph.CloseAllInputStreams());
  MP_ASSERT_OK(graph.WaitUntilDone());
}

}  // namespace mediapipe
