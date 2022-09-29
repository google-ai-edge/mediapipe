// Copyright 2022 The MediaPipe Authors.
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

#include <string>
#include <vector>

#include "absl/strings/str_cat.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/port/gmock.h"
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/framework/port/logging.h"
#include "mediapipe/framework/tool/container_util.h"

namespace mediapipe {
namespace {

// Returns a CalculatorGraph to run a single calculator.
CalculatorGraph BuildCalculatorGraph(CalculatorGraphConfig::Node node_config) {
  CalculatorGraphConfig config;
  *config.add_node() = node_config;
  *config.mutable_input_stream() = node_config.input_stream();
  *config.mutable_output_stream() = node_config.output_stream();
  *config.mutable_input_side_packet() = node_config.input_side_packet();
  *config.mutable_output_side_packet() = node_config.output_side_packet();
  return CalculatorGraph(config);
}

// Creates a string packet.
Packet pack(std::string data, int timestamp) {
  return MakePacket<std::string>(data).At(Timestamp(timestamp));
}

// Creates an int packet.
Packet pack(int data, int timestamp) {
  return MakePacket<int>(data).At(Timestamp(timestamp));
}

// Tests showing packet channel synchronization through SwitchDemuxCalculator.
class SwitchDemuxCalculatorTest : public ::testing::Test {
 protected:
  SwitchDemuxCalculatorTest() {}
  ~SwitchDemuxCalculatorTest() override {}
  void SetUp() override {}
  void TearDown() override {}

  // Defines a SwitchDemuxCalculator CalculatorGraphConfig::Node.
  CalculatorGraphConfig::Node BuildNodeConfig() {
    CalculatorGraphConfig::Node result;
    *result.mutable_calculator() = "SwitchDemuxCalculator";
    *result.add_input_stream() = "SELECT:select";
    for (int c = 0; c < 2; ++c) {
      *result.add_output_stream() =
          absl::StrCat(tool::ChannelTag("FRAME", c), ":frame_", c);
      *result.add_output_stream() =
          absl::StrCat(tool::ChannelTag("MASK", c), ":mask_", c);
    }
    *result.add_input_stream() = "FRAME:frame";
    *result.add_input_stream() = "MASK:mask";
    return result;
  }
};

// Shows the SwitchMuxCalculator is available.
TEST_F(SwitchDemuxCalculatorTest, IsRegistered) {
  EXPECT_TRUE(CalculatorBaseRegistry::IsRegistered("SwitchDemuxCalculator"));
}

TEST_F(SwitchDemuxCalculatorTest, BasicDataFlow) {
  CalculatorGraphConfig::Node node_config = BuildNodeConfig();
  CalculatorGraph graph = BuildCalculatorGraph(node_config);
  std::vector<Packet> output_frames0;
  EXPECT_TRUE(graph
                  .ObserveOutputStream("frame_0",
                                       [&](const Packet& p) {
                                         output_frames0.push_back(p);
                                         return absl::OkStatus();
                                       })
                  .ok());
  std::vector<Packet> output_frames1;
  EXPECT_TRUE(graph
                  .ObserveOutputStream("frame_1",
                                       [&](const Packet& p) {
                                         output_frames1.push_back(p);
                                         return absl::OkStatus();
                                       })
                  .ok());
  EXPECT_TRUE(
      graph.StartRun({}, {{"frame", MakePacket<std::string>("frame_header")}})
          .ok());

  // Finalize input for the "mask" input stream.
  EXPECT_TRUE(graph.CloseInputStream("mask").ok());

  // Channel 0 is selected just before corresponding packets arrive.
  EXPECT_TRUE(graph.AddPacketToInputStream("select", pack(0, 1)).ok());
  EXPECT_TRUE(graph.AddPacketToInputStream("select", pack(0, 10)).ok());
  EXPECT_TRUE(graph.AddPacketToInputStream("frame", pack("p0_t10", 10)).ok());
  EXPECT_TRUE(graph.WaitUntilIdle().ok());
  EXPECT_EQ(output_frames0.size(), 1);
  EXPECT_EQ(output_frames1.size(), 0);
  EXPECT_EQ(output_frames0[0].Get<std::string>(), "p0_t10");

  // Channel 1 is selected just before corresponding packets arrive.
  EXPECT_TRUE(graph.AddPacketToInputStream("select", pack(1, 11)).ok());
  EXPECT_TRUE(graph.AddPacketToInputStream("select", pack(1, 20)).ok());
  EXPECT_TRUE(graph.AddPacketToInputStream("frame", pack("p1_t20", 20)).ok());
  EXPECT_TRUE(graph.WaitUntilIdle().ok());
  EXPECT_EQ(output_frames0.size(), 1);
  EXPECT_EQ(output_frames1.size(), 1);
  EXPECT_EQ(output_frames1[0].Get<std::string>(), "p1_t20");

  EXPECT_EQ(
      graph.FindOutputStreamManager("frame_0")->Header().Get<std::string>(),
      "frame_header");
  EXPECT_EQ(
      graph.FindOutputStreamManager("frame_1")->Header().Get<std::string>(),
      "frame_header");

  EXPECT_TRUE(graph.CloseAllPacketSources().ok());
  EXPECT_TRUE(graph.WaitUntilDone().ok());
}

}  // namespace
}  // namespace mediapipe
