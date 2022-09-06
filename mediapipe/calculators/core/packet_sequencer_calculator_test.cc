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
#include "mediapipe/framework/calculator.pb.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/port/gmock.h"
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/framework/port/logging.h"
#include "mediapipe/framework/port/parse_text_proto.h"
#include "mediapipe/framework/port/ret_check.h"
#include "mediapipe/framework/port/status_matchers.h"

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

// Tests showing packet timestamp synchronization through
// PacketSequencerCalculator.
class PacketSequencerCalculatorTest : public ::testing::Test {
 protected:
  PacketSequencerCalculatorTest() {}
  ~PacketSequencerCalculatorTest() override {}
  void SetUp() override {}
  void TearDown() override {}

  // Defines a PacketSequencerCalculator CalculatorGraphConfig::Node.
  CalculatorGraphConfig::Node BuildNodeConfig() {
    CalculatorGraphConfig::Node result;
    *result.mutable_calculator() = "PacketSequencerCalculator";
    *result.add_input_stream() = "INPUT:select";
    *result.add_input_stream() = "TICK:0:frame";
    *result.add_input_stream() = "TICK:1:mask";
    *result.add_output_stream() = "OUTPUT:select_timed";
    return result;
  }
};

// Shows the PacketSequencerCalculator is available.
TEST_F(PacketSequencerCalculatorTest, IsRegistered) {
  EXPECT_TRUE(
      CalculatorBaseRegistry::IsRegistered("PacketSequencerCalculator"));
}

// Shows how control packets recieve timestamps before and after frame packets
// have arrived.
TEST_F(PacketSequencerCalculatorTest, ChannelEarly) {
  CalculatorGraphConfig::Node node_config = BuildNodeConfig();
  CalculatorGraph graph = BuildCalculatorGraph(node_config);
  std::vector<Packet> outputs;
  MP_ASSERT_OK(graph.ObserveOutputStream("select_timed", [&](const Packet& p) {
    outputs.push_back(p);
    return absl::OkStatus();
  }));
  MP_ASSERT_OK(graph.StartRun({}));

  // Some control packets arrive.
  MP_ASSERT_OK(graph.AddPacketToInputStream("select", pack("p0_t10", 10)));
  MP_ASSERT_OK(graph.AddPacketToInputStream("select", pack("p0_t20", 20)));
  MP_ASSERT_OK(graph.WaitUntilIdle());

  // The control packets are assigned low timestamps.
  ASSERT_EQ(outputs.size(), 2);
  EXPECT_EQ(outputs[0].Get<std::string>(), "p0_t10");
  EXPECT_EQ(outputs[0].Timestamp(), Timestamp::Min());
  EXPECT_EQ(outputs[1].Timestamp(), Timestamp::Min() + 1);

  // Some frame packets arrive.
  MP_ASSERT_OK(graph.AddPacketToInputStream("mask", pack("p2_t10", 10)));
  MP_ASSERT_OK(graph.AddPacketToInputStream("frame", pack("p1_t20", 20)));
  MP_ASSERT_OK(graph.WaitUntilIdle());

  // Some more control packets arrive.
  MP_ASSERT_OK(graph.AddPacketToInputStream("select", pack("p0_t30", 30)));
  MP_ASSERT_OK(graph.AddPacketToInputStream("select", pack("p0_t40", 40)));
  MP_ASSERT_OK(graph.WaitUntilIdle());

  // New control packets are assigned timestamps following Timestamp(20).
  ASSERT_EQ(outputs.size(), 4);
  EXPECT_EQ(outputs[2].Get<std::string>(), "p0_t30");
  EXPECT_EQ(outputs[2].Timestamp(), Timestamp(21));
  EXPECT_EQ(outputs[3].Timestamp(), Timestamp(22));

  MP_ASSERT_OK(graph.CloseAllPacketSources());
  MP_ASSERT_OK(graph.WaitUntilDone());
}

}  // namespace
}  // namespace mediapipe
