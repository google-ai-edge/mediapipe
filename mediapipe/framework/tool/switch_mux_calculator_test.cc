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

// Tests showing packet channel synchronization through SwitchMuxCalculator.
class SwitchMuxCalculatorTest : public ::testing::Test {
 protected:
  SwitchMuxCalculatorTest() {}
  ~SwitchMuxCalculatorTest() override {}
  void SetUp() override {}
  void TearDown() override {}

  // Defines a SwitchMuxCalculator CalculatorGraphConfig::Node.
  CalculatorGraphConfig::Node BuildNodeConfig() {
    CalculatorGraphConfig::Node result;
    *result.mutable_calculator() = "SwitchMuxCalculator";
    *result.add_input_stream() = "SELECT:select";
    for (int c = 0; c < 3; ++c) {
      *result.add_input_stream() =
          absl::StrCat(tool::ChannelTag("FRAME", c), ":frame_", c);
      *result.add_input_stream() =
          absl::StrCat(tool::ChannelTag("MASK", c), ":mask_", c);
    }
    *result.add_output_stream() = "FRAME:frame";
    *result.add_output_stream() = "MASK:mask";
    return result;
  }
};

// Shows the SwitchMuxCalculator is available.
TEST_F(SwitchMuxCalculatorTest, IsRegistered) {
  EXPECT_TRUE(CalculatorBaseRegistry::IsRegistered("SwitchMuxCalculator"));
}

// Shows that channels are queued until packets arrive.
TEST_F(SwitchMuxCalculatorTest, ChannelEarly) {
  CalculatorGraphConfig::Node node_config = BuildNodeConfig();
  CalculatorGraph graph = BuildCalculatorGraph(node_config);
  std::vector<Packet> output_frames;
  MP_ASSERT_OK(graph.ObserveOutputStream("frame", [&](const Packet& p) {
    output_frames.push_back(p);
    return absl::OkStatus();
  }));
  MP_ASSERT_OK(graph.StartRun({}));

  // Finalize input for the "mask" input stream.
  MP_ASSERT_OK(graph.CloseInputStream("mask_0"));
  MP_ASSERT_OK(graph.CloseInputStream("mask_1"));
  MP_ASSERT_OK(graph.CloseInputStream("mask_2"));

  // All channels are specified before any frame packets arrive.
  MP_ASSERT_OK(graph.AddPacketToInputStream("select", pack(0, 1)));
  MP_ASSERT_OK(graph.AddPacketToInputStream("select", pack(0, 10)));
  MP_ASSERT_OK(graph.AddPacketToInputStream("select", pack(1, 11)));
  MP_ASSERT_OK(graph.AddPacketToInputStream("select", pack(1, 20)));
  MP_ASSERT_OK(graph.AddPacketToInputStream("select", pack(0, 21)));
  MP_ASSERT_OK(graph.AddPacketToInputStream("select", pack(0, 30)));
  MP_ASSERT_OK(graph.WaitUntilIdle());

  // The packet at timestamp 10 is passed from channel 0.
  MP_ASSERT_OK(graph.AddPacketToInputStream("frame_0", pack("p0_t10", 10)));
  MP_ASSERT_OK(graph.WaitUntilIdle());
  ASSERT_EQ(output_frames.size(), 1);
  EXPECT_EQ(output_frames[0].Get<std::string>(), "p0_t10");

  // The packet at timestamp 20 is passed from channel 1.
  MP_ASSERT_OK(graph.AddPacketToInputStream("frame_1", pack("p1_t20", 20)));
  MP_ASSERT_OK(graph.WaitUntilIdle());
  ASSERT_EQ(output_frames.size(), 2);
  EXPECT_EQ(output_frames[1].Get<std::string>(), "p1_t20");

  // The packet at timestamp 30 is passed from channel 0.
  MP_ASSERT_OK(graph.AddPacketToInputStream("frame_0", pack("p0_t30", 30)));
  MP_ASSERT_OK(graph.WaitUntilIdle());
  ASSERT_EQ(output_frames.size(), 3);
  EXPECT_EQ(output_frames[2].Get<std::string>(), "p0_t30");

  MP_ASSERT_OK(graph.CloseAllPacketSources());
  MP_ASSERT_OK(graph.WaitUntilDone());
}

// Shows that packets are queued until channels are specified.
TEST_F(SwitchMuxCalculatorTest, ChannelsLate) {
  CalculatorGraphConfig::Node node_config = BuildNodeConfig();
  CalculatorGraph graph = BuildCalculatorGraph(node_config);
  std::vector<Packet> output_frames;
  MP_ASSERT_OK(graph.ObserveOutputStream("frame", [&](const Packet& p) {
    output_frames.push_back(p);
    return absl::OkStatus();
  }));
  MP_ASSERT_OK(graph.StartRun({}));

  // Finalize input for the "mask" input stream.
  MP_ASSERT_OK(graph.CloseInputStream("mask_0"));
  MP_ASSERT_OK(graph.CloseInputStream("mask_1"));
  MP_ASSERT_OK(graph.CloseInputStream("mask_2"));

  // All frame packets arrive before any channels are specified.
  // All packets are queued awaiting channel choices.
  MP_ASSERT_OK(graph.AddPacketToInputStream("frame_0", pack("p0_t10", 10)));
  MP_ASSERT_OK(graph.AddPacketToInputStream("frame_1", pack("p1_t20", 20)));
  MP_ASSERT_OK(graph.AddPacketToInputStream("frame_0", pack("p0_t30", 30)));
  MP_ASSERT_OK(graph.WaitUntilIdle());
  ASSERT_EQ(output_frames.size(), 0);

  // The packet at timestamp 10 is released from channel 0.
  MP_ASSERT_OK(graph.AddPacketToInputStream("select", pack(0, 1)));
  MP_ASSERT_OK(graph.AddPacketToInputStream("select", pack(0, 10)));
  MP_ASSERT_OK(graph.WaitUntilIdle());
  ASSERT_EQ(output_frames.size(), 1);
  EXPECT_EQ(output_frames[0].Get<std::string>(), "p0_t10");

  // The packet at timestamp 20 is released from channel 1.
  MP_ASSERT_OK(graph.AddPacketToInputStream("select", pack(1, 11)));
  MP_ASSERT_OK(graph.AddPacketToInputStream("select", pack(1, 20)));
  MP_ASSERT_OK(graph.WaitUntilIdle());
  ASSERT_EQ(output_frames.size(), 2);
  EXPECT_EQ(output_frames[1].Get<std::string>(), "p1_t20");

  // The packet at timestamp 30 is released from channel 0.
  MP_ASSERT_OK(graph.AddPacketToInputStream("select", pack(0, 21)));
  MP_ASSERT_OK(graph.AddPacketToInputStream("select", pack(0, 30)));
  MP_ASSERT_OK(graph.WaitUntilIdle());
  ASSERT_EQ(output_frames.size(), 3);
  EXPECT_EQ(output_frames[2].Get<std::string>(), "p0_t30");

  MP_ASSERT_OK(graph.CloseAllPacketSources());
  MP_ASSERT_OK(graph.WaitUntilDone());
}

// Shows that no packets are queued when channels are specified just in time.
TEST_F(SwitchMuxCalculatorTest, ChannelsOnTime) {
  CalculatorGraphConfig::Node node_config = BuildNodeConfig();
  CalculatorGraph graph = BuildCalculatorGraph(node_config);
  std::vector<Packet> output_frames;
  MP_ASSERT_OK(graph.ObserveOutputStream("frame", [&](const Packet& p) {
    output_frames.push_back(p);
    return absl::OkStatus();
  }));
  MP_ASSERT_OK(graph.StartRun({}));

  // Finalize input for the "mask" input stream.
  MP_ASSERT_OK(graph.CloseInputStream("mask_0"));
  MP_ASSERT_OK(graph.CloseInputStream("mask_1"));
  MP_ASSERT_OK(graph.CloseInputStream("mask_2"));

  // Channel 0 is selected just before corresponding packets arrive.
  MP_ASSERT_OK(graph.AddPacketToInputStream("select", pack(0, 1)));
  MP_ASSERT_OK(graph.AddPacketToInputStream("select", pack(0, 10)));
  MP_ASSERT_OK(graph.AddPacketToInputStream("frame_0", pack("p0_t10", 10)));
  MP_ASSERT_OK(graph.WaitUntilIdle());
  ASSERT_EQ(output_frames.size(), 1);
  EXPECT_EQ(output_frames[0].Get<std::string>(), "p0_t10");

  // Channel 1 is selected just before corresponding packets arrive.
  MP_ASSERT_OK(graph.AddPacketToInputStream("select", pack(1, 11)));
  MP_ASSERT_OK(graph.AddPacketToInputStream("select", pack(1, 20)));
  MP_ASSERT_OK(graph.AddPacketToInputStream("frame_1", pack("p1_t20", 20)));
  MP_ASSERT_OK(graph.WaitUntilIdle());
  ASSERT_EQ(output_frames.size(), 2);
  EXPECT_EQ(output_frames[1].Get<std::string>(), "p1_t20");

  // Channel 0 is selected just before corresponding packets arrive.
  MP_ASSERT_OK(graph.AddPacketToInputStream("select", pack(0, 21)));
  MP_ASSERT_OK(graph.AddPacketToInputStream("select", pack(0, 30)));
  MP_ASSERT_OK(graph.AddPacketToInputStream("frame_0", pack("p0_t30", 30)));
  MP_ASSERT_OK(graph.WaitUntilIdle());
  ASSERT_EQ(output_frames.size(), 3);
  EXPECT_EQ(output_frames[2].Get<std::string>(), "p0_t30");

  MP_ASSERT_OK(graph.CloseAllPacketSources());
  MP_ASSERT_OK(graph.WaitUntilDone());
}

// Shows what happens if the last packet from a channel never arrives.
TEST_F(SwitchMuxCalculatorTest, ChannelNeverCompletes) {
  CalculatorGraphConfig::Node node_config = BuildNodeConfig();
  CalculatorGraph graph = BuildCalculatorGraph(node_config);
  std::vector<Packet> output_frames;
  MP_ASSERT_OK(graph.ObserveOutputStream("frame", [&](const Packet& p) {
    output_frames.push_back(p);
    return absl::OkStatus();
  }));
  MP_ASSERT_OK(graph.StartRun({}));

  // Finalize input for the "mask" input stream.
  MP_ASSERT_OK(graph.CloseInputStream("mask_0"));
  MP_ASSERT_OK(graph.CloseInputStream("mask_1"));
  MP_ASSERT_OK(graph.CloseInputStream("mask_2"));

  // Channel 0 is selected, but it's closing packet never arrives.
  MP_ASSERT_OK(graph.AddPacketToInputStream("select", pack(0, 1)));
  MP_ASSERT_OK(graph.AddPacketToInputStream("select", pack(0, 10)));
  MP_ASSERT_OK(graph.WaitUntilIdle());
  ASSERT_EQ(output_frames.size(), 0);

  // Channel 1 is selected, but we still wait for channel 0 to finish.
  MP_ASSERT_OK(graph.AddPacketToInputStream("select", pack(1, 11)));
  MP_ASSERT_OK(graph.AddPacketToInputStream("select", pack(1, 20)));
  MP_ASSERT_OK(graph.AddPacketToInputStream("frame_1", pack("p1_t20", 20)));
  MP_ASSERT_OK(graph.WaitUntilIdle());
  ASSERT_EQ(output_frames.size(), 0);

  // Finally channel 0 advances, and channel 1 can be delivered.
  // Note that "p0_t15" is discarded because its channel is deselected.
  MP_ASSERT_OK(graph.AddPacketToInputStream("frame_0", pack("p0_t15", 15)));
  MP_ASSERT_OK(graph.WaitUntilIdle());
  ASSERT_EQ(output_frames.size(), 1);
  EXPECT_EQ(output_frames[0].Get<std::string>(), "p1_t20");

  MP_ASSERT_OK(graph.CloseAllPacketSources());
  MP_ASSERT_OK(graph.WaitUntilDone());
}

// Shows how output is sequenced when one channel is delayed substantially.
// Queues up "SELECT" packets for channel 0, 1, and 2.
// Queues up "frame" packets for channel 0 and 2.
// The output packets from channel 1, 2, and 0 wait for channel 1.
TEST_F(SwitchMuxCalculatorTest, OneChannelIsSlow) {
  CalculatorGraphConfig::Node node_config = BuildNodeConfig();
  CalculatorGraph graph = BuildCalculatorGraph(node_config);
  std::vector<Packet> output_frames;
  MP_ASSERT_OK(graph.ObserveOutputStream("frame", [&](const Packet& p) {
    output_frames.push_back(p);
    return absl::OkStatus();
  }));
  MP_ASSERT_OK(graph.StartRun({}));

  // Finalize input for the "mask" input stream.
  MP_ASSERT_OK(graph.CloseInputStream("mask_0"));
  MP_ASSERT_OK(graph.CloseInputStream("mask_1"));
  MP_ASSERT_OK(graph.CloseInputStream("mask_2"));

  // Channel 0 is selected, and some packets arrive.
  MP_ASSERT_OK(graph.AddPacketToInputStream("select", pack(0, 1)));
  MP_ASSERT_OK(graph.AddPacketToInputStream("select", pack(0, 10)));
  MP_ASSERT_OK(graph.AddPacketToInputStream("frame_0", pack("p0_t20", 10)));
  MP_ASSERT_OK(graph.WaitUntilIdle());
  ASSERT_EQ(output_frames.size(), 1);

  // Channel 1 is selected, but its packets are delayed.
  MP_ASSERT_OK(graph.AddPacketToInputStream("select", pack(1, 11)));
  MP_ASSERT_OK(graph.AddPacketToInputStream("select", pack(1, 20)));
  MP_ASSERT_OK(graph.AddPacketToInputStream("frame_0", pack("p0_t20", 20)));
  MP_ASSERT_OK(graph.AddPacketToInputStream("frame_2", pack("p2_t20", 20)));
  MP_ASSERT_OK(graph.WaitUntilIdle());
  ASSERT_EQ(output_frames.size(), 1);

  // Channel 2 is selected, packets arrive, but wait for channel 1.
  MP_ASSERT_OK(graph.AddPacketToInputStream("select", pack(2, 21)));
  MP_ASSERT_OK(graph.AddPacketToInputStream("select", pack(2, 30)));
  MP_ASSERT_OK(graph.AddPacketToInputStream("frame_2", pack("p2_t30", 30)));
  MP_ASSERT_OK(graph.WaitUntilIdle());
  ASSERT_EQ(output_frames.size(), 1);

  // Channel 0 is selected again, packets arrive, but wait for channel 1.
  MP_ASSERT_OK(graph.AddPacketToInputStream("select", pack(0, 31)));
  MP_ASSERT_OK(graph.AddPacketToInputStream("select", pack(0, 40)));
  MP_ASSERT_OK(graph.AddPacketToInputStream("frame_0", pack("p0_t40", 40)));
  MP_ASSERT_OK(graph.WaitUntilIdle());
  ASSERT_EQ(output_frames.size(), 1);

  // Channel 1 is selected again, but its packets are still delayed.
  MP_ASSERT_OK(graph.AddPacketToInputStream("select", pack(1, 41)));
  MP_ASSERT_OK(graph.AddPacketToInputStream("select", pack(1, 50)));
  MP_ASSERT_OK(graph.WaitUntilIdle());
  ASSERT_EQ(output_frames.size(), 1);

  // Finally, the delayed packets from channel 1 arrive.
  // And all packets for all five "SELECT"" inetervals are delivered.
  MP_ASSERT_OK(graph.AddPacketToInputStream("frame_1", pack("p1_t20", 20)));
  MP_ASSERT_OK(graph.AddPacketToInputStream("frame_1", pack("p1_t50", 50)));
  MP_ASSERT_OK(graph.WaitUntilIdle());
  ASSERT_EQ(output_frames.size(), 5);

  MP_ASSERT_OK(graph.CloseAllPacketSources());
  MP_ASSERT_OK(graph.WaitUntilDone());
}

}  // namespace
}  // namespace mediapipe
