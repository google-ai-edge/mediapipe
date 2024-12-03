// Copyright 2020 The MediaPipe Authors.
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
#include <string>
#include <vector>

#include "absl/memory/memory.h"
#include "absl/strings/match.h"
#include "absl/strings/str_replace.h"
#include "absl/strings/string_view.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/port/gmock.h"
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/framework/port/parse_text_proto.h"
#include "mediapipe/framework/port/status.h"
#include "mediapipe/framework/port/status_matchers.h"
#include "mediapipe/framework/tool/options_util.h"
#include "mediapipe/util/packet_test_util.h"

namespace mediapipe {
namespace {

using ::testing::ElementsAre;
using ::testing::Eq;
using ::testing::HasSubstr;
using ::testing::IsEmpty;

TEST(SidePacketToStreamCalculator, WrongConfigWithMissingTick) {
  CalculatorGraphConfig graph_config =
      ParseTextProtoOrDie<CalculatorGraphConfig>(
          R"pb(
            input_stream: "tick"
            input_side_packet: "side_packet"
            output_stream: "packet"
            node {
              calculator: "SidePacketToStreamCalculator"
              input_side_packet: "side_packet"
              output_stream: "AT_TICK:packet"
            }
          )pb");
  CalculatorGraph graph;
  auto status = graph.Initialize(graph_config);
  EXPECT_FALSE(status.ok());
  EXPECT_THAT(
      status.message(),
      HasSubstr(
          "Either both TICK input and tick (AT_TICK/AT_FIRST_TICK) output "
          "should be used or none of them."));
}

TEST(SidePacketToStreamCalculator,
     WrongConfigWithMissingTickForFirstTickProcessing) {
  CalculatorGraphConfig graph_config =
      ParseTextProtoOrDie<CalculatorGraphConfig>(
          R"pb(
            input_stream: "tick"
            input_side_packet: "side_packet"
            output_stream: "packet"
            node {
              calculator: "SidePacketToStreamCalculator"
              input_side_packet: "side_packet"
              output_stream: "AT_FIRST_TICK:packet"
            }
          )pb");
  CalculatorGraph graph;
  auto status = graph.Initialize(graph_config);
  EXPECT_FALSE(status.ok());
  EXPECT_THAT(
      status.message(),
      HasSubstr(
          "Either both TICK input and tick (AT_TICK/AT_FIRST_TICK) output "
          "should be used or none of them."));
}

TEST(SidePacketToStreamCalculator, WrongConfigWithMissingTimestampSideInput) {
  CalculatorGraphConfig graph_config =
      ParseTextProtoOrDie<CalculatorGraphConfig>(
          R"pb(
            input_stream: "timestamp"
            input_side_packet: "side_packet"
            output_stream: "packet"
            node {
              calculator: "SidePacketToStreamCalculator"
              input_side_packet: "side_packet"
              output_stream: "AT_TIMESTAMP:packet"
            }
          )pb");
  CalculatorGraph graph;
  auto status = graph.Initialize(graph_config);
  EXPECT_FALSE(status.ok());
  EXPECT_THAT(status.message(),
              HasSubstr("Either both TIMESTAMP and AT_TIMESTAMP should be used "
                        "or none of them."));
}

TEST(SidePacketToStreamCalculator, WrongConfigWithNonExistentTag) {
  CalculatorGraphConfig graph_config =
      ParseTextProtoOrDie<CalculatorGraphConfig>(
          R"pb(
            input_stream: "tick"
            input_side_packet: "side_packet"
            output_stream: "packet"
            node {
              calculator: "SidePacketToStreamCalculator"
              input_side_packet: "side_packet"
              output_stream: "DOES_NOT_EXIST:packet"
            }
          )pb");
  CalculatorGraph graph;
  auto status = graph.Initialize(graph_config);
  EXPECT_FALSE(status.ok());
  EXPECT_THAT(status.message(),
              HasSubstr("Only one of AT_PRESTREAM, AT_POSTSTREAM, AT_ZERO, "
                        "AT_TICK, AT_FIRST_TICK and AT_TIMESTAMP tags is "
                        "allowed and required to specify output stream(s)."));
}

TEST(SidePacketToStreamCalculator, WrongConfigWithMixedTags) {
  CalculatorGraphConfig graph_config =
      ParseTextProtoOrDie<CalculatorGraphConfig>(
          R"pb(
            input_stream: "tick"
            input_side_packet: "side_packet0"
            input_side_packet: "side_packet1"
            node {
              calculator: "SidePacketToStreamCalculator"
              input_side_packet: "side_packet0"
              input_side_packet: "side_packet1"
              output_stream: "AT_TICK:packet0"
              output_stream: "AT_PRE_STREAM:packet1"
            }
          )pb");
  CalculatorGraph graph;
  auto status = graph.Initialize(graph_config);
  EXPECT_FALSE(status.ok());
  EXPECT_THAT(status.message(),
              HasSubstr("Only one of AT_PRESTREAM, AT_POSTSTREAM, AT_ZERO, "
                        "AT_TICK, AT_FIRST_TICK and AT_TIMESTAMP tags is "
                        "allowed and required to specify output stream(s)."));
}

TEST(SidePacketToStreamCalculator, WrongConfigWithNotEnoughSidePackets) {
  CalculatorGraphConfig graph_config =
      ParseTextProtoOrDie<CalculatorGraphConfig>(
          R"pb(
            input_side_packet: "side_packet0"
            input_side_packet: "side_packet1"
            node {
              calculator: "SidePacketToStreamCalculator"
              input_side_packet: "side_packet0"
              output_stream: "AT_PRESTREAM:0:packet0"
              output_stream: "AT_PRESTREAM:1:packet1"
            }
          )pb");
  CalculatorGraph graph;
  auto status = graph.Initialize(graph_config);
  EXPECT_FALSE(status.ok());
  EXPECT_THAT(
      status.message(),
      HasSubstr(
          "Same number of input side packets and output streams is required."));
}

TEST(SidePacketToStreamCalculator, WrongConfigWithNotEnoughOutputStreams) {
  CalculatorGraphConfig graph_config =
      ParseTextProtoOrDie<CalculatorGraphConfig>(
          R"pb(
            input_side_packet: "side_packet0"
            input_side_packet: "side_packet1"
            node {
              calculator: "SidePacketToStreamCalculator"
              input_side_packet: "side_packet0"
              input_side_packet: "side_packet1"
              output_stream: "AT_PRESTREAM:packet0"
            }
          )pb");
  CalculatorGraph graph;
  auto status = graph.Initialize(graph_config);
  EXPECT_FALSE(status.ok());
  EXPECT_THAT(
      status.message(),
      HasSubstr(
          "Same number of input side packets and output streams is required."));
}

void DoTestNonAtTickOutputTag(absl::string_view tag,
                              Timestamp expected_timestamp) {
  CalculatorGraphConfig graph_config =
      ParseTextProtoOrDie<CalculatorGraphConfig>(absl::StrReplaceAll(
          R"(
            input_side_packet: "side_packet"
            output_stream: "packet"
            node {
              calculator: "SidePacketToStreamCalculator"
              input_side_packet: "side_packet"
              output_stream: "$tag:packet"
            }
          )",
          {{"$tag", tag}}));
  CalculatorGraph graph;
  MP_ASSERT_OK(graph.Initialize(graph_config));
  const int expected_value = 10;
  std::vector<Packet> output_packets;
  MP_ASSERT_OK(graph.ObserveOutputStream(
      "packet", [&output_packets](const Packet& packet) {
        output_packets.push_back(packet);
        return absl::OkStatus();
      }));
  MP_ASSERT_OK(
      graph.StartRun({{"side_packet", MakePacket<int>(expected_value)}}));
  MP_ASSERT_OK(graph.WaitForObservedOutput());

  ASSERT_FALSE(output_packets.empty());
  EXPECT_EQ(expected_timestamp, output_packets.back().Timestamp());
  EXPECT_EQ(expected_value, output_packets.back().Get<int>());
}

TEST(SidePacketToStreamCalculator, NoAtTickOutputTags) {
  DoTestNonAtTickOutputTag("AT_PRESTREAM", Timestamp::PreStream());
  DoTestNonAtTickOutputTag("AT_POSTSTREAM", Timestamp::PostStream());
  DoTestNonAtTickOutputTag("AT_ZERO", Timestamp(0));
}

TEST(SidePacketToStreamCalculator, AtTick) {
  CalculatorGraphConfig graph_config =
      ParseTextProtoOrDie<CalculatorGraphConfig>(
          R"pb(
            input_stream: "tick"
            input_side_packet: "side_packet"
            output_stream: "packet"
            node {
              calculator: "SidePacketToStreamCalculator"
              input_stream: "TICK:tick"
              input_side_packet: "side_packet"
              output_stream: "AT_TICK:packet"
            }
          )pb");
  std::vector<Packet> output_packets;
  tool::AddVectorSink("packet", &graph_config, &output_packets);
  CalculatorGraph graph;

  MP_ASSERT_OK(graph.Initialize(graph_config));
  const int expected_value = 20;
  MP_ASSERT_OK(
      graph.StartRun({{"side_packet", MakePacket<int>(expected_value)}}));

  auto tick_and_verify = [&graph, &output_packets,
                          expected_value](int at_timestamp) {
    MP_ASSERT_OK(graph.AddPacketToInputStream(
        "tick",
        MakePacket<int>(/*doesn't matter*/ 1).At(Timestamp(at_timestamp))));
    MP_ASSERT_OK(graph.WaitUntilIdle());

    ASSERT_FALSE(output_packets.empty());
    EXPECT_EQ(Timestamp(at_timestamp), output_packets.back().Timestamp());
    EXPECT_EQ(expected_value, output_packets.back().Get<int>());
  };

  tick_and_verify(/*at_timestamp=*/0);
  tick_and_verify(/*at_timestamp=*/1);
  tick_and_verify(/*at_timestamp=*/128);
  tick_and_verify(/*at_timestamp=*/1024);
  tick_and_verify(/*at_timestamp=*/1025);
}

TEST(SidePacketToStreamCalculator, AtFirstTick) {
  CalculatorGraphConfig graph_config =
      ParseTextProtoOrDie<CalculatorGraphConfig>(
          R"pb(
            input_stream: "tick"
            input_side_packet: "side_packet"
            output_stream: "packet"
            node {
              calculator: "SidePacketToStreamCalculator"
              input_stream: "TICK:tick"
              input_side_packet: "side_packet"
              output_stream: "AT_FIRST_TICK:packet"
            }
          )pb");
  std::vector<Packet> output_packets;
  tool::AddVectorSink("packet", &graph_config, &output_packets);
  CalculatorGraph graph;

  MP_ASSERT_OK(graph.Initialize(graph_config));
  const int expected_value = 20;
  const Timestamp kTestTimestamp(1234);
  MP_ASSERT_OK(
      graph.StartRun({{"side_packet", MakePacket<int>(expected_value)}}));

  auto insert_tick = [&graph](Timestamp at_timestamp) {
    MP_ASSERT_OK(graph.AddPacketToInputStream(
        "tick", MakePacket<int>(/*doesn't matter*/ 1).At(at_timestamp)));
    MP_ASSERT_OK(graph.WaitUntilIdle());
  };

  insert_tick(kTestTimestamp);

  EXPECT_THAT(output_packets,
              ElementsAre(PacketContainsTimestampAndPayload<int>(
                  Eq(kTestTimestamp), Eq(expected_value))));

  output_packets.clear();

  // Should not result in an additional output.
  insert_tick(kTestTimestamp + 1);
  EXPECT_THAT(output_packets, IsEmpty());
}

TEST(SidePacketToStreamCalculator, AtTickWithMultipleSidePackets) {
  CalculatorGraphConfig graph_config =
      ParseTextProtoOrDie<CalculatorGraphConfig>(
          R"pb(
            input_stream: "tick"
            input_side_packet: "side_packet0"
            input_side_packet: "side_packet1"
            output_stream: "packet0"
            output_stream: "packet1"
            node {
              calculator: "SidePacketToStreamCalculator"
              input_stream: "TICK:tick"
              input_side_packet: "side_packet0"
              input_side_packet: "side_packet1"
              output_stream: "AT_TICK:0:packet0"
              output_stream: "AT_TICK:1:packet1"
            }
          )pb");
  std::vector<Packet> output_packets0;
  tool::AddVectorSink("packet0", &graph_config, &output_packets0);
  std::vector<Packet> output_packets1;
  tool::AddVectorSink("packet1", &graph_config, &output_packets1);
  CalculatorGraph graph;

  MP_ASSERT_OK(graph.Initialize(graph_config));
  const int expected_value0 = 20;
  const int expected_value1 = 128;
  MP_ASSERT_OK(
      graph.StartRun({{"side_packet0", MakePacket<int>(expected_value0)},
                      {"side_packet1", MakePacket<int>(expected_value1)}}));

  auto tick_and_verify = [&graph, &output_packets0, &output_packets1,
                          expected_value0, expected_value1](int at_timestamp) {
    MP_ASSERT_OK(graph.AddPacketToInputStream(
        "tick",
        MakePacket<int>(/*doesn't matter*/ 1).At(Timestamp(at_timestamp))));
    MP_ASSERT_OK(graph.WaitUntilIdle());

    ASSERT_FALSE(output_packets0.empty());
    ASSERT_FALSE(output_packets1.empty());

    EXPECT_EQ(Timestamp(at_timestamp), output_packets0.back().Timestamp());
    EXPECT_EQ(expected_value0, output_packets0.back().Get<int>());
    EXPECT_EQ(Timestamp(at_timestamp), output_packets1.back().Timestamp());
    EXPECT_EQ(expected_value1, output_packets1.back().Get<int>());
  };

  tick_and_verify(/*at_timestamp=*/0);
  tick_and_verify(/*at_timestamp=*/1);
  tick_and_verify(/*at_timestamp=*/128);
  tick_and_verify(/*at_timestamp=*/1024);
  tick_and_verify(/*at_timestamp=*/1025);
}

TEST(SidePacketToStreamCalculator, AtFirstTickWithMultipleSidePackets) {
  CalculatorGraphConfig graph_config =
      ParseTextProtoOrDie<CalculatorGraphConfig>(
          R"pb(
            input_stream: "tick"
            input_side_packet: "side_packet0"
            input_side_packet: "side_packet1"
            output_stream: "packet0"
            output_stream: "packet1"
            node {
              calculator: "SidePacketToStreamCalculator"
              input_stream: "TICK:tick"
              input_side_packet: "side_packet0"
              input_side_packet: "side_packet1"
              output_stream: "AT_FIRST_TICK:0:packet0"
              output_stream: "AT_FIRST_TICK:1:packet1"
            }
          )pb");
  std::vector<Packet> output_packets0;
  tool::AddVectorSink("packet0", &graph_config, &output_packets0);
  std::vector<Packet> output_packets1;
  tool::AddVectorSink("packet1", &graph_config, &output_packets1);
  CalculatorGraph graph;

  MP_ASSERT_OK(graph.Initialize(graph_config));
  const int expected_value0 = 20;
  const int expected_value1 = 128;
  const Timestamp kTestTimestamp(1234);
  MP_ASSERT_OK(
      graph.StartRun({{"side_packet0", MakePacket<int>(expected_value0)},
                      {"side_packet1", MakePacket<int>(expected_value1)}}));

  auto insert_tick = [&graph](Timestamp at_timestamp) {
    MP_ASSERT_OK(graph.AddPacketToInputStream(
        "tick", MakePacket<int>(/*doesn't matter*/ 1).At(at_timestamp)));
    MP_ASSERT_OK(graph.WaitUntilIdle());
  };

  insert_tick(kTestTimestamp);

  EXPECT_THAT(output_packets0,
              ElementsAre(PacketContainsTimestampAndPayload<int>(
                  Eq(kTestTimestamp), Eq(expected_value0))));
  EXPECT_THAT(output_packets1,
              ElementsAre(PacketContainsTimestampAndPayload<int>(
                  Eq(kTestTimestamp), Eq(expected_value1))));

  output_packets0.clear();
  output_packets1.clear();

  // Should not result in an additional output.
  insert_tick(kTestTimestamp + 1);
  EXPECT_THAT(output_packets0, IsEmpty());
  EXPECT_THAT(output_packets1, IsEmpty());
}

TEST(SidePacketToStreamCalculator, AtTimestamp) {
  CalculatorGraphConfig graph_config =
      ParseTextProtoOrDie<CalculatorGraphConfig>(
          R"pb(
            input_side_packet: "timestamp"
            input_side_packet: "side_packet"
            output_stream: "packet"
            node {
              calculator: "SidePacketToStreamCalculator"
              input_side_packet: "TIMESTAMP:timestamp"
              input_side_packet: "side_packet"
              output_stream: "AT_TIMESTAMP:packet"
            }
          )pb");
  std::vector<Packet> output_packets;
  tool::AddVectorSink("packet", &graph_config, &output_packets);
  CalculatorGraph graph;

  MP_ASSERT_OK(graph.Initialize(graph_config));
  const int expected_value = 20;
  const int64_t expected_timestamp = 5;
  MP_ASSERT_OK(
      graph.StartRun({{"side_packet", MakePacket<int>(expected_value)},
                      {"timestamp", MakePacket<int64_t>(expected_timestamp)}}));

  MP_ASSERT_OK(graph.WaitUntilDone());

  ASSERT_FALSE(output_packets.empty());
  EXPECT_EQ(Timestamp(expected_timestamp), output_packets.back().Timestamp());
  EXPECT_EQ(expected_value, output_packets.back().Get<int>());
}

TEST(SidePacketToStreamCalculator, AtTimestampWithMultipleOutputs) {
  CalculatorGraphConfig graph_config =
      ParseTextProtoOrDie<CalculatorGraphConfig>(
          R"pb(
            input_side_packet: "timestamp"
            input_side_packet: "side_packet0"
            input_side_packet: "side_packet1"
            output_stream: "packet"
            node {
              calculator: "SidePacketToStreamCalculator"
              input_side_packet: "TIMESTAMP:timestamp"
              input_side_packet: "side_packet0"
              input_side_packet: "side_packet1"
              output_stream: "AT_TIMESTAMP:0:packet0"
              output_stream: "AT_TIMESTAMP:1:packet1"
            }
          )pb");
  std::vector<Packet> output_packets0;
  tool::AddVectorSink("packet0", &graph_config, &output_packets0);
  std::vector<Packet> output_packets1;
  tool::AddVectorSink("packet1", &graph_config, &output_packets1);
  CalculatorGraph graph;

  MP_ASSERT_OK(graph.Initialize(graph_config));
  const int expected_value0 = 20;
  const int expected_value1 = 15;
  const int64_t expected_timestamp = 5;
  MP_ASSERT_OK(
      graph.StartRun({{"side_packet0", MakePacket<int>(expected_value0)},
                      {"side_packet1", MakePacket<int>(expected_value1)},
                      {"timestamp", MakePacket<int64_t>(expected_timestamp)}}));

  MP_ASSERT_OK(graph.WaitUntilDone());

  ASSERT_FALSE(output_packets0.empty());
  EXPECT_EQ(Timestamp(expected_timestamp), output_packets0.back().Timestamp());
  EXPECT_EQ(expected_value0, output_packets0.back().Get<int>());
  ASSERT_FALSE(output_packets1.empty());
  EXPECT_EQ(Timestamp(expected_timestamp), output_packets1.back().Timestamp());
  EXPECT_EQ(expected_value1, output_packets1.back().Get<int>());
}

}  // namespace
}  // namespace mediapipe
