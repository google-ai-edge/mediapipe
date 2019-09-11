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

#include "mediapipe/calculators/util/packet_frequency.pb.h"
#include "mediapipe/framework/calculator_runner.h"
#include "mediapipe/framework/port/gmock.h"
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/framework/port/parse_text_proto.h"
#include "mediapipe/framework/port/status_matchers.h"
#include "mediapipe/framework/timestamp.h"

namespace mediapipe {
namespace {

CalculatorGraphConfig::Node GetDefaultNode() {
  return ParseTextProtoOrDie<CalculatorGraphConfig::Node>(R"(
    calculator: "PacketFrequencyCalculator"
    input_stream: "packet_stream"
    output_stream: "packet_frequency"
    options {
      [mediapipe.PacketFrequencyCalculatorOptions.ext] {
        time_window_sec: 3.0
        label: "stream_description"
      }
    }
  )");
}

CalculatorGraphConfig::Node GetNodeWithMultipleStreams() {
  return ParseTextProtoOrDie<CalculatorGraphConfig::Node>(R"(
    calculator: "PacketFrequencyCalculator"
    input_stream: "packet_stream_0"
    input_stream: "packet_stream_1"
    input_stream: "packet_stream_2"
    output_stream: "packet_frequency_0"
    output_stream: "packet_frequency_1"
    output_stream: "packet_frequency_2"
    input_stream_handler { input_stream_handler: "ImmediateInputStreamHandler" }
    options {
      [mediapipe.PacketFrequencyCalculatorOptions.ext] {
        time_window_sec: 3.0
        label: "stream_description_0"
        label: "stream_description_1"
        label: "stream_description_2"
      }
    }
  )");
}

// Tests packet frequency.
TEST(PacketFrequencyCalculatorTest, MultiPacketTest) {
  // Setup the calculator runner and provide integer packets as input (note that
  // it doesn't have to be integer; the calculator can take any type as input).
  CalculatorRunner runner(GetDefaultNode());

  // Packet 1.
  runner.MutableInputs()->Index(0).packets.push_back(
      Adopt(new int).At(Timestamp(0)));
  // Packet 2.
  runner.MutableInputs()->Index(0).packets.push_back(
      Adopt(new int).At(Timestamp(500000)));
  // Packet 3.
  runner.MutableInputs()->Index(0).packets.push_back(
      Adopt(new int).At(Timestamp(1000000)));
  // Packet 4.
  runner.MutableInputs()->Index(0).packets.push_back(
      Adopt(new int).At(Timestamp(1500000)));
  // Packet 5.
  runner.MutableInputs()->Index(0).packets.push_back(
      Adopt(new int).At(Timestamp(3000000)));
  // Packet 6.
  runner.MutableInputs()->Index(0).packets.push_back(
      Adopt(new int).At(Timestamp(4000000)));
  // Packet 7.
  runner.MutableInputs()->Index(0).packets.push_back(
      Adopt(new int).At(Timestamp(9000000)));

  // Run the calculator.
  MP_ASSERT_OK(runner.Run()) << "Calculator execution failed.";
  const std::vector<Packet>& output_packets = runner.Outputs().Index(0).packets;

  // Very first packet. So frequency is zero.
  const auto& output1 = output_packets[0].Get<PacketFrequency>();
  EXPECT_FLOAT_EQ(output1.packet_frequency_hz(), 0.0);
  EXPECT_EQ(output1.label(), "stream_description");

  // 2 packets in the first 500ms.
  const auto& output2 = output_packets[1].Get<PacketFrequency>();
  EXPECT_FLOAT_EQ(output2.packet_frequency_hz(), 4.000000);
  EXPECT_EQ(output2.label(), "stream_description");

  // 3 packets in the first 1 sec.
  const auto& output3 = output_packets[2].Get<PacketFrequency>();
  EXPECT_FLOAT_EQ(output3.packet_frequency_hz(), 3.000000);
  EXPECT_EQ(output3.label(), "stream_description");

  // 4 packets in the first 1.5 sec.
  const auto& output4 = output_packets[3].Get<PacketFrequency>();
  EXPECT_FLOAT_EQ(output4.packet_frequency_hz(), 2.666667);
  EXPECT_EQ(output4.label(), "stream_description");

  // 5 packets in the first 3 sec.
  const auto& output5 = output_packets[4].Get<PacketFrequency>();
  EXPECT_FLOAT_EQ(output5.packet_frequency_hz(), 1.666667);
  EXPECT_EQ(output5.label(), "stream_description");

  // 4 packets in the past 3 sec window.
  const auto& output6 = output_packets[5].Get<PacketFrequency>();
  EXPECT_FLOAT_EQ(output6.packet_frequency_hz(), 1.333333);
  EXPECT_EQ(output6.label(), "stream_description");

  // 1 packet in the past 3 sec window.
  const auto& output7 = output_packets[6].Get<PacketFrequency>();
  EXPECT_FLOAT_EQ(output7.packet_frequency_hz(), 0.33333334);
  EXPECT_EQ(output7.label(), "stream_description");
}

// Tests packet frequency with multiple input/output streams.
TEST(PacketFrequencyCalculatorTest, MultiStreamTest) {
  // Setup the calculator runner and provide strings as input on all streams
  // (note that it doesn't have to be std::string; the calculator can take any
  // type as input).
  CalculatorRunner runner(GetNodeWithMultipleStreams());

  // Packet 1 on stream 1.
  runner.MutableInputs()->Index(0).packets.push_back(
      Adopt(new std::string).At(Timestamp(250000)));
  // Packet 2 on stream 1.
  runner.MutableInputs()->Index(0).packets.push_back(
      Adopt(new std::string).At(Timestamp(500000)));
  // Packet 1 on stream 2.
  runner.MutableInputs()->Index(1).packets.push_back(
      Adopt(new std::string).At(Timestamp(100000)));
  // Packet 2 on stream 2.
  runner.MutableInputs()->Index(1).packets.push_back(
      Adopt(new std::string).At(Timestamp(5000000)));
  // Packet 1 on stream 3.
  runner.MutableInputs()->Index(2).packets.push_back(
      Adopt(new std::string).At(Timestamp(0)));
  // Packet 2 on stream 3.
  runner.MutableInputs()->Index(2).packets.push_back(
      Adopt(new std::string).At(Timestamp(3000000)));

  // Run the calculator.
  MP_ASSERT_OK(runner.Run()) << "Calculator execution failed.";
  const std::vector<Packet>& output_packets_stream_1 =
      runner.Outputs().Index(0).packets;
  const std::vector<Packet>& output_packets_stream_2 =
      runner.Outputs().Index(1).packets;
  const std::vector<Packet>& output_packets_stream_3 =
      runner.Outputs().Index(2).packets;

  // First packet on stream 1. So frequency is zero.
  const auto& output1 = output_packets_stream_1[0].Get<PacketFrequency>();
  EXPECT_FLOAT_EQ(output1.packet_frequency_hz(), 0.0);
  EXPECT_EQ(output1.label(), "stream_description_0");

  // Second packet on stream 1.
  const auto& output2 = output_packets_stream_1[1].Get<PacketFrequency>();
  EXPECT_FLOAT_EQ(output2.packet_frequency_hz(), 8.000000);
  EXPECT_EQ(output2.label(), "stream_description_0");

  // First packet on stream 2. So frequency is zero.
  const auto& output3 = output_packets_stream_2[0].Get<PacketFrequency>();
  EXPECT_FLOAT_EQ(output3.packet_frequency_hz(), 0.0);
  EXPECT_EQ(output3.label(), "stream_description_1");

  // Second packet on stream 2.
  const auto& output4 = output_packets_stream_2[1].Get<PacketFrequency>();
  EXPECT_FLOAT_EQ(output4.packet_frequency_hz(), 0.33333334);
  EXPECT_EQ(output4.label(), "stream_description_1");

  // First packet on stream 3. So frequency is zero.
  const auto& output5 = output_packets_stream_3[0].Get<PacketFrequency>();
  EXPECT_FLOAT_EQ(output5.packet_frequency_hz(), 0.0);
  EXPECT_EQ(output5.label(), "stream_description_2");

  // Second packet on stream 3.
  const auto& output6 = output_packets_stream_3[1].Get<PacketFrequency>();
  EXPECT_FLOAT_EQ(output6.packet_frequency_hz(), 0.66666669);
  EXPECT_EQ(output6.label(), "stream_description_2");
}

}  // namespace
}  // namespace mediapipe
