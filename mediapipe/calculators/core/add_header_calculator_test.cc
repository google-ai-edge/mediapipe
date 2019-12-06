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

#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/calculator_runner.h"
#include "mediapipe/framework/port/canonical_errors.h"
#include "mediapipe/framework/port/gmock.h"
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/framework/port/status.h"
#include "mediapipe/framework/port/status_matchers.h"
#include "mediapipe/framework/timestamp.h"
#include "mediapipe/framework/tool/validate_type.h"

namespace mediapipe {

class AddHeaderCalculatorTest : public ::testing::Test {};

TEST_F(AddHeaderCalculatorTest, HeaderStream) {
  CalculatorGraphConfig::Node node;
  node.set_calculator("AddHeaderCalculator");
  node.add_input_stream("HEADER:header_stream");
  node.add_input_stream("DATA:data_stream");
  node.add_output_stream("merged_stream");

  CalculatorRunner runner(node);

  // Set header and add 5 packets.
  runner.MutableInputs()->Tag("HEADER").header =
      Adopt(new std::string("my_header"));
  for (int i = 0; i < 5; ++i) {
    Packet packet = Adopt(new int(i)).At(Timestamp(i * 1000));
    runner.MutableInputs()->Tag("DATA").packets.push_back(packet);
  }

  // Run calculator.
  MP_ASSERT_OK(runner.Run());

  ASSERT_EQ(1, runner.Outputs().NumEntries());

  // Test output.
  EXPECT_EQ(std::string("my_header"),
            runner.Outputs().Index(0).header.Get<std::string>());
  const std::vector<Packet>& output_packets = runner.Outputs().Index(0).packets;
  ASSERT_EQ(5, output_packets.size());
  for (int i = 0; i < 5; ++i) {
    const int val = output_packets[i].Get<int>();
    EXPECT_EQ(i, val);
    EXPECT_EQ(Timestamp(i * 1000), output_packets[i].Timestamp());
  }
}

TEST_F(AddHeaderCalculatorTest, HandlesEmptyHeaderStream) {
  CalculatorGraphConfig::Node node;
  node.set_calculator("AddHeaderCalculator");
  node.add_input_stream("HEADER:header_stream");
  node.add_input_stream("DATA:data_stream");
  node.add_output_stream("merged_stream");

  CalculatorRunner runner(node);

  // No header and no packets.
  // Run calculator.
  MP_ASSERT_OK(runner.Run());
  EXPECT_TRUE(runner.Outputs().Index(0).header.IsEmpty());
}

TEST_F(AddHeaderCalculatorTest, NoPacketsOnHeaderStream) {
  CalculatorGraphConfig::Node node;
  node.set_calculator("AddHeaderCalculator");
  node.add_input_stream("HEADER:header_stream");
  node.add_input_stream("DATA:data_stream");
  node.add_output_stream("merged_stream");

  CalculatorRunner runner(node);

  // Set header and add 5 packets.
  runner.MutableInputs()->Tag("HEADER").header =
      Adopt(new std::string("my_header"));
  runner.MutableInputs()->Tag("HEADER").packets.push_back(
      Adopt(new std::string("not allowed")));
  for (int i = 0; i < 5; ++i) {
    Packet packet = Adopt(new int(i)).At(Timestamp(i * 1000));
    runner.MutableInputs()->Tag("DATA").packets.push_back(packet);
  }

  // Run calculator.
  ASSERT_FALSE(runner.Run().ok());
}

TEST_F(AddHeaderCalculatorTest, InputSidePacket) {
  CalculatorGraphConfig::Node node;
  node.set_calculator("AddHeaderCalculator");
  node.add_input_stream("DATA:data_stream");
  node.add_output_stream("merged_stream");
  node.add_input_side_packet("HEADER:header");

  CalculatorRunner runner(node);

  // Set header and add 5 packets.
  runner.MutableSidePackets()->Tag("HEADER") =
      Adopt(new std::string("my_header"));
  for (int i = 0; i < 5; ++i) {
    Packet packet = Adopt(new int(i)).At(Timestamp(i * 1000));
    runner.MutableInputs()->Tag("DATA").packets.push_back(packet);
  }

  // Run calculator.
  MP_ASSERT_OK(runner.Run());

  ASSERT_EQ(1, runner.Outputs().NumEntries());

  // Test output.
  EXPECT_EQ(std::string("my_header"),
            runner.Outputs().Index(0).header.Get<std::string>());
  const std::vector<Packet>& output_packets = runner.Outputs().Index(0).packets;
  ASSERT_EQ(5, output_packets.size());
  for (int i = 0; i < 5; ++i) {
    const int val = output_packets[i].Get<int>();
    EXPECT_EQ(i, val);
    EXPECT_EQ(Timestamp(i * 1000), output_packets[i].Timestamp());
  }
}

TEST_F(AddHeaderCalculatorTest, UsingBothSideInputAndStream) {
  CalculatorGraphConfig::Node node;
  node.set_calculator("AddHeaderCalculator");
  node.add_input_stream("HEADER:header_stream");
  node.add_input_stream("DATA:data_stream");
  node.add_output_stream("merged_stream");
  node.add_input_side_packet("HEADER:header");

  CalculatorRunner runner(node);

  // Set both headers and add 5 packets.
  runner.MutableSidePackets()->Tag("HEADER") =
      Adopt(new std::string("my_header"));
  runner.MutableSidePackets()->Tag("HEADER") =
      Adopt(new std::string("my_header"));
  for (int i = 0; i < 5; ++i) {
    Packet packet = Adopt(new int(i)).At(Timestamp(i * 1000));
    runner.MutableInputs()->Tag("DATA").packets.push_back(packet);
  }

  // Run should fail because header can only be provided one way.
  EXPECT_EQ(runner.Run().code(), ::mediapipe::InvalidArgumentError("").code());
}

}  // namespace mediapipe
