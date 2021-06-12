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

#include <vector>

#include "mediapipe/framework/calculator_runner.h"
#include "mediapipe/framework/port/gmock.h"
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/framework/port/status_matchers.h"
#include "mediapipe/framework/tool/validate_type.h"

namespace mediapipe {
namespace {

inline Packet PacketFrom(int i) { return Adopt(new int(i)).At(Timestamp(i)); }

TEST(PacketInnerJoinCalculatorTest, AllMatching) {
  // Test case.
  const std::vector<int> packets_on_stream1 = {0, 1, 2, 3};
  const std::vector<int> packets_on_stream2 = {0, 1, 2, 3};
  // Run.
  CalculatorRunner runner("PacketInnerJoinCalculator", "", 2, 2, 0);
  for (int packet_load : packets_on_stream1) {
    runner.MutableInputs()->Index(0).packets.push_back(PacketFrom(packet_load));
  }
  for (int packet_load : packets_on_stream2) {
    runner.MutableInputs()->Index(1).packets.push_back(PacketFrom(packet_load));
  }
  MP_ASSERT_OK(runner.Run());
  // Check.
  const std::vector<int> expected = {0, 1, 2, 3};
  ASSERT_EQ(expected.size(), runner.Outputs().Index(0).packets.size());
  ASSERT_EQ(expected.size(), runner.Outputs().Index(1).packets.size());
  for (int i = 0; i < expected.size(); ++i) {
    const Packet packet1 = runner.Outputs().Index(0).packets[i];
    EXPECT_EQ(expected[i], packet1.Get<int>());
    EXPECT_EQ(expected[i], packet1.Timestamp().Value());
    const Packet packet2 = runner.Outputs().Index(1).packets[i];
    EXPECT_EQ(expected[i], packet2.Get<int>());
    EXPECT_EQ(expected[i], packet2.Timestamp().Value());
  }
}

TEST(PacketInnerJoinCalculatorTest, NoneMatching) {
  // Test case.
  const std::vector<int> packets_on_stream1 = {0, 2};
  const std::vector<int> packets_on_stream2 = {1, 3};
  // Run.
  CalculatorRunner runner("PacketInnerJoinCalculator", "", 2, 2, 0);
  for (int packet_load : packets_on_stream1) {
    runner.MutableInputs()->Index(0).packets.push_back(PacketFrom(packet_load));
  }
  for (int packet_load : packets_on_stream2) {
    runner.MutableInputs()->Index(1).packets.push_back(PacketFrom(packet_load));
  }
  MP_ASSERT_OK(runner.Run());
  // Check.
  EXPECT_TRUE(runner.Outputs().Index(0).packets.empty());
  EXPECT_TRUE(runner.Outputs().Index(1).packets.empty());
}

TEST(PacketInnerJoinCalculatorTest, SomeMatching) {
  // Test case.
  const std::vector<int> packets_on_stream1 = {0, 1, 2, 3, 4, 6};
  const std::vector<int> packets_on_stream2 = {0, 2, 4, 5, 6};
  // Run.
  CalculatorRunner runner("PacketInnerJoinCalculator", "", 2, 2, 0);
  for (int packet_load : packets_on_stream1) {
    runner.MutableInputs()->Index(0).packets.push_back(PacketFrom(packet_load));
  }
  for (int packet_load : packets_on_stream2) {
    runner.MutableInputs()->Index(1).packets.push_back(PacketFrom(packet_load));
  }
  MP_ASSERT_OK(runner.Run());
  // Check.
  const std::vector<int> expected = {0, 2, 4, 6};
  ASSERT_EQ(expected.size(), runner.Outputs().Index(0).packets.size());
  ASSERT_EQ(expected.size(), runner.Outputs().Index(1).packets.size());
  for (int i = 0; i < expected.size(); ++i) {
    const Packet packet1 = runner.Outputs().Index(0).packets[i];
    EXPECT_EQ(expected[i], packet1.Get<int>());
    EXPECT_EQ(expected[i], packet1.Timestamp().Value());
    const Packet packet2 = runner.Outputs().Index(1).packets[i];
    EXPECT_EQ(expected[i], packet2.Get<int>());
    EXPECT_EQ(expected[i], packet2.Timestamp().Value());
  }
}

}  // namespace
}  // namespace mediapipe
