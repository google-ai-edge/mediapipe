// Copyright 2018 The MediaPipe Authors.
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

#include "mediapipe/framework/tool/fill_packet_set.h"

#include "mediapipe/framework/calculator.pb.h"
#include "mediapipe/framework/port/gmock.h"
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/framework/port/status_matchers.h"

namespace mediapipe {

namespace {

TEST(FillPacketSetTest, Success) {
  CalculatorGraphConfig::Node node;
  node.add_input_side_packet("side_packet1");
  node.add_input_side_packet("side_packet2");
  node.add_input_side_packet("DOUBLE:side_packet3");
  node.add_input_side_packet("DOUBLE:1:side_packet4");

  PacketTypeSet input_side_packet_types(
      tool::TagMap::Create(node.input_side_packet()).ValueOrDie());
  input_side_packet_types.Index(0).Set<int>(
      // An age
  );
  input_side_packet_types.Index(1).Set<std::string>(
      // A name
  );
  input_side_packet_types.Tag("DOUBLE").Set<double>(
      // double1
  );
  input_side_packet_types.Get("DOUBLE", 1)
      .Set<double>(
          // double2
      );
  MP_EXPECT_OK(ValidatePacketTypeSet(input_side_packet_types));

  std::map<std::string, Packet> all_side_packets;
  all_side_packets["side_packet1"] = MakePacket<int>(70);
  all_side_packets["side_packet2"] = MakePacket<std::string>("Dennis Ritchie");
  all_side_packets["side_packet3"] = MakePacket<double>(42.0);
  all_side_packets["side_packet4"] = MakePacket<double>(-43.0);
  all_side_packets["foo_side_packet"] = MakePacket<std::string>("asdfgh");
  all_side_packets["bar_side_packet"] = MakePacket<double>(-1.0);

  std::unique_ptr<PacketSet> input_side_packets =
      tool::FillPacketSet(input_side_packet_types, all_side_packets, nullptr)
          .ValueOrDie();
  ASSERT_EQ(4, input_side_packets->NumEntries());
  EXPECT_EQ(input_side_packets->Index(0).Get<int>(), 70);
  EXPECT_EQ(input_side_packets->Index(1).Get<std::string>(), "Dennis Ritchie");
  EXPECT_EQ(input_side_packets->Tag("DOUBLE").Get<double>(), 42.0);
  EXPECT_EQ(input_side_packets->Get("DOUBLE", 1).Get<double>(), -43.0);
}

TEST(FillPacketSetTest, MissingSidePacketError) {
  CalculatorGraphConfig::Node node;
  node.add_input_side_packet("side_packet1");
  node.add_input_side_packet("side_packet2");
  node.add_input_side_packet("DOUBLE:side_packet3");
  node.add_input_side_packet("DOUBLE:1:side_packet4");

  PacketTypeSet input_side_packet_types(
      tool::TagMap::Create(node.input_side_packet()).ValueOrDie());
  input_side_packet_types.Index(0).Set<int>(
      // An age
  );
  input_side_packet_types.Index(1).Set<std::string>(
      // A name
  );
  input_side_packet_types.Tag("DOUBLE").Set<double>(
      // double1
  );
  input_side_packet_types.Get("DOUBLE", 1)
      .Set<double>(
          // double2
      );
  MP_EXPECT_OK(ValidatePacketTypeSet(input_side_packet_types));

  std::map<std::string, Packet> all_side_packets;
  all_side_packets["side_packet1"] = MakePacket<int>(70);
  all_side_packets["side_packet2"] = MakePacket<std::string>("Dennis Ritchie");
  all_side_packets["side_packet4"] = MakePacket<double>(-43.0);
  all_side_packets["foo_side_packet"] = MakePacket<std::string>("asdfgh");
  all_side_packets["bar_side_packet"] = MakePacket<double>(-1.0);

  EXPECT_THAT(
      tool::FillPacketSet(input_side_packet_types, all_side_packets, nullptr)
          .status()
          .message(),
      testing::HasSubstr("side_packet3"));
}

TEST(FillPacketSetTest, MissingSidePacketOk) {
  CalculatorGraphConfig::Node node;
  node.add_input_side_packet("side_packet1");
  node.add_input_side_packet("side_packet2");
  node.add_input_side_packet("DOUBLE:side_packet3");
  node.add_input_side_packet("DOUBLE:1:side_packet4");

  PacketTypeSet input_side_packet_types(
      tool::TagMap::Create(node.input_side_packet()).ValueOrDie());
  input_side_packet_types.Index(0).Set<int>(
      // An age
  );
  input_side_packet_types.Index(1).Set<std::string>(
      // A name
  );
  input_side_packet_types.Tag("DOUBLE").Set<double>(
      // double1
  );
  input_side_packet_types.Get("DOUBLE", 1)
      .Set<double>(
          // double2
      );
  MP_EXPECT_OK(ValidatePacketTypeSet(input_side_packet_types));

  std::map<std::string, Packet> all_side_packets;
  all_side_packets["side_packet1"] = MakePacket<int>(70);
  all_side_packets["side_packet2"] = MakePacket<std::string>("Dennis Ritchie");
  all_side_packets["side_packet4"] = MakePacket<double>(-43.0);
  all_side_packets["foo_side_packet"] = MakePacket<std::string>("asdfgh");
  all_side_packets["bar_side_packet"] = MakePacket<double>(-1.0);

  int missing_packet_count;
  std::unique_ptr<PacketSet> input_side_packets =
      tool::FillPacketSet(input_side_packet_types, all_side_packets,
                          &missing_packet_count)
          .ValueOrDie();
  ASSERT_EQ(4, input_side_packets->NumEntries());
  EXPECT_EQ(1, missing_packet_count);
  EXPECT_EQ(input_side_packets->Index(0).Get<int>(), 70);
  EXPECT_EQ(input_side_packets->Index(1).Get<std::string>(), "Dennis Ritchie");
  EXPECT_TRUE(input_side_packets->Tag("DOUBLE").IsEmpty());
  EXPECT_EQ(input_side_packets->Get("DOUBLE", 1).Get<double>(), -43.0);
}

TEST(FillPacketSetTest, WrongSidePacketType) {
  CalculatorGraphConfig::Node node;
  node.add_input_side_packet("side_packet1");
  node.add_input_side_packet("side_packet2");
  node.add_input_side_packet("DOUBLE:side_packet3");
  node.add_input_side_packet("DOUBLE:1:side_packet4");

  PacketTypeSet input_side_packet_types(
      tool::TagMap::Create(node.input_side_packet()).ValueOrDie());
  input_side_packet_types.Index(0).Set<int>(
      // An age
  );
  input_side_packet_types.Index(1).Set<std::string>(
      // A name
  );
  input_side_packet_types.Tag("DOUBLE").Set<double>(
      // double1
  );
  input_side_packet_types.Get("DOUBLE", 1)
      .Set<double>(
          // double2
      );
  MP_EXPECT_OK(ValidatePacketTypeSet(input_side_packet_types));

  std::map<std::string, Packet> all_side_packets;
  all_side_packets["side_packet1"] = MakePacket<float>(3.0f);  // Wrong Type.
  all_side_packets["side_packet2"] = MakePacket<std::string>("Dennis Ritchie");
  all_side_packets["side_packet3"] = MakePacket<double>(42.0);
  all_side_packets["side_packet4"] = MakePacket<double>(-43.0);
  all_side_packets["foo_side_packet"] = MakePacket<std::string>("asdfgh");
  all_side_packets["bar_side_packet"] = MakePacket<double>(-1.0);

  EXPECT_THAT(
      tool::FillPacketSet(input_side_packet_types, all_side_packets, nullptr)
          .status()
          .message(),
      testing::AllOf(
          // Problematic side packet.
          testing::HasSubstr("side_packet1"),
          // Actual type.
          testing::HasSubstr("float"),
          // Expected type.
          testing::HasSubstr("int")));
}

}  // namespace
}  // namespace mediapipe
