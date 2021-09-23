// Copyright 2021 The MediaPipe Authors.
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
//
#include "mediapipe/util/packet_test_util.h"

#include <string>

#include "mediapipe/framework/packet.h"
#include "mediapipe/framework/port.h"
#include "mediapipe/framework/port/gmock.h"
#include "mediapipe/framework/port/gtest-spi.h"
#include "mediapipe/framework/timestamp.h"

namespace mediapipe {
namespace {

using ::testing::Eq;

TEST(PacketTestUtilTest, Matches) {
  const Packet int_packet = MakePacket<int>(42);
  EXPECT_THAT(int_packet, PacketContains<int>(Eq(42)));
}

TEST(PacketTestUtilTest, MatchesContentWithMatchingTimestamp) {
  const Packet int_packet = MakePacket<int>(42).At(Timestamp::PostStream());
  EXPECT_THAT(int_packet, PacketContainsTimestampAndPayload<int>(
                              Eq(Timestamp::PostStream()), Eq(42)));
}

TEST(PacketTestUtilTest, MatchesContentWithMismatchingTimestamp) {
  const Packet int_packet = MakePacket<int>(42).At(Timestamp(0ll));
  EXPECT_NONFATAL_FAILURE(
      {
        EXPECT_THAT(int_packet, PacketContainsTimestampAndPayload<int>(
                                    Eq(Timestamp::PostStream()), Eq(42)));
      },
      "`Packet::Timestamp` is equal to Timestamp::PostStream()");
}

TEST(PacketTestUtilTest, DoesNotMatch) {
  const Packet int_packet = MakePacket<int>(42);
  EXPECT_NONFATAL_FAILURE(
      { EXPECT_THAT(int_packet, PacketContains<int>(Eq(47))); },
      "containing value 42");
}

TEST(PacketTestUtilTest, DoesNotMatchContentWithMatchingTimestamp) {
  const Packet int_packet = MakePacket<int>(42).At(Timestamp(0ll));
  EXPECT_NONFATAL_FAILURE(
      {
        EXPECT_THAT(int_packet, PacketContainsTimestampAndPayload<int>(
                                    Eq(Timestamp(0ll)), Eq(47)));
      },
      "type int that is equal to 47");
}

TEST(PacketTestUtilTest, DoesNotMatchContentWithMismatchingTimestamp) {
  const Packet int_packet = MakePacket<int>(42).At(Timestamp(0ll));
  EXPECT_NONFATAL_FAILURE(
      {
        EXPECT_THAT(int_packet, PacketContainsTimestampAndPayload<int>(
                                    Eq(Timestamp(20ll)), Eq(47)));
      },
      "`Packet::Timestamp` is equal to 20) and (packet contains value of type "
      "int that is equal to 47");
}

TEST(PacketTestUtilTest, TypeMismatch) {
  const Packet string_packet = MakePacket<std::string>("42");
  EXPECT_NONFATAL_FAILURE(
      { EXPECT_THAT(string_packet, PacketContains<int>(Eq(42))); },
      "does not contain expected type int");
}

TEST(PacketTestUtilTest, TypeMismatchContentWithMatchingTimestamp) {
  const Packet int_packet = MakePacket<std::string>("42").At(Timestamp(0ll));
  EXPECT_NONFATAL_FAILURE(
      {
        EXPECT_THAT(int_packet, PacketContainsTimestampAndPayload<int>(
                                    Eq(Timestamp(0ll)), Eq(47)));
      },
      "does not contain expected type int");
}

TEST(PacketTestUtilTest, TypeMismatchContentWithMismatchingTimestamp) {
  const Packet int_packet = MakePacket<std::string>("42").At(Timestamp(0ll));
  EXPECT_NONFATAL_FAILURE(
      {
        EXPECT_THAT(int_packet, PacketContainsTimestampAndPayload<int>(
                                    Eq(Timestamp::PreStream()), Eq(47)));
      },
      "`Packet::Timestamp` is equal to Timestamp::PreStream()) and (packet "
      "contains value of type int that is equal to 47");
}

}  // namespace
}  // namespace mediapipe
