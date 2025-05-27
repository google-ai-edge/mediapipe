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

#include "mediapipe/framework/api3/packet.h"

#include <memory>

#include "mediapipe/framework/api3/packet_matcher.h"
#include "mediapipe/framework/port/gmock.h"
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/framework/timestamp.h"

namespace mediapipe::api3 {
namespace {

using ::testing::Eq;
using ::testing::Pointee;

TEST(PacketTest, MakePacketWorks) {
  Packet<int> p = MakePacket<int>(10);
  EXPECT_THAT(p, PacketEq<int>(Eq(10), Eq(Timestamp::Unset())));

  p = MakePacket<int>(20).At(Timestamp(2000));
  EXPECT_THAT(p, PacketEq<int>(Eq(20), Eq(Timestamp(2000))));
}

TEST(PacketTest, MakePacketWorksForUniquePtr) {
  Packet<int> p = MakePacket<int>(std::make_unique<int>(100));
  EXPECT_THAT(p, PacketEq<int>(Eq(100), Eq(Timestamp::Unset())));

  Packet<std::unique_ptr<int>> p_u =
      MakePacket<std::unique_ptr<int>>(std::make_unique<int>(100));
  EXPECT_THAT(p_u, (PacketEq<std::unique_ptr<int>>(Pointee(100),
                                                   Eq(Timestamp::Unset()))));
}

TEST(PacketTest, PointToForeignWorks) {
  int ptr = 100;
  Packet<int> p = PointToForeign(&ptr);
  EXPECT_THAT(p, PacketEq<int>(Eq(100), Eq(Timestamp::Unset())));
  p = Packet<int>();
}

}  // namespace
}  // namespace mediapipe::api3
