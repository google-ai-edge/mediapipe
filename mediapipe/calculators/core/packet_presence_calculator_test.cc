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

#include <functional>
#include <string>
#include <vector>

#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/calculator_runner.h"
#include "mediapipe/framework/port/gmock.h"
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/framework/port/parse_text_proto.h"
#include "mediapipe/framework/port/status.h"
#include "mediapipe/framework/port/status_matchers.h"
#include "mediapipe/framework/timestamp.h"
#include "mediapipe/framework/tool/sink.h"

namespace mediapipe {
using ::testing::ElementsAre;
using ::testing::Eq;
using ::testing::Value;
namespace {

MATCHER_P2(BoolPacket, value, timestamp, "") {
  return Value(arg.template Get<bool>(), Eq(value)) &&
         Value(arg.Timestamp(), Eq(timestamp));
}

TEST(PreviousLoopbackCalculator, CorrectTimestamps) {
  std::vector<Packet> output_packets;
  CalculatorGraphConfig graph_config =
      ParseTextProtoOrDie<CalculatorGraphConfig>(R"pb(
        input_stream: 'allow'
        input_stream: 'value'
        node {
          calculator: "GateCalculator"
          input_stream: 'value'
          input_stream: 'ALLOW:allow'
          output_stream: 'gated_value'
        }
        node {
          calculator: 'PacketPresenceCalculator'
          input_stream: 'PACKET:gated_value'
          output_stream: 'PRESENCE:presence'
        }
      )pb");
  tool::AddVectorSink("presence", &graph_config, &output_packets);

  CalculatorGraph graph;
  MP_ASSERT_OK(graph.Initialize(graph_config, {}));
  MP_ASSERT_OK(graph.StartRun({}));

  auto send_packet = [&graph](int value, bool allow, Timestamp timestamp) {
    MP_ASSERT_OK(graph.AddPacketToInputStream(
        "value", MakePacket<int>(value).At(timestamp)));
    MP_ASSERT_OK(graph.AddPacketToInputStream(
        "allow", MakePacket<bool>(allow).At(timestamp)));
  };

  send_packet(10, false, Timestamp(10));
  MP_EXPECT_OK(graph.WaitUntilIdle());
  EXPECT_THAT(output_packets, ElementsAre(BoolPacket(false, Timestamp(10))));

  output_packets.clear();
  send_packet(20, true, Timestamp(11));
  MP_EXPECT_OK(graph.WaitUntilIdle());
  EXPECT_THAT(output_packets, ElementsAre(BoolPacket(true, Timestamp(11))));

  MP_EXPECT_OK(graph.CloseAllInputStreams());
  MP_EXPECT_OK(graph.WaitUntilDone());
}

}  // namespace
}  // namespace mediapipe
