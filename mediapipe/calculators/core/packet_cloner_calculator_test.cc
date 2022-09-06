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

#include <algorithm>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "absl/strings/str_cat.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/port/gmock.h"
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/framework/port/parse_text_proto.h"
#include "mediapipe/framework/port/status.h"
#include "mediapipe/framework/port/status_matchers.h"
#include "mediapipe/framework/timestamp.h"
#include "mediapipe/framework/tool/sink.h"

namespace mediapipe {
namespace {

using ::testing::ElementsAre;
using ::testing::Eq;
using ::testing::IsTrue;
using ::testing::Value;

MATCHER_P2(IntPacket, value, ts, "") {
  return Value(arg.template Get<int>(), Eq(value)) &&
         Value(arg.Timestamp(), Eq(Timestamp(ts)));
}

MATCHER_P2(FloatPacket, value, ts, "") {
  return Value(arg.template Get<float>(), Eq(value)) &&
         Value(arg.Timestamp(), Eq(Timestamp(ts)));
}

MATCHER_P(EmptyPacket, ts, "") {
  return Value(arg.IsEmpty(), IsTrue()) &&
         Value(arg.Timestamp(), Eq(Timestamp(ts)));
}

template <typename T>
absl::Status SendPacket(const std::string& input_name, T value, int ts,
                        CalculatorGraph& graph) {
  return graph.AddPacketToInputStream(input_name,
                                      MakePacket<T>(value).At(Timestamp(ts)));
}

struct Params {
  bool use_tick_tag = false;
};

class PacketClonerCalculatorTest : public testing::TestWithParam<Params> {};

TEST_P(PacketClonerCalculatorTest, ClonesSingleInputSameTimestamps) {
  CalculatorGraphConfig graph_config =
      ParseTextProtoOrDie<CalculatorGraphConfig>([&]() {
        if (GetParam().use_tick_tag) {
          return R"pb(
            input_stream: 'in1'
            input_stream: 'tick'
            node {
              calculator: 'PacketClonerCalculator'
              input_stream: 'in1'
              input_stream: 'TICK:tick'
              output_stream: 'out1'
            })pb";
        }
        return R"pb(
          input_stream: 'in1'
          input_stream: 'tick'
          node {
            calculator: 'PacketClonerCalculator'
            input_stream: 'in1'
            input_stream: 'tick'
            output_stream: 'out1'
          })pb";
      }());
  std::vector<Packet> out1;
  tool::AddVectorSink("out1", &graph_config, &out1);

  CalculatorGraph graph;
  MP_ASSERT_OK(graph.Initialize(graph_config, {}));
  MP_ASSERT_OK(graph.StartRun({}));

  MP_ASSERT_OK(SendPacket("in1", 1, /*ts=*/10000, graph));
  MP_ASSERT_OK(SendPacket("tick", 1000, /*ts=*/10000, graph));
  MP_ASSERT_OK(graph.WaitUntilIdle());

  EXPECT_THAT(out1, ElementsAre(IntPacket(1, 10000)));
}

TEST_P(PacketClonerCalculatorTest, ClonesSingleInputEarlierTimestamps) {
  CalculatorGraphConfig graph_config =
      ParseTextProtoOrDie<CalculatorGraphConfig>([&]() {
        if (GetParam().use_tick_tag) {
          return R"pb(
            input_stream: 'in1'
            input_stream: 'tick'
            node {
              calculator: 'PacketClonerCalculator'
              input_stream: 'in1'
              input_stream: 'TICK:tick'
              output_stream: 'out1'
            })pb";
        }
        return R"pb(
          input_stream: 'in1'
          input_stream: 'tick'
          node {
            calculator: 'PacketClonerCalculator'
            input_stream: 'in1'
            input_stream: 'tick'
            output_stream: 'out1'
          })pb";
      }());
  std::vector<Packet> out1;
  tool::AddVectorSink("out1", &graph_config, &out1);

  CalculatorGraph graph;
  MP_ASSERT_OK(graph.Initialize(graph_config, {}));
  MP_ASSERT_OK(graph.StartRun({}));

  // PacketClonerCalculator is non-ImmediateInputStreamHandler
  // PacketClonerCalculator waits for "in1" to arrive for ts=5000
  MP_ASSERT_OK(SendPacket("in1", 1, /*ts=*/5000, graph));
  // Newer tick at ts=10000, should NOT trigger output for ts=5000
  // PacketClonerCalculator waits for "in1" to arrive for ts=10000
  MP_ASSERT_OK(SendPacket("tick", 1000, /*ts=*/10000, graph));
  MP_ASSERT_OK(SendPacket("tick", 1001, /*ts=*/10001, graph));
  MP_ASSERT_OK(SendPacket("tick", 1002, /*ts=*/10002, graph));
  // Newer "in1" at ts=15000, should trigger output for ts=10000
  MP_ASSERT_OK(SendPacket("in1", 2, /*ts=*/15000, graph));
  MP_ASSERT_OK(graph.WaitUntilIdle());

  EXPECT_THAT(out1, ElementsAre(IntPacket(1, 10000), IntPacket(1, 10001),
                                IntPacket(1, 10002)));
}

TEST_P(PacketClonerCalculatorTest, ClonesFiveInputs) {
  CalculatorGraphConfig graph_config =
      ParseTextProtoOrDie<CalculatorGraphConfig>([&]() {
        if (GetParam().use_tick_tag) {
          return R"pb(
            input_stream: 'in1'
            input_stream: 'in2'
            input_stream: 'in3'
            input_stream: 'in4'
            input_stream: 'in5'
            input_stream: 'tick'
            node {
              calculator: 'PacketClonerCalculator'
              input_stream: 'in1'
              input_stream: 'in2'
              input_stream: 'in3'
              input_stream: 'in4'
              input_stream: 'in5'
              output_stream: 'out1'
              output_stream: 'out2'
              output_stream: 'out3'
              input_stream: 'TICK:tick'  # arbitrary location
              output_stream: 'out4'
              output_stream: 'out5'
            }
          )pb";
        }
        return R"pb(
          input_stream: 'in1'
          input_stream: 'in2'
          input_stream: 'in3'
          input_stream: 'in4'
          input_stream: 'in5'
          input_stream: 'tick'
          node {
            calculator: 'PacketClonerCalculator'
            input_stream: 'in1'
            input_stream: 'in2'
            input_stream: 'in3'
            input_stream: 'in4'
            input_stream: 'in5'
            input_stream: 'tick'
            output_stream: 'out1'
            output_stream: 'out2'
            output_stream: 'out3'
            output_stream: 'out4'
            output_stream: 'out5'
          }
        )pb";
      }());
  constexpr int kNumToClone = 5;
  std::array<std::vector<Packet>, kNumToClone> outs;
  for (int i = 0; i < kNumToClone; ++i) {
    tool::AddVectorSink(absl::StrCat("out", i + 1), &graph_config, &outs[i]);
  }

  CalculatorGraph graph;
  MP_ASSERT_OK(graph.Initialize(graph_config, {}));
  MP_ASSERT_OK(graph.StartRun({}));

  MP_ASSERT_OK(SendPacket("in1", 10, /*ts=*/10000, graph));
  MP_ASSERT_OK(SendPacket("in2", 20.0f, /*ts=*/10000, graph));
  MP_ASSERT_OK(SendPacket("in3", 30, /*ts=*/10000, graph));
  MP_ASSERT_OK(SendPacket("in4", 40.0f, /*ts=*/10000, graph));
  MP_ASSERT_OK(SendPacket("in5", 50, /*ts=*/10000, graph));
  MP_ASSERT_OK(SendPacket("tick", 1000, /*ts=*/10000, graph));
  // Below "tick" packets won't trigger output, until newer inputs are sent,
  // because inputs are missing and ImmediateInputStreamHandler is not
  // configured.
  MP_ASSERT_OK(SendPacket("tick", 1001, /*ts=*/10001, graph));
  MP_ASSERT_OK(SendPacket("tick", 1002, /*ts=*/10002, graph));
  MP_ASSERT_OK(graph.WaitUntilIdle());

  EXPECT_THAT(outs, ElementsAre(ElementsAre(IntPacket(10, 10000)),
                                ElementsAre(FloatPacket(20.0f, 10000)),
                                ElementsAre(IntPacket(30, 10000)),
                                ElementsAre(FloatPacket(40.0f, 10000)),
                                ElementsAre(IntPacket(50, 10000))));

  MP_ASSERT_OK(SendPacket("in1", 100, /*ts=*/20000, graph));
  MP_ASSERT_OK(SendPacket("in2", 200.0f, /*ts=*/20000, graph));
  MP_ASSERT_OK(SendPacket("in3", 300, /*ts=*/20000, graph));
  MP_ASSERT_OK(SendPacket("in4", 400.0f, /*ts=*/20000, graph));
  MP_ASSERT_OK(SendPacket("in5", 500, /*ts=*/20000, graph));
  MP_ASSERT_OK(SendPacket("tick", 2000, /*ts=*/20000, graph));
  // Below "tick" packets won't trigger output, because inputs are missing and
  // ImmediateInputStreamHandler is not configured.
  MP_ASSERT_OK(SendPacket("tick", 2001, /*ts=*/20001, graph));
  MP_ASSERT_OK(SendPacket("tick", 2002, /*ts=*/20002, graph));
  MP_ASSERT_OK(graph.WaitUntilIdle());

  EXPECT_THAT(
      outs,
      ElementsAre(
          ElementsAre(IntPacket(10, 10000), IntPacket(10, 10001),
                      IntPacket(10, 10002), IntPacket(100, 20000)),
          ElementsAre(FloatPacket(20.0f, 10000), FloatPacket(20.0f, 10001),
                      FloatPacket(20.0f, 10002), FloatPacket(200.0f, 20000)),
          ElementsAre(IntPacket(30, 10000), IntPacket(30, 10001),
                      IntPacket(30, 10002), IntPacket(300, 20000)),
          ElementsAre(FloatPacket(40.0f, 10000), FloatPacket(40.0f, 10001),
                      FloatPacket(40.0f, 10002), FloatPacket(400.0f, 20000)),
          ElementsAre(IntPacket(50, 10000), IntPacket(50, 10001),
                      IntPacket(50, 10002), IntPacket(500, 20000))));
}

TEST_P(PacketClonerCalculatorTest,
       ClonesTwoInputsWithImmediateInputStreamHandler) {
  CalculatorGraphConfig graph_config =
      ParseTextProtoOrDie<CalculatorGraphConfig>([&]() {
        if (GetParam().use_tick_tag) {
          return R"pb(
            input_stream: 'in1'
            input_stream: 'in2'
            input_stream: 'tick'
            node {
              calculator: 'PacketClonerCalculator'
              input_stream: 'TICK:tick'
              input_stream: 'in1'
              input_stream: 'in2'
              output_stream: 'out1'
              output_stream: 'out2'
              input_stream_handler {
                input_stream_handler: "ImmediateInputStreamHandler"
              }
            })pb";
        }
        return R"pb(
          input_stream: 'in1'
          input_stream: 'in2'
          input_stream: 'tick'
          node {
            calculator: 'PacketClonerCalculator'
            input_stream: 'in1'
            input_stream: 'in2'
            input_stream: 'tick'
            output_stream: 'out1'
            output_stream: 'out2'
            input_stream_handler {
              input_stream_handler: "ImmediateInputStreamHandler"
            }
          })pb";
      }());
  constexpr int kNumToClone = 2;
  std::array<std::vector<Packet>, kNumToClone> outs;
  for (int i = 0; i < kNumToClone; ++i) {
    tool::AddVectorSink(absl::StrCat("out", i + 1), &graph_config, &outs[i]);
  }

  CalculatorGraph graph;
  MP_ASSERT_OK(graph.Initialize(graph_config, {}));
  MP_ASSERT_OK(graph.StartRun({}));

  // No packets to clone.
  MP_ASSERT_OK(SendPacket("tick", 0, /*ts=*/0, graph));
  MP_ASSERT_OK(graph.WaitUntilIdle());

  // Cloning current packets.
  MP_ASSERT_OK(SendPacket("in1", 1, /*ts=*/10000, graph));
  MP_ASSERT_OK(SendPacket("in2", 10.0f, /*ts=*/10000, graph));
  MP_ASSERT_OK(SendPacket("tick", 1000, /*ts=*/10000, graph));
  MP_ASSERT_OK(graph.WaitUntilIdle());

  // Cloning past packets.
  MP_ASSERT_OK(SendPacket("tick", 1500, /*ts=*/15000, graph));
  MP_ASSERT_OK(graph.WaitUntilIdle());

  // Cloning past packets.
  MP_ASSERT_OK(SendPacket("in1", 2, /*ts=*/10001, graph));
  MP_ASSERT_OK(SendPacket("in2", 20.0f, /*ts=*/10001, graph));
  MP_ASSERT_OK(SendPacket("tick", 2000, /*ts=*/20000, graph));
  MP_ASSERT_OK(graph.WaitUntilIdle());

  // Cloning future packets.
  MP_ASSERT_OK(SendPacket("in1", 3, /*ts=*/30000, graph));
  MP_ASSERT_OK(SendPacket("in2", 30.0f, /*ts=*/30000, graph));
  // Waiting to ensure newer packets (ts=30000) to clone would get into the
  // cloner before tick (ts=25000) does.
  MP_ASSERT_OK(graph.WaitUntilIdle());
  MP_ASSERT_OK(SendPacket("tick", 3000, /*ts=*/25000, graph));
  MP_ASSERT_OK(graph.WaitUntilIdle());

  // Cloning packets having different timestamps.
  MP_ASSERT_OK(SendPacket("in1", 4, /*ts=*/38000, graph));
  MP_ASSERT_OK(SendPacket("in2", 40.0f, /*ts=*/39000, graph));
  MP_ASSERT_OK(SendPacket("tick", 4000, /*ts=*/40000, graph));
  MP_ASSERT_OK(graph.WaitUntilIdle());

  EXPECT_THAT(
      outs,
      ElementsAre(
          ElementsAre(IntPacket(1, 10000), IntPacket(1, 15000),
                      IntPacket(2, 20000), IntPacket(3, 25000),
                      IntPacket(4, 40000)),
          ElementsAre(FloatPacket(10.0f, 10000), FloatPacket(10.0f, 15000),
                      FloatPacket(20.0f, 20000), FloatPacket(30.0f, 25000),
                      FloatPacket(40.0f, 40000))));
}

class PacketClonerCalculatorGatedInputTest : public ::testing::Test {
 protected:
  void SetUp() override {
    CalculatorGraphConfig graph_config =
        ParseTextProtoOrDie<CalculatorGraphConfig>([&]() {
          return R"pb(
            input_stream: 'input'
            input_stream: 'input_enabled'
            input_stream: 'tick'
            input_stream: 'tick_enabled'
            node {
              calculator: 'GateCalculator'
              input_stream: 'tick'
              input_stream: 'ALLOW:tick_enabled'
              output_stream: 'tick_gated'
            }
            node {
              calculator: 'GateCalculator'
              input_stream: 'input'
              input_stream: 'ALLOW:input_enabled'
              output_stream: 'input_gated'
            }
            node {
              calculator: 'PacketClonerCalculator'
              input_stream: 'input_gated'
              input_stream: 'TICK:tick_gated'
              output_stream: 'output'
            })pb";
        }());

    MP_ASSERT_OK(graph.Initialize(graph_config, {}));
    MP_ASSERT_OK(graph.ObserveOutputStream(
        "output",
        [this](Packet const& packet) {
          output.push_back(packet);
          return absl::OkStatus();
        },
        true));
    MP_ASSERT_OK(graph.StartRun({}));
  }

  CalculatorGraph graph;
  std::vector<Packet> output;
};

TEST_F(PacketClonerCalculatorGatedInputTest,
       PropagatesTimestampBoundsWithEmptyInput) {
  MP_ASSERT_OK(SendPacket("tick_enabled", false, /*ts=*/100, graph));
  MP_ASSERT_OK(SendPacket("tick", 0, /*ts=*/100, graph));

  MP_ASSERT_OK(SendPacket("input_enabled", false, /*ts=*/200, graph));
  MP_ASSERT_OK(SendPacket("input", 1, /*ts=*/200, graph));

  MP_ASSERT_OK(graph.WaitUntilIdle());

  EXPECT_THAT(output, ElementsAre(EmptyPacket(100)));
}

TEST_F(PacketClonerCalculatorGatedInputTest,
       PropagatesTimestampBoundsWithInput) {
  MP_ASSERT_OK(SendPacket("input_enabled", true, /*ts=*/100, graph));
  MP_ASSERT_OK(SendPacket("input", 1, /*ts=*/100, graph));

  MP_ASSERT_OK(SendPacket("tick_enabled", true, /*ts=*/100, graph));
  MP_ASSERT_OK(SendPacket("tick", 0, /*ts=*/100, graph));

  MP_ASSERT_OK(SendPacket("tick_enabled", false, /*ts=*/110, graph));
  MP_ASSERT_OK(SendPacket("tick", 0, /*ts=*/110, graph));

  MP_ASSERT_OK(SendPacket("input_enabled", false, /*ts=*/200, graph));
  MP_ASSERT_OK(SendPacket("input", 2, /*ts=*/200, graph));

  MP_ASSERT_OK(graph.WaitUntilIdle());

  EXPECT_THAT(output, ElementsAre(IntPacket(1, 100), EmptyPacket(110)));
}

TEST_F(PacketClonerCalculatorGatedInputTest,
       PropagatesTimestampBoundsFromTick) {
  MP_ASSERT_OK(SendPacket("input_enabled", true, /*ts=*/100, graph));
  MP_ASSERT_OK(SendPacket("input", 1, /*ts=*/100, graph));

  MP_ASSERT_OK(SendPacket("tick_enabled", true, /*ts=*/100, graph));
  MP_ASSERT_OK(SendPacket("tick", 0, /*ts=*/100, graph));

  MP_ASSERT_OK(SendPacket("input_enabled", true, /*ts=*/110, graph));
  MP_ASSERT_OK(SendPacket("input", 2, /*ts=*/110, graph));

  MP_ASSERT_OK(SendPacket("tick_enabled", false, /*ts=*/200, graph));
  MP_ASSERT_OK(SendPacket("tick", 0, /*ts=*/200, graph));

  MP_ASSERT_OK(SendPacket("input_enabled", false, /*ts=*/200, graph));
  MP_ASSERT_OK(SendPacket("input", 2, /*ts=*/200, graph));

  MP_ASSERT_OK(graph.WaitUntilIdle());

  EXPECT_THAT(output, ElementsAre(IntPacket(1, 100), EmptyPacket(200)));
}

INSTANTIATE_TEST_SUITE_P(PacketClonerCalculator, PacketClonerCalculatorTest,
                         testing::ValuesIn({Params{.use_tick_tag = false},
                                            Params{.use_tick_tag = true}}));
}  // anonymous namespace
}  // namespace mediapipe
