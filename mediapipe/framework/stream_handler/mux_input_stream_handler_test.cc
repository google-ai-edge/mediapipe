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

#include "absl/status/status.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/port/gmock.h"
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/framework/port/parse_text_proto.h"
#include "mediapipe/framework/port/status_matchers.h"

namespace mediapipe {
namespace {

using ::testing::ElementsAre;

// A regression test for b/31620439. MuxInputStreamHandler's accesses to the
// control and data streams should be atomic so that it has a consistent view
// of the two streams. None of the CHECKs in the GetNodeReadiness() method of
// MuxInputStreamHandler should fail when running this test.
TEST(MuxInputStreamHandlerTest, AtomicAccessToControlAndDataStreams) {
  CalculatorGraphConfig config =
      mediapipe::ParseTextProtoOrDie<CalculatorGraphConfig>(R"pb(
        input_stream: "input"
        node {
          calculator: "RoundRobinDemuxCalculator"
          input_stream: "input"
          output_stream: "OUTPUT:0:input0"
          output_stream: "OUTPUT:1:input1"
          output_stream: "OUTPUT:2:input2"
          output_stream: "OUTPUT:3:input3"
          output_stream: "OUTPUT:4:input4"
          output_stream: "SELECT:select"
        }
        node {
          calculator: "PassThroughCalculator"
          input_stream: "input0"
          output_stream: "output0"
        }
        node {
          calculator: "PassThroughCalculator"
          input_stream: "input1"
          output_stream: "output1"
        }
        node {
          calculator: "PassThroughCalculator"
          input_stream: "input2"
          output_stream: "output2"
        }
        node {
          calculator: "PassThroughCalculator"
          input_stream: "input3"
          output_stream: "output3"
        }
        node {
          calculator: "PassThroughCalculator"
          input_stream: "input4"
          output_stream: "output4"
        }
        node {
          calculator: "MuxCalculator"
          input_stream: "INPUT:0:output0"
          input_stream: "INPUT:1:output1"
          input_stream: "INPUT:2:output2"
          input_stream: "INPUT:3:output3"
          input_stream: "INPUT:4:output4"
          input_stream: "SELECT:select"
          output_stream: "OUTPUT:output"
          input_stream_handler { input_stream_handler: "MuxInputStreamHandler" }
        })pb");
  CalculatorGraph graph;
  MP_ASSERT_OK(graph.Initialize(config));
  MP_ASSERT_OK(graph.StartRun({}));
  for (int i = 0; i < 2000; ++i) {
    MP_ASSERT_OK(graph.AddPacketToInputStream(
        "input", Adopt(new int(i)).At(Timestamp(i))));
  }
  MP_ASSERT_OK(graph.CloseAllInputStreams());
  MP_ASSERT_OK(graph.WaitUntilDone());
}

MATCHER_P2(IntPacket, value, ts, "") {
  return arg.template Get<int>() == value && arg.Timestamp() == ts;
}

struct GateAndMuxGraphInput {
  int input0;
  int input1;
  int input2;
  int select;
  bool allow0;
  bool allow1;
  bool allow2;
  Timestamp at;
};

constexpr char kGateAndMuxGraph[] = R"pb(
  input_stream: "input0"
  input_stream: "input1"
  input_stream: "input2"
  input_stream: "select"
  input_stream: "allow0"
  input_stream: "allow1"
  input_stream: "allow2"
  node {
    calculator: "GateCalculator"
    input_stream: "ALLOW:allow0"
    input_stream: "input0"
    output_stream: "output0"
  }
  node {
    calculator: "GateCalculator"
    input_stream: "ALLOW:allow1"
    input_stream: "input1"
    output_stream: "output1"
  }
  node {
    calculator: "GateCalculator"
    input_stream: "ALLOW:allow2"
    input_stream: "input2"
    output_stream: "output2"
  }
  node {
    calculator: "MuxCalculator"
    input_stream: "INPUT:0:output0"
    input_stream: "INPUT:1:output1"
    input_stream: "INPUT:2:output2"
    input_stream: "SELECT:select"
    output_stream: "OUTPUT:output"
    input_stream_handler { input_stream_handler: "MuxInputStreamHandler" }
  })pb";

absl::Status SendInput(GateAndMuxGraphInput in, CalculatorGraph& graph) {
  MP_RETURN_IF_ERROR(graph.AddPacketToInputStream(
      "input0", MakePacket<int>(in.input0).At(in.at)));
  MP_RETURN_IF_ERROR(graph.AddPacketToInputStream(
      "input1", MakePacket<int>(in.input1).At(in.at)));
  MP_RETURN_IF_ERROR(graph.AddPacketToInputStream(
      "input2", MakePacket<int>(in.input2).At(in.at)));
  MP_RETURN_IF_ERROR(graph.AddPacketToInputStream(
      "select", MakePacket<int>(in.select).At(in.at)));
  MP_RETURN_IF_ERROR(graph.AddPacketToInputStream(
      "allow0", MakePacket<bool>(in.allow0).At(in.at)));
  MP_RETURN_IF_ERROR(graph.AddPacketToInputStream(
      "allow1", MakePacket<bool>(in.allow1).At(in.at)));
  MP_RETURN_IF_ERROR(graph.AddPacketToInputStream(
      "allow2", MakePacket<bool>(in.allow2).At(in.at)));
  return graph.WaitUntilIdle();
}

TEST(MuxInputStreamHandlerTest, BasicMuxing) {
  CalculatorGraphConfig config =
      mediapipe::ParseTextProtoOrDie<CalculatorGraphConfig>(kGateAndMuxGraph);
  std::vector<Packet> output_packets;
  tool::AddVectorSink("output", &config, &output_packets);

  CalculatorGraph graph;
  MP_ASSERT_OK(graph.Initialize(config));
  MP_ASSERT_OK(graph.StartRun({}));

  MP_ASSERT_OK(SendInput({.input0 = 1000,
                          .input1 = 900,
                          .input2 = 800,
                          .select = 0,
                          .allow0 = true,
                          .allow1 = false,
                          .allow2 = false,
                          .at = Timestamp(1)},
                         graph));
  EXPECT_THAT(output_packets, ElementsAre(IntPacket(1000, Timestamp(1))));

  MP_ASSERT_OK(SendInput({.input0 = 1000,
                          .input1 = 900,
                          .input2 = 800,
                          .select = 1,
                          .allow0 = false,
                          .allow1 = true,
                          .allow2 = false,
                          .at = Timestamp(2)},
                         graph));
  EXPECT_THAT(output_packets, ElementsAre(IntPacket(1000, Timestamp(1)),
                                          IntPacket(900, Timestamp(2))));

  MP_ASSERT_OK(SendInput({.input0 = 1000,
                          .input1 = 900,
                          .input2 = 800,
                          .select = 2,
                          .allow0 = false,
                          .allow1 = false,
                          .allow2 = true,
                          .at = Timestamp(3)},
                         graph));
  EXPECT_THAT(output_packets, ElementsAre(IntPacket(1000, Timestamp(1)),
                                          IntPacket(900, Timestamp(2)),
                                          IntPacket(800, Timestamp(3))));

  MP_ASSERT_OK(graph.CloseAllInputStreams());
  MP_ASSERT_OK(graph.WaitUntilDone());
}

TEST(MuxInputStreamHandlerTest, MuxingNonEmptyInputs) {
  CalculatorGraphConfig config =
      mediapipe::ParseTextProtoOrDie<CalculatorGraphConfig>(kGateAndMuxGraph);
  std::vector<Packet> output_packets;
  tool::AddVectorSink("output", &config, &output_packets);

  CalculatorGraph graph;
  MP_ASSERT_OK(graph.Initialize(config));
  MP_ASSERT_OK(graph.StartRun({}));

  MP_ASSERT_OK(SendInput({.input0 = 1000,
                          .input1 = 900,
                          .input2 = 800,
                          .select = 0,
                          .allow0 = true,
                          .allow1 = true,
                          .allow2 = true,
                          .at = Timestamp(1)},
                         graph));
  EXPECT_THAT(output_packets, ElementsAre(IntPacket(1000, Timestamp(1))));

  MP_ASSERT_OK(SendInput({.input0 = 1000,
                          .input1 = 900,
                          .input2 = 800,
                          .select = 1,
                          .allow0 = true,
                          .allow1 = true,
                          .allow2 = true,
                          .at = Timestamp(2)},
                         graph));
  EXPECT_THAT(output_packets, ElementsAre(IntPacket(1000, Timestamp(1)),
                                          IntPacket(900, Timestamp(2))));

  MP_ASSERT_OK(SendInput({.input0 = 1000,
                          .input1 = 900,
                          .input2 = 800,
                          .select = 2,
                          .allow0 = true,
                          .allow1 = true,
                          .allow2 = true,
                          .at = Timestamp(3)},
                         graph));
  EXPECT_THAT(output_packets, ElementsAre(IntPacket(1000, Timestamp(1)),
                                          IntPacket(900, Timestamp(2)),
                                          IntPacket(800, Timestamp(3))));

  MP_ASSERT_OK(graph.CloseAllInputStreams());
  MP_ASSERT_OK(graph.WaitUntilDone());
}

TEST(MuxInputStreamHandlerTest, MuxingAllTimestampBoundUpdates) {
  CalculatorGraphConfig config =
      mediapipe::ParseTextProtoOrDie<CalculatorGraphConfig>(kGateAndMuxGraph);
  std::vector<Packet> output_packets;
  tool::AddVectorSink("output", &config, &output_packets);

  CalculatorGraph graph;
  MP_ASSERT_OK(graph.Initialize(config));
  MP_ASSERT_OK(graph.StartRun({}));

  MP_ASSERT_OK(SendInput({.input0 = 1000,
                          .input1 = 900,
                          .input2 = 800,
                          .select = 0,
                          .allow0 = false,
                          .allow1 = false,
                          .allow2 = false,
                          .at = Timestamp(1)},
                         graph));
  EXPECT_TRUE(output_packets.empty());

  MP_ASSERT_OK(SendInput({.input0 = 1000,
                          .input1 = 900,
                          .input2 = 800,
                          .select = 1,
                          .allow0 = false,
                          .allow1 = false,
                          .allow2 = false,
                          .at = Timestamp(2)},
                         graph));
  EXPECT_TRUE(output_packets.empty());

  MP_ASSERT_OK(SendInput({.input0 = 1000,
                          .input1 = 900,
                          .input2 = 800,
                          .select = 2,
                          .allow0 = false,
                          .allow1 = false,
                          .allow2 = false,
                          .at = Timestamp(3)},
                         graph));
  EXPECT_TRUE(output_packets.empty());

  MP_ASSERT_OK(graph.CloseAllInputStreams());
  MP_ASSERT_OK(graph.WaitUntilDone());
}

TEST(MuxInputStreamHandlerTest, MuxingSlectedTimestampBoundUpdates) {
  CalculatorGraphConfig config =
      mediapipe::ParseTextProtoOrDie<CalculatorGraphConfig>(kGateAndMuxGraph);
  std::vector<Packet> output_packets;
  tool::AddVectorSink("output", &config, &output_packets);

  CalculatorGraph graph;
  MP_ASSERT_OK(graph.Initialize(config));
  MP_ASSERT_OK(graph.StartRun({}));

  MP_ASSERT_OK(SendInput({.input0 = 1000,
                          .input1 = 900,
                          .input2 = 800,
                          .select = 0,
                          .allow0 = false,
                          .allow1 = true,
                          .allow2 = true,
                          .at = Timestamp(1)},
                         graph));
  EXPECT_TRUE(output_packets.empty());

  MP_ASSERT_OK(SendInput({.input0 = 1000,
                          .input1 = 900,
                          .input2 = 800,
                          .select = 1,
                          .allow0 = true,
                          .allow1 = false,
                          .allow2 = true,
                          .at = Timestamp(2)},
                         graph));
  EXPECT_TRUE(output_packets.empty());

  MP_ASSERT_OK(SendInput({.input0 = 1000,
                          .input1 = 900,
                          .input2 = 800,
                          .select = 2,
                          .allow0 = true,
                          .allow1 = true,
                          .allow2 = false,
                          .at = Timestamp(3)},
                         graph));
  EXPECT_TRUE(output_packets.empty());

  MP_ASSERT_OK(graph.CloseAllInputStreams());
  MP_ASSERT_OK(graph.WaitUntilDone());
}

TEST(MuxInputStreamHandlerTest, MuxingSometimesTimestampBoundUpdates) {
  CalculatorGraphConfig config =
      mediapipe::ParseTextProtoOrDie<CalculatorGraphConfig>(kGateAndMuxGraph);
  std::vector<Packet> output_packets;
  tool::AddVectorSink("output", &config, &output_packets);

  CalculatorGraph graph;
  MP_ASSERT_OK(graph.Initialize(config));
  MP_ASSERT_OK(graph.StartRun({}));

  MP_ASSERT_OK(SendInput({.input0 = 1000,
                          .input1 = 900,
                          .input2 = 800,
                          .select = 0,
                          .allow0 = false,
                          .allow1 = false,
                          .allow2 = false,
                          .at = Timestamp(1)},
                         graph));
  EXPECT_TRUE(output_packets.empty());

  MP_ASSERT_OK(SendInput({.input0 = 1000,
                          .input1 = 900,
                          .input2 = 800,
                          .select = 1,
                          .allow0 = false,
                          .allow1 = true,
                          .allow2 = false,
                          .at = Timestamp(2)},
                         graph));
  EXPECT_THAT(output_packets, ElementsAre(IntPacket(900, Timestamp(2))));

  MP_ASSERT_OK(SendInput({.input0 = 1000,
                          .input1 = 900,
                          .input2 = 800,
                          .select = 2,
                          .allow0 = true,
                          .allow1 = true,
                          .allow2 = false,
                          .at = Timestamp(3)},
                         graph));
  EXPECT_THAT(output_packets, ElementsAre(IntPacket(900, Timestamp(2))));

  MP_ASSERT_OK(SendInput({.input0 = 1000,
                          .input1 = 900,
                          .input2 = 800,
                          .select = 2,
                          .allow0 = true,
                          .allow1 = true,
                          .allow2 = true,
                          .at = Timestamp(4)},
                         graph));
  EXPECT_THAT(output_packets, ElementsAre(IntPacket(900, Timestamp(2)),
                                          IntPacket(800, Timestamp(4))));

  MP_ASSERT_OK(SendInput({.input0 = 700,
                          .input1 = 600,
                          .input2 = 500,
                          .select = 0,
                          .allow0 = true,
                          .allow1 = false,
                          .allow2 = false,
                          .at = Timestamp(5)},
                         graph));
  EXPECT_THAT(output_packets, ElementsAre(IntPacket(900, Timestamp(2)),
                                          IntPacket(800, Timestamp(4)),
                                          IntPacket(700, Timestamp(5))));

  MP_ASSERT_OK(graph.CloseAllInputStreams());
  MP_ASSERT_OK(graph.WaitUntilDone());
}

MATCHER_P(EmptyPacket, ts, "") {
  return arg.IsEmpty() && arg.Timestamp() == ts;
}

MATCHER_P2(Pair, m1, m2, "") {
  const auto& p = arg.template Get<std::pair<Packet, Packet>>();
  return testing::Matches(m1)(p.first) && testing::Matches(m2)(p.second);
}

TEST(MuxInputStreamHandlerTest,
     TimestampBoundUpdateWhenControlPacketEarlierThanDataPacket) {
  CalculatorGraphConfig config =
      mediapipe::ParseTextProtoOrDie<CalculatorGraphConfig>(R"pb(
        input_stream: "input0"
        input_stream: "input1"
        input_stream: "select"
        node {
          calculator: "MuxCalculator"
          input_stream: "INPUT:0:input0"
          input_stream: "INPUT:1:input1"
          input_stream: "SELECT:select"
          output_stream: "OUTPUT:output"
          input_stream_handler { input_stream_handler: "MuxInputStreamHandler" }
        }
        node {
          calculator: "MakePairCalculator"
          input_stream: "select"
          input_stream: "output"
          output_stream: "pair"
        }
      )pb");
  std::vector<Packet> pair_packets;
  tool::AddVectorSink("pair", &config, &pair_packets);

  CalculatorGraph graph;
  MP_ASSERT_OK(graph.Initialize(config));
  MP_ASSERT_OK(graph.StartRun({}));

  MP_ASSERT_OK(graph.AddPacketToInputStream(
      "select", MakePacket<int>(0).At(Timestamp(1))));
  MP_ASSERT_OK(graph.WaitUntilIdle());
  EXPECT_TRUE(pair_packets.empty());

  MP_ASSERT_OK(graph.AddPacketToInputStream(
      "input0", MakePacket<int>(1000).At(Timestamp(2))));

  MP_ASSERT_OK(graph.WaitUntilIdle());
  EXPECT_THAT(pair_packets, ElementsAre(Pair(IntPacket(0, Timestamp(1)),
                                             EmptyPacket(Timestamp(1)))));

  MP_ASSERT_OK(graph.AddPacketToInputStream(
      "input1", MakePacket<int>(900).At(Timestamp(1))));
  MP_ASSERT_OK(graph.AddPacketToInputStream(
      "input1", MakePacket<int>(800).At(Timestamp(4))));

  MP_ASSERT_OK(graph.WaitUntilIdle());
  EXPECT_THAT(pair_packets, ElementsAre(Pair(IntPacket(0, Timestamp(1)),
                                             EmptyPacket(Timestamp(1)))));

  MP_ASSERT_OK(graph.AddPacketToInputStream(
      "select", MakePacket<int>(0).At(Timestamp(2))));
  MP_ASSERT_OK(graph.WaitUntilIdle());
  EXPECT_THAT(
      pair_packets,
      ElementsAre(
          Pair(IntPacket(0, Timestamp(1)), EmptyPacket(Timestamp(1))),
          Pair(IntPacket(0, Timestamp(2)), IntPacket(1000, Timestamp(2)))));

  MP_ASSERT_OK(graph.AddPacketToInputStream(
      "select", MakePacket<int>(1).At(Timestamp(3))));
  MP_ASSERT_OK(graph.WaitUntilIdle());
  EXPECT_THAT(
      pair_packets,
      ElementsAre(
          Pair(IntPacket(0, Timestamp(1)), EmptyPacket(Timestamp(1))),
          Pair(IntPacket(0, Timestamp(2)), IntPacket(1000, Timestamp(2))),
          Pair(IntPacket(1, Timestamp(3)), EmptyPacket(Timestamp(3)))));

  MP_ASSERT_OK(graph.AddPacketToInputStream(
      "select", MakePacket<int>(1).At(Timestamp(4))));
  MP_ASSERT_OK(graph.WaitUntilIdle());
  EXPECT_THAT(
      pair_packets,
      ElementsAre(
          Pair(IntPacket(0, Timestamp(1)), EmptyPacket(Timestamp(1))),
          Pair(IntPacket(0, Timestamp(2)), IntPacket(1000, Timestamp(2))),
          Pair(IntPacket(1, Timestamp(3)), EmptyPacket(Timestamp(3))),
          Pair(IntPacket(1, Timestamp(4)), IntPacket(800, Timestamp(4)))));

  MP_ASSERT_OK(graph.CloseAllInputStreams());
  MP_ASSERT_OK(graph.WaitUntilDone());
}

TEST(MuxInputStreamHandlerTest,
     TimestampBoundUpdateWhenControlPacketEarlierThanDataPacketPacketsAtOnce) {
  CalculatorGraphConfig config =
      mediapipe::ParseTextProtoOrDie<CalculatorGraphConfig>(R"pb(
        input_stream: "input0"
        input_stream: "input1"
        input_stream: "select"
        node {
          calculator: "MuxCalculator"
          input_stream: "INPUT:0:input0"
          input_stream: "INPUT:1:input1"
          input_stream: "SELECT:select"
          output_stream: "OUTPUT:output"
          input_stream_handler { input_stream_handler: "MuxInputStreamHandler" }
        }
        node {
          calculator: "MakePairCalculator"
          input_stream: "select"
          input_stream: "output"
          output_stream: "pair"
        }
      )pb");
  std::vector<Packet> pair_packets;
  tool::AddVectorSink("pair", &config, &pair_packets);

  CalculatorGraph graph;
  MP_ASSERT_OK(graph.Initialize(config));
  MP_ASSERT_OK(graph.StartRun({}));

  MP_ASSERT_OK(graph.AddPacketToInputStream(
      "select", MakePacket<int>(0).At(Timestamp(1))));
  MP_ASSERT_OK(graph.AddPacketToInputStream(
      "input0", MakePacket<int>(1000).At(Timestamp(2))));
  MP_ASSERT_OK(graph.AddPacketToInputStream(
      "input1", MakePacket<int>(900).At(Timestamp(1))));
  MP_ASSERT_OK(graph.AddPacketToInputStream(
      "input1", MakePacket<int>(800).At(Timestamp(4))));
  MP_ASSERT_OK(graph.AddPacketToInputStream(
      "select", MakePacket<int>(0).At(Timestamp(2))));
  MP_ASSERT_OK(graph.AddPacketToInputStream(
      "select", MakePacket<int>(1).At(Timestamp(3))));
  MP_ASSERT_OK(graph.AddPacketToInputStream(
      "select", MakePacket<int>(1).At(Timestamp(4))));
  MP_ASSERT_OK(graph.WaitUntilIdle());
  EXPECT_THAT(
      pair_packets,
      ElementsAre(
          Pair(IntPacket(0, Timestamp(1)), EmptyPacket(Timestamp(1))),
          Pair(IntPacket(0, Timestamp(2)), IntPacket(1000, Timestamp(2))),
          Pair(IntPacket(1, Timestamp(3)), EmptyPacket(Timestamp(3))),
          Pair(IntPacket(1, Timestamp(4)), IntPacket(800, Timestamp(4)))));

  MP_ASSERT_OK(graph.CloseAllInputStreams());
  MP_ASSERT_OK(graph.WaitUntilDone());
}

TEST(MuxInputStreamHandlerTest,
     TimestampBoundUpdateTriggersTimestampBoundUpdate) {
  CalculatorGraphConfig config =
      mediapipe::ParseTextProtoOrDie<CalculatorGraphConfig>(R"pb(
        input_stream: "input0"
        input_stream: "input1"
        input_stream: "select"
        input_stream: "allow0"
        input_stream: "allow1"
        node {
          calculator: "GateCalculator"
          input_stream: "ALLOW:allow0"
          input_stream: "input0"
          output_stream: "output0"
        }
        node {
          calculator: "GateCalculator"
          input_stream: "ALLOW:allow1"
          input_stream: "input1"
          output_stream: "output1"
        }
        node {
          calculator: "MuxCalculator"
          input_stream: "INPUT:0:output0"
          input_stream: "INPUT:1:output1"
          input_stream: "SELECT:select"
          output_stream: "OUTPUT:output"
          input_stream_handler { input_stream_handler: "MuxInputStreamHandler" }
        }
        node {
          calculator: "MakePairCalculator"
          input_stream: "select"
          input_stream: "output"
          output_stream: "pair"
        }
      )pb");
  std::vector<Packet> pair_packets;
  tool::AddVectorSink("pair", &config, &pair_packets);

  CalculatorGraph graph;
  MP_ASSERT_OK(graph.Initialize(config));
  MP_ASSERT_OK(graph.StartRun({}));

  MP_ASSERT_OK(graph.AddPacketToInputStream(
      "select", MakePacket<int>(0).At(Timestamp(1))));
  MP_ASSERT_OK(graph.AddPacketToInputStream(
      "input0", MakePacket<int>(1000).At(Timestamp(1))));
  MP_ASSERT_OK(graph.AddPacketToInputStream(
      "allow0", MakePacket<bool>(false).At(Timestamp(1))));

  MP_ASSERT_OK(graph.WaitUntilIdle());
  EXPECT_THAT(pair_packets, ElementsAre(Pair(IntPacket(0, Timestamp(1)),
                                             EmptyPacket(Timestamp(1)))));

  MP_ASSERT_OK(graph.AddPacketToInputStream(
      "select", MakePacket<int>(0).At(Timestamp(2))));
  MP_ASSERT_OK(graph.AddPacketToInputStream(
      "input0", MakePacket<int>(900).At(Timestamp(2))));
  MP_ASSERT_OK(graph.AddPacketToInputStream(
      "allow0", MakePacket<bool>(true).At(Timestamp(2))));

  MP_ASSERT_OK(graph.WaitUntilIdle());
  EXPECT_THAT(
      pair_packets,
      ElementsAre(
          Pair(IntPacket(0, Timestamp(1)), EmptyPacket(Timestamp(1))),
          Pair(IntPacket(0, Timestamp(2)), IntPacket(900, Timestamp(2)))));

  MP_ASSERT_OK(graph.CloseAllInputStreams());
  MP_ASSERT_OK(graph.WaitUntilDone());
}

}  // namespace
}  // namespace mediapipe
