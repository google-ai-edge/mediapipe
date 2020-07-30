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

#include <algorithm>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/calculator_runner.h"
#include "mediapipe/framework/port/gmock.h"
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/framework/port/integral_types.h"
#include "mediapipe/framework/port/parse_text_proto.h"
#include "mediapipe/framework/port/status.h"
#include "mediapipe/framework/port/status_matchers.h"
#include "mediapipe/framework/timestamp.h"
#include "mediapipe/framework/tool/sink.h"

namespace mediapipe {

using ::testing::ElementsAre;
using ::testing::Eq;
using ::testing::Pair;
using ::testing::Value;
namespace {

// Returns the timestamp values for a vector of Packets.
// TODO: puth this kind of test util in a common place.
std::vector<int64> TimestampValues(const std::vector<Packet>& packets) {
  std::vector<int64> result;
  for (const Packet& packet : packets) {
    result.push_back(packet.Timestamp().Value());
  }
  return result;
}

MATCHER(EmptyPacket, negation ? "isn't empty" : "is empty") {
  if (arg.IsEmpty()) {
    return true;
  }
  return false;
}

MATCHER_P(IntPacket, value, "") {
  return Value(arg.template Get<int>(), Eq(value));
}

MATCHER_P2(PairPacket, timestamp, pair, "") {
  Timestamp actual_timestamp = arg.Timestamp();
  const auto& actual_pair = arg.template Get<std::pair<Packet, Packet>>();
  return Value(actual_timestamp, Eq(timestamp)) && Value(actual_pair, pair);
}

TEST(PreviousLoopbackCalculator, CorrectTimestamps) {
  std::vector<Packet> in_prev;
  CalculatorGraphConfig graph_config_ =
      ParseTextProtoOrDie<CalculatorGraphConfig>(R"(
        input_stream: 'in'
        node {
          calculator: 'PreviousLoopbackCalculator'
          input_stream: 'MAIN:in'
          input_stream: 'LOOP:out'
          input_stream_info: { tag_index: 'LOOP' back_edge: true }
          output_stream: 'PREV_LOOP:previous'
        }
        # This calculator synchronizes its inputs as normal, so it is used
        # to check that both "in" and "previous" are ready.
        node {
          calculator: 'PassThroughCalculator'
          input_stream: 'in'
          input_stream: 'previous'
          output_stream: 'out'
          output_stream: 'previous2'
        }
        node {
          calculator: 'MakePairCalculator'
          input_stream: 'out'
          input_stream: 'previous2'
          output_stream: 'pair'
        }
      )");
  tool::AddVectorSink("pair", &graph_config_, &in_prev);

  CalculatorGraph graph_;
  MP_ASSERT_OK(graph_.Initialize(graph_config_, {}));
  MP_ASSERT_OK(graph_.StartRun({}));

  auto send_packet = [&graph_](const std::string& input_name, int n) {
    MP_EXPECT_OK(graph_.AddPacketToInputStream(
        input_name, MakePacket<int>(n).At(Timestamp(n))));
  };

  send_packet("in", 1);
  MP_EXPECT_OK(graph_.WaitUntilIdle());
  EXPECT_THAT(TimestampValues(in_prev), ElementsAre(1));
  EXPECT_THAT(in_prev.back(),
              PairPacket(Timestamp(1), Pair(IntPacket(1), EmptyPacket())));

  send_packet("in", 2);
  MP_EXPECT_OK(graph_.WaitUntilIdle());
  EXPECT_THAT(TimestampValues(in_prev), ElementsAre(1, 2));
  EXPECT_THAT(in_prev.back(),
              PairPacket(Timestamp(2), Pair(IntPacket(2), IntPacket(1))));

  send_packet("in", 5);
  MP_EXPECT_OK(graph_.WaitUntilIdle());
  EXPECT_THAT(TimestampValues(in_prev), ElementsAre(1, 2, 5));
  EXPECT_THAT(in_prev.back(),
              PairPacket(Timestamp(5), Pair(IntPacket(5), IntPacket(2))));

  send_packet("in", 15);
  MP_EXPECT_OK(graph_.WaitUntilIdle());
  EXPECT_THAT(TimestampValues(in_prev), ElementsAre(1, 2, 5, 15));
  EXPECT_THAT(in_prev.back(),
              PairPacket(Timestamp(15), Pair(IntPacket(15), IntPacket(5))));

  MP_EXPECT_OK(graph_.CloseAllInputStreams());
  MP_EXPECT_OK(graph_.WaitUntilDone());
}

// A Calculator that outputs a summary packet in CalculatorBase::Close().
class PacketOnCloseCalculator : public CalculatorBase {
 public:
  static ::mediapipe::Status GetContract(CalculatorContract* cc) {
    cc->Inputs().Index(0).Set<int>();
    cc->Outputs().Index(0).Set<int>();
    return ::mediapipe::OkStatus();
  }

  ::mediapipe::Status Open(CalculatorContext* cc) final {
    cc->SetOffset(TimestampDiff(0));
    return ::mediapipe::OkStatus();
  }

  ::mediapipe::Status Process(CalculatorContext* cc) final {
    sum_ += cc->Inputs().Index(0).Value().Get<int>();
    cc->Outputs().Index(0).AddPacket(cc->Inputs().Index(0).Value());
    return ::mediapipe::OkStatus();
  }

  ::mediapipe::Status Close(CalculatorContext* cc) final {
    cc->Outputs().Index(0).AddPacket(
        MakePacket<int>(sum_).At(Timestamp::Max()));
    return ::mediapipe::OkStatus();
  }

 private:
  int sum_ = 0;
};
REGISTER_CALCULATOR(PacketOnCloseCalculator);

// Demonstrates that all ouput and input streams in PreviousLoopbackCalculator
// will close as expected when all graph input streams are closed.
TEST(PreviousLoopbackCalculator, ClosesCorrectly) {
  std::vector<Packet> outputs;
  CalculatorGraphConfig graph_config_ =
      ParseTextProtoOrDie<CalculatorGraphConfig>(R"(
        input_stream: 'in'
        node {
          calculator: 'PreviousLoopbackCalculator'
          input_stream: 'MAIN:in'
          input_stream: 'LOOP:out'
          input_stream_info: { tag_index: 'LOOP' back_edge: true }
          output_stream: 'PREV_LOOP:previous'
        }
        # This calculator synchronizes its inputs as normal, so it is used
        # to check that both "in" and "previous" are ready.
        node {
          calculator: 'PassThroughCalculator'
          input_stream: 'in'
          input_stream: 'previous'
          output_stream: 'out'
          output_stream: 'previous2'
        }
        node {
          calculator: 'PacketOnCloseCalculator'
          input_stream: 'out'
          output_stream: 'close_out'
        }
      )");
  tool::AddVectorSink("close_out", &graph_config_, &outputs);

  CalculatorGraph graph_;
  MP_ASSERT_OK(graph_.Initialize(graph_config_, {}));
  MP_ASSERT_OK(graph_.StartRun({}));

  auto send_packet = [&graph_](const std::string& input_name, int n) {
    MP_EXPECT_OK(graph_.AddPacketToInputStream(
        input_name, MakePacket<int>(n).At(Timestamp(n))));
  };

  send_packet("in", 1);
  MP_EXPECT_OK(graph_.WaitUntilIdle());
  EXPECT_THAT(TimestampValues(outputs), ElementsAre(1));

  send_packet("in", 2);
  MP_EXPECT_OK(graph_.WaitUntilIdle());
  EXPECT_THAT(TimestampValues(outputs), ElementsAre(1, 2));

  send_packet("in", 5);
  MP_EXPECT_OK(graph_.WaitUntilIdle());
  EXPECT_THAT(TimestampValues(outputs), ElementsAre(1, 2, 5));

  send_packet("in", 15);
  MP_EXPECT_OK(graph_.WaitUntilIdle());
  EXPECT_THAT(TimestampValues(outputs), ElementsAre(1, 2, 5, 15));

  MP_EXPECT_OK(graph_.CloseAllInputStreams());
  MP_EXPECT_OK(graph_.WaitUntilIdle());
  EXPECT_THAT(TimestampValues(outputs),
              ElementsAre(1, 2, 5, 15, Timestamp::Max().Value()));

  MP_EXPECT_OK(graph_.WaitUntilDone());
}

TEST(PreviousLoopbackCalculator, ProcessesMaxTimestamp) {
  std::vector<Packet> out_and_previous_packets;
  CalculatorGraphConfig graph_config =
      ParseTextProtoOrDie<CalculatorGraphConfig>(R"(
        input_stream: 'in'
        node {
          calculator: 'PreviousLoopbackCalculator'
          input_stream: 'MAIN:in'
          input_stream: 'LOOP:out'
          input_stream_info: { tag_index: 'LOOP' back_edge: true }
          output_stream: 'PREV_LOOP:previous'
        }
        node {
          calculator: 'PassThroughCalculator'
          input_stream: 'in'
          input_stream: 'previous'
          output_stream: 'out'
          output_stream: 'previous2'
        }
        node {
          calculator: 'MakePairCalculator'
          input_stream: 'out'
          input_stream: 'previous'
          output_stream: 'out_and_previous'
        }
      )");
  tool::AddVectorSink("out_and_previous", &graph_config,
                      &out_and_previous_packets);

  CalculatorGraph graph;
  MP_ASSERT_OK(graph.Initialize(graph_config, {}));
  MP_ASSERT_OK(graph.StartRun({}));

  MP_EXPECT_OK(graph.AddPacketToInputStream(
      "in", MakePacket<int>(1).At(Timestamp::Max())));

  MP_EXPECT_OK(graph.WaitUntilIdle());

  EXPECT_THAT(out_and_previous_packets,
              ElementsAre(PairPacket(Timestamp::Max(),
                                     Pair(IntPacket(1), EmptyPacket()))));

  MP_EXPECT_OK(graph.CloseAllInputStreams());
  MP_EXPECT_OK(graph.WaitUntilIdle());
  MP_EXPECT_OK(graph.WaitUntilDone());
}

TEST(PreviousLoopbackCalculator, ProcessesMaxTimestampNonEmptyPrevious) {
  std::vector<Packet> out_and_previous_packets;
  CalculatorGraphConfig graph_config =
      ParseTextProtoOrDie<CalculatorGraphConfig>(R"(
        input_stream: 'in'
        node {
          calculator: 'PreviousLoopbackCalculator'
          input_stream: 'MAIN:in'
          input_stream: 'LOOP:out'
          input_stream_info: { tag_index: 'LOOP' back_edge: true }
          output_stream: 'PREV_LOOP:previous'
        }
        node {
          calculator: 'PassThroughCalculator'
          input_stream: 'in'
          input_stream: 'previous'
          output_stream: 'out'
          output_stream: 'previous2'
        }
        node {
          calculator: 'MakePairCalculator'
          input_stream: 'out'
          input_stream: 'previous'
          output_stream: 'out_and_previous'
        }
      )");
  tool::AddVectorSink("out_and_previous", &graph_config,
                      &out_and_previous_packets);

  CalculatorGraph graph;
  MP_ASSERT_OK(graph.Initialize(graph_config, {}));
  MP_ASSERT_OK(graph.StartRun({}));

  MP_EXPECT_OK(graph.AddPacketToInputStream(
      "in", MakePacket<int>(1).At(Timestamp::Min())));
  MP_EXPECT_OK(graph.AddPacketToInputStream(
      "in", MakePacket<int>(2).At(Timestamp::Max())));

  MP_EXPECT_OK(graph.WaitUntilIdle());

  EXPECT_THAT(
      out_and_previous_packets,
      ElementsAre(
          PairPacket(Timestamp::Min(), Pair(IntPacket(1), EmptyPacket())),
          PairPacket(Timestamp::Max(), Pair(IntPacket(2), IntPacket(1)))));

  MP_EXPECT_OK(graph.CloseAllInputStreams());
  MP_EXPECT_OK(graph.WaitUntilIdle());
  MP_EXPECT_OK(graph.WaitUntilDone());
}

// Demonstrates that downstream calculators won't be blocked by
// always-empty-LOOP-stream.
TEST(PreviousLoopbackCalculator, EmptyLoopForever) {
  std::vector<Packet> outputs;
  CalculatorGraphConfig graph_config_ =
      ParseTextProtoOrDie<CalculatorGraphConfig>(R"(
        input_stream: 'in'
        node {
          calculator: 'PreviousLoopbackCalculator'
          input_stream: 'MAIN:in'
          input_stream: 'LOOP:previous'
          input_stream_info: { tag_index: 'LOOP' back_edge: true }
          output_stream: 'PREV_LOOP:previous'
        }
        # This calculator synchronizes its inputs as normal, so it is used
        # to check that both "in" and "previous" are ready.
        node {
          calculator: 'PassThroughCalculator'
          input_stream: 'in'
          input_stream: 'previous'
          output_stream: 'out'
          output_stream: 'previous2'
        }
        node {
          calculator: 'PacketOnCloseCalculator'
          input_stream: 'out'
          output_stream: 'close_out'
        }
      )");
  tool::AddVectorSink("close_out", &graph_config_, &outputs);

  CalculatorGraph graph_;
  MP_ASSERT_OK(graph_.Initialize(graph_config_, {}));
  MP_ASSERT_OK(graph_.StartRun({}));

  auto send_packet = [&graph_](const std::string& input_name, int n) {
    MP_EXPECT_OK(graph_.AddPacketToInputStream(
        input_name, MakePacket<int>(n).At(Timestamp(n))));
  };

  for (int main_ts = 0; main_ts < 50; ++main_ts) {
    send_packet("in", main_ts);
    MP_EXPECT_OK(graph_.WaitUntilIdle());
    std::vector<int64> ts_values = TimestampValues(outputs);
    EXPECT_EQ(ts_values.size(), main_ts + 1);
    for (int j = 0; j < main_ts + 1; ++j) {
      EXPECT_EQ(ts_values[j], j);
    }
  }

  MP_EXPECT_OK(graph_.CloseAllInputStreams());
  MP_EXPECT_OK(graph_.WaitUntilIdle());
  MP_EXPECT_OK(graph_.WaitUntilDone());
}

class PreviousLoopbackCalculatorProcessingTimestampsTest
    : public testing::Test {
 protected:
  void SetUp() override {
    CalculatorGraphConfig graph_config =
        ParseTextProtoOrDie<CalculatorGraphConfig>(R"(
          input_stream: 'input'
          input_stream: 'force_main_empty'
          input_stream: 'force_loop_empty'
          # Used to indicate "main" timestamp bound updates.
          node {
            calculator: 'GateCalculator'
            input_stream: 'input'
            input_stream: 'DISALLOW:force_main_empty'
            output_stream: 'main'
          }
          node {
            calculator: 'PreviousLoopbackCalculator'
            input_stream: 'MAIN:main'
            input_stream: 'LOOP:loop'
            input_stream_info: { tag_index: 'LOOP' back_edge: true }
            output_stream: 'PREV_LOOP:prev_loop'
          }
          node {
            calculator: 'PassThroughCalculator'
            input_stream: 'input'
            input_stream: 'prev_loop'
            output_stream: 'passed_through_input'
            output_stream: 'passed_through_prev_loop'
          }
          # Used to indicate "loop" timestamp bound updates.
          node {
            calculator: 'GateCalculator'
            input_stream: 'input'
            input_stream: 'DISALLOW:force_loop_empty'
            output_stream: 'loop'
          }
          node {
            calculator: 'MakePairCalculator'
            input_stream: 'passed_through_input'
            input_stream: 'passed_through_prev_loop'
            output_stream: 'passed_through_input_and_prev_loop'
          }
        )");
    tool::AddVectorSink("passed_through_input_and_prev_loop", &graph_config,
                        &output_packets_);
    MP_ASSERT_OK(graph_.Initialize(graph_config, {}));
    MP_ASSERT_OK(graph_.StartRun({}));
  }

  void SendPackets(int timestamp, int input, bool force_main_empty,
                   bool force_loop_empty) {
    MP_ASSERT_OK(graph_.AddPacketToInputStream(
        "input", MakePacket<int>(input).At(Timestamp(timestamp))));
    MP_ASSERT_OK(graph_.AddPacketToInputStream(
        "force_main_empty",
        MakePacket<bool>(force_main_empty).At(Timestamp(timestamp))));
    MP_ASSERT_OK(graph_.AddPacketToInputStream(
        "force_loop_empty",
        MakePacket<bool>(force_loop_empty).At(Timestamp(timestamp))));
  }

  CalculatorGraph graph_;
  std::vector<Packet> output_packets_;
};

TEST_F(PreviousLoopbackCalculatorProcessingTimestampsTest,
       MultiplePacketsEmptyMainNonEmptyLoop) {
  SendPackets(/*timestamp=*/1, /*input=*/1, /*force_main_empty=*/true,
              /*force_loop_empty=*/false);
  MP_EXPECT_OK(graph_.WaitUntilIdle());
  EXPECT_THAT(
      output_packets_,
      ElementsAre(PairPacket(Timestamp(1), Pair(IntPacket(1), EmptyPacket()))));

  SendPackets(/*timestamp=*/2, /*input=*/2, /*force_main_empty=*/true,
              /*force_loop_empty=*/false);
  MP_EXPECT_OK(graph_.WaitUntilIdle());
  EXPECT_THAT(
      output_packets_,
      ElementsAre(PairPacket(Timestamp(1), Pair(IntPacket(1), EmptyPacket())),
                  PairPacket(Timestamp(2), Pair(IntPacket(2), EmptyPacket()))));

  SendPackets(/*timestamp=*/3, /*input=*/3, /*force_main_empty=*/true,
              /*force_loop_empty=*/false);
  MP_EXPECT_OK(graph_.WaitUntilIdle());
  EXPECT_THAT(
      output_packets_,
      ElementsAre(PairPacket(Timestamp(1), Pair(IntPacket(1), EmptyPacket())),
                  PairPacket(Timestamp(2), Pair(IntPacket(2), EmptyPacket())),
                  PairPacket(Timestamp(3), Pair(IntPacket(3), EmptyPacket()))));

  SendPackets(/*timestamp=*/5, /*input=*/5, /*force_main_empty=*/true,
              /*force_loop_empty=*/false);
  MP_EXPECT_OK(graph_.WaitUntilIdle());
  EXPECT_THAT(
      output_packets_,
      ElementsAre(PairPacket(Timestamp(1), Pair(IntPacket(1), EmptyPacket())),
                  PairPacket(Timestamp(2), Pair(IntPacket(2), EmptyPacket())),
                  PairPacket(Timestamp(3), Pair(IntPacket(3), EmptyPacket())),
                  PairPacket(Timestamp(5), Pair(IntPacket(5), EmptyPacket()))));

  SendPackets(/*timestamp=*/15, /*input=*/15,
              /*force_main_empty=*/true,
              /*force_loop_empty=*/false);
  MP_EXPECT_OK(graph_.WaitUntilIdle());
  EXPECT_THAT(
      output_packets_,
      ElementsAre(
          PairPacket(Timestamp(1), Pair(IntPacket(1), EmptyPacket())),
          PairPacket(Timestamp(2), Pair(IntPacket(2), EmptyPacket())),
          PairPacket(Timestamp(3), Pair(IntPacket(3), EmptyPacket())),
          PairPacket(Timestamp(5), Pair(IntPacket(5), EmptyPacket())),
          PairPacket(Timestamp(15), Pair(IntPacket(15), EmptyPacket()))));

  MP_EXPECT_OK(graph_.CloseAllInputStreams());
  MP_EXPECT_OK(graph_.WaitUntilDone());
}

TEST_F(PreviousLoopbackCalculatorProcessingTimestampsTest,
       MultiplePacketsNonEmptyMainEmptyLoop) {
  SendPackets(/*timestamp=*/1, /*input=*/1,
              /*force_main_empty=*/false,
              /*force_loop_empty=*/true);
  MP_EXPECT_OK(graph_.WaitUntilIdle());
  EXPECT_THAT(
      output_packets_,
      ElementsAre(PairPacket(Timestamp(1), Pair(IntPacket(1), EmptyPacket()))));

  SendPackets(/*timestamp=*/2, /*input=*/2,
              /*force_main_empty=*/false,
              /*force_loop_empty=*/true);
  MP_EXPECT_OK(graph_.WaitUntilIdle());
  EXPECT_THAT(
      output_packets_,
      ElementsAre(PairPacket(Timestamp(1), Pair(IntPacket(1), EmptyPacket())),
                  PairPacket(Timestamp(2), Pair(IntPacket(2), EmptyPacket()))));

  SendPackets(/*timestamp=*/3, /*input=*/3,
              /*force_main_empty=*/false,
              /*force_loop_empty=*/true);
  MP_EXPECT_OK(graph_.WaitUntilIdle());
  EXPECT_THAT(
      output_packets_,
      ElementsAre(PairPacket(Timestamp(1), Pair(IntPacket(1), EmptyPacket())),
                  PairPacket(Timestamp(2), Pair(IntPacket(2), EmptyPacket())),
                  PairPacket(Timestamp(3), Pair(IntPacket(3), EmptyPacket()))));

  SendPackets(/*timestamp=*/5, /*input=*/5,
              /*force_main_empty=*/false,
              /*force_loop_empty=*/true);
  MP_EXPECT_OK(graph_.WaitUntilIdle());
  EXPECT_THAT(
      output_packets_,
      ElementsAre(PairPacket(Timestamp(1), Pair(IntPacket(1), EmptyPacket())),
                  PairPacket(Timestamp(2), Pair(IntPacket(2), EmptyPacket())),
                  PairPacket(Timestamp(3), Pair(IntPacket(3), EmptyPacket())),
                  PairPacket(Timestamp(5), Pair(IntPacket(5), EmptyPacket()))));

  SendPackets(/*timestamp=*/15, /*input=*/15,
              /*force_main_empty=*/false,
              /*force_loop_empty=*/true);
  MP_EXPECT_OK(graph_.WaitUntilIdle());
  EXPECT_THAT(
      output_packets_,
      ElementsAre(
          PairPacket(Timestamp(1), Pair(IntPacket(1), EmptyPacket())),
          PairPacket(Timestamp(2), Pair(IntPacket(2), EmptyPacket())),
          PairPacket(Timestamp(3), Pair(IntPacket(3), EmptyPacket())),
          PairPacket(Timestamp(5), Pair(IntPacket(5), EmptyPacket())),
          PairPacket(Timestamp(15), Pair(IntPacket(15), EmptyPacket()))));

  MP_EXPECT_OK(graph_.CloseAllInputStreams());
  MP_EXPECT_OK(graph_.WaitUntilDone());
}

TEST_F(PreviousLoopbackCalculatorProcessingTimestampsTest,
       MultiplePacketsAlteringMainNonEmptyLoop) {
  SendPackets(/*timestamp=*/1, /*input=*/1,
              /*force_main_empty=*/false,
              /*force_loop_empty=*/false);
  MP_EXPECT_OK(graph_.WaitUntilIdle());
  EXPECT_THAT(
      output_packets_,
      ElementsAre(PairPacket(Timestamp(1), Pair(IntPacket(1), EmptyPacket()))));

  SendPackets(/*timestamp=*/2, /*input=*/2, /*force_main_empty=*/true,
              /*force_loop_empty=*/false);
  MP_EXPECT_OK(graph_.WaitUntilIdle());
  EXPECT_THAT(
      output_packets_,
      ElementsAre(PairPacket(Timestamp(1), Pair(IntPacket(1), EmptyPacket())),
                  PairPacket(Timestamp(2), Pair(IntPacket(2), EmptyPacket()))));

  SendPackets(/*timestamp=*/3, /*input=*/3,
              /*force_main_empty=*/false,
              /*force_loop_empty=*/false);
  MP_EXPECT_OK(graph_.WaitUntilIdle());
  EXPECT_THAT(
      output_packets_,
      ElementsAre(PairPacket(Timestamp(1), Pair(IntPacket(1), EmptyPacket())),
                  PairPacket(Timestamp(2), Pair(IntPacket(2), EmptyPacket())),
                  PairPacket(Timestamp(3), Pair(IntPacket(3), IntPacket(1)))));

  SendPackets(/*timestamp=*/5, /*input=*/5,
              /*force_main_empty=*/false,
              /*force_loop_empty=*/false);
  MP_EXPECT_OK(graph_.WaitUntilIdle());
  EXPECT_THAT(
      output_packets_,
      ElementsAre(PairPacket(Timestamp(1), Pair(IntPacket(1), EmptyPacket())),
                  PairPacket(Timestamp(2), Pair(IntPacket(2), EmptyPacket())),
                  PairPacket(Timestamp(3), Pair(IntPacket(3), IntPacket(1))),
                  PairPacket(Timestamp(5), Pair(IntPacket(5), IntPacket(3)))));

  SendPackets(/*timestamp=*/15, /*input=*/15,
              /*force_main_empty=*/true,
              /*force_loop_empty=*/false);
  MP_EXPECT_OK(graph_.WaitUntilIdle());
  EXPECT_THAT(
      output_packets_,
      ElementsAre(
          PairPacket(Timestamp(1), Pair(IntPacket(1), EmptyPacket())),
          PairPacket(Timestamp(2), Pair(IntPacket(2), EmptyPacket())),
          PairPacket(Timestamp(3), Pair(IntPacket(3), IntPacket(1))),
          PairPacket(Timestamp(5), Pair(IntPacket(5), IntPacket(3))),
          PairPacket(Timestamp(15), Pair(IntPacket(15), EmptyPacket()))));

  MP_EXPECT_OK(graph_.CloseAllInputStreams());
  MP_EXPECT_OK(graph_.WaitUntilDone());
}

TEST_F(PreviousLoopbackCalculatorProcessingTimestampsTest,
       MultiplePacketsNonEmptyMainAlteringLoop) {
  SendPackets(/*timestamp=*/1, /*input=*/1,
              /*force_main_empty=*/false,
              /*force_loop_empty=*/false);
  MP_EXPECT_OK(graph_.WaitUntilIdle());
  EXPECT_THAT(
      output_packets_,
      ElementsAre(PairPacket(Timestamp(1), Pair(IntPacket(1), EmptyPacket()))));

  SendPackets(/*timestamp=*/2, /*input=*/2,
              /*force_main_empty=*/false,
              /*force_loop_empty=*/true);
  MP_EXPECT_OK(graph_.WaitUntilIdle());
  EXPECT_THAT(
      output_packets_,
      ElementsAre(PairPacket(Timestamp(1), Pair(IntPacket(1), EmptyPacket())),
                  PairPacket(Timestamp(2), Pair(IntPacket(2), IntPacket(1)))));

  SendPackets(/*timestamp=*/3, /*input=*/3,
              /*force_main_empty=*/false,
              /*force_loop_empty=*/false);
  MP_EXPECT_OK(graph_.WaitUntilIdle());
  EXPECT_THAT(
      output_packets_,
      ElementsAre(PairPacket(Timestamp(1), Pair(IntPacket(1), EmptyPacket())),
                  PairPacket(Timestamp(2), Pair(IntPacket(2), IntPacket(1))),
                  PairPacket(Timestamp(3), Pair(IntPacket(3), EmptyPacket()))));

  SendPackets(/*timestamp=*/5, /*input=*/5,
              /*force_main_empty=*/false,
              /*force_loop_empty=*/true);
  MP_EXPECT_OK(graph_.WaitUntilIdle());
  EXPECT_THAT(
      output_packets_,
      ElementsAre(PairPacket(Timestamp(1), Pair(IntPacket(1), EmptyPacket())),
                  PairPacket(Timestamp(2), Pair(IntPacket(2), IntPacket(1))),
                  PairPacket(Timestamp(3), Pair(IntPacket(3), EmptyPacket())),
                  PairPacket(Timestamp(5), Pair(IntPacket(5), IntPacket(3)))));

  SendPackets(/*timestamp=*/15, /*input=*/15,
              /*force_main_empty=*/false,
              /*force_loop_empty=*/false);
  MP_EXPECT_OK(graph_.WaitUntilIdle());
  EXPECT_THAT(
      output_packets_,
      ElementsAre(
          PairPacket(Timestamp(1), Pair(IntPacket(1), EmptyPacket())),
          PairPacket(Timestamp(2), Pair(IntPacket(2), IntPacket(1))),
          PairPacket(Timestamp(3), Pair(IntPacket(3), EmptyPacket())),
          PairPacket(Timestamp(5), Pair(IntPacket(5), IntPacket(3))),
          PairPacket(Timestamp(15), Pair(IntPacket(15), EmptyPacket()))));

  MP_EXPECT_OK(graph_.CloseAllInputStreams());
  MP_EXPECT_OK(graph_.WaitUntilDone());
}

TEST_F(PreviousLoopbackCalculatorProcessingTimestampsTest,
       MultiplePacketsCheckIfLastCorrectAlteringMainAlteringLoop) {
  int num_packets = 1000;
  for (int i = 0; i < num_packets; ++i) {
    bool force_main_empty = i % 3 == 0 ? true : false;
    bool force_loop_empty = i % 2 == 0 ? true : false;
    SendPackets(/*timestamp=*/i + 1, /*input=*/i + 1, force_main_empty,
                force_loop_empty);
  }
  SendPackets(/*timestamp=*/num_packets + 1,
              /*input=*/num_packets + 1, /*force_main_empty=*/false,
              /*force_loop_empty=*/false);
  SendPackets(/*timestamp=*/num_packets + 2,
              /*input=*/num_packets + 2, /*force_main_empty=*/false,
              /*force_loop_empty=*/false);

  MP_EXPECT_OK(graph_.WaitUntilIdle());
  ASSERT_FALSE(output_packets_.empty());
  EXPECT_THAT(
      output_packets_.back(),
      PairPacket(Timestamp(num_packets + 2),
                 Pair(IntPacket(num_packets + 2), IntPacket(num_packets + 1))));

  MP_EXPECT_OK(graph_.CloseAllInputStreams());
  MP_EXPECT_OK(graph_.WaitUntilDone());
}

// Similar to GateCalculator, but it doesn't propagate timestamp bound updates.
class DroppingGateCalculator : public CalculatorBase {
 public:
  static ::mediapipe::Status GetContract(CalculatorContract* cc) {
    cc->Inputs().Index(0).SetAny();
    cc->Inputs().Tag("DISALLOW").Set<bool>();
    cc->Outputs().Index(0).SetSameAs(&cc->Inputs().Index(0));
    return ::mediapipe::OkStatus();
  }

  ::mediapipe::Status Process(CalculatorContext* cc) final {
    if (!cc->Inputs().Index(0).IsEmpty() &&
        !cc->Inputs().Tag("DISALLOW").Get<bool>()) {
      cc->Outputs().Index(0).AddPacket(cc->Inputs().Index(0).Value());
    }
    return ::mediapipe::OkStatus();
  }
};
REGISTER_CALCULATOR(DroppingGateCalculator);

// Tests PreviousLoopbackCalculator in cases when there are no "LOOP" timestamp
// bound updates and non-empty packets for a while and the aforementioned start
// to arrive at some point. So, "PREV_LOOP" is delayed for a couple of inputs.
class PreviousLoopbackCalculatorDelayBehaviorTest : public testing::Test {
 protected:
  void SetUp() override {
    CalculatorGraphConfig graph_config =
        ParseTextProtoOrDie<CalculatorGraphConfig>(R"(
          input_stream: 'input'
          # Drops "loop" when set to "true", delaying output of prev_loop, hence
          # delaying output of the graph.
          input_stream: 'delay_next_output'
          node {
            calculator: 'PreviousLoopbackCalculator'
            input_stream: 'MAIN:input'
            input_stream: 'LOOP:loop'
            input_stream_info: { tag_index: 'LOOP' back_edge: true }
            output_stream: 'PREV_LOOP:prev_loop'
          }
          node {
            calculator: 'PassThroughCalculator'
            input_stream: 'input'
            input_stream: 'prev_loop'
            output_stream: 'passed_through_input'
            output_stream: 'passed_through_prev_loop'
          }
          node {
            calculator: 'DroppingGateCalculator'
            input_stream: 'input'
            input_stream: 'DISALLOW:delay_next_output'
            output_stream: 'loop'
          }
          node {
            calculator: 'MakePairCalculator'
            input_stream: 'passed_through_input'
            input_stream: 'passed_through_prev_loop'
            output_stream: 'passed_through_input_and_prev_loop'
          }
        )");
    tool::AddVectorSink("passed_through_input_and_prev_loop", &graph_config,
                        &output_packets_);
    MP_ASSERT_OK(graph_.Initialize(graph_config, {}));
    MP_ASSERT_OK(graph_.StartRun({}));
  }

  void SendPackets(int timestamp, int input, bool delay_next_output) {
    MP_ASSERT_OK(graph_.AddPacketToInputStream(
        "input", MakePacket<int>(input).At(Timestamp(timestamp))));
    MP_ASSERT_OK(graph_.AddPacketToInputStream(
        "delay_next_output",
        MakePacket<bool>(delay_next_output).At(Timestamp(timestamp))));
  }

  CalculatorGraph graph_;
  std::vector<Packet> output_packets_;
};

TEST_F(PreviousLoopbackCalculatorDelayBehaviorTest, MultipleDelayedOutputs) {
  SendPackets(/*timestamp=*/1, /*input=*/1, /*delay_next_output=*/true);
  MP_EXPECT_OK(graph_.WaitUntilIdle());
  EXPECT_THAT(
      output_packets_,
      ElementsAre(PairPacket(Timestamp(1), Pair(IntPacket(1), EmptyPacket()))));

  SendPackets(/*timestamp=*/2, /*input=*/2, /*delay_next_output=*/true);
  MP_EXPECT_OK(graph_.WaitUntilIdle());
  EXPECT_THAT(
      output_packets_,
      ElementsAre(PairPacket(Timestamp(1), Pair(IntPacket(1), EmptyPacket()))));

  SendPackets(/*timestamp=*/3, /*input=*/3, /*delay_next_output=*/true);
  MP_EXPECT_OK(graph_.WaitUntilIdle());
  EXPECT_THAT(
      output_packets_,
      ElementsAre(PairPacket(Timestamp(1), Pair(IntPacket(1), EmptyPacket()))));

  SendPackets(/*timestamp=*/5, /*input=*/5, /*delay_next_output=*/false);
  MP_EXPECT_OK(graph_.WaitUntilIdle());
  EXPECT_THAT(
      output_packets_,
      ElementsAre(PairPacket(Timestamp(1), Pair(IntPacket(1), EmptyPacket())),
                  PairPacket(Timestamp(2), Pair(IntPacket(2), EmptyPacket())),
                  PairPacket(Timestamp(3), Pair(IntPacket(3), EmptyPacket())),
                  PairPacket(Timestamp(5), Pair(IntPacket(5), EmptyPacket()))));

  SendPackets(/*timestamp=*/15, /*input=*/15, /*delay_next_output=*/false);
  MP_EXPECT_OK(graph_.WaitUntilIdle());
  EXPECT_THAT(
      output_packets_,
      ElementsAre(
          PairPacket(Timestamp(1), Pair(IntPacket(1), EmptyPacket())),
          PairPacket(Timestamp(2), Pair(IntPacket(2), EmptyPacket())),
          PairPacket(Timestamp(3), Pair(IntPacket(3), EmptyPacket())),
          PairPacket(Timestamp(5), Pair(IntPacket(5), EmptyPacket())),
          PairPacket(Timestamp(15), Pair(IntPacket(15), IntPacket(5)))));

  MP_EXPECT_OK(graph_.CloseAllInputStreams());
  MP_EXPECT_OK(graph_.WaitUntilDone());
}

TEST_F(PreviousLoopbackCalculatorDelayBehaviorTest,
       NonDelayedOutputFollowedByMultipleDelayedOutputs) {
  SendPackets(/*timestamp=*/1, /*input=*/1, /*delay_next_output=*/false);
  MP_EXPECT_OK(graph_.WaitUntilIdle());
  EXPECT_THAT(
      output_packets_,
      ElementsAre(PairPacket(Timestamp(1), Pair(IntPacket(1), EmptyPacket()))));

  SendPackets(/*timestamp=*/2, /*input=*/2, /*delay_next_output=*/true);
  MP_EXPECT_OK(graph_.WaitUntilIdle());
  EXPECT_THAT(
      output_packets_,
      ElementsAre(PairPacket(Timestamp(1), Pair(IntPacket(1), EmptyPacket())),
                  PairPacket(Timestamp(2), Pair(IntPacket(2), IntPacket(1)))));

  SendPackets(/*timestamp=*/3, /*input=*/3, /*delay_next_output=*/true);
  MP_EXPECT_OK(graph_.WaitUntilIdle());
  EXPECT_THAT(
      output_packets_,
      ElementsAre(PairPacket(Timestamp(1), Pair(IntPacket(1), EmptyPacket())),
                  PairPacket(Timestamp(2), Pair(IntPacket(2), IntPacket(1)))));

  SendPackets(/*timestamp=*/5, /*input=*/5, /*delay_next_output=*/false);
  MP_EXPECT_OK(graph_.WaitUntilIdle());
  EXPECT_THAT(
      output_packets_,
      ElementsAre(PairPacket(Timestamp(1), Pair(IntPacket(1), EmptyPacket())),
                  PairPacket(Timestamp(2), Pair(IntPacket(2), IntPacket(1))),
                  PairPacket(Timestamp(3), Pair(IntPacket(3), EmptyPacket())),
                  PairPacket(Timestamp(5), Pair(IntPacket(5), EmptyPacket()))));

  SendPackets(/*timestamp=*/15, /*input=*/15, /*delay_next_output=*/false);
  MP_EXPECT_OK(graph_.WaitUntilIdle());
  EXPECT_THAT(
      output_packets_,
      ElementsAre(
          PairPacket(Timestamp(1), Pair(IntPacket(1), EmptyPacket())),
          PairPacket(Timestamp(2), Pair(IntPacket(2), IntPacket(1))),
          PairPacket(Timestamp(3), Pair(IntPacket(3), EmptyPacket())),
          PairPacket(Timestamp(5), Pair(IntPacket(5), EmptyPacket())),
          PairPacket(Timestamp(15), Pair(IntPacket(15), IntPacket(5)))));

  MP_EXPECT_OK(graph_.CloseAllInputStreams());
  MP_EXPECT_OK(graph_.WaitUntilDone());
}

}  // anonymous namespace
}  // namespace mediapipe
