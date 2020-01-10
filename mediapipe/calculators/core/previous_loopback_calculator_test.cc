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
#include "mediapipe/framework/port/status_matchers.h"
#include "mediapipe/framework/timestamp.h"
#include "mediapipe/framework/tool/sink.h"

namespace mediapipe {

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
  auto pair_values = [](const Packet& packet) {
    auto pair = packet.Get<std::pair<Packet, Packet>>();
    int first = pair.first.IsEmpty() ? -1 : pair.first.Get<int>();
    int second = pair.second.IsEmpty() ? -1 : pair.second.Get<int>();
    return std::make_pair(first, second);
  };

  send_packet("in", 1);
  MP_EXPECT_OK(graph_.WaitUntilIdle());
  EXPECT_EQ(TimestampValues(in_prev), (std::vector<int64>{1}));
  EXPECT_EQ(pair_values(in_prev.back()), std::make_pair(1, -1));

  send_packet("in", 2);
  MP_EXPECT_OK(graph_.WaitUntilIdle());
  EXPECT_EQ(TimestampValues(in_prev), (std::vector<int64>{1, 2}));
  EXPECT_EQ(pair_values(in_prev.back()), std::make_pair(2, 1));

  send_packet("in", 5);
  MP_EXPECT_OK(graph_.WaitUntilIdle());
  EXPECT_EQ(TimestampValues(in_prev), (std::vector<int64>{1, 2, 5}));
  EXPECT_EQ(pair_values(in_prev.back()), std::make_pair(5, 2));

  send_packet("in", 15);
  MP_EXPECT_OK(graph_.WaitUntilIdle());
  EXPECT_EQ(TimestampValues(in_prev), (std::vector<int64>{1, 2, 5, 15}));
  EXPECT_EQ(pair_values(in_prev.back()), std::make_pair(15, 5));

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
  EXPECT_EQ(TimestampValues(outputs), (std::vector<int64>{1}));

  send_packet("in", 2);
  MP_EXPECT_OK(graph_.WaitUntilIdle());
  EXPECT_EQ(TimestampValues(outputs), (std::vector<int64>{1, 2}));

  send_packet("in", 5);
  MP_EXPECT_OK(graph_.WaitUntilIdle());
  EXPECT_EQ(TimestampValues(outputs), (std::vector<int64>{1, 2, 5}));

  send_packet("in", 15);
  MP_EXPECT_OK(graph_.WaitUntilIdle());
  EXPECT_EQ(TimestampValues(outputs), (std::vector<int64>{1, 2, 5, 15}));

  MP_EXPECT_OK(graph_.CloseAllInputStreams());
  MP_EXPECT_OK(graph_.WaitUntilIdle());
  EXPECT_EQ(TimestampValues(outputs),
            (std::vector<int64>{1, 2, 5, 15, Timestamp::Max().Value()}));

  MP_EXPECT_OK(graph_.WaitUntilDone());
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

  send_packet("in", 0);
  MP_EXPECT_OK(graph_.WaitUntilIdle());
  EXPECT_EQ(TimestampValues(outputs), (std::vector<int64>{0}));

  for (int main_ts = 1; main_ts < 50; ++main_ts) {
    send_packet("in", main_ts);
    MP_EXPECT_OK(graph_.WaitUntilIdle());
    std::vector<int64> ts_values = TimestampValues(outputs);
    EXPECT_EQ(ts_values.size(), main_ts + 1);
    for (int j = 0; j < main_ts; ++j) {
      EXPECT_EQ(ts_values[j], j);
    }
  }

  MP_EXPECT_OK(graph_.CloseAllInputStreams());
  MP_EXPECT_OK(graph_.WaitUntilIdle());
  MP_EXPECT_OK(graph_.WaitUntilDone());
}

}  // anonymous namespace
}  // namespace mediapipe
