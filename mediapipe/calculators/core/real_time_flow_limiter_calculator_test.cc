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
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/calculator_runner.h"
#include "mediapipe/framework/formats/image_frame.h"
#include "mediapipe/framework/port/gmock.h"
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/framework/port/parse_text_proto.h"
#include "mediapipe/framework/port/status_matchers.h"
#include "mediapipe/framework/timestamp.h"
#include "mediapipe/framework/tool/sink.h"

namespace mediapipe {

namespace {

constexpr char kFinishedTag[] = "FINISHED";

// A simple Semaphore for synchronizing test threads.
class AtomicSemaphore {
 public:
  AtomicSemaphore(int64_t supply) : supply_(supply) {}
  void Acquire(int64_t amount) {
    while (supply_.fetch_sub(amount) - amount < 0) {
      Release(amount);
    }
  }
  void Release(int64_t amount) { supply_.fetch_add(amount); }

 private:
  std::atomic<int64_t> supply_;
};

// Returns the timestamp values for a vector of Packets.
std::vector<int64_t> TimestampValues(const std::vector<Packet>& packets) {
  std::vector<int64_t> result;
  for (const Packet& packet : packets) {
    result.push_back(packet.Timestamp().Value());
  }
  return result;
}

// Returns the packet values for a vector of Packets.
template <typename T>
std::vector<T> PacketValues(const std::vector<Packet>& packets) {
  std::vector<T> result;
  for (const Packet& packet : packets) {
    result.push_back(packet.Get<T>());
  }
  return result;
}

constexpr int kNumImageFrames = 5;
constexpr int kNumFinished = 3;
CalculatorGraphConfig::Node GetDefaultNode() {
  return ParseTextProtoOrDie<CalculatorGraphConfig::Node>(R"pb(
    calculator: "RealTimeFlowLimiterCalculator"
    input_stream: "raw_frames"
    input_stream: "FINISHED:finished"
    input_stream_info: { tag_index: "FINISHED" back_edge: true }
    output_stream: "gated_frames"
  )pb");
}

// Simple test to make sure that the RealTimeFlowLimiterCalculator outputs just
// one packet when MAX_IN_FLIGHT is 1.
TEST(RealTimeFlowLimiterCalculator, OneOutputTest) {
  // Setup the calculator runner and add only ImageFrame packets.
  CalculatorRunner runner(GetDefaultNode());
  for (int i = 0; i < kNumImageFrames; ++i) {
    Timestamp timestamp = Timestamp(i * Timestamp::kTimestampUnitsPerSecond);
    runner.MutableInputs()->Index(0).packets.push_back(
        MakePacket<ImageFrame>().At(timestamp));
  }

  // Run the calculator.
  MP_ASSERT_OK(runner.Run()) << "Calculator execution failed.";
  const std::vector<Packet>& frame_output_packets =
      runner.Outputs().Index(0).packets;

  EXPECT_EQ(frame_output_packets.size(), 1);
}

// Simple test to make sure that the RealTimeFlowLimiterCalculator waits for all
// input streams to have at least one packet available before publishing.
TEST(RealTimeFlowLimiterCalculator, BasicTest) {
  // Setup the calculator runner and add both ImageFrame and finish packets.
  CalculatorRunner runner(GetDefaultNode());
  for (int i = 0; i < kNumImageFrames; ++i) {
    Timestamp timestamp = Timestamp(i * Timestamp::kTimestampUnitsPerSecond);
    runner.MutableInputs()->Index(0).packets.push_back(
        MakePacket<ImageFrame>().At(timestamp));
  }
  for (int i = 0; i < kNumFinished; ++i) {
    Timestamp timestamp =
        Timestamp((i + 1) * Timestamp::kTimestampUnitsPerSecond);
    runner.MutableInputs()
        ->Tag(kFinishedTag)
        .packets.push_back(MakePacket<bool>(true).At(timestamp));
  }

  // Run the calculator.
  MP_ASSERT_OK(runner.Run()) << "Calculator execution failed.";
  const std::vector<Packet>& frame_output_packets =
      runner.Outputs().Index(0).packets;

  // Only outputs packets if both input streams are available.
  int expected_num_packets = std::min(kNumImageFrames, kNumFinished + 1);
  EXPECT_EQ(frame_output_packets.size(), expected_num_packets);
}

// A Calculator::Process callback function.
typedef std::function<absl::Status(const InputStreamShardSet&,
                                   OutputStreamShardSet*)>
    ProcessFunction;

// A testing callback function that passes through all packets.
absl::Status PassthroughFunction(const InputStreamShardSet& inputs,
                                 OutputStreamShardSet* outputs) {
  for (int i = 0; i < inputs.NumEntries(); ++i) {
    if (!inputs.Index(i).Value().IsEmpty()) {
      outputs->Index(i).AddPacket(inputs.Index(i).Value());
    }
  }
  return absl::OkStatus();
}

// A Calculator that runs a testing callback function in Close.
class CloseCallbackCalculator : public CalculatorBase {
 public:
  static absl::Status GetContract(CalculatorContract* cc) {
    for (CollectionItemId id = cc->Inputs().BeginId();
         id < cc->Inputs().EndId(); ++id) {
      cc->Inputs().Get(id).SetAny();
    }
    for (CollectionItemId id = cc->Outputs().BeginId();
         id < cc->Outputs().EndId(); ++id) {
      cc->Outputs().Get(id).SetAny();
    }
    cc->InputSidePackets().Index(0).Set<std::function<absl::Status()>>();
    return absl::OkStatus();
  }

  absl::Status Process(CalculatorContext* cc) override {
    return PassthroughFunction(cc->Inputs(), &(cc->Outputs()));
  }

  absl::Status Close(CalculatorContext* cc) override {
    const auto& callback =
        cc->InputSidePackets().Index(0).Get<std::function<absl::Status()>>();
    return callback();
  }
};
REGISTER_CALCULATOR(CloseCallbackCalculator);

// Tests demostrating an RealTimeFlowLimiterCalculator operating in a cyclic
// graph.
// TODO: clean up these tests.
class RealTimeFlowLimiterCalculatorTest : public testing::Test {
 public:
  RealTimeFlowLimiterCalculatorTest()
      : enter_semaphore_(0), exit_semaphore_(0) {}

  void SetUp() override {
    graph_config_ = InflightGraphConfig();
    tool::AddVectorSink("out_1", &graph_config_, &out_1_packets_);
    tool::AddVectorSink("out_2", &graph_config_, &out_2_packets_);
  }

  void InitializeGraph(int max_in_flight) {
    ProcessFunction semaphore_0_func = [&](const InputStreamShardSet& inputs,
                                           OutputStreamShardSet* outputs) {
      enter_semaphore_.Release(1);
      return PassthroughFunction(inputs, outputs);
    };
    ProcessFunction semaphore_1_func = [&](const InputStreamShardSet& inputs,
                                           OutputStreamShardSet* outputs) {
      exit_semaphore_.Acquire(1);
      return PassthroughFunction(inputs, outputs);
    };
    std::function<absl::Status()> close_func = [this]() {
      close_count_++;
      return absl::OkStatus();
    };
    MP_ASSERT_OK(graph_.Initialize(
        graph_config_, {
                           {"max_in_flight", MakePacket<int>(max_in_flight)},
                           {"callback_0", Adopt(new auto(semaphore_0_func))},
                           {"callback_1", Adopt(new auto(semaphore_1_func))},
                           {"callback_2", Adopt(new auto(close_func))},
                       }));
  }

  // Adds a packet to a graph input stream.
  void AddPacket(const std::string& input_name, int value) {
    MP_EXPECT_OK(graph_.AddPacketToInputStream(
        input_name, MakePacket<int>(value).At(Timestamp(value))));
  }

  // A calculator graph starting with an RealTimeFlowLimiterCalculator and
  // ending with a InFlightFinishCalculator.
  // Back-edge "finished" limits processing to one frame in-flight.
  // The two LambdaCalculators are used to keep certain packet sets in flight.
  CalculatorGraphConfig InflightGraphConfig() {
    return ParseTextProtoOrDie<CalculatorGraphConfig>(R"pb(
      input_stream: 'in_1'
      input_stream: 'in_2'
      node {
        calculator: 'RealTimeFlowLimiterCalculator'
        input_side_packet: 'MAX_IN_FLIGHT:max_in_flight'
        input_stream: 'in_1'
        input_stream: 'in_2'
        input_stream: 'FINISHED:out_1'
        input_stream_info: { tag_index: 'FINISHED' back_edge: true }
        output_stream: 'in_1_sampled'
        output_stream: 'in_2_sampled'
      }
      node {
        calculator: 'LambdaCalculator'
        input_side_packet: 'callback_0'
        input_stream: 'in_1_sampled'
        input_stream: 'in_2_sampled'
        output_stream: 'queue_1'
        output_stream: 'queue_2'
      }
      node {
        calculator: 'LambdaCalculator'
        input_side_packet: 'callback_1'
        input_stream: 'queue_1'
        input_stream: 'queue_2'
        output_stream: 'close_1'
        output_stream: 'close_2'
      }
      node {
        calculator: 'CloseCallbackCalculator'
        input_side_packet: 'callback_2'
        input_stream: 'close_1'
        input_stream: 'close_2'
        output_stream: 'out_1'
        output_stream: 'out_2'
      }
    )pb");
  }

 protected:
  CalculatorGraphConfig graph_config_;
  CalculatorGraph graph_;
  AtomicSemaphore enter_semaphore_;
  AtomicSemaphore exit_semaphore_;
  std::vector<Packet> out_1_packets_;
  std::vector<Packet> out_2_packets_;
  int close_count_ = 0;
};

// A test demonstrating an RealTimeFlowLimiterCalculator operating in a cyclic
// graph. This test shows that:
//
// (1) Timestamps are passed through unaltered.
// (2) All output streams including the back_edge stream are closed when
//     the first input stream is closed.
//
TEST_F(RealTimeFlowLimiterCalculatorTest, BackEdgeCloses) {
  InitializeGraph(1);
  MP_ASSERT_OK(graph_.StartRun({}));

  auto send_packet = [this](const std::string& input_name, int64_t n) {
    MP_EXPECT_OK(graph_.AddPacketToInputStream(
        input_name, MakePacket<int64_t>(n).At(Timestamp(n))));
  };

  for (int i = 0; i < 10; i++) {
    send_packet("in_1", i * 10);
    // This next input should be dropped.
    send_packet("in_1", i * 10 + 5);
    MP_EXPECT_OK(graph_.WaitUntilIdle());
    send_packet("in_2", i * 10);
    exit_semaphore_.Release(1);
    MP_EXPECT_OK(graph_.WaitUntilIdle());
  }
  MP_EXPECT_OK(graph_.CloseInputStream("in_1"));
  MP_EXPECT_OK(graph_.CloseInputStream("in_2"));
  MP_EXPECT_OK(graph_.WaitUntilIdle());

  // All output streams are closed and all output packets are delivered,
  // with stream "in_1" and stream "in_2" closed.
  EXPECT_EQ(10, out_1_packets_.size());
  EXPECT_EQ(10, out_2_packets_.size());

  // Timestamps have not been messed with.
  EXPECT_EQ(PacketValues<int64_t>(out_1_packets_),
            TimestampValues(out_1_packets_));
  EXPECT_EQ(PacketValues<int64_t>(out_2_packets_),
            TimestampValues(out_2_packets_));

  // Extra inputs on in_1 have been dropped
  EXPECT_EQ(TimestampValues(out_1_packets_),
            (std::vector<int64_t>{0, 10, 20, 30, 40, 50, 60, 70, 80, 90}));
  EXPECT_EQ(TimestampValues(out_1_packets_), TimestampValues(out_2_packets_));

  // The closing of the stream has been propagated.
  EXPECT_EQ(1, close_count_);
}

// A test demonstrating that all output streams are closed when all
// input streams are closed after the last input packet has been processed.
TEST_F(RealTimeFlowLimiterCalculatorTest, AllStreamsClose) {
  InitializeGraph(1);
  MP_ASSERT_OK(graph_.StartRun({}));

  exit_semaphore_.Release(10);
  for (int i = 0; i < 10; i++) {
    AddPacket("in_1", i);
    MP_EXPECT_OK(graph_.WaitUntilIdle());
    AddPacket("in_2", i);
    MP_EXPECT_OK(graph_.WaitUntilIdle());
  }
  MP_EXPECT_OK(graph_.CloseAllInputStreams());
  MP_EXPECT_OK(graph_.WaitUntilIdle());

  EXPECT_EQ(TimestampValues(out_1_packets_), TimestampValues(out_2_packets_));
  EXPECT_EQ(TimestampValues(out_1_packets_),
            (std::vector<int64_t>{0, 1, 2, 3, 4, 5, 6, 7, 8, 9}));
  EXPECT_EQ(1, close_count_);
}

TEST(RealTimeFlowLimiterCalculator, TwoStreams) {
  std::vector<Packet> a_passed;
  std::vector<Packet> b_passed;
  CalculatorGraphConfig graph_config_ =
      ParseTextProtoOrDie<CalculatorGraphConfig>(R"pb(
        input_stream: 'in_a'
        input_stream: 'in_b'
        input_stream: 'finished'
        node {
          name: 'input_dropper'
          calculator: 'RealTimeFlowLimiterCalculator'
          input_side_packet: 'MAX_IN_FLIGHT:max_in_flight'
          input_stream: 'in_a'
          input_stream: 'in_b'
          input_stream: 'FINISHED:finished'
          input_stream_info: { tag_index: 'FINISHED' back_edge: true }
          output_stream: 'in_a_sampled'
          output_stream: 'in_b_sampled'
          output_stream: 'ALLOW:allow'
        }
      )pb");
  std::string allow_cb_name;
  tool::AddVectorSink("in_a_sampled", &graph_config_, &a_passed);
  tool::AddVectorSink("in_b_sampled", &graph_config_, &b_passed);
  tool::AddCallbackCalculator("allow", &graph_config_, &allow_cb_name, true);

  bool allow = true;
  auto allow_cb = [&allow](const Packet& packet) {
    allow = packet.Get<bool>();
  };

  CalculatorGraph graph_;
  MP_EXPECT_OK(graph_.Initialize(
      graph_config_,
      {
          {"max_in_flight", MakePacket<int>(1)},
          {allow_cb_name,
           MakePacket<std::function<void(const Packet&)>>(allow_cb)},
      }));

  MP_EXPECT_OK(graph_.StartRun({}));

  auto send_packet = [&graph_](const std::string& input_name, int n) {
    MP_EXPECT_OK(graph_.AddPacketToInputStream(
        input_name, MakePacket<int>(n).At(Timestamp(n))));
  };
  send_packet("in_a", 1);
  MP_EXPECT_OK(graph_.WaitUntilIdle());
  EXPECT_EQ(allow, false);
  EXPECT_EQ(TimestampValues(a_passed), (std::vector<int64_t>{1}));
  EXPECT_EQ(TimestampValues(b_passed), (std::vector<int64_t>{}));

  send_packet("in_a", 2);
  send_packet("in_b", 1);
  MP_EXPECT_OK(graph_.WaitUntilIdle());
  EXPECT_EQ(TimestampValues(a_passed), (std::vector<int64_t>{1}));
  EXPECT_EQ(TimestampValues(b_passed), (std::vector<int64_t>{1}));
  EXPECT_EQ(allow, false);

  send_packet("finished", 1);
  MP_EXPECT_OK(graph_.WaitUntilIdle());
  EXPECT_EQ(TimestampValues(a_passed), (std::vector<int64_t>{1}));
  EXPECT_EQ(TimestampValues(b_passed), (std::vector<int64_t>{1}));
  EXPECT_EQ(allow, true);

  send_packet("in_b", 2);
  MP_EXPECT_OK(graph_.WaitUntilIdle());
  EXPECT_EQ(TimestampValues(a_passed), (std::vector<int64_t>{1}));
  EXPECT_EQ(TimestampValues(b_passed), (std::vector<int64_t>{1}));
  EXPECT_EQ(allow, true);

  send_packet("in_b", 3);
  MP_EXPECT_OK(graph_.WaitUntilIdle());
  EXPECT_EQ(TimestampValues(a_passed), (std::vector<int64_t>{1}));
  EXPECT_EQ(TimestampValues(b_passed), (std::vector<int64_t>{1, 3}));
  EXPECT_EQ(allow, false);

  send_packet("in_b", 4);
  MP_EXPECT_OK(graph_.WaitUntilIdle());
  EXPECT_EQ(TimestampValues(a_passed), (std::vector<int64_t>{1}));
  EXPECT_EQ(TimestampValues(b_passed), (std::vector<int64_t>{1, 3}));
  EXPECT_EQ(allow, false);

  send_packet("in_a", 3);
  MP_EXPECT_OK(graph_.WaitUntilIdle());
  EXPECT_EQ(TimestampValues(a_passed), (std::vector<int64_t>{1, 3}));
  EXPECT_EQ(TimestampValues(b_passed), (std::vector<int64_t>{1, 3}));
  EXPECT_EQ(allow, false);

  send_packet("finished", 3);
  MP_EXPECT_OK(graph_.WaitUntilIdle());
  EXPECT_EQ(TimestampValues(a_passed), (std::vector<int64_t>{1, 3}));
  EXPECT_EQ(TimestampValues(b_passed), (std::vector<int64_t>{1, 3}));
  EXPECT_EQ(allow, true);

  MP_EXPECT_OK(graph_.CloseAllInputStreams());
  MP_EXPECT_OK(graph_.WaitUntilDone());
}

TEST(RealTimeFlowLimiterCalculator, CanConsume) {
  std::vector<Packet> in_sampled_packets_;
  CalculatorGraphConfig graph_config_ =
      ParseTextProtoOrDie<CalculatorGraphConfig>(R"pb(
        input_stream: 'in'
        input_stream: 'finished'
        node {
          name: 'input_dropper'
          calculator: 'RealTimeFlowLimiterCalculator'
          input_side_packet: 'MAX_IN_FLIGHT:max_in_flight'
          input_stream: 'in'
          input_stream: 'FINISHED:finished'
          input_stream_info: { tag_index: 'FINISHED' back_edge: true }
          output_stream: 'in_sampled'
          output_stream: 'ALLOW:allow'
        }
      )pb");
  std::string allow_cb_name;
  tool::AddVectorSink("in_sampled", &graph_config_, &in_sampled_packets_);
  tool::AddCallbackCalculator("allow", &graph_config_, &allow_cb_name, true);

  bool allow = true;
  auto allow_cb = [&allow](const Packet& packet) {
    allow = packet.Get<bool>();
  };

  CalculatorGraph graph_;
  MP_EXPECT_OK(graph_.Initialize(
      graph_config_,
      {
          {"max_in_flight", MakePacket<int>(1)},
          {allow_cb_name,
           MakePacket<std::function<void(const Packet&)>>(allow_cb)},
      }));

  MP_EXPECT_OK(graph_.StartRun({}));

  auto send_packet = [&graph_](const std::string& input_name, int n) {
    MP_EXPECT_OK(graph_.AddPacketToInputStream(
        input_name, MakePacket<int>(n).At(Timestamp(n))));
  };
  send_packet("in", 1);
  MP_EXPECT_OK(graph_.WaitUntilIdle());
  EXPECT_EQ(allow, false);
  EXPECT_EQ(TimestampValues(in_sampled_packets_), (std::vector<int64_t>{1}));

  MP_EXPECT_OK(in_sampled_packets_[0].Consume<int>());

  MP_EXPECT_OK(graph_.CloseAllInputStreams());
  MP_EXPECT_OK(graph_.WaitUntilDone());
}

}  // anonymous namespace
}  // namespace mediapipe
