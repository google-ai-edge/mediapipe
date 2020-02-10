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

#include "mediapipe/framework/tool/simulation_clock.h"

#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/memory/memory.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/executor.h"
#include "mediapipe/framework/input_stream.h"
#include "mediapipe/framework/output_stream.h"
#include "mediapipe/framework/port/core_proto_inc.h"
#include "mediapipe/framework/port/gmock.h"
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/framework/port/logging.h"
#include "mediapipe/framework/port/parse_text_proto.h"
#include "mediapipe/framework/port/status.h"
#include "mediapipe/framework/port/status_matchers.h"
#include "mediapipe/framework/tool/simulation_clock_executor.h"

using testing::ElementsAre;

namespace mediapipe {

namespace {

class SimulationClockTest : public ::testing::Test {
 protected:
  void SetUpInFlightGraph() {
    graph_config_ = ParseTextProtoOrDie<CalculatorGraphConfig>(R"(
      input_stream: "input_packets_0"
      node {
        calculator: 'FlowLimiterCalculator'
        input_stream_handler {
          input_stream_handler: 'ImmediateInputStreamHandler'
        }
        input_side_packet: 'MAX_IN_FLIGHT:max_in_flight'
        input_stream: 'input_packets_0'
        input_stream: 'FINISHED:finish_indicator'
        input_stream_info: { tag_index: 'FINISHED' back_edge: true }
        output_stream: 'input_0_sampled'
      }
      node {
        calculator: "RoundRobinDemuxCalculator"
        input_stream: "input_0_sampled"
        output_stream: "OUTPUT:0:input_0"
        output_stream: "OUTPUT:1:input_1"
      }
      node {
        calculator: "LambdaCalculator"
        input_side_packet: 'callback_0'
        input_stream: "input_0"
        output_stream: "output_0"
      }
      node {
        calculator: "LambdaCalculator"
        input_side_packet: 'callback_1'
        input_stream: "input_1"
        output_stream: "output_1"
      }
      node {
        calculator: "ImmediateMuxCalculator"
        input_stream_handler {
          input_stream_handler: "ImmediateInputStreamHandler"
        }
        input_stream: "output_0"
        input_stream: "output_1"
        output_stream: 'output_packets_0'
        output_stream: 'finish_indicator'
      }
    )");
  }

  // Initialize the test clock as a SimulationClock.
  void SetupSimulationClock() {
    auto executor = std::make_shared<SimulationClockExecutor>(4);
    simulation_clock_ = executor->GetClock();
    clock_ = simulation_clock_.get();
    MP_ASSERT_OK(graph_.SetExecutor("", executor));
  }

  // Initialize the test clock as a RealClock.
  void SetupRealClock() { clock_ = ::mediapipe::Clock::RealClock(); }

  // Return the values of the timestamps of a vector of Packets.
  static std::vector<int64> TimestampValues(
      const std::vector<Packet>& packets) {
    std::vector<int64> result;
    for (const Packet& p : packets) {
      result.push_back(p.Timestamp().Value());
    }
    return result;
  }

  static std::vector<int64> TimeValues(const std::vector<absl::Time>& times) {
    std::vector<int64> result;
    for (const absl::Time& t : times) {
      result.push_back(absl::ToUnixMicros(t));
    }
    return result;
  }

  std::shared_ptr<SimulationClock> simulation_clock_;
  CalculatorGraphConfig graph_config_;
  CalculatorGraph graph_;
  ::mediapipe::Clock* clock_;
};

// Just directly calls SimulationClock::Sleep on several threads.
TEST_F(SimulationClockTest, SleepUntil) {
  std::vector<absl::Time> start_times;
  auto executor = std::make_shared<SimulationClockExecutor>(4);
  simulation_clock_ = executor->GetClock();
  clock_ = simulation_clock_.get();

  std::function<void(int)> run_chain = [&](int count) {
    if (count > 0) {
      start_times.push_back(clock_->TimeNow());
      clock_->Sleep(absl::Microseconds(10000));
      run_chain(count - 1);
    }
  };
  simulation_clock_->ThreadStart();
  for (int i = 0; i < 3; i++) {
    executor->Schedule([&] { run_chain(3); });
    clock_->Sleep(absl::Microseconds(2000));
  }
  clock_->Sleep(absl::Microseconds(100000));
  simulation_clock_->ThreadFinish();
  EXPECT_THAT(
      TimeValues(start_times),  //
      ElementsAre(0, 2000, 4000, 10000, 12000, 14000, 20000, 22000, 24000));
}

// Directly calls SimulationClock::Sleep with duplicate wake times.
TEST_F(SimulationClockTest, DuplicateWakeTimes) {
  std::vector<absl::Time> start_times;
  std::vector<int> start_counts;
  auto executor = std::make_shared<SimulationClockExecutor>(4);
  simulation_clock_ = executor->GetClock();
  clock_ = simulation_clock_.get();
  std::function<void(int)> run_chain = [&](int count) {
    if (count > 0) {
      start_times.push_back(clock_->TimeNow());
      start_counts.push_back(count);
      clock_->Sleep(absl::Microseconds(10000));
      run_chain(count - 1);
    }
  };
  simulation_clock_->ThreadStart();
  for (int i = 0; i < 3; i++) {
    executor->Schedule([&] { run_chain(3); });
    clock_->Sleep(absl::Microseconds(10000));
  }
  clock_->Sleep(absl::Microseconds(100000));
  simulation_clock_->ThreadFinish();
  EXPECT_THAT(
      TimeValues(start_times),
      ElementsAre(0, 10000, 10000, 20000, 20000, 20000, 30000, 30000, 40000));
  EXPECT_THAT(start_counts, ElementsAre(3, 2, 3, 1, 2, 3, 1, 2, 1));
}

// A Calculator::Process callback function.
typedef std::function<::mediapipe::Status(const InputStreamShardSet&,
                                          OutputStreamShardSet*)>
    ProcessFunction;

// A testing callback function that passes through all packets.
::mediapipe::Status PassThrough(const InputStreamShardSet& inputs,
                                OutputStreamShardSet* outputs) {
  for (int i = 0; i < inputs.NumEntries(); ++i) {
    if (!inputs.Index(i).Value().IsEmpty()) {
      outputs->Index(i).AddPacket(inputs.Index(i).Value());
    }
  }
  return ::mediapipe::OkStatus();
}

// This test shows sim clock synchronizing a bunch of parallel tasks.
TEST_F(SimulationClockTest, InFlight) {
  // Callbacks to control the MockCalculators.
  // SetupSimulationClock can be replaced by SetupRealClock to run
  // the test over 200 ms of real time rather simulated time.
  SetupSimulationClock();
  ProcessFunction wait_0 = [&](const InputStreamShardSet& inputs,
                               OutputStreamShardSet* outputs) {
    clock_->Sleep(absl::Microseconds(20000));
    return PassThrough(inputs, outputs);
  };
  ProcessFunction wait_1 = [&](const InputStreamShardSet& inputs,
                               OutputStreamShardSet* outputs) {
    clock_->Sleep(absl::Microseconds(30000));
    return PassThrough(inputs, outputs);
  };

  // Start the graph with the callbacks.
  SetUpInFlightGraph();
  std::vector<Packet> out_packets;
  tool::AddVectorSink("output_packets_0", &graph_config_, &out_packets);
  MP_ASSERT_OK(graph_.Initialize(graph_config_,
                                 {
                                     {"max_in_flight", MakePacket<int>(2)},
                                     {"callback_0", Adopt(new auto(wait_0))},
                                     {"callback_1", Adopt(new auto(wait_1))},
                                 }));
  MP_ASSERT_OK(graph_.StartRun({}));
  simulation_clock_->ThreadStart();

  // Add 10 input packets to the graph, one each 10 ms, starting after 11 ms
  // of clock time.  Timestamps lag clock times by 1 ms.
  clock_->Sleep(absl::Microseconds(11000));
  for (uint64 ts = 10000; ts <= 100000; ts += 10000) {
    MP_EXPECT_OK(graph_.AddPacketToInputStream(
        "input_packets_0", MakePacket<uint64>(ts).At(Timestamp(ts))));
    clock_->Sleep(absl::Microseconds(10000));
  }

  // Wait for 100 ms of clock time, then close the graph.
  clock_->Sleep(absl::Microseconds(100000));
  simulation_clock_->ThreadFinish();
  MP_ASSERT_OK(graph_.CloseAllInputStreams());
  MP_ASSERT_OK(graph_.WaitUntilDone());

  // Validate the graph run.
  EXPECT_THAT(TimestampValues(out_packets),
              ElementsAre(10000, 20000, 40000, 60000, 70000, 100000));
}

// Shows successful destruction of CalculatorGraph, SimulationClockExecutor,
// and SimulationClock.  With tsan, this test reveals a race condition unless
// the SimulationClock destructor calls ThreadFinish to waits for all threads.
TEST_F(SimulationClockTest, DestroyClock) {
  auto graph_config = ParseTextProtoOrDie<CalculatorGraphConfig>(R"(
    node {
      calculator: "LambdaCalculator"
      input_side_packet: 'callback_0'
      output_stream: "input_1"
    }
    node {
      calculator: "LambdaCalculator"
      input_side_packet: 'callback_1'
      input_stream: "input_1"
      output_stream: "output_1"
    }
  )");

  int input_count = 0;
  ProcessFunction wait_0 = [&](const InputStreamShardSet& inputs,
                               OutputStreamShardSet* outputs) {
    clock_->Sleep(absl::Microseconds(20000));
    if (++input_count < 4) {
      outputs->Index(0).AddPacket(
          MakePacket<uint64>(input_count).At(Timestamp(input_count)));
      return ::mediapipe::OkStatus();
    } else {
      return tool::StatusStop();
    }
  };
  ProcessFunction wait_1 = [&](const InputStreamShardSet& inputs,
                               OutputStreamShardSet* outputs) {
    clock_->Sleep(absl::Microseconds(30000));
    return PassThrough(inputs, outputs);
  };

  std::vector<Packet> out_packets;
  ::mediapipe::Status status;
  {
    CalculatorGraph graph;
    auto executor = std::make_shared<SimulationClockExecutor>(4);
    clock_ = executor->GetClock().get();
    MP_ASSERT_OK(graph.SetExecutor("", executor));
    tool::AddVectorSink("output_1", &graph_config, &out_packets);
    MP_ASSERT_OK(graph.Initialize(graph_config,
                                  {
                                      {"callback_0", Adopt(new auto(wait_0))},
                                      {"callback_1", Adopt(new auto(wait_1))},
                                  }));
    MP_EXPECT_OK(graph.Run());
  }
  EXPECT_EQ(out_packets.size(), 3);
}

}  // namespace
}  // namespace mediapipe
