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

#include <stdint.h>

#include <atomic>
#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/strings/str_cat.h"
#include "absl/synchronization/mutex.h"
#include "mediapipe/framework/calculator.pb.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/port/gmock.h"
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/framework/port/integral_types.h"
#include "mediapipe/framework/port/logging.h"
#include "mediapipe/framework/port/status.h"
#include "mediapipe/framework/port/status_matchers.h"
#include "mediapipe/framework/port/threadpool.h"
#include "mediapipe/framework/tool/sink.h"

// Tests for ImmediateMuxCalculator.  These tests show how parallel output
// packets are handled when they arrive in various orders.
using testing::ElementsAre;

namespace mediapipe {

namespace {

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

// A mediapipe::Executor that signals the start and finish of each task.
// Provides 4 worker threads.
class CountingExecutor : public Executor {
 public:
  CountingExecutor(std::function<void()> start_callback,
                   std::function<void()> finish_callback)
      : thread_pool_(4),
        start_callback_(std::move(start_callback)),
        finish_callback_(std::move(finish_callback)) {
    thread_pool_.StartWorkers();
  }
  void Schedule(std::function<void()> task) override {
    start_callback_();
    thread_pool_.Schedule([this, task] {
      task();
      finish_callback_();
    });
  }

 private:
  ThreadPool thread_pool_;
  std::function<void()> start_callback_;
  std::function<void()> finish_callback_;
};

// Returns a new mediapipe::Executor with 4 worker threads.
std::shared_ptr<Executor> MakeExecutor(std::function<void()> start_callback,
                                       std::function<void()> finish_callback) {
  return std::make_shared<CountingExecutor>(start_callback, finish_callback);
}

// Tests showing ImmediateMuxCalculator dropping packets in various sequences.
class ImmediateMuxCalculatorTest : public ::testing::Test {
 protected:
  void SetUpMuxGraph() {
    ASSERT_TRUE(proto_ns::TextFormat::ParseFromString(R"(
          input_stream: "input_packets_0"
          input_stream: "input_packets_1"
          node {
            calculator: "ImmediateMuxCalculator"
            input_stream_handler {
              input_stream_handler: "ImmediateInputStreamHandler"
            }
            input_stream: "input_packets_0"
            input_stream: "input_packets_1"
            output_stream: "output_packets_0"
          }
        )",
                                                      &graph_config_));
  }

  void SetUpDemuxGraph() {
    ASSERT_TRUE(proto_ns::TextFormat::ParseFromString(R"(
        input_stream: "input_packets_0"
        node {
          calculator: "RoundRobinDemuxCalculator"
          input_stream: "input_packets_0"
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
          output_stream: "output_packets_0"
        }
        )",
                                                      &graph_config_));
  }

  void SetUpDemuxInFlightGraph() {
    ASSERT_TRUE(proto_ns::TextFormat::ParseFromString(R"(
        input_stream: "input_packets_0"
        node {
          calculator: 'FlowLimiterCalculator'
          input_stream_handler {
            input_stream_handler: 'ImmediateInputStreamHandler'
          }
          input_side_packet: 'MAX_IN_FLIGHT:max_in_flight'
          input_stream: 'input_packets_0'
          input_stream: 'FINISHED:finish_indicator'
          input_stream_info: {
            tag_index: 'FINISHED'
            back_edge: true
          }
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
        )",
                                                      &graph_config_));
  }

  static Packet PacketAt(int64 ts) {
    return Adopt(new int64(999)).At(Timestamp(ts));
  }
  static Packet None() { return Packet().At(Timestamp::OneOverPostStream()); }
  static bool IsNone(const Packet& packet) {
    return packet.Timestamp() == Timestamp::OneOverPostStream();
  }
  // Return the values of the timestamps of a vector of Packets.
  static std::vector<int64> TimestampValues(
      const std::vector<Packet>& packets) {
    std::vector<int64> result;
    for (const Packet& p : packets) {
      result.push_back(p.Timestamp().Value());
    }
    return result;
  }

  // Runs a CalculatorGraph with a series of packet sets.
  // Returns a vector of packets from each graph output stream.
  void RunGraph(const std::vector<std::vector<Packet>>& input_sets,
                std::vector<Packet>* output_packets) {
    // Register output packet observers.
    tool::AddVectorSink("output_packets_0", &graph_config_, output_packets);

    // Start running the graph.
    CalculatorGraph graph;
    MP_ASSERT_OK(graph.Initialize(graph_config_));
    MP_ASSERT_OK(graph.StartRun({}));

    // Send each packet to the graph in the specified order.
    for (int t = 0; t < input_sets.size(); t++) {
      const std::vector<Packet>& input_set = input_sets[t];
      MP_EXPECT_OK(graph.WaitUntilIdle());
      for (int i = 0; i < input_set.size(); i++) {
        const Packet& packet = input_set[i];
        if (!IsNone(packet)) {
          MP_EXPECT_OK(graph.AddPacketToInputStream(
              absl::StrCat("input_packets_", i), packet));
        }
      }
    }
    MP_ASSERT_OK(graph.CloseAllInputStreams());
    MP_ASSERT_OK(graph.WaitUntilDone());
  }

  CalculatorGraphConfig graph_config_;
};

TEST_F(ImmediateMuxCalculatorTest, IncreasingTimestamps) {
  // Run the graph with a series of packet sets.
  std::vector<std::vector<Packet>> input_sets = {
      {PacketAt(10000), None()},  //
      {PacketAt(20000), None()},  //
      {None(), PacketAt(30000)},  //
      {None(), PacketAt(40000)},
  };
  SetUpMuxGraph();
  std::vector<Packet> output_packets;
  RunGraph(input_sets, &output_packets);

  // Validate the output packets.
  EXPECT_THAT(TimestampValues(output_packets),
              ElementsAre(10000, 20000, 30000, 40000));
}

TEST_F(ImmediateMuxCalculatorTest, SupersededTimestamp) {
  // Run the graph with a series of packet sets.
  std::vector<std::vector<Packet>> input_sets = {
      {PacketAt(10000), None()},  //
      {PacketAt(30000), None()},  //
      {None(), PacketAt(20000)},  //
      {None(), PacketAt(40000)},
  };
  SetUpMuxGraph();
  std::vector<Packet> output_packets;
  RunGraph(input_sets, &output_packets);

  // Output packet 20000 is superseded and dropped.
  EXPECT_THAT(TimestampValues(output_packets),
              ElementsAre(10000, 30000, 40000));
}

TEST_F(ImmediateMuxCalculatorTest, SimultaneousTimestamps) {
  // Run the graph with a series of packet sets.
  std::vector<std::vector<Packet>> input_sets = {
      {PacketAt(10000), None()},           //
      {PacketAt(40000), PacketAt(20000)},  //
      {None(), PacketAt(30000)},
  };
  SetUpMuxGraph();
  std::vector<Packet> output_packets;
  RunGraph(input_sets, &output_packets);

  // Output packet 20000 is superseded and dropped.
  EXPECT_THAT(TimestampValues(output_packets), ElementsAre(10000, 40000));
}

// A Calculator::Process callback function.
typedef std::function<absl::Status(const InputStreamShardSet&,
                                   OutputStreamShardSet*)>
    ProcessFunction;

// A testing callback function that passes through all packets.
absl::Status PassThrough(const InputStreamShardSet& inputs,
                         OutputStreamShardSet* outputs) {
  for (int i = 0; i < inputs.NumEntries(); ++i) {
    if (!inputs.Index(i).Value().IsEmpty()) {
      outputs->Index(i).AddPacket(inputs.Index(i).Value());
    }
  }
  return absl::OkStatus();
}

TEST_F(ImmediateMuxCalculatorTest, Demux) {
  // Semaphores to sequence the parallel Process outputs.
  AtomicSemaphore semaphore_0(0);
  AtomicSemaphore semaphore_1(0);
  ProcessFunction wait_0 = [&semaphore_0](const InputStreamShardSet& inputs,
                                          OutputStreamShardSet* outputs) {
    semaphore_0.Acquire(1);
    return PassThrough(inputs, outputs);
  };
  ProcessFunction wait_1 = [&semaphore_1](const InputStreamShardSet& inputs,
                                          OutputStreamShardSet* outputs) {
    semaphore_1.Acquire(1);
    return PassThrough(inputs, outputs);
  };

  // A callback to await and capture output packets.
  std::vector<Packet> out_packets;
  absl::Mutex out_mutex;
  auto out_cb = [&](const Packet& p) {
    absl::MutexLock lock(&out_mutex);
    out_packets.push_back(p);
    return absl::OkStatus();
  };
  auto wait_for = [&](std::function<bool()> cond) {
    absl::MutexLock lock(&out_mutex);
    out_mutex.Await(absl::Condition(&cond));
  };
  SetUpDemuxGraph();

  // Start the graph and add five input packets.
  CalculatorGraph graph;
  MP_ASSERT_OK(graph.Initialize(graph_config_,
                                {
                                    {"callback_0", Adopt(new auto(wait_0))},
                                    {"callback_1", Adopt(new auto(wait_1))},
                                }));
  MP_ASSERT_OK(graph.ObserveOutputStream("output_packets_0", out_cb));
  MP_ASSERT_OK(graph.StartRun({}));
  MP_EXPECT_OK(
      graph.AddPacketToInputStream("input_packets_0", PacketAt(10000)));
  MP_EXPECT_OK(
      graph.AddPacketToInputStream("input_packets_0", PacketAt(20000)));
  MP_EXPECT_OK(
      graph.AddPacketToInputStream("input_packets_0", PacketAt(30000)));
  MP_EXPECT_OK(
      graph.AddPacketToInputStream("input_packets_0", PacketAt(40000)));
  MP_EXPECT_OK(
      graph.AddPacketToInputStream("input_packets_0", PacketAt(50000)));

  // Release the outputs in order 20000, 10000, 30000, 50000, 40000.
  semaphore_1.Release(1);  // 20000
  wait_for([&] { return !out_packets.empty(); });
  semaphore_0.Release(1);  // 10000
  semaphore_0.Release(1);  // 30000
  wait_for([&] { return out_packets.size() >= 2; });
  semaphore_0.Release(1);  // 50000
  wait_for([&] { return out_packets.size() >= 3; });
  semaphore_1.Release(1);  // 40000
  MP_ASSERT_OK(graph.CloseAllInputStreams());
  MP_ASSERT_OK(graph.WaitUntilDone());

  // Output packets 10000 and 40000 are superseded and dropped.
  EXPECT_THAT(TimestampValues(out_packets), ElementsAre(20000, 30000, 50000));
}

}  // namespace
}  // namespace mediapipe
