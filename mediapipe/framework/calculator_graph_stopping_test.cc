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

#include <map>
#include <string>
#include <vector>

#include "absl/strings/string_view.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/calculator_graph.h"
#include "mediapipe/framework/port/canonical_errors.h"
#include "mediapipe/framework/port/core_proto_inc.h"
#include "mediapipe/framework/port/gmock.h"
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/framework/port/integral_types.h"
#include "mediapipe/framework/port/logging.h"
#include "mediapipe/framework/port/status_matchers.h"
#include "mediapipe/framework/tool/sink.h"
#include "mediapipe/framework/tool/status_util.h"

namespace mediapipe {}

namespace testing_ns {
using ::mediapipe::CalculatorBase;
using ::mediapipe::CalculatorContext;
using ::mediapipe::CalculatorContract;
using ::mediapipe::CalculatorGraphConfig;
using ::mediapipe::GetFromUniquePtr;
using ::mediapipe::InputStreamShardSet;
using ::mediapipe::MakePacket;
using ::mediapipe::OutputStreamShardSet;
using ::mediapipe::Timestamp;
namespace proto_ns = ::mediapipe::proto_ns;
using ::mediapipe::CalculatorGraph;
using ::mediapipe::Packet;

class InfiniteSequenceCalculator : public mediapipe::CalculatorBase {
 public:
  static ::mediapipe::Status GetContract(mediapipe::CalculatorContract* cc) {
    cc->Outputs().Tag("OUT").Set<int>();
    cc->Outputs().Tag("EVENT").Set<int>();
    return ::mediapipe::OkStatus();
  }
  ::mediapipe::Status Open(CalculatorContext* cc) override {
    cc->Outputs().Tag("EVENT").AddPacket(MakePacket<int>(1).At(Timestamp(1)));
    return ::mediapipe::OkStatus();
  }
  ::mediapipe::Status Process(CalculatorContext* cc) override {
    cc->Outputs().Tag("OUT").AddPacket(
        MakePacket<int>(count_).At(Timestamp(count_)));
    count_++;
    return ::mediapipe::OkStatus();
  }
  ::mediapipe::Status Close(CalculatorContext* cc) override {
    cc->Outputs().Tag("EVENT").AddPacket(MakePacket<int>(2).At(Timestamp(2)));
    return ::mediapipe::OkStatus();
  }

 private:
  int count_ = 0;
};
REGISTER_CALCULATOR(::testing_ns::InfiniteSequenceCalculator);

class StoppingPassThroughCalculator : public mediapipe::CalculatorBase {
 public:
  static ::mediapipe::Status GetContract(CalculatorContract* cc) {
    for (int i = 0; i < cc->Inputs().NumEntries(""); ++i) {
      cc->Inputs().Get("", i).SetAny();
      cc->Outputs().Get("", i).SetSameAs(&cc->Inputs().Get("", i));
    }
    cc->Outputs().Tag("EVENT").Set<int>();
    return ::mediapipe::OkStatus();
  }
  ::mediapipe::Status Open(CalculatorContext* cc) override {
    cc->Outputs().Tag("EVENT").AddPacket(MakePacket<int>(1).At(Timestamp(1)));
    return ::mediapipe::OkStatus();
  }
  ::mediapipe::Status Process(CalculatorContext* cc) override {
    for (int i = 0; i < cc->Inputs().NumEntries(""); ++i) {
      if (!cc->Inputs().Get("", i).IsEmpty()) {
        cc->Outputs().Get("", i).AddPacket(cc->Inputs().Get("", i).Value());
      }
    }
    return (++count_ <= max_count_) ? ::mediapipe::OkStatus()
                                    : ::mediapipe::tool::StatusStop();
  }
  ::mediapipe::Status Close(CalculatorContext* cc) override {
    cc->Outputs().Tag("EVENT").AddPacket(MakePacket<int>(2).At(Timestamp(2)));
    return ::mediapipe::OkStatus();
  }

 private:
  int count_ = 0;
  int max_count_ = 10;
};
REGISTER_CALCULATOR(::testing_ns::StoppingPassThroughCalculator);

// A simple Semaphore for synchronizing test threads.
class AtomicSemaphore {
 public:
  AtomicSemaphore(int64_t supply) : supply_(supply) {}
  void Acquire(int64_t amount) {
    while (supply_.fetch_sub(amount) - amount < 0) {
      Release(amount);
    }
  }
  void Release(int64_t amount) { supply_ += amount; }

 private:
  std::atomic<int64_t> supply_;
};

// A ProcessFunction that passes through all packets.
::mediapipe::Status DoProcess(const InputStreamShardSet& inputs,
                              OutputStreamShardSet* outputs) {
  for (int i = 0; i < inputs.NumEntries(); ++i) {
    if (!inputs.Index(i).Value().IsEmpty()) {
      outputs->Index(i).AddPacket(inputs.Index(i).Value());
    }
  }
  return ::mediapipe::OkStatus();
}

typedef std::function<::mediapipe::Status(const InputStreamShardSet&,
                                          OutputStreamShardSet*)>
    ProcessFunction;

// A Calculator that delegates its Process function to a callback function.
class ProcessCallbackCalculator : public CalculatorBase {
 public:
  static ::mediapipe::Status GetContract(CalculatorContract* cc) {
    for (int i = 0; i < cc->Inputs().NumEntries(); ++i) {
      cc->Inputs().Index(i).SetAny();
      cc->Outputs().Index(i).SetSameAs(&cc->Inputs().Index(0));
    }
    cc->InputSidePackets().Index(0).Set<std::unique_ptr<ProcessFunction>>();
    return ::mediapipe::OkStatus();
  }

  ::mediapipe::Status Open(CalculatorContext* cc) final {
    callback_ =
        *GetFromUniquePtr<ProcessFunction>(cc->InputSidePackets().Index(0));
    return ::mediapipe::OkStatus();
  }

  ::mediapipe::Status Process(CalculatorContext* cc) final {
    return callback_(cc->Inputs(), &(cc->Outputs()));
  }

 private:
  ProcessFunction callback_;
};
REGISTER_CALCULATOR(::testing_ns::ProcessCallbackCalculator);

// Tests CloseAllPacketSources.
TEST(CalculatorGraphStoppingTest, CloseAllPacketSources) {
  CalculatorGraphConfig graph_config;
  ASSERT_TRUE(proto_ns::TextFormat::ParseFromString(R"(
      max_queue_size: 5
      input_stream: 'input'
      node {
        calculator: 'InfiniteSequenceCalculator'
        output_stream: 'OUT:count'
        output_stream: 'EVENT:event'
      }
      node {
        calculator: 'StoppingPassThroughCalculator'
        input_stream: 'count'
        input_stream: 'input'
        output_stream: 'count_out'
        output_stream: 'input_out'
        output_stream: 'EVENT:event_out'
      }
      package: 'testing_ns'
  )",
                                                    &graph_config));
  CalculatorGraph graph;
  MP_ASSERT_OK(graph.Initialize(graph_config, {}));

  // Observe output packets, and call CloseAllPacketSources after kNumPackets.
  std::vector<Packet> out_packets;
  std::vector<Packet> count_packets;
  std::vector<int> event_packets;
  std::vector<int> event_out_packets;
  int kNumPackets = 8;
  MP_ASSERT_OK(graph.ObserveOutputStream(  //
      "input_out", [&](const Packet& packet) {
        out_packets.push_back(packet);
        if (out_packets.size() >= kNumPackets) {
          MP_EXPECT_OK(graph.CloseAllPacketSources());
        }
        return ::mediapipe::OkStatus();
      }));
  MP_ASSERT_OK(graph.ObserveOutputStream(  //
      "count_out", [&](const Packet& packet) {
        count_packets.push_back(packet);
        return ::mediapipe::OkStatus();
      }));
  MP_ASSERT_OK(graph.ObserveOutputStream(  //
      "event", [&](const Packet& packet) {
        event_packets.push_back(packet.Get<int>());
        return ::mediapipe::OkStatus();
      }));
  MP_ASSERT_OK(graph.ObserveOutputStream(  //
      "event_out", [&](const Packet& packet) {
        event_out_packets.push_back(packet.Get<int>());
        return ::mediapipe::OkStatus();
      }));
  MP_ASSERT_OK(graph.StartRun({}));
  for (int i = 0; i < kNumPackets; ++i) {
    MP_EXPECT_OK(graph.AddPacketToInputStream(
        "input", MakePacket<int>(i).At(Timestamp(i))));
  }

  // The graph run should complete with no error status.
  MP_EXPECT_OK(graph.WaitUntilDone());
  EXPECT_EQ(kNumPackets, out_packets.size());
  EXPECT_LE(kNumPackets, count_packets.size());
  std::vector<int> expected_events = {1, 2};
  EXPECT_EQ(event_packets, expected_events);
  EXPECT_EQ(event_out_packets, expected_events);
}

// Verify that deadlock due to throttling can be reported.
TEST(CalculatorGraphStoppingTest, DeadlockReporting) {
  CalculatorGraphConfig config;
  ASSERT_TRUE(proto_ns::TextFormat::ParseFromString(R"(
        input_stream: 'in_1'
        input_stream: 'in_2'
        max_queue_size: 2
        node {
          calculator: 'ProcessCallbackCalculator'
          input_stream: 'in_1'
          input_stream: 'in_2'
          output_stream: 'out_1'
          output_stream: 'out_2'
          input_side_packet: 'callback_1'
        }
        package: 'testing_ns'
        report_deadlock: true
      )",
                                                    &config));
  CalculatorGraph graph;
  MP_ASSERT_OK(graph.Initialize(config));
  graph.SetGraphInputStreamAddMode(
      CalculatorGraph::GraphInputStreamAddMode::WAIT_TILL_NOT_FULL);
  std::vector<Packet> out_packets;
  MP_ASSERT_OK(
      graph.ObserveOutputStream("out_1", [&out_packets](const Packet& packet) {
        out_packets.push_back(packet);
        return ::mediapipe::OkStatus();
      }));

  // Lambda that waits for a local semaphore.
  AtomicSemaphore semaphore(0);
  ProcessFunction callback_1 = [&semaphore](const InputStreamShardSet& inputs,
                                            OutputStreamShardSet* outputs) {
    semaphore.Acquire(1);
    return DoProcess(inputs, outputs);
  };

  // Lambda that adds a packet to the calculator graph.
  auto add_packet = [&graph](std::string s, int i) {
    return graph.AddPacketToInputStream(s, MakePacket<int>(i).At(Timestamp(i)));
  };

  // Start the graph.
  MP_ASSERT_OK(graph.StartRun({
      {"callback_1", AdoptAsUniquePtr(new auto(callback_1))},
  }));

  // Add 3 packets to "in_1" with no packets on "in_2".
  // This causes throttling and deadlock with max_queue_size 2.
  semaphore.Release(3);
  MP_EXPECT_OK(add_packet("in_1", 1));
  MP_EXPECT_OK(add_packet("in_1", 2));
  EXPECT_FALSE(add_packet("in_1", 3).ok());

  ::mediapipe::Status status = graph.WaitUntilIdle();
  EXPECT_EQ(status.code(), ::mediapipe::StatusCode::kUnavailable);
  EXPECT_THAT(
      status.message(),
      testing::HasSubstr("Detected a deadlock due to input throttling"));

  MP_ASSERT_OK(graph.CloseAllInputStreams());
  EXPECT_FALSE(graph.WaitUntilDone().ok());
  ASSERT_EQ(0, out_packets.size());
}

// Verify that input streams grow due to deadlock resolution.
TEST(CalculatorGraphStoppingTest, DeadlockResolution) {
  CalculatorGraphConfig config;
  ASSERT_TRUE(proto_ns::TextFormat::ParseFromString(R"(
        input_stream: 'in_1'
        input_stream: 'in_2'
        max_queue_size: 2
        node {
          calculator: 'ProcessCallbackCalculator'
          input_stream: 'in_1'
          input_stream: 'in_2'
          output_stream: 'out_1'
          output_stream: 'out_2'
          input_side_packet: 'callback_1'
        }
        package: 'testing_ns'
      )",
                                                    &config));
  CalculatorGraph graph;
  MP_ASSERT_OK(graph.Initialize(config));
  graph.SetGraphInputStreamAddMode(
      CalculatorGraph::GraphInputStreamAddMode::WAIT_TILL_NOT_FULL);
  std::vector<Packet> out_packets;
  MP_ASSERT_OK(
      graph.ObserveOutputStream("out_1", [&out_packets](const Packet& packet) {
        out_packets.push_back(packet);
        return ::mediapipe::OkStatus();
      }));

  // Lambda that waits for a local semaphore.
  AtomicSemaphore semaphore(0);
  ProcessFunction callback_1 = [&semaphore](const InputStreamShardSet& inputs,
                                            OutputStreamShardSet* outputs) {
    semaphore.Acquire(1);
    return DoProcess(inputs, outputs);
  };

  // Lambda that adds a packet to the calculator graph.
  auto add_packet = [&graph](std::string s, int i) {
    return graph.AddPacketToInputStream(s, MakePacket<int>(i).At(Timestamp(i)));
  };

  // Start the graph.
  MP_ASSERT_OK(graph.StartRun({
      {"callback_1", AdoptAsUniquePtr(new auto(callback_1))},
  }));

  // Add 9 packets to "in_1" with no packets on "in_2".
  // This grows the input stream "in_1" to max-queue-size 10.
  semaphore.Release(9);
  for (int i = 1; i <= 9; ++i) {
    MP_EXPECT_OK(add_packet("in_1", i));
    MP_ASSERT_OK(graph.WaitUntilIdle());
  }

  // Advance the timestamp-bound and flush "in_1".
  semaphore.Release(1);
  MP_EXPECT_OK(add_packet("in_2", 30));
  MP_ASSERT_OK(graph.WaitUntilIdle());

  // Fill up input stream "in_1", with the semaphore blocked and deadlock
  // resolution disabled.
  for (int i = 11; i < 23; ++i) {
    MP_EXPECT_OK(add_packet("in_1", i));
  }

  // Adding any more packets fails with error "Graph is throttled".
  graph.SetGraphInputStreamAddMode(
      CalculatorGraph::GraphInputStreamAddMode::ADD_IF_NOT_FULL);
  EXPECT_FALSE(add_packet("in_1", 23).ok());

  // Allow the 12 blocked calls to "callback_1" to complete.
  semaphore.Release(12);

  MP_ASSERT_OK(graph.WaitUntilIdle());
  MP_ASSERT_OK(graph.CloseAllInputStreams());
  MP_ASSERT_OK(graph.WaitUntilDone());
  ASSERT_EQ(21, out_packets.size());
}

}  // namespace testing_ns
