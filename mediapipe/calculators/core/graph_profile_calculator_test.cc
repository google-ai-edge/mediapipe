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

#include <memory>
#include <string>
#include <vector>

#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/time/time.h"
#include "mediapipe/framework/calculator.pb.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/calculator_profile.pb.h"
#include "mediapipe/framework/deps/clock.h"
#include "mediapipe/framework/port/gmock.h"
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/framework/port/integral_types.h"
#include "mediapipe/framework/port/logging.h"
#include "mediapipe/framework/port/parse_text_proto.h"
#include "mediapipe/framework/port/proto_ns.h"
#include "mediapipe/framework/port/status_matchers.h"
#include "mediapipe/framework/port/threadpool.h"
#include "mediapipe/framework/tool/simulation_clock_executor.h"

// Tests for GraphProfileCalculator.
using testing::ElementsAre;

namespace mediapipe {
namespace {

constexpr char kClockTag[] = "CLOCK";

using mediapipe::Clock;

// A Calculator with a fixed Process call latency.
class SleepCalculator : public CalculatorBase {
 public:
  static absl::Status GetContract(CalculatorContract* cc) {
    cc->InputSidePackets().Tag(kClockTag).Set<std::shared_ptr<Clock>>();
    cc->Inputs().Index(0).SetAny();
    cc->Outputs().Index(0).SetSameAs(&cc->Inputs().Index(0));
    cc->SetTimestampOffset(TimestampDiff(0));
    return absl::OkStatus();
  }
  absl::Status Open(CalculatorContext* cc) final {
    clock_ =
        cc->InputSidePackets().Tag(kClockTag).Get<std::shared_ptr<Clock>>();
    return absl::OkStatus();
  }

  absl::Status Process(CalculatorContext* cc) final {
    clock_->Sleep(absl::Milliseconds(5));
    cc->Outputs().Index(0).AddPacket(cc->Inputs().Index(0).Value());
    return absl::OkStatus();
  }
  std::shared_ptr<::mediapipe::Clock> clock_ = nullptr;
};
REGISTER_CALCULATOR(SleepCalculator);

// Tests showing GraphProfileCalculator reporting GraphProfile output packets.
class GraphProfileCalculatorTest : public ::testing::Test {
 protected:
  void SetUpProfileGraph() {
    ASSERT_TRUE(proto_ns::TextFormat::ParseFromString(R"(
        input_stream: "input_packets_0"
        node {
          calculator: 'SleepCalculator'
          input_side_packet: 'CLOCK:sync_clock'
          input_stream: 'input_packets_0'
          output_stream: 'output_packets_1'
        }
        node {
          calculator: "GraphProfileCalculator"
          options: {
            [mediapipe.GraphProfileCalculatorOptions.ext]: {
              profile_interval: 25000
            }
          }
          input_stream: "FRAME:output_packets_1"
          output_stream: "PROFILE:output_packets_0"
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
    std::shared_ptr<SimulationClockExecutor> executor(
        new SimulationClockExecutor(3 /*num_threads*/));
    CalculatorGraph graph;
    MP_ASSERT_OK(graph.SetExecutor("", executor));
    graph.profiler()->SetClock(executor->GetClock());
    MP_ASSERT_OK(graph.Initialize(graph_config_));
    executor->GetClock()->ThreadStart();
    MP_ASSERT_OK(graph.StartRun({
        {"sync_clock",
         Adopt(new std::shared_ptr<::mediapipe::Clock>(executor->GetClock()))},
    }));

    // Send each packet to the graph in the specified order.
    for (int t = 0; t < input_sets.size(); t++) {
      const std::vector<Packet>& input_set = input_sets[t];
      for (int i = 0; i < input_set.size(); i++) {
        const Packet& packet = input_set[i];
        if (!IsNone(packet)) {
          MP_EXPECT_OK(graph.AddPacketToInputStream(
              absl::StrCat("input_packets_", i), packet));
        }
        executor->GetClock()->Sleep(absl::Milliseconds(10));
      }
    }
    MP_ASSERT_OK(graph.CloseAllInputStreams());
    executor->GetClock()->Sleep(absl::Milliseconds(100));
    executor->GetClock()->ThreadFinish();
    MP_ASSERT_OK(graph.WaitUntilDone());
  }

  CalculatorGraphConfig graph_config_;
};

TEST_F(GraphProfileCalculatorTest, GraphProfile) {
  SetUpProfileGraph();
  auto profiler_config = graph_config_.mutable_profiler_config();
  profiler_config->set_enable_profiler(true);
  profiler_config->set_trace_enabled(false);
  profiler_config->set_trace_log_disabled(true);
  profiler_config->set_enable_stream_latency(true);
  profiler_config->set_calculator_filter(".*Calculator");

  // Run the graph with a series of packet sets.
  std::vector<std::vector<Packet>> input_sets = {
      {PacketAt(10000)},  //
      {PacketAt(20000)},  //
      {PacketAt(30000)},  //
      {PacketAt(40000)},
  };
  std::vector<Packet> output_packets;
  RunGraph(input_sets, &output_packets);

  // Validate the output packets.
  EXPECT_THAT(TimestampValues(output_packets),  //
              ElementsAre(10000, 40000));

  GraphProfile expected_profile =
      mediapipe::ParseTextProtoOrDie<GraphProfile>(R"pb(
        calculator_profiles {
          name: "GraphProfileCalculator"
          open_runtime: 0
          process_runtime { total: 0 count: 3 }
          process_input_latency { total: 15000 count: 3 }
          process_output_latency { total: 15000 count: 3 }
          input_stream_profiles {
            name: "output_packets_1"
            back_edge: false
            latency { total: 0 count: 3 }
          }
        }
        calculator_profiles {
          name: "SleepCalculator"
          open_runtime: 0
          process_runtime { total: 15000 count: 3 }
          process_input_latency { total: 0 count: 3 }
          process_output_latency { total: 15000 count: 3 }
          input_stream_profiles {
            name: "input_packets_0"
            back_edge: false
            latency { total: 0 count: 3 }
          }
        })pb");

  ASSERT_EQ(output_packets.size(), 2);
  EXPECT_TRUE(output_packets[0].Get<GraphProfile>().has_config());
  EXPECT_THAT(output_packets[1].Get<GraphProfile>(),
              mediapipe::EqualsProto(expected_profile));
}

}  // namespace
}  // namespace mediapipe
