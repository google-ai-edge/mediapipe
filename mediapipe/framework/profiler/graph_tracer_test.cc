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

#include "mediapipe/framework/profiler/graph_tracer.h"

#include <fstream>
#include <functional>
#include <map>
#include <memory>
#include <utility>
#include <vector>

#include "absl/flags/flag.h"
#include "absl/time/time.h"
#include "mediapipe/framework/calculator.pb.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/calculator_profile.pb.h"
#include "mediapipe/framework/deps/clock.h"
#include "mediapipe/framework/port/advanced_proto_inc.h"
#include "mediapipe/framework/port/file_helpers.h"
#include "mediapipe/framework/port/gmock.h"
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/framework/port/integral_types.h"
#include "mediapipe/framework/port/logging.h"
#include "mediapipe/framework/port/parse_text_proto.h"
#include "mediapipe/framework/port/status.h"
#include "mediapipe/framework/port/status_matchers.h"
#include "mediapipe/framework/profiler/graph_profiler.h"
#include "mediapipe/framework/profiler/test_context_builder.h"
#include "mediapipe/framework/tool/simulation_clock.h"
#include "mediapipe/framework/tool/simulation_clock_executor.h"
#include "mediapipe/framework/tool/status_util.h"

namespace mediapipe {

using PacketInfoMap =
    ShardedMap<std::string, std::list<std::pair<int64, PacketInfo>>>;

class GraphProfilerTestPeer {
 public:
  static PacketInfoMap* GetPacketsInfoMap(GraphProfiler* profiler) {
    return &profiler->packets_info_;
  }
};

namespace {

using testing::ElementsAre;

class GraphTracerTest : public ::testing::Test {
 protected:
  GraphTracerTest() {
    std::string err;
    ParseTime("%Y-%m-%d-%H-%M-%E*S", "2020-12-25-15-45-00", &start_time_, &err);
    start_timestamp_ = Timestamp(ToUnixMicros(start_time_));
  }

  // Initializes the GraphTracer.
  void SetUpGraphTracer() {
    ProfilerConfig profiler_config;
    profiler_config.set_trace_enabled(true);
    tracer_ = absl::make_unique<GraphTracer>(profiler_config);
  }

  // Initializes the input and output stream specs for a calculator node.
  void SetUpCalculatorContext(const std::string& node_name, int node_id,
                              const std::vector<std::string>& inputs,
                              const std::vector<std::string>& outputs) {
    context_builders_[node_name].Init(node_name, node_id, inputs, outputs);
  }

  void ClearCalculatorContext(const std::string& node_name) {
    context_builders_[node_name].Clear();
  }

  // Invokes LogInputEvents with some input packets.
  void LogInputPackets(const std::string& node_name,
                       GraphTrace::EventType event_type, absl::Time event_time,
                       const std::vector<Packet>& packets) {
    context_builders_[node_name].AddInputs(packets);
    tracer_->LogInputEvents(event_type, context_builders_[node_name].get(),
                            event_time);
  }

  // Invokes LogOutputEvents some output packets.
  void LogOutputPackets(const std::string& node_name,
                        GraphTrace::EventType event_type, absl::Time event_time,
                        const std::vector<std::vector<Packet>>& packets) {
    context_builders_[node_name].AddOutputs(packets);
    tracer_->LogOutputEvents(event_type, context_builders_[node_name].get(),
                             event_time);
  }

  // Returns the GraphTrace for the logged events.
  GraphTrace GetTrace() {
    GraphTrace result;
    tracer_->GetTrace(absl::InfinitePast(), absl::InfiniteFuture(), &result);
    return result;
  }

  std::unique_ptr<GraphTracer> tracer_;
  std::map<std::string, TestContextBuilder> context_builders_;
  absl::Time start_time_;
  Timestamp start_timestamp_;
};

TEST_F(GraphTracerTest, EmptyTrace) {
  // Define the GraphTracer.
  SetUpGraphTracer();

  // Validate the GraphTrace data.
  EXPECT_THAT(GetTrace(),
              EqualsProto(mediapipe::ParseTextProtoOrDie<GraphTrace>(R"pb(
                base_time: 0
                base_timestamp: 0
                stream_name: ""
              )pb")));
}

TEST_F(GraphTracerTest, CalculatorTrace) {
  // Define the GraphTracer, the CalculatorState, and the stream specs.
  SetUpGraphTracer();
  SetUpCalculatorContext("PCalculator_1", /*node_id=*/0, {"input_stream"},
                         {"output_stream"});
  absl::Time curr_time = start_time_;

  // PCalculator_1 processes one packet from stream "input_stream".
  LogInputPackets("PCalculator_1", GraphTrace::PROCESS, curr_time,
                  {MakePacket<std::string>("hello").At(start_timestamp_)});
  curr_time += absl::Microseconds(10000);
  LogOutputPackets("PCalculator_1", GraphTrace::PROCESS, curr_time,
                   {{MakePacket<std::string>("goodbye").At(start_timestamp_)}});

  // Validate the GraphTrace data.
  EXPECT_THAT(
      GetTrace(), EqualsProto(mediapipe::ParseTextProtoOrDie<GraphTrace>(R"pb(
        base_time: 1608911100000000
        base_timestamp: 1608911100000000
        stream_name: ""
        stream_name: "input_stream"
        stream_name: "output_stream"
        calculator_trace {
          node_id: 0
          input_timestamp: 0
          event_type: PROCESS
          start_time: 0
          finish_time: 10000
          thread_id: 0
          input_trace {
            finish_time: 0
            packet_timestamp: 0
            stream_id: 1
            event_data: 1
          }
          output_trace { packet_timestamp: 0 stream_id: 2 event_data: 2 }
        }
      )pb")));
}

TEST_F(GraphTracerTest, GraphTrace) {
  // Define the GraphTracer, the CalculatorState, and the stream specs.
  SetUpGraphTracer();
  SetUpCalculatorContext("PCalculator_1", /*node_id=*/0, {"input_stream"},
                         {"up_1", "up_2"});
  absl::Time curr_time = start_time_;

  // PCalculator_1 sends one packet to stream "up_1", and two to "up_2".
  LogInputPackets("PCalculator_1", GraphTrace::PROCESS, curr_time,
                  {MakePacket<std::string>("hello").At(start_timestamp_)});
  curr_time += absl::Microseconds(10000);
  LogOutputPackets(
      "PCalculator_1", GraphTrace::PROCESS, curr_time,
      {
          {MakePacket<std::string>("up").At(start_timestamp_)},
          {MakePacket<std::string>("up").At(start_timestamp_),
           MakePacket<std::string>("pup").At(start_timestamp_ + 5)},
      });
  curr_time += absl::Microseconds(1000);

  // PCalculator_2 processes one packet from stream "up_1".
  SetUpCalculatorContext("PCalculator_2", /*node_id=*/1, {"up_1"}, {"down_1"});
  LogInputPackets("PCalculator_2", GraphTrace::PROCESS, curr_time,
                  {MakePacket<std::string>("up").At(start_timestamp_)});
  curr_time += absl::Microseconds(10000);
  LogOutputPackets("PCalculator_2", GraphTrace::PROCESS, curr_time,
                   {{MakePacket<std::string>("down_1").At(start_timestamp_)}});
  curr_time -= absl::Microseconds(5000);

  // PCalculator_3 processes two packets from stream "up_2".
  SetUpCalculatorContext("PCalculator_3", /*node_id=*/2, {"up_2"}, {"down_2"});
  LogInputPackets("PCalculator_3", GraphTrace::PROCESS, curr_time,
                  {MakePacket<std::string>("up").At(start_timestamp_)});
  curr_time += absl::Microseconds(20000);
  LogOutputPackets("PCalculator_3", GraphTrace::PROCESS, curr_time,
                   {{MakePacket<std::string>("out").At(start_timestamp_)}});
  curr_time += absl::Microseconds(2000);

  // Note: the packet data ID is based on the packet's payload address, which
  // means the same ID can be reused if data is allocated in the same location
  // as a previously expired packet (b/160212191). This means the generated
  // trace can change depending on the allocator. To keep results stable, we
  // must keep the packets used in this test alive until the end. Each
  // TestContextBuilder happens to keep a reference to all packets for the last
  // context, so for now we just create a separate TestContextBuilder instead of
  // clearing it. TODO: revise this test.
  SetUpCalculatorContext("PCalculator_3a", /*node_id=*/2, {"up_2"}, {"down_2"});
  LogInputPackets("PCalculator_3a", GraphTrace::PROCESS, curr_time,
                  {MakePacket<std::string>("pup").At(start_timestamp_ + 5)});
  curr_time += absl::Microseconds(20000);
  LogOutputPackets(
      "PCalculator_3a", GraphTrace::PROCESS, curr_time,
      {{MakePacket<std::string>("pout").At(start_timestamp_ + 5)}});
  curr_time += absl::Microseconds(1000);

  // Validate the GraphTrace data.
  EXPECT_THAT(
      GetTrace(), EqualsProto(mediapipe::ParseTextProtoOrDie<GraphTrace>(R"pb(
        base_time: 1608911100000000
        base_timestamp: 1608911100000000
        stream_name: ""
        stream_name: "input_stream"
        stream_name: "up_1"
        stream_name: "up_2"
        stream_name: "down_1"
        stream_name: "down_2"
        calculator_trace {
          node_id: 0
          input_timestamp: 0
          event_type: PROCESS
          start_time: 0
          finish_time: 10000
          thread_id: 0
          input_trace {
            finish_time: 0
            packet_timestamp: 0
            stream_id: 1
            event_data: 1
          }
          output_trace { packet_timestamp: 0 stream_id: 2 event_data: 2 }
          output_trace { packet_timestamp: 0 stream_id: 3 event_data: 3 }
          output_trace { packet_timestamp: 5 stream_id: 3 event_data: 4 }
        }
        calculator_trace {
          node_id: 1
          input_timestamp: 0
          event_type: PROCESS
          start_time: 11000
          finish_time: 21000
          thread_id: 0
          input_trace {
            start_time: 10000
            finish_time: 11000
            packet_timestamp: 0
            stream_id: 2
            event_data: 5
          }
          output_trace { packet_timestamp: 0 stream_id: 4 event_data: 6 }
        }
        calculator_trace {
          node_id: 2
          input_timestamp: 0
          event_type: PROCESS
          start_time: 16000
          finish_time: 36000
          thread_id: 0
          input_trace {
            start_time: 10000
            finish_time: 16000
            packet_timestamp: 0
            stream_id: 3
            event_data: 7
          }
          output_trace { packet_timestamp: 0 stream_id: 5 event_data: 8 }
        }
        calculator_trace {
          node_id: 2
          input_timestamp: 5
          event_type: PROCESS
          start_time: 38000
          finish_time: 58000
          thread_id: 0
          input_trace {
            start_time: 10000
            finish_time: 38000
            packet_timestamp: 5
            stream_id: 3
            event_data: 9
          }
          output_trace { packet_timestamp: 5 stream_id: 5 event_data: 10 }
        }
      )pb")));

  // No timestamps are completed before start_time_.
  // One timestamp is completed before start_time_ + 10ms.
  // Two timestamps are completed before start_time_ + 48ms.
  Timestamp ts_0 = tracer_->TimestampAfter(start_time_);
  EXPECT_EQ(Timestamp::Min() + 1, ts_0);
  Timestamp ts_1 =
      tracer_->TimestampAfter(start_time_ + absl::Microseconds(10000));
  EXPECT_EQ(start_timestamp_ + 1, ts_1);
  Timestamp ts_2 =
      tracer_->TimestampAfter(start_time_ + absl::Microseconds(48000));
  EXPECT_EQ(start_timestamp_ + 5 + 1, ts_2);

  // 3 calculators run at start_timestamp_.
  // 1 calculator runs at start_timestamp_ + 5.
  // 4 calculators run between start_timestamp_ and start_timestamp_ + 5 + 1;
  absl::Time t_0 = start_time_;
  absl::Time t_1 = start_time_ + absl::Microseconds(10000);
  absl::Time t_2 = start_time_ + absl::Microseconds(48000);
  GraphTrace trace;
  tracer_->GetTrace(t_0, t_1, &trace);
  EXPECT_EQ(1, trace.calculator_trace().size());
  tracer_->GetTrace(t_1, t_2, &trace);
  EXPECT_EQ(4, trace.calculator_trace().size());
  tracer_->GetTrace(t_0, t_2, &trace);
  EXPECT_EQ(4, trace.calculator_trace().size());
}

// Tests showing GraphTracer logging packet latencies.
class GraphTracerE2ETest : public ::testing::Test {
 protected:
  void SetUpPassThroughGraph() {
    CHECK(proto_ns::TextFormat::ParseFromString(R"(
        input_stream: "input_0"
        node {
          calculator: "LambdaCalculator"
          input_side_packet: 'callback_0'
          input_stream: "input_0"
          output_stream: "output_0"
        }
        profiler_config {
          histogram_interval_size_usec: 1000
          num_histogram_intervals: 100
          trace_enabled: true
        }
        )",
                                                &graph_config_));
  }

  void SetUpDemuxInFlightGraph() {
    CHECK(proto_ns::TextFormat::ParseFromString(R"(
        node {
          calculator: "LambdaCalculator"
          input_side_packet: 'callback_2'
          output_stream: "input_packets_0"
        }
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
        profiler_config {
          histogram_interval_size_usec: 1000
          num_histogram_intervals: 100
          trace_enabled: true
        }
        )",
                                                &graph_config_));
  }

  absl::Time ParseTime(const std::string& date_time_str) {
    absl::Time result;
    absl::ParseTime(absl::RFC3339_sec, date_time_str, &result, nullptr);
    return result;
  }

  absl::Time StartTime() { return ParseTime("2018-12-06T09:00:00Z"); }

  void SetUpSimulationClock() {
    auto executor = std::make_shared<SimulationClockExecutor>(8);
    simulation_clock_ = executor->GetClock();
    clock_ = simulation_clock_.get();
    simulation_clock_->ThreadStart();
    clock_->SleepUntil(StartTime());
    simulation_clock_->ThreadFinish();
    MP_ASSERT_OK(graph_.SetExecutor("", executor));
  }

  void SetUpRealClock() { clock_ = mediapipe::Clock::RealClock(); }

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
  GraphTrace NodeTimestamps(const GraphTrace& trace) {
    GraphTrace result;
    for (auto& name : trace.calculator_name()) {
      result.add_calculator_name(name);
    }
    for (auto& ct : trace.calculator_trace()) {
      auto rt = result.add_calculator_trace();
      rt->set_input_timestamp(ct.input_timestamp() + trace.base_timestamp());
      rt->set_node_id(ct.node_id());
    }
    return result;
  }
  void StripThreadIds(GraphTrace* trace) {
    for (auto& ct : *trace->mutable_calculator_trace()) {
      ct.clear_thread_id();
    }
  }
  void StripDataIds(GraphTrace* trace) {
    TraceBuilder builder;
    for (auto& ct : *trace->mutable_calculator_trace()) {
      if ((*builder.trace_event_registry())[ct.event_type()].id_event_data()) {
        for (auto& st : *ct.mutable_input_trace()) {
          st.clear_event_data();
        }
        for (auto& st : *ct.mutable_output_trace()) {
          st.clear_event_data();
        }
      }
    }
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

  void RunPassThroughGraph() {
    // SetUpSimulationClock can be replaced by SetUpRealClock.
    SetUpSimulationClock();

    // Callbacks to control the LambdaCalculators.
    ProcessFunction wait_0 = [&](const InputStreamShardSet& inputs,
                                 OutputStreamShardSet* outputs) {
      clock_->Sleep(absl::Microseconds(20001));
      return PassThrough(inputs, outputs);
    };

    // Start the graph with the callbacks.
    MP_ASSERT_OK(graph_.Initialize(graph_config_,
                                   {
                                       {"callback_0", Adopt(new auto(wait_0))},
                                   }));
    graph_.profiler()->SetClock(simulation_clock_);
    std::vector<Packet> out_packets;
    MP_ASSERT_OK(
        graph_.ObserveOutputStream("output_0", [&](const Packet& packet) {
          out_packets.push_back(packet);
          return absl::OkStatus();
        }));
    simulation_clock_->ThreadStart();
    MP_ASSERT_OK(graph_.StartRun({}));

    // The first 6 packets to send into the graph at 5001 us intervals.
    for (int ts = 10000; ts < 70000; ts += 10000) {
      clock_->Sleep(absl::Microseconds(5001));
      MP_EXPECT_OK(graph_.AddPacketToInputStream("input_0", PacketAt(ts)));
    }

    // Wait for all packets to be processed.
    MP_ASSERT_OK(graph_.CloseAllPacketSources());
    clock_->Sleep(absl::Microseconds(240000 + 0));
    MP_ASSERT_OK(graph_.WaitUntilDone());
    simulation_clock_->ThreadFinish();

    // Validate the graph run.
    EXPECT_THAT(TimestampValues(out_packets),
                ElementsAre(10000, 20000, 30000, 40000, 50000, 60000));
  }

  void RunDemuxInFlightGraph() {
    // SetUpSimulationClock can be replaced by SetUpRealClock.
    SetUpSimulationClock();

    // Callbacks to control the LambdaCalculators.
    ProcessFunction wait_0 = [&](const InputStreamShardSet& inputs,
                                 OutputStreamShardSet* outputs) {
      clock_->Sleep(absl::Microseconds(20001));
      return PassThrough(inputs, outputs);
    };
    ProcessFunction wait_1 = [&](const InputStreamShardSet& inputs,
                                 OutputStreamShardSet* outputs) {
      clock_->Sleep(absl::Microseconds(30001));
      return PassThrough(inputs, outputs);
    };

    // A callback to control the source LambdaCalculator.
    std::vector<std::pair<int64, Packet>> packets;
    ProcessFunction wait_2 = [&](const InputStreamShardSet& inputs,
                                 OutputStreamShardSet* outputs) {
      if (!packets.empty()) {
        clock_->Sleep(absl::Microseconds(packets.front().first));
        outputs->Index(0).AddPacket(packets.front().second);
        packets.erase(packets.begin());
        return absl::OkStatus();
      }
      return tool::StatusStop();
    };

    // The first 6 packets to send into the graph at 5001 us intervals.
    for (int ts = 10000; ts < 70000; ts += 10000) {
      packets.emplace_back(5001, PacketAt(ts));
    }

    // Start the graph with the callbacks.
    MP_ASSERT_OK(graph_.Initialize(graph_config_,
                                   {
                                       {"max_in_flight", MakePacket<int>(4)},
                                       {"callback_0", Adopt(new auto(wait_0))},
                                       {"callback_1", Adopt(new auto(wait_1))},
                                       {"callback_2", Adopt(new auto(wait_2))},
                                   }));
    graph_.profiler()->SetClock(simulation_clock_);
    std::vector<Packet> out_packets;
    MP_ASSERT_OK(graph_.ObserveOutputStream("output_packets_0",
                                            [&](const Packet& packet) {
                                              out_packets.push_back(packet);
                                              return absl::OkStatus();
                                            }));
    simulation_clock_->ThreadStart();
    MP_ASSERT_OK(graph_.StartRun({}));

    // Wait for all packets to be added and processed.
    clock_->Sleep(absl::Microseconds(160000 + 0));
    MP_ASSERT_OK(graph_.WaitUntilDone());
    simulation_clock_->ThreadFinish();

    // Validate the graph run.
    EXPECT_THAT(TimestampValues(out_packets),
                ElementsAre(10000, 20000, 30000, 50000));
  }

  CalculatorGraphConfig graph_config_;
  CalculatorGraph graph_;
  mediapipe::Clock* clock_;
  std::shared_ptr<SimulationClock> simulation_clock_;
};

// Initialize a TimeHistogram protobuf with some latency values.
void FillHistogram(const std::vector<int64>& values, TimeHistogram* result) {
  result->set_num_intervals(100);
  result->set_interval_size_usec(1000);
  result->mutable_count()->Resize(result->num_intervals(), 0);
  for (int64 v : values) {
    result->set_total(result->total() + v);
    int bin = v / result->interval_size_usec();
    bin = std::min(bin, (int)result->num_intervals() - 1);
    result->set_count(bin, result->count(bin) + 1);
  }
}

// Verify profiler histograms with the PassThrough graph.
TEST_F(GraphTracerE2ETest, PassThroughGraphProfile) {
  SetUpPassThroughGraph();
  graph_config_.mutable_profiler_config()->set_enable_profiler(true);
  graph_config_.mutable_profiler_config()->set_enable_stream_latency(true);
  // Trace log writing should be disabled, otherwise if a default trace log path
  // is set, the GraphProfiler will dump the profiles to that path and empty out
  // the CalculatorProfiles.
  graph_config_.mutable_profiler_config()->set_trace_log_disabled(true);
  RunPassThroughGraph();
  std::vector<CalculatorProfile> profiles;
  MP_EXPECT_OK(graph_.profiler()->GetCalculatorProfiles(&profiles));
  EXPECT_EQ(1, profiles.size());
  CalculatorProfile expected =
      mediapipe::ParseTextProtoOrDie<CalculatorProfile>(R"pb(
        name: "LambdaCalculator"
        open_runtime: 0
        close_runtime: 0
        input_stream_profiles { name: "input_0" back_edge: false })pb");

  FillHistogram({20001, 20001, 20001, 20001, 20001, 20001},
                expected.mutable_process_runtime());
  FillHistogram({0, 15000, 30000, 45000, 60000, 75000},
                expected.mutable_process_input_latency());
  FillHistogram({20001, 35001, 50001, 65001, 80001, 95001},
                expected.mutable_process_output_latency());
  FillHistogram({0, 15000, 30000, 45000, 60000, 75000},
                expected.mutable_input_stream_profiles(0)->mutable_latency());

  EXPECT_THAT(profiles[0], EqualsProto(expected));
  EXPECT_EQ(GraphProfilerTestPeer::GetPacketsInfoMap(graph_.profiler())->size(),
            2);
}

TEST_F(GraphTracerE2ETest, DemuxGraphLog) {
  SetUpDemuxInFlightGraph();
  RunDemuxInFlightGraph();

  // Validate a summary of the event trace.
  GraphTrace trace;
  graph_.profiler()->tracer()->GetLog(absl::InfinitePast(),
                                      absl::InfiniteFuture(), &trace);
  GraphTrace node_timestamps = NodeTimestamps(trace);
  EXPECT_THAT(node_timestamps,
              EqualsProto(mediapipe::ParseTextProtoOrDie<GraphTrace>(R"pb(
                calculator_trace { node_id: 1 input_timestamp: 10000 }
                calculator_trace { node_id: 2 input_timestamp: 10000 }
                calculator_trace { node_id: 3 input_timestamp: 10000 }
                calculator_trace { node_id: 4 input_timestamp: 10000 }
                calculator_trace { node_id: 5 input_timestamp: 10000 }
                calculator_trace { node_id: 0 input_timestamp: 10000 }
                calculator_trace { node_id: 1 input_timestamp: 10000 }
                calculator_trace { node_id: 1 input_timestamp: 10000 }
                calculator_trace { node_id: 1 input_timestamp: 10000 }
                calculator_trace { node_id: 1 input_timestamp: 10000 }
                calculator_trace { node_id: 2 input_timestamp: 10000 }
                calculator_trace { node_id: 2 input_timestamp: 10000 }
                calculator_trace { node_id: 1 input_timestamp: 10000 }
                calculator_trace { node_id: 2 input_timestamp: 10000 }
                calculator_trace { node_id: 2 input_timestamp: 10000 }
                calculator_trace { node_id: 3 input_timestamp: 10000 }
                calculator_trace { node_id: 3 input_timestamp: 10000 }
                calculator_trace { node_id: 4 input_timestamp: 10000 }
                calculator_trace { node_id: 2 input_timestamp: 10000 }
                calculator_trace { node_id: 3 input_timestamp: 10000 }
                calculator_trace { node_id: 0 input_timestamp: 20000 }
                calculator_trace { node_id: 1 input_timestamp: 20000 }
                calculator_trace { node_id: 1 input_timestamp: 10000 }
                calculator_trace { node_id: 1 input_timestamp: 20000 }
                calculator_trace { node_id: 1 input_timestamp: 20000 }
                calculator_trace { node_id: 2 input_timestamp: 20000 }
                calculator_trace { node_id: 2 input_timestamp: 10000 }
                calculator_trace { node_id: 1 input_timestamp: 10000 }
                calculator_trace { node_id: 2 input_timestamp: 20000 }
                calculator_trace { node_id: 2 input_timestamp: 20000 }
                calculator_trace { node_id: 4 input_timestamp: 20000 }
                calculator_trace { node_id: 4 input_timestamp: 10000 }
                calculator_trace { node_id: 2 input_timestamp: 10000 }
                calculator_trace { node_id: 4 input_timestamp: 20000 }
                calculator_trace { node_id: 0 input_timestamp: 30000 }
                calculator_trace { node_id: 1 input_timestamp: 30000 }
                calculator_trace { node_id: 1 input_timestamp: 10000 }
                calculator_trace { node_id: 1 input_timestamp: 30000 }
                calculator_trace { node_id: 1 input_timestamp: 30000 }
                calculator_trace { node_id: 2 input_timestamp: 30000 }
                calculator_trace { node_id: 2 input_timestamp: 10000 }
                calculator_trace { node_id: 1 input_timestamp: 10000 }
                calculator_trace { node_id: 2 input_timestamp: 30000 }
                calculator_trace { node_id: 2 input_timestamp: 30000 }
                calculator_trace { node_id: 3 input_timestamp: 30000 }
                calculator_trace { node_id: 2 input_timestamp: 10000 }
                calculator_trace { node_id: 0 input_timestamp: 40000 }
                calculator_trace { node_id: 1 input_timestamp: 40000 }
                calculator_trace { node_id: 1 input_timestamp: 10000 }
                calculator_trace { node_id: 1 input_timestamp: 40000 }
                calculator_trace { node_id: 1 input_timestamp: 40000 }
                calculator_trace { node_id: 2 input_timestamp: 40000 }
                calculator_trace { node_id: 2 input_timestamp: 10000 }
                calculator_trace { node_id: 1 input_timestamp: 10000 }
                calculator_trace { node_id: 2 input_timestamp: 40000 }
                calculator_trace { node_id: 2 input_timestamp: 40000 }
                calculator_trace { node_id: 4 input_timestamp: 40000 }
                calculator_trace { node_id: 2 input_timestamp: 10000 }
                calculator_trace { node_id: 3 input_timestamp: 10000 }
                calculator_trace { node_id: 5 input_timestamp: 10000 }
                calculator_trace { node_id: 5 input_timestamp: 10000 }
                calculator_trace { node_id: 3 input_timestamp: 10000 }
                calculator_trace { node_id: 5 input_timestamp: 10000 }
                calculator_trace { node_id: 5 input_timestamp: 10000 }
                calculator_trace { node_id: 5 input_timestamp: 10000 }
                calculator_trace { node_id: 1 input_timestamp: 10000 }
                calculator_trace { node_id: 1 input_timestamp: 10000 }
                calculator_trace { node_id: 5 input_timestamp: 10000 }
                calculator_trace { node_id: 3 input_timestamp: 30000 }
                calculator_trace { node_id: 1 input_timestamp: 10000 }
                calculator_trace { node_id: 1 input_timestamp: 10000 }
                calculator_trace { node_id: 0 input_timestamp: 50000 }
                calculator_trace { node_id: 1 input_timestamp: 50000 }
                calculator_trace { node_id: 1 input_timestamp: 10000 }
                calculator_trace { node_id: 1 input_timestamp: 50000 }
                calculator_trace { node_id: 1 input_timestamp: 50000 }
                calculator_trace { node_id: 2 input_timestamp: 50000 }
                calculator_trace { node_id: 2 input_timestamp: 10000 }
                calculator_trace { node_id: 1 input_timestamp: 10000 }
                calculator_trace { node_id: 2 input_timestamp: 50000 }
                calculator_trace { node_id: 2 input_timestamp: 50000 }
                calculator_trace { node_id: 3 input_timestamp: 50000 }
                calculator_trace { node_id: 2 input_timestamp: 10000 }
                calculator_trace { node_id: 0 input_timestamp: 60000 }
                calculator_trace { node_id: 1 input_timestamp: 60000 }
                calculator_trace { node_id: 1 input_timestamp: 10000 }
                calculator_trace { node_id: 1 input_timestamp: 60000 }
                calculator_trace { node_id: 2 input_timestamp: 10000 }
                calculator_trace { node_id: 1 input_timestamp: 10000 }
                calculator_trace { node_id: 1 input_timestamp: 10000 }
                calculator_trace { node_id: 2 input_timestamp: 10000 }
                calculator_trace { node_id: 1 input_timestamp: 10000 }
                calculator_trace { node_id: 4 input_timestamp: 20000 }
                calculator_trace { node_id: 5 input_timestamp: 20000 }
                calculator_trace { node_id: 5 input_timestamp: 10000 }
                calculator_trace { node_id: 4 input_timestamp: 10000 }
                calculator_trace { node_id: 5 input_timestamp: 20000 }
                calculator_trace { node_id: 5 input_timestamp: 20000 }
                calculator_trace { node_id: 5 input_timestamp: 20000 }
                calculator_trace { node_id: 1 input_timestamp: 20000 }
                calculator_trace { node_id: 1 input_timestamp: 10000 }
                calculator_trace { node_id: 5 input_timestamp: 10000 }
                calculator_trace { node_id: 4 input_timestamp: 40000 }
                calculator_trace { node_id: 1 input_timestamp: 20000 }
                calculator_trace { node_id: 1 input_timestamp: 10000 }
                calculator_trace { node_id: 3 input_timestamp: 30000 }
                calculator_trace { node_id: 5 input_timestamp: 30000 }
                calculator_trace { node_id: 5 input_timestamp: 10000 }
                calculator_trace { node_id: 3 input_timestamp: 10000 }
                calculator_trace { node_id: 5 input_timestamp: 30000 }
                calculator_trace { node_id: 5 input_timestamp: 30000 }
                calculator_trace { node_id: 5 input_timestamp: 30000 }
                calculator_trace { node_id: 1 input_timestamp: 30000 }
                calculator_trace { node_id: 1 input_timestamp: 10000 }
                calculator_trace { node_id: 5 input_timestamp: 10000 }
                calculator_trace { node_id: 3 input_timestamp: 50000 }
                calculator_trace { node_id: 1 input_timestamp: 30000 }
                calculator_trace { node_id: 1 input_timestamp: 10000 }
                calculator_trace { node_id: 3 input_timestamp: 50000 }
                calculator_trace { node_id: 5 input_timestamp: 50000 }
                calculator_trace { node_id: 5 input_timestamp: 10000 }
                calculator_trace { node_id: 3 input_timestamp: 10000 }
                calculator_trace { node_id: 5 input_timestamp: 50000 }
                calculator_trace { node_id: 5 input_timestamp: 50000 }
                calculator_trace { node_id: 5 input_timestamp: 50000 }
                calculator_trace { node_id: 1 input_timestamp: 50000 }
                calculator_trace { node_id: 1 input_timestamp: 10000 }
                calculator_trace { node_id: 5 input_timestamp: 10000 }
                calculator_trace { node_id: 5 input_timestamp: 10000 }
                calculator_trace { node_id: 5 input_timestamp: 10000 }
                calculator_trace { node_id: 1 input_timestamp: 50000 }
                calculator_trace { node_id: 1 input_timestamp: 10000 }
                calculator_trace { node_id: 4 input_timestamp: 40000 }
                calculator_trace { node_id: 5 input_timestamp: 40000 }
                calculator_trace { node_id: 5 input_timestamp: 10000 }
                calculator_trace { node_id: 4 input_timestamp: 10000 }
                calculator_trace { node_id: 5 input_timestamp: 40000 }
                calculator_trace { node_id: 5 input_timestamp: 40000 }
                calculator_trace { node_id: 1 input_timestamp: 50001 }
                calculator_trace { node_id: 1 input_timestamp: 10000 }
                calculator_trace { node_id: 5 input_timestamp: 10000 }
                calculator_trace { node_id: 5 input_timestamp: 10000 }
                calculator_trace { node_id: 1 input_timestamp: 50001 }
                calculator_trace { node_id: 1 input_timestamp: 10000 }
                calculator_trace { node_id: 1 input_timestamp: 10000 }
              )pb")));

  // Validate a one-timestamp slice of the event trace.
  GraphTrace trace_2;
  graph_.profiler()->tracer()->GetLog(StartTime() + absl::Microseconds(25000),
                                      StartTime() + absl::Microseconds(30005),
                                      &trace_2);
  StripThreadIds(&trace_2);
  StripDataIds(&trace_2);
  EXPECT_THAT(
      trace_2,
      EqualsProto(mediapipe::ParseTextProtoOrDie<GraphTrace>(
          R"pb(
            base_time: 1544086800000000
            base_timestamp: 10000
            stream_name: ""
            stream_name: "input_packets_0"
            stream_name: "input_0_sampled"
            stream_name: "input_0"
            stream_name: "input_1"
            stream_name: "output_0"
            stream_name: "output_packets_0"
            stream_name: "finish_indicator"
            stream_name: "output_1"
            calculator_trace {
              node_id: 3
              input_timestamp: 0
              event_type: PROCESS
              finish_time: 25002
              output_trace { packet_timestamp: 0 stream_id: 5 }
            }
            calculator_trace {
              node_id: 5
              input_timestamp: 0
              event_type: PACKET_QUEUED
              start_time: 25002
              input_trace { packet_timestamp: 0 stream_id: 5 event_data: 1 }
            }
            calculator_trace {
              node_id: 5
              event_type: READY_FOR_PROCESS
              start_time: 25002
            }
            calculator_trace {
              node_id: 3
              event_type: READY_FOR_PROCESS
              start_time: 25002
            }
            calculator_trace {
              node_id: 5
              input_timestamp: 0
              event_type: PROCESS
              start_time: 25002
              input_trace { packet_timestamp: 0 stream_id: 5 }
            }
            calculator_trace {
              node_id: 5
              input_timestamp: 0
              event_type: PROCESS
              finish_time: 25002
              output_trace { packet_timestamp: 0 stream_id: 6 }
            }
            calculator_trace {
              node_id: 5
              input_timestamp: 0
              event_type: PROCESS
              finish_time: 25002
              output_trace { packet_timestamp: 0 stream_id: 7 }
            }
            calculator_trace {
              node_id: 1
              input_timestamp: 0
              event_type: PACKET_QUEUED
              start_time: 25002
              input_trace { packet_timestamp: 0 stream_id: 7 event_data: 1 }
            }
            calculator_trace {
              node_id: 1
              event_type: READY_FOR_PROCESS
              start_time: 25002
            }
            calculator_trace {
              node_id: 5
              event_type: NOT_READY
              start_time: 25002
            }
            calculator_trace {
              node_id: 3
              input_timestamp: 20000
              event_type: PROCESS
              start_time: 25002
              input_trace { packet_timestamp: 20000 stream_id: 3 }
            }
            calculator_trace {
              node_id: 1
              input_timestamp: 0
              event_type: PROCESS
              start_time: 25002
              input_trace { packet_timestamp: 0 stream_id: 7 }
            }
            calculator_trace {
              node_id: 1
              event_type: NOT_READY
              start_time: 25002
            }
            calculator_trace {
              node_id: 0
              input_timestamp: 40000
              event_type: PROCESS
              finish_time: 25005
              output_trace { packet_timestamp: 40000 stream_id: 1 }
            }
            calculator_trace {
              node_id: 1
              input_timestamp: 40000
              event_type: PACKET_QUEUED
              start_time: 25005
              input_trace { packet_timestamp: 40000 stream_id: 1 event_data: 1 }
            }
            calculator_trace {
              node_id: 1
              event_type: READY_FOR_PROCESS
              start_time: 25005
            }
            calculator_trace {
              node_id: 1
              input_timestamp: 40000
              event_type: PROCESS
              start_time: 25005
              input_trace { packet_timestamp: 40000 stream_id: 1 }
            }
            calculator_trace {
              node_id: 1
              input_timestamp: 40000
              event_type: PROCESS
              finish_time: 25005
              output_trace { packet_timestamp: 40000 stream_id: 2 }
            }
            calculator_trace {
              node_id: 2
              input_timestamp: 40000
              event_type: PACKET_QUEUED
              start_time: 25005
              input_trace { packet_timestamp: 40000 stream_id: 2 event_data: 1 }
            }
            calculator_trace {
              node_id: 2
              event_type: READY_FOR_PROCESS
              start_time: 25005
            }
            calculator_trace {
              node_id: 1
              event_type: NOT_READY
              start_time: 25005
            }
            calculator_trace {
              node_id: 2
              input_timestamp: 40000
              event_type: PROCESS
              start_time: 25005
              input_trace { packet_timestamp: 40000 stream_id: 2 }
            }
            calculator_trace {
              node_id: 2
              input_timestamp: 40000
              event_type: PROCESS
              finish_time: 25005
              output_trace { packet_timestamp: 40000 stream_id: 3 }
            }
            calculator_trace {
              node_id: 3
              input_timestamp: 40000
              event_type: PACKET_QUEUED
              start_time: 25005
              input_trace { packet_timestamp: 40000 stream_id: 3 event_data: 1 }
            }
            calculator_trace {
              node_id: 2
              event_type: NOT_READY
              start_time: 25005
            }
          )pb")));
}

// Read a GraphProfile from a file path.
absl::Status ReadGraphProfile(const std::string& path, GraphProfile* profile) {
  std::ifstream ifs;
  ifs.open(path);
  proto_ns::io::IstreamInputStream in_stream(&ifs);
  profile->ParseFromZeroCopyStream(&in_stream);
  return ifs.is_open() ? absl::OkStatus()
                       : absl::UnavailableError("Cannot open");
}

TEST_F(GraphTracerE2ETest, DemuxGraphLogFile) {
  std::string log_path = absl::StrCat(getenv("TEST_TMPDIR"), "/log_file_");
  SetUpDemuxInFlightGraph();
  graph_config_.mutable_profiler_config()->set_trace_log_path(log_path);
  graph_config_.mutable_profiler_config()->set_trace_log_interval_usec(-1);
  RunDemuxInFlightGraph();
  GraphProfile profile;
  MP_EXPECT_OK(
      ReadGraphProfile(absl::StrCat(log_path, 0, ".binarypb"), &profile));
  EXPECT_EQ(113, profile.graph_trace(0).calculator_trace().size());
}

TEST_F(GraphTracerE2ETest, DemuxGraphLogFiles) {
  std::string log_path = absl::StrCat(getenv("TEST_TMPDIR"), "/log_files_");
  SetUpDemuxInFlightGraph();
  graph_config_.mutable_profiler_config()->set_trace_log_path(log_path);
  graph_config_.mutable_profiler_config()->set_trace_log_count(100);
  graph_config_.mutable_profiler_config()->set_trace_log_interval_count(5);
  graph_config_.mutable_profiler_config()->set_trace_log_interval_usec(2500);
  RunDemuxInFlightGraph();
  std::vector<int> event_counts;
  std::vector<GraphProfile> graph_profiles;
  for (int i = 0; i < 7; ++i) {
    GraphProfile profile;
    std::string log_file_name = absl::StrCat(log_path, i, ".binarypb");
    if (ReadGraphProfile(log_file_name, &profile).ok()) {
      int count = 0;
      for (auto trace : *profile.mutable_graph_trace()) {
        count += trace.calculator_trace().size();
      }
      event_counts.push_back(count);
      graph_profiles.push_back(profile);
    }
  }

  // The expected counts of calculator_trace records in each of the log files.
  // The processing spans three 12.5ms log files, because
  // RunDemuxInFlightGraph adds packets over 30ms.
  std::vector<int> expected = {50, 64, 12};
  EXPECT_EQ(event_counts, expected);
  GraphProfile& profile_2 = graph_profiles[2];
  profile_2.clear_calculator_profiles();
  profile_2.mutable_config()->mutable_profiler_config()->clear_trace_log_path();
  for (auto& trace : *profile_2.mutable_graph_trace()) {
    StripThreadIds(&trace);
    StripDataIds(&trace);
  }
  EXPECT_THAT(profile_2,
              EqualsProto(mediapipe::ParseTextProtoOrDie<GraphProfile>(R"pb(
                graph_trace {
                  base_time: 1544086800000000
                  base_timestamp: 0
                  calculator_name: "LambdaCalculator_1"
                  calculator_name: "FlowLimiterCalculator"
                  calculator_name: "RoundRobinDemuxCalculator"
                  calculator_name: "LambdaCalculator_2"
                  calculator_name: "LambdaCalculator_3"
                  calculator_name: "ImmediateMuxCalculator"
                  stream_name: ""
                  stream_name: "input_packets_0"
                  stream_name: "input_0_sampled"
                  stream_name: "input_0"
                  stream_name: "input_1"
                  stream_name: "output_0"
                  stream_name: "output_packets_0"
                  stream_name: "finish_indicator"
                  stream_name: "output_1"
                  calculator_trace {
                    node_id: 4
                    input_timestamp: 40000
                    event_type: PROCESS
                    finish_time: 70004
                    output_trace { packet_timestamp: 40000 stream_id: 8 }
                  }
                  calculator_trace {
                    node_id: 5
                    input_timestamp: 40000
                    event_type: PACKET_QUEUED
                    start_time: 70004
                    input_trace {
                      finish_time: 70004
                      packet_timestamp: 40000
                      stream_id: 8
                      event_data: 1
                    }
                  }
                  calculator_trace {
                    node_id: 5
                    event_type: READY_FOR_PROCESS
                    start_time: 70004
                  }
                  calculator_trace {
                    node_id: 4
                    event_type: READY_FOR_CLOSE
                    start_time: 70004
                  }
                  calculator_trace {
                    node_id: 5
                    input_timestamp: 40000
                    event_type: PROCESS
                    start_time: 70004
                    finish_time: 70004
                    input_trace {
                      start_time: 70004
                      finish_time: 70004
                      packet_timestamp: 40000
                      stream_id: 8
                    }
                    output_trace { packet_timestamp: 50001 stream_id: 7 }
                  }
                  calculator_trace {
                    node_id: 1
                    input_timestamp: 50001
                    event_type: PACKET_QUEUED
                    start_time: 70004
                    input_trace {
                      finish_time: 70004
                      packet_timestamp: 50001
                      stream_id: 7
                      event_data: 1
                    }
                  }
                  calculator_trace {
                    node_id: 1
                    event_type: READY_FOR_PROCESS
                    start_time: 70004
                  }
                  calculator_trace {
                    node_id: 5
                    event_type: NOT_READY
                    start_time: 70004
                  }
                  calculator_trace {
                    node_id: 5
                    event_type: READY_FOR_CLOSE
                    start_time: 70004
                  }
                  calculator_trace {
                    node_id: 1
                    input_timestamp: 50001
                    event_type: PROCESS
                    start_time: 70004
                    input_trace {
                      start_time: 70004
                      finish_time: 70004
                      packet_timestamp: 50001
                      stream_id: 7
                    }
                  }
                  calculator_trace {
                    node_id: 1
                    event_type: READY_FOR_PROCESS
                    start_time: 70004
                  }
                  calculator_trace {
                    node_id: 1
                    event_type: READY_FOR_CLOSE
                    start_time: 70004
                  }
                }
                graph_trace {
                  base_time: 1544086800000000
                  base_timestamp: 0
                  stream_name: ""
                  stream_name: "input_packets_0"
                  stream_name: "input_0_sampled"
                  stream_name: "input_0"
                  stream_name: "input_1"
                  stream_name: "output_0"
                  stream_name: "output_packets_0"
                  stream_name: "finish_indicator"
                  stream_name: "output_1"
                }
                config {
                  node {
                    name: "LambdaCalculator_1"
                    calculator: "LambdaCalculator"
                    output_stream: "input_packets_0"
                    input_side_packet: "callback_2"
                  }
                  node {
                    name: "FlowLimiterCalculator"
                    calculator: "FlowLimiterCalculator"
                    input_stream: "input_packets_0"
                    input_stream: "FINISHED:finish_indicator"
                    output_stream: "input_0_sampled"
                    input_side_packet: "MAX_IN_FLIGHT:max_in_flight"
                    input_stream_handler {
                      input_stream_handler: "ImmediateInputStreamHandler"
                    }
                    input_stream_info { tag_index: "FINISHED" back_edge: true }
                  }
                  node {
                    name: "RoundRobinDemuxCalculator"
                    calculator: "RoundRobinDemuxCalculator"
                    input_stream: "input_0_sampled"
                    output_stream: "OUTPUT:0:input_0"
                    output_stream: "OUTPUT:1:input_1"
                  }
                  node {
                    name: "LambdaCalculator_2"
                    calculator: "LambdaCalculator"
                    input_stream: "input_0"
                    output_stream: "output_0"
                    input_side_packet: "callback_0"
                  }
                  node {
                    name: "LambdaCalculator_3"
                    calculator: "LambdaCalculator"
                    input_stream: "input_1"
                    output_stream: "output_1"
                    input_side_packet: "callback_1"
                  }
                  node {
                    name: "ImmediateMuxCalculator"
                    calculator: "ImmediateMuxCalculator"
                    input_stream: "output_0"
                    input_stream: "output_1"
                    output_stream: "output_packets_0"
                    output_stream: "finish_indicator"
                    input_stream_handler {
                      input_stream_handler: "ImmediateInputStreamHandler"
                    }
                  }
                  executor {}
                  profiler_config {
                    histogram_interval_size_usec: 1000
                    num_histogram_intervals: 100
                    trace_log_count: 100
                    trace_log_interval_usec: 2500
                    trace_log_interval_count: 5
                    trace_enabled: true
                  }
                }
              )pb")));
}

TEST_F(GraphTracerE2ETest, DisableLoggingToDisk) {
  std::string log_path =
      absl::StrCat(getenv("TEST_TMPDIR"), "/log_file_disabled_");
  SetUpDemuxInFlightGraph();
  graph_config_.mutable_profiler_config()->set_trace_log_path(log_path);
  graph_config_.mutable_profiler_config()->set_trace_log_disabled(true);
  RunDemuxInFlightGraph();
  EXPECT_TRUE(absl::IsNotFound(
      mediapipe::file::Exists(absl::StrCat(log_path, 0, ".binarypb"))));
}

TEST_F(GraphTracerE2ETest, LoggingHappensWithDefaultPath) {
  std::string log_path = "/tmp/mediapipe_trace_0.binarypb";
  SetUpDemuxInFlightGraph();
  graph_config_.mutable_profiler_config()->set_trace_log_disabled(false);
  RunDemuxInFlightGraph();
  MP_EXPECT_OK(mediapipe::file::Exists(log_path));
}

TEST_F(GraphTracerE2ETest, GpuTaskTrace) {
  std::string stream_1 = "stream_1";
  std::string stream_2 = "stream_2";
  TraceBuffer buffer(10000);
  buffer.push_back(TraceEvent(TraceEvent::PROCESS)
                       .set_event_time(absl::FromUnixMicros(1100))
                       .set_node_id(333)
                       .set_stream_id(&stream_1)
                       .set_input_ts(Timestamp(1000))
                       .set_packet_ts(Timestamp(1000))
                       .set_is_finish(false));
  buffer.push_back(TraceEvent(TraceEvent::GPU_TASK)
                       .set_event_time(absl::FromUnixMicros(1200))
                       .set_node_id(333)
                       .set_stream_id(&stream_1)
                       .set_input_ts(Timestamp(1000))
                       .set_packet_ts(Timestamp(1000))
                       .set_is_finish(false));
  buffer.push_back(TraceEvent(TraceEvent::GPU_TASK)
                       .set_event_time(absl::FromUnixMicros(3200))
                       .set_node_id(333)
                       .set_stream_id(&stream_1)
                       .set_input_ts(Timestamp(1000))
                       .set_packet_ts(Timestamp(1000))
                       .set_is_finish(true));
  buffer.push_back(TraceEvent(TraceEvent::PROCESS)
                       .set_event_time(absl::FromUnixMicros(2100))
                       .set_node_id(333)
                       .set_stream_id(&stream_2)
                       .set_input_ts(Timestamp(1000))
                       .set_packet_ts(Timestamp(1000))
                       .set_is_finish(true));

  TraceBuilder builder;
  GraphTrace trace_1;
  builder.CreateTrace(buffer, absl::InfinitePast(), absl::InfiniteFuture(),
                      &trace_1);
  EXPECT_THAT(
      trace_1,
      EqualsProto(mediapipe::ParseTextProtoOrDie<GraphTrace>(
          R"pb(
            base_time: 1100
            base_timestamp: 1000
            stream_name: ""
            stream_name: "stream_1"
            stream_name: "stream_2"
            calculator_trace {
              node_id: 333
              input_timestamp: 0
              event_type: PROCESS
              start_time: 0
              finish_time: 1000
              input_trace {
                finish_time: 0
                packet_timestamp: 0
                stream_id: 1
                event_data: 0
              }
              output_trace { packet_timestamp: 0 stream_id: 2 event_data: 0 }
              thread_id: 0
            }
            calculator_trace {
              node_id: 333
              input_timestamp: 0
              event_type: GPU_TASK
              start_time: 100
              finish_time: 2100
              thread_id: 0
            }
          )pb")));

  GraphTrace trace_2;
  builder.CreateLog(buffer, absl::InfinitePast(), absl::InfiniteFuture(),
                    &trace_2);
  EXPECT_THAT(
      trace_2,
      EqualsProto(mediapipe::ParseTextProtoOrDie<GraphTrace>(
          R"pb(
            base_time: 1100
            base_timestamp: 1000
            stream_name: ""
            stream_name: "stream_1"
            stream_name: "stream_2"
            calculator_trace {
              node_id: 333
              input_timestamp: 0
              event_type: PROCESS
              start_time: 0
              input_trace { packet_timestamp: 0 stream_id: 1 event_data: 0 }
              thread_id: 0
            }
            calculator_trace {
              node_id: 333
              input_timestamp: 0
              event_type: GPU_TASK
              start_time: 100
              thread_id: 0
            }
            calculator_trace {
              node_id: 333
              input_timestamp: 0
              event_type: GPU_TASK
              finish_time: 2100
              thread_id: 0
            }
            calculator_trace {
              node_id: 333
              input_timestamp: 0
              event_type: PROCESS
              finish_time: 1000
              output_trace { packet_timestamp: 0 stream_id: 2 event_data: 0 }
              thread_id: 0
            }
          )pb")));
}

// Show that trace_enabled activates the GlContextProfiler.
TEST_F(GraphTracerE2ETest, GpuTracing) {
  CHECK(proto_ns::TextFormat::ParseFromString(R"(
        input_stream: "input_buffer"
        input_stream: "render_data"
        output_stream: "annotated_buffer"
        node {
          calculator: "AnnotationOverlayCalculator"
          input_stream: "IMAGE:input_buffer"
          input_stream: "render_data"
          output_stream: "IMAGE:annotated_buffer"
        }
        profiler_config {
          trace_enabled: true
        }
        )",
                                              &graph_config_));

  // Create the CalculatorGraph with only trace_enabled set.
  MP_ASSERT_OK(graph_.Initialize(graph_config_, {}));
  // Check that GPU profiling is enabled wihout running the graph.
  // This graph with GlFlatColorCalculator cannot run on desktop.
  EXPECT_NE(nullptr, graph_.profiler()->CreateGlProfilingHelper());
}

// This test shows that ~CalculatorGraph() can complete successfully, even when
// the periodic profiler output is enabled.  If periodic profiler output is not
// stopped in ~CalculatorGraph(), it will deadlock at ~Executor().
TEST_F(GraphTracerE2ETest, DestructGraph) {
  std::string log_path = absl::StrCat(getenv("TEST_TMPDIR"), "/log_file_");
  SetUpPassThroughGraph();
  graph_config_.mutable_profiler_config()->set_trace_enabled(true);
  graph_config_.mutable_profiler_config()->set_trace_log_path(log_path);
  graph_config_.set_num_threads(4);

  // Callbacks to control the LambdaCalculator.
  ProcessFunction wait_0 = [&](const InputStreamShardSet& inputs,
                               OutputStreamShardSet* outputs) {
    return PassThrough(inputs, outputs);
  };

  {
    CalculatorGraph graph;
    // Start the graph with the callback.
    MP_ASSERT_OK(graph.Initialize(graph_config_,
                                  {
                                      {"callback_0", Adopt(new auto(wait_0))},
                                  }));
    MP_ASSERT_OK(graph.StartRun({}));
    // Destroy the graph immediately.
  }
}

}  // namespace
}  // namespace mediapipe
