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

#include "mediapipe/framework/profiler/graph_profiler.h"

#include "absl/log/absl_log.h"
#include "absl/status/statusor.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/time.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/mediapipe_profiling.h"
#include "mediapipe/framework/port/core_proto_inc.h"
#include "mediapipe/framework/port/gmock.h"
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/framework/port/status_matchers.h"
#include "mediapipe/framework/port/statusor.h"
#include "mediapipe/framework/profiler/test_context_builder.h"
#include "mediapipe/framework/tool/simulation_clock.h"
#include "mediapipe/framework/tool/tag_map_helper.h"

using ::testing::proto::Partially;

namespace mediapipe {

namespace {

constexpr char kDummyTestCalculatorName[] = "DummyTestCalculator";

CalculatorGraphConfig::Node CreateNodeConfig(
    const std::string& raw_node_config) {
  CalculatorGraphConfig::Node node_config;
  QCHECK(google::protobuf::TextFormat::ParseFromString(raw_node_config,
                                                       &node_config));
  return node_config;
}

CalculatorGraphConfig CreateGraphConfig(const std::string& raw_graph_config) {
  CalculatorGraphConfig graph_config;
  QCHECK(google::protobuf::TextFormat::ParseFromString(raw_graph_config,
                                                       &graph_config));
  return graph_config;
}

CalculatorProfile GetProfileWithName(
    const std::vector<CalculatorProfile>& profiles,
    const std::string& calculator_name) {
  for (const CalculatorProfile& p : profiles) {
    if (p.name() == calculator_name) {
      return p;
    }
  }
  ABSL_LOG(FATAL) << "Cannot find calulator profile with name "
                  << calculator_name;
  return CalculatorProfile::default_instance();
}

TimeHistogram CreateTimeHistogram(int64_t total, std::vector<int64_t> counts) {
  TimeHistogram time_histogram;
  time_histogram.set_total(total);
  for (int64_t c : counts) {
    time_histogram.add_count(c);
  }
  return time_histogram;
}

using PacketInfoMap =
    ShardedMap<std::string, std::list<std::pair<int64_t, PacketInfo>>>;

// Returns a PacketInfo from a PacketInfoMap.
PacketInfo* GetPacketInfo(PacketInfoMap* map, const PacketId& packet_id) {
  auto entry = map->find(packet_id.stream_name);
  if (entry == map->end()) {
    return nullptr;
  }
  auto& list = entry->second;
  for (auto iter = list.rbegin(); iter != list.rend(); ++iter) {
    if (iter->first == packet_id.timestamp_usec) {
      return &iter->second;
    }
  }
  return nullptr;
}
}  // namespace

class GraphProfilerTestPeer : public testing::Test {
 public:
  void SetUp() override { packet_type_.SetAny(); }

  bool GetIsInitialized() {
    absl::ReaderMutexLock lock(&profiler_.profiler_mutex_);
    return profiler_.is_initialized_;
  }

  bool GetIsProfiling() {
    absl::ReaderMutexLock lock(&profiler_.profiler_mutex_);
    return profiler_.is_profiling_;
  }

  bool GetIsProfilingStreamLatency() {
    absl::ReaderMutexLock lock(&profiler_.profiler_mutex_);
    return profiler_.profiler_config_.enable_stream_latency();
  }

  bool GetTraceLogDisabled() {
    absl::ReaderMutexLock lock(&profiler_.profiler_mutex_);
    return profiler_.profiler_config_.trace_log_disabled();
  }

  bool GetUsePacketTimeStampForAddedPacket() {
    absl::ReaderMutexLock lock(&profiler_.profiler_mutex_);
    return profiler_.profiler_config_.use_packet_timestamp_for_added_packet();
  }

  GraphProfiler::CalculatorProfileMap* GetCalculatorProfilesMap() {
    return &profiler_.calculator_profiles_;
  }

  CalculatorProfile FindCalculatorProfile(const std::string& expected_name) {
    return GetCalculatorProfilesMap()->find(expected_name)->second;
  }

  GraphProfiler::PacketInfoMap* GetPacketsInfoMap() {
    return &profiler_.packets_info_;
  }

  static void InitializeTimeHistogram(int64_t interval_size_usec,
                                      int64_t num_intervals,
                                      TimeHistogram* histogram) {
    GraphProfiler::InitializeTimeHistogram(interval_size_usec, num_intervals,
                                           histogram);
  }

  static void AddTimeSample(int64_t start_time_usec, int64_t end_time_usec,
                            TimeHistogram* histogram) {
    GraphProfiler::AddTimeSample(start_time_usec, end_time_usec, histogram);
  }

  void InitializeOutputStreams(const CalculatorGraphConfig::Node& node_config) {
    profiler_.InitializeOutputStreams(node_config);
  }

  void InitializeInputStreams(const CalculatorGraphConfig::Node& node_config,
                              int64_t interval_size_usec, int64_t num_intervals,
                              CalculatorProfile* calculator_profile) {
    profiler_.InitializeInputStreams(node_config, interval_size_usec,
                                     num_intervals, calculator_profile);
  }

  void InitializeProfilerWithGraphConfig(const std::string& raw_graph_config) {
    auto graph_config = CreateGraphConfig(raw_graph_config);
    mediapipe::ValidatedGraphConfig validated_graph;
    QCHECK_OK(validated_graph.Initialize(graph_config));
    profiler_.Initialize(validated_graph);
    QCHECK_OK(profiler_.Start(nullptr));
  }

  void SetOpenRuntime(const CalculatorContext& calculator_context,
                      int64_t start_time_usec, int64_t end_time_usec) {
    profiler_.SetOpenRuntime(calculator_context, start_time_usec,
                             end_time_usec);
  }

  void SetCloseRuntime(const CalculatorContext& calculator_context,
                       int64_t start_time_usec, int64_t end_time_usec) {
    profiler_.SetCloseRuntime(calculator_context, start_time_usec,
                              end_time_usec);
  }

  // Updates the Process() data for calculator.
  // Requires ReaderLock for is_profiling_.
  void AddProcessSample(const CalculatorContext& calculator_context,
                        int64_t start_time_usec, int64_t end_time_usec) {
    profiler_.AddProcessSample(calculator_context, start_time_usec,
                               end_time_usec);
  }

  OutputStreamSpec CreateOutputStreamSpec(const std::string& name) {
    OutputStreamSpec output_stream_spec;
    output_stream_spec.name = name;
    output_stream_spec.packet_type = &packet_type_;
    return output_stream_spec;
  }

  void CheckHasProfilesWithInputStreamName(
      const std::string& expected_name,
      const std::vector<std::string>& expected_stream_names) {
    CalculatorProfile profile =
        GetCalculatorProfilesMap()->find(expected_name)->second;
    ASSERT_EQ(profile.name(), expected_name);
    ASSERT_EQ(profile.input_stream_profiles().size(),
              expected_stream_names.size())
        << " for " << expected_name;
    for (int i = 0; i < expected_stream_names.size(); ++i) {
      ASSERT_EQ(profile.input_stream_profiles(i).name(),
                expected_stream_names[i]);
    }
  }

  std::vector<CalculatorProfile> Profiles() {
    std::vector<CalculatorProfile> result;
    MP_EXPECT_OK(profiler_.GetCalculatorProfiles(&result));
    return result;
  }

  std::shared_ptr<ProfilingContext> profiler_ptr_ =
      std::make_shared<ProfilingContext>();
  ProfilingContext& profiler_ = *profiler_ptr_;
  PacketType packet_type_;
};

namespace {

// Tests that Initialize() uses the ProfilerConfig in the graph definition
// including enable_stream_latency.
TEST_F(GraphProfilerTestPeer, InitializeConfig) {
  // Checks defaults before initialization.
  ASSERT_EQ(GetIsInitialized(), false);
  ASSERT_EQ(GetIsProfiling(), false);
  ASSERT_EQ(GetIsProfilingStreamLatency(), false);
  ASSERT_EQ(GetTraceLogDisabled(), false);
  ASSERT_EQ(GetUsePacketTimeStampForAddedPacket(), false);
  InitializeProfilerWithGraphConfig(R"(
    profiler_config {
      histogram_interval_size_usec: 1000
      num_histogram_intervals: 3
      enable_profiler: true
      enable_stream_latency: true
      use_packet_timestamp_for_added_packet: true
      trace_log_disabled: true
    }
    input_stream: "input_stream"
    node {
      calculator: "DummyTestCalculator"
      input_stream: "input_stream"
    })");
  ASSERT_EQ(GetIsInitialized(), true);
  ASSERT_EQ(GetIsProfiling(), true);
  ASSERT_EQ(GetIsProfilingStreamLatency(), true);
  ASSERT_EQ(GetTraceLogDisabled(), true);
  ASSERT_EQ(GetUsePacketTimeStampForAddedPacket(), true);
  // Checks histogram_interval_size_usec and num_histogram_intervals.
  CalculatorProfile actual =
      GetCalculatorProfilesMap()->find(kDummyTestCalculatorName)->second;
  EXPECT_THAT(actual, EqualsProto(R"pb(
                name: "DummyTestCalculator"
                process_runtime {
                  total: 0
                  interval_size_usec: 1000
                  num_intervals: 3
                  count: 0
                  count: 0
                  count: 0
                }
                process_input_latency {
                  total: 0
                  interval_size_usec: 1000
                  num_intervals: 3
                  count: 0
                  count: 0
                  count: 0
                }
                process_output_latency {
                  total: 0
                  interval_size_usec: 1000
                  num_intervals: 3
                  count: 0
                  count: 0
                  count: 0
                }
                input_stream_profiles {
                  name: "input_stream"
                  back_edge: false
                  latency {
                    total: 0
                    interval_size_usec: 1000
                    num_intervals: 3
                    count: 0
                    count: 0
                    count: 0
                  }
                }
              )pb"));
}

// Tests that Initialize() uses the ProfilerConfig in the graph definition.
TEST_F(GraphProfilerTestPeer, InitializeConfigWithoutStreamLatency) {
  // Checks defaults before initialization.
  ASSERT_EQ(GetIsProfiling(), false);
  ASSERT_EQ(GetIsProfilingStreamLatency(), false);
  ASSERT_EQ(GetUsePacketTimeStampForAddedPacket(), false);
  InitializeProfilerWithGraphConfig(R"(
    profiler_config {
      histogram_interval_size_usec: 1000
      num_histogram_intervals: 3
      enable_profiler: true
    }
    input_stream: "input_stream"
    node {
      calculator: "DummyTestCalculator"
      input_stream: "input_stream"
    })");
  ASSERT_EQ(GetIsProfiling(), true);
  ASSERT_EQ(GetIsProfilingStreamLatency(), false);
  ASSERT_EQ(GetUsePacketTimeStampForAddedPacket(), false);
  // Checks histogram_interval_size_usec and num_histogram_intervals.
  CalculatorProfile actual =
      GetCalculatorProfilesMap()->find(kDummyTestCalculatorName)->second;
  EXPECT_THAT(actual, EqualsProto(R"pb(
                name: "DummyTestCalculator"
                process_runtime {
                  total: 0
                  interval_size_usec: 1000
                  num_intervals: 3
                  count: 0
                  count: 0
                  count: 0
                }
              )pb"));
}

// Tests that Initialize() reads all the configs defined in the graph
// definition.
TEST_F(GraphProfilerTestPeer, Initialize) {
  InitializeProfilerWithGraphConfig(R"(
    profiler_config {
      histogram_interval_size_usec: 100
      num_histogram_intervals: 3
      enable_profiler: true
      enable_stream_latency: true
    }

    input_stream: "input_stream"
    input_stream: "dangling_stream"
    input_side_packet: "input_side_packet"
    output_stream: "output_stream"

    # Defining the calculator names explicitly to avoid relying on
    # definition order and auto postfix for duplicate calculators.
    node {
      calculator: "DummyTestCalculator"
      name: "A_Source_Calc"
      output_stream: "TAG:source_stream1"
    }
    node {
      calculator: "DummyTestCalculator"
      name: "A_Normal_Calc"
      input_stream: "input_stream"
      input_stream: "MY_TAG:source_stream1"
      output_stream: "my_stream"
    }
    node {
      calculator: "DummyTestCalculator"
      name: "Another_Source_Calc"
      input_side_packet: "input_side_packet"
      output_stream: "source_stream2"
    }
    node {
      calculator: "FlowLimiterCalculator"
      input_stream: "FINISHED:my_other_stream"
      input_stream: "source_stream2"
      input_stream_info: {
        tag_index: "FINISHED"
        back_edge: true
      }
      output_stream: "gated_source_stream2"
    }
    node {
      calculator: "DummyTestCalculator"
      name: "Another_Normal_Calc"
      input_stream: "my_stream"
      input_stream: "gated_source_stream2"
      output_stream: "my_other_stream"
    }
    node {
      calculator: "DummyTestCalculator"
      name: "A_Sink_Calc"
      input_stream: "my_other_stream"
    }
    node {
      calculator: "PassThroughCalculator"
      name: "An_Isolated_Calc_With_Identical_Inputs"
      input_stream: "input_stream"
      input_stream: "input_stream"
      output_stream: "output_stream"
      output_stream: "dangling_output_stream"
    })");

  // Checks calculator_profiles_ map.
  ASSERT_EQ(GetCalculatorProfilesMap()->size(), 7);
  CheckHasProfilesWithInputStreamName("A_Source_Calc", {});
  CheckHasProfilesWithInputStreamName("A_Normal_Calc",
                                      {"input_stream", "source_stream1"});
  CheckHasProfilesWithInputStreamName("Another_Source_Calc", {});
  CheckHasProfilesWithInputStreamName("FlowLimiterCalculator",
                                      {"source_stream2", "my_other_stream"});
  CheckHasProfilesWithInputStreamName("Another_Normal_Calc",
                                      {"my_stream", "gated_source_stream2"});
  CheckHasProfilesWithInputStreamName("A_Sink_Calc", {"my_other_stream"});
  CheckHasProfilesWithInputStreamName("An_Isolated_Calc_With_Identical_Inputs",
                                      {"input_stream", "input_stream"});

  // Checks packets_info_ map.
  // Should not be affected by calling Initialize().
  ASSERT_EQ(GetPacketsInfoMap()->size(), 0);
}

// Tests that GraphProfiler checks not to be initialized multiple times.
// A second attempt to intitialze GraphProfiler should cause a CHECK failure.
TEST_F(GraphProfilerTestPeer, InitializeMultipleTimes) {
  ASSERT_EQ(GetIsInitialized(), false);
  InitializeProfilerWithGraphConfig(R"(
    profiler_config {
      histogram_interval_size_usec: 1000
      num_histogram_intervals: 3
      enable_profiler: true
    }
    input_stream: "input_stream"
    node {
      calculator: "DummyTestCalculator"
      input_stream: "input_stream"
    })");
  ASSERT_EQ(GetIsInitialized(), true);

  ASSERT_DEATH(InitializeProfilerWithGraphConfig(R"(
    profiler_config {
      histogram_interval_size_usec: 1000
      num_histogram_intervals: 3
      enable_profiler: true
    }
    input_stream: "input_stream"
    node {
      calculator: "DummyTestCalculator"
      input_stream: "input_stream"
    })"),
               "Cannot initialize .* multiple times.");
}

// Tests that graph identifiers are not reused, even after destruction.
TEST_F(GraphProfilerTestPeer, InitializeMultipleProfilers) {
  auto raw_graph_config = R"(
    profiler_config {
      enable_profiler: true
    }
    input_stream: "input_stream"
    node {
      calculator: "DummyTestCalculator"
      input_stream: "input_stream"
    })";
  const int n_iterations = 100;
  absl::flat_hash_set<int> seen_ids;
  for (int i = 0; i < n_iterations; ++i) {
    std::shared_ptr<ProfilingContext> profiler =
        std::make_shared<ProfilingContext>();
    auto graph_config = CreateGraphConfig(raw_graph_config);
    mediapipe::ValidatedGraphConfig validated_graph;
    QCHECK_OK(validated_graph.Initialize(graph_config));
    profiler->Initialize(validated_graph);

    int id = profiler->GetGraphId();
    ASSERT_THAT(seen_ids, testing::Not(testing::Contains(id)));
    seen_ids.insert(id);
  }
}
// Tests that Pause(), Resume(), and Reset() works.
TEST_F(GraphProfilerTestPeer, PauseResumeReset) {
  InitializeProfilerWithGraphConfig(R"(
    profiler_config {
      enable_profiler: true
    }
    input_stream: "input_stream"
    node {
      calculator: "DummyTestCalculator"
      input_stream: "input_stream"
      output_stream: "output_stream"
    })");
  std::shared_ptr<mediapipe::SimulationClock> simulation_clock(
      new SimulationClock());
  simulation_clock->ThreadStart();
  profiler_.SetClock(simulation_clock);

  TestContextBuilder context(kDummyTestCalculatorName, /*node_id=*/0,
                             {"input_stream"}, {"output_stream"});
  context.AddInputs({MakePacket<std::string>("15").At(Timestamp(100))});

  // Checks works without making any change.
  {
    GraphProfiler::Scope profiler_scope(GraphTrace::PROCESS, context.get(),
                                        &profiler_);
    simulation_clock->Sleep(absl::Microseconds(10));
  }
  ASSERT_THAT(Profiles()[0].process_runtime(),
              Partially(EqualsProto(CreateTimeHistogram(/*total=*/10, {1}))));

  // Pause: profile should not change on calling Process().
  profiler_.Pause();
  {
    GraphProfiler::Scope profiler_scope(GraphTrace::PROCESS, context.get(),
                                        &profiler_);
    simulation_clock->Sleep(absl::Microseconds(100));
  }
  ASSERT_THAT(Profiles()[0].process_runtime(),
              Partially(EqualsProto(CreateTimeHistogram(/*total=*/10, {1}))));

  // Resume: profile should update again on calling Process().
  profiler_.Resume();
  {
    GraphProfiler::Scope profiler_scope(GraphTrace::PROCESS, context.get(),
                                        &profiler_);
    simulation_clock->Sleep(absl::Microseconds(1000));
  }
  ASSERT_THAT(Profiles()[0].process_runtime(),
              Partially(EqualsProto(CreateTimeHistogram(/*total=*/1010, {2}))));

  // Resest: profile should be clean.
  profiler_.Reset();
  ASSERT_THAT(Profiles()[0].process_runtime(),
              Partially(EqualsProto(CreateTimeHistogram(/*total=*/0, {0}))));

  // Checks still works after calling Reset().
  {
    GraphProfiler::Scope profiler_scope(GraphTrace::PROCESS, context.get(),
                                        &profiler_);
    simulation_clock->Sleep(absl::Microseconds(10000));
  }
  ASSERT_THAT(
      Profiles()[0].process_runtime(),
      Partially(EqualsProto(CreateTimeHistogram(/*total=*/10000, {1}))));

  simulation_clock->ThreadFinish();
}

// Tests that AddPacketInfo() uses packet timestamp when
// use_packet_timestamp_for_added_packet is true.
TEST_F(GraphProfilerTestPeer, AddPacketInfoUsingPacketTimestamp) {
  InitializeProfilerWithGraphConfig(R"(
    profiler_config {
      enable_profiler: true
      enable_stream_latency: true
      use_packet_timestamp_for_added_packet: true
    }
    input_stream: "input_stream"
    node {
      calculator: "DummyTestCalculator"
      input_stream: "input_stream"
    })");
  // Checks packets_info_ map before adding any packet.
  ASSERT_EQ(GetPacketsInfoMap()->size(), 0);

  std::string input_stream_name = "input_stream";
  Packet packet = MakePacket<std::string>("hello").At(Timestamp(100));
  profiler_.LogEvent(TraceEvent(GraphTrace::PROCESS)
                         .set_stream_id(&input_stream_name)
                         .set_input_ts(packet.Timestamp())
                         .set_packet_ts(packet.Timestamp())
                         .set_packet_data_id(&packet));

  PacketInfo expected_packet_info = {0,
                                     /*production_time_usec=*/100,
                                     /*source_process_start_usec=*/100};
  ASSERT_EQ(*GetPacketInfo(GetPacketsInfoMap(), {"input_stream", 100}),
            expected_packet_info);
}

// Tests that AddPacketInfo() uses profiler's clock when
// use_packet_timestamp_for_added_packet is false.
TEST_F(GraphProfilerTestPeer, AddPacketInfoUsingProfilerClock) {
  std::shared_ptr<mediapipe::SimulationClock> simulation_clock(
      new SimulationClock());
  simulation_clock->ThreadStart();

  InitializeProfilerWithGraphConfig(R"(
    profiler_config {
      enable_profiler: true
      enable_stream_latency: true
      use_packet_timestamp_for_added_packet: false
    }
    input_stream: "input_stream"
    node {
      calculator: "DummyTestCalculator"
      input_stream: "input_stream"
    })");
  profiler_.SetClock(simulation_clock);

  // Checks packets_info_ map before adding any packet.
  ASSERT_EQ(GetPacketsInfoMap()->size(), 0);

  simulation_clock->Sleep(absl::Microseconds(200));
  std::string input_stream_name = "input_stream";
  Packet packet = MakePacket<std::string>("hello").At(Timestamp(110));
  profiler_.LogEvent(TraceEvent(GraphTrace::PROCESS)
                         .set_stream_id(&input_stream_name)
                         .set_input_ts(packet.Timestamp())
                         .set_packet_ts(packet.Timestamp())
                         .set_packet_data_id(&packet));
  int64_t profiler_now_usec = ToUnixMicros(simulation_clock->TimeNow());

  PacketInfo expected_packet_info = {
      0,
      /*production_time_usec=*/profiler_now_usec,
      /*source_process_start_usec=*/profiler_now_usec};
  ASSERT_EQ(*GetPacketInfo(GetPacketsInfoMap(), {"input_stream", 110}),
            expected_packet_info);

  simulation_clock->ThreadFinish();
}

// Tests that AddPacketInfo() does not add packet info for a stream that has no
// consumer.
TEST_F(GraphProfilerTestPeer, AddPacketInfoWhenNoConsumer) {
  InitializeProfilerWithGraphConfig(R"(
    profiler_config {
      enable_profiler: true
      enable_stream_latency: true
      use_packet_timestamp_for_added_packet: true
    }
    input_stream: "input_stream1"
    input_stream: "input_stream2"
    node {
      calculator: "DummyTestCalculator"
      input_stream: "input_stream2"
    })");
  // Checks packets_info_ map before adding any packet.
  ASSERT_EQ(GetPacketsInfoMap()->size(), 0);

  std::string input_stream_name = "input_stream1";
  Packet packet = MakePacket<std::string>("hello").At(Timestamp(100));
  profiler_.LogEvent(TraceEvent(GraphTrace::PROCESS)
                         .set_stream_id(&input_stream_name)
                         .set_input_ts(packet.Timestamp())
                         .set_packet_ts(packet.Timestamp())
                         .set_packet_data_id(&packet));
  ASSERT_EQ(GetPacketInfo(GetPacketsInfoMap(), {"input_stream", 100}), nullptr);

  std::string input_stream_name2 = "input_stream2";
  profiler_.LogEvent(TraceEvent(GraphTrace::PROCESS)
                         .set_stream_id(&input_stream_name2)
                         .set_input_ts(packet.Timestamp())
                         .set_packet_ts(packet.Timestamp())
                         .set_packet_data_id(&packet));
  ASSERT_NE(GetPacketInfo(GetPacketsInfoMap(), {"input_stream2", 100}),
            nullptr);
}

// Tests that SetOpenRuntime() updates |open_runtime| and doesn't affect other
// histograms.
TEST_F(GraphProfilerTestPeer, SetOpenRuntime) {
  InitializeProfilerWithGraphConfig(R"(
    profiler_config {
      enable_profiler: true
    }
    input_stream: "input_stream"
    node {
      calculator: "DummyTestCalculator"
      input_stream: "input_stream"
      output_stream: "output_stream"
    })");
  std::shared_ptr<mediapipe::SimulationClock> simulation_clock(
      new SimulationClock());
  simulation_clock->ThreadStart();
  profiler_.SetClock(simulation_clock);

  TestContextBuilder context(kDummyTestCalculatorName, /*node_id=*/0,
                             {"input_stream"}, {"output_stream"});
  context.AddInputs({MakePacket<std::string>("15").At(Timestamp(100))});
  {
    GraphProfiler::Scope profiler_scope(GraphTrace::OPEN, context.get(),
                                        &profiler_);
    simulation_clock->Sleep(absl::Microseconds(100));
  }

  std::vector<CalculatorProfile> profiles = Profiles();
  simulation_clock->ThreadFinish();

  ASSERT_EQ(profiles.size(), 1);
  EXPECT_THAT(profiles[0], Partially(EqualsProto(R"pb(
                name: "DummyTestCalculator"
                open_runtime: 100
                process_runtime { total: 0 }
              )pb")));
  // Checks packets_info_ map hasn't changed.
  ASSERT_EQ(GetPacketsInfoMap()->size(), 0);
}

// Tests that SetOpenRuntime() updates |open_runtime| and also updates the
// packet info map when stream latency is enabled and the calculator produces
// output packet in Open().
TEST_F(GraphProfilerTestPeer, SetOpenRuntimeWithStreamLatency) {
  InitializeProfilerWithGraphConfig(R"(
    profiler_config {
      enable_profiler: true
      enable_stream_latency: true
    }
    node {
      calculator: "DummyTestCalculator"
      name: "source_calc"
      output_stream: "stream_0"
      output_stream: "stream_1"
    }
    # This is needed to have a consumer for the output packets.
    # Otherwise, the profiler skips them.
    node {
      calculator: "DummyTestCalculator"
      name: "consumer_calc"
      input_stream: "stream_0"
      input_stream: "stream_1"
    })");
  std::shared_ptr<mediapipe::SimulationClock> simulation_clock(
      new SimulationClock());
  simulation_clock->ThreadStart();
  profiler_.SetClock(simulation_clock);

  TestContextBuilder source_context("source_calc", /*node_id=*/0, {},
                                    {"stream_0", "stream_1"});
  source_context.AddInputs({});
  source_context.AddOutputs(
      {{}, {MakePacket<std::string>("15").At(Timestamp(100))}});

  simulation_clock->SleepUntil(absl::FromUnixMicros(1000));
  {
    GraphProfiler::Scope profiler_scope(GraphTrace::OPEN, source_context.get(),
                                        &profiler_);
    simulation_clock->Sleep(absl::Microseconds(150));
  }

  std::vector<CalculatorProfile> profiles = Profiles();
  simulation_clock->ThreadFinish();

  ASSERT_EQ(profiles.size(), 2);
  CalculatorProfile source_profile =
      GetProfileWithName(profiles, "source_calc");

  EXPECT_THAT(source_profile, EqualsProto(R"pb(
                name: "source_calc"
                open_runtime: 150
                process_runtime {
                  total: 0
                  interval_size_usec: 1000000
                  num_intervals: 1
                  count: 0
                }
                process_input_latency {
                  total: 0
                  interval_size_usec: 1000000
                  num_intervals: 1
                  count: 0
                }
                process_output_latency {
                  total: 0
                  interval_size_usec: 1000000
                  num_intervals: 1
                  count: 0
                }
              )pb"));

  // Check packets_info_ map has been updated.
  ASSERT_EQ(GetPacketsInfoMap()->size(), 1);
  PacketInfo expected_packet_info = {0,
                                     /*production_time_usec=*/1000 + 150,
                                     /*source_process_start_usec=*/1000 + 0};
  ASSERT_EQ(*GetPacketInfo(GetPacketsInfoMap(), {"stream_1", 100}),
            expected_packet_info);
}

// Tests that SetCloseRuntime() updates |close_runtime| and doesn't affect other
// histograms.
TEST_F(GraphProfilerTestPeer, SetCloseRuntime) {
  InitializeProfilerWithGraphConfig(R"(
    profiler_config {
      enable_profiler: true
    }
    input_stream: "input_stream"
    node {
      calculator: "DummyTestCalculator"
      input_stream: "input_stream"
      output_stream: "output_stream"
    })");
  std::shared_ptr<mediapipe::SimulationClock> simulation_clock(
      new SimulationClock());
  simulation_clock->ThreadStart();
  profiler_.SetClock(simulation_clock);

  TestContextBuilder context(kDummyTestCalculatorName, /*node_id=*/0,
                             {"input_stream"}, {"output_stream"});
  context.AddInputs({MakePacket<std::string>("15").At(Timestamp(100))});
  {
    GraphProfiler::Scope profiler_scope(GraphTrace::CLOSE, context.get(),
                                        &profiler_);
    simulation_clock->Sleep(absl::Microseconds(100));
  }

  std::vector<CalculatorProfile> profiles = Profiles();
  simulation_clock->ThreadFinish();

  EXPECT_THAT(profiles[0], EqualsProto(R"pb(
                name: "DummyTestCalculator"
                close_runtime: 100
                process_runtime {
                  total: 0
                  interval_size_usec: 1000000
                  num_intervals: 1
                  count: 0
                }
              )pb"));
}

// Tests that SetCloseRuntime() updates |close_runtime| and doesn't affect other
// histograms.
TEST_F(GraphProfilerTestPeer, SetCloseRuntimeWithStreamLatency) {
  InitializeProfilerWithGraphConfig(R"(
    profiler_config {
      enable_profiler: true
      enable_stream_latency: true
    }
    input_stream: "input_stream"
    node {
      calculator: "DummyTestCalculator"
      name: "source_calc"
      input_stream: "input_stream"
      output_stream: "output_stream"
    }
    # This is needed to have a consumer for the output packets.
    # Otherwise, the profiler skips them.
    node {
      calculator: "DummyTestCalculator"
      name: "consumer_calc"
      input_stream: "output_stream"
    })");
  std::shared_ptr<mediapipe::SimulationClock> simulation_clock(
      new SimulationClock());
  simulation_clock->ThreadStart();
  profiler_.SetClock(simulation_clock);

  TestContextBuilder source_context("source_calc", /*node_id=*/0,
                                    {"input_stream"}, {"output_stream"});
  source_context.AddOutputs(
      {{MakePacket<std::string>("15").At(Timestamp::PostStream())}});
  CalculatorContextManager().PushInputTimestampToContext(
      source_context.get(), Timestamp::PostStream());
  simulation_clock->SleepUntil(absl::FromUnixMicros(1000));
  {
    GraphProfiler::Scope profiler_scope(GraphTrace::CLOSE, source_context.get(),
                                        &profiler_);
    simulation_clock->Sleep(absl::Microseconds(100));
  }

  std::vector<CalculatorProfile> profiles = Profiles();
  simulation_clock->ThreadFinish();

  ASSERT_EQ(profiles.size(), 2);
  CalculatorProfile source_profile =
      GetProfileWithName(profiles, "source_calc");

  EXPECT_THAT(source_profile, EqualsProto(R"pb(
                name: "source_calc"
                close_runtime: 100
                process_runtime {
                  total: 0
                  interval_size_usec: 1000000
                  num_intervals: 1
                  count: 0
                }
                process_input_latency {
                  total: 0
                  interval_size_usec: 1000000
                  num_intervals: 1
                  count: 0
                }
                process_output_latency {
                  total: 0
                  interval_size_usec: 1000000
                  num_intervals: 1
                  count: 0
                }
                input_stream_profiles {
                  name: "input_stream"
                  back_edge: false
                  latency {
                    total: 0
                    interval_size_usec: 1000000
                    num_intervals: 1
                    count: 0
                  }
                }
              )pb"));
  PacketInfo expected_packet_info = {0,
                                     /*production_time_usec=*/1000 + 100,
                                     /*source_process_start_usec=*/1000 + 0};
  PacketId packet_id = {"output_stream", Timestamp::PostStream().Value()};
  ASSERT_EQ(*GetPacketInfo(GetPacketsInfoMap(), packet_id),
            expected_packet_info);
}

// Tests that InitializeTimeHistogram set the histogram values and counts
// properly.
TEST_F(GraphProfilerTestPeer, InitializeTimeHistogram) {
  TimeHistogram histogram;
  GraphProfilerTestPeer::InitializeTimeHistogram(/*interval_size_usec=*/50,
                                                 /*num_intervals=*/3,
                                                 &histogram);
  ASSERT_EQ(histogram.interval_size_usec(), 50);
  ASSERT_EQ(histogram.num_intervals(), 3);
  ASSERT_THAT(histogram, Partially(EqualsProto(CreateTimeHistogram(
                             /*total=*/0, /*counts=*/{0, 0, 0}))));
}

// Test AddTimeSample() update the correct bucket.
TEST_F(GraphProfilerTestPeer, AddTimeSample) {
  TimeHistogram histogram;
  GraphProfilerTestPeer::InitializeTimeHistogram(/*interval_size_usec=*/100,
                                                 /*num_intervals=*/3,
                                                 &histogram);
  // Took 30us -> 1st bucket.
  GraphProfilerTestPeer::AddTimeSample(/*start_time_usec=*/100,
                                       /*end_time_usec=*/130, &histogram);
  ASSERT_THAT(
      histogram,
      Partially(EqualsProto(CreateTimeHistogram(/*total=*/30, {1, 0, 0}))));
  // Took 100us -> 2st bucket.
  GraphProfilerTestPeer::AddTimeSample(/*start_time_usec=*/100,
                                       /*end_time_usec=*/200, &histogram);
  ASSERT_THAT(histogram, Partially(EqualsProto(CreateTimeHistogram(
                             /*total=*/30 + 100, {1, 1, 0}))));
  // Took 500us -> last bucket.
  GraphProfilerTestPeer::AddTimeSample(/*start_time_usec=*/100,
                                       /*end_time_usec=*/600, &histogram);
  ASSERT_THAT(histogram, Partially(EqualsProto(CreateTimeHistogram(
                             /*total=*/30 + 100 + 500, {1, 1, 1}))));
}

// Tests that InitializeOutputStreams adds all the outputs of a node to the
// stream consumer count map.
TEST_F(GraphProfilerTestPeer, InitializeOutputStreams) {
  // Without any output stream.
  auto node_config = CreateNodeConfig(R"(
    calculator: "SinkCalculator"
    input_stream: "input_stream"
    input_side_packet: "input_side_packet")");
  InitializeOutputStreams(node_config);
  // With output stream.
  node_config = CreateNodeConfig(R"(
    calculator: "MyCalculator"
    input_stream: "input_stream"
    input_side_packet: "input_side_packet"
    output_stream: "without_tag"
    output_stream: "MY_TAG:with_tag")");
  InitializeOutputStreams(node_config);
}

// Tests that InitializeInputStreams adds all (and only) the input stream,
// excluding the back edges or input side packets.
TEST_F(GraphProfilerTestPeer, InitializeInputStreams) {
  CalculatorProfile profile;
  int64_t interval_size_usec = 100;
  int64_t num_intervals = 1;
  // Without any input stream.
  auto node_config = CreateNodeConfig(R"(
    calculator: "SourceCalculator"
    input_side_packet: "input_side_packet"
    output_stream: "output_stream")");
  InitializeInputStreams(node_config, interval_size_usec, num_intervals,
                         &profile);
  ASSERT_EQ(profile.input_stream_profiles().size(), 0);
  // With input stream and backedges.
  node_config = CreateNodeConfig(R"(
    calculator: "MyCalculator"
    input_stream: "without_tag"
    input_stream: "TAG1:back_edge"
    input_stream: "TAG2:with_tag"
    input_stream: "TAG3:with_info"
    input_side_packet: "input_side_packet"
    output_stream: "output_stream"
    input_stream_info: {
      tag_index: "TAG1"
      back_edge: true
    }
    input_stream_info: {  # This is not a back edge.
      tag_index: "TAG3"
      back_edge: false
    })");
  InitializeInputStreams(node_config, interval_size_usec, num_intervals,
                         &profile);
  // GraphProfiler adds the back edge to the profile as well to keep the
  // ordering. So, it expect to see 4 input stream profiles.
  ASSERT_EQ(profile.input_stream_profiles().size(), 4);
  ASSERT_EQ(profile.input_stream_profiles(0).name(), "without_tag");
  ASSERT_EQ(profile.input_stream_profiles(1).name(), "back_edge");
  ASSERT_EQ(profile.input_stream_profiles(2).name(), "with_tag");
  ASSERT_EQ(profile.input_stream_profiles(3).name(), "with_info");
}

// Tests that AddProcessSample() updates |process_runtime| and doesn't affect
// other histograms or packet info map if stream latency is not enabled.
TEST_F(GraphProfilerTestPeer, AddProcessSample) {
  InitializeProfilerWithGraphConfig(R"(
    profiler_config {
      enable_profiler: true
    }
    input_stream: "input_stream"
    node {
      calculator: "DummyTestCalculator"
      input_stream: "input_stream"
      output_stream: "output_stream"
    })");
  std::shared_ptr<mediapipe::SimulationClock> simulation_clock(
      new SimulationClock());
  simulation_clock->ThreadStart();
  profiler_.SetClock(simulation_clock);

  TestContextBuilder context(kDummyTestCalculatorName, /*node_id=*/0,
                             {"input_stream"}, {"output_stream"});
  context.AddInputs({MakePacket<std::string>("5").At(Timestamp(100))});
  context.AddOutputs({{MakePacket<std::string>("15").At(Timestamp(100))}});

  {
    GraphProfiler::Scope profiler_scope(GraphTrace::PROCESS, context.get(),
                                        &profiler_);
    simulation_clock->Sleep(absl::Microseconds(150));
  }

  std::vector<CalculatorProfile> profiles = Profiles();
  simulation_clock->ThreadFinish();

  ASSERT_EQ(profiles.size(), 1);
  EXPECT_THAT(profiles[0], EqualsProto(R"pb(
                name: "DummyTestCalculator"
                process_runtime {
                  total: 150
                  interval_size_usec: 1000000
                  num_intervals: 1
                  count: 1
                }
              )pb"));
  // Checks packets_info_ map hasn't changed.
  ASSERT_EQ(GetPacketsInfoMap()->size(), 0);
}

// Tests that AddProcessSample() updates |process_runtime| and also updates the
// packet info map when stream latency is enabled.
TEST_F(GraphProfilerTestPeer, AddProcessSampleWithStreamLatency) {
  InitializeProfilerWithGraphConfig(R"(
    profiler_config {
      enable_profiler: true
      enable_stream_latency: true
    }
    node {
      calculator: "DummyTestCalculator"
      name: "source_calc"
      output_stream: "stream_0"
      output_stream: "stream_1"
    }
    node {
      calculator: "DummyTestCalculator"
      name: "consumer_calc"
      input_stream: "stream_0"
      input_stream: "stream_1"
    })");
  std::shared_ptr<mediapipe::SimulationClock> simulation_clock(
      new SimulationClock());
  simulation_clock->ThreadStart();
  profiler_.SetClock(simulation_clock);

  TestContextBuilder source_context("source_calc", /*node_id=*/0, {},
                                    {"stream_0", "stream_1"});
  source_context.AddInputs({});
  source_context.AddOutputs(
      {{}, {MakePacket<std::string>("15").At(Timestamp(100))}});

  int64_t when_source_started = 1000;
  int64_t when_source_finished = when_source_started + 150;
  simulation_clock->SleepUntil(absl::FromUnixMicros(when_source_started));
  {
    GraphProfiler::Scope profiler_scope(GraphTrace::PROCESS,
                                        source_context.get(), &profiler_);
    simulation_clock->Sleep(absl::Microseconds(150));
  }
  std::vector<CalculatorProfile> profiles = Profiles();

  ASSERT_EQ(profiles.size(), 2);
  CalculatorProfile source_profile =
      GetProfileWithName(profiles, "source_calc");

  EXPECT_THAT(profiles[0], Partially(EqualsProto(R"pb(
                process_runtime {
                  total: 150
                  interval_size_usec: 1000000
                  num_intervals: 1
                  count: 1
                }
                process_input_latency {
                  total: 0
                  interval_size_usec: 1000000
                  num_intervals: 1
                  count: 1
                }
                process_output_latency {
                  total: 150
                  interval_size_usec: 1000000
                  num_intervals: 1
                  count: 1
                }
              )pb")));

  // Check packets_info_ map has been updated.
  ASSERT_EQ(GetPacketsInfoMap()->size(), 1);
  PacketInfo expected_packet_info = {
      0,
      /*production_time_usec=*/when_source_finished,
      /*source_process_start_usec=*/when_source_started};
  ASSERT_EQ(*GetPacketInfo(GetPacketsInfoMap(), {"stream_1", 100}),
            expected_packet_info);

  // Run process for consumer calculator and checks its profile.
  TestContextBuilder consumer_context("consumer_calc", /*node_id=*/0,
                                      {"stream_0", "stream_1"}, {});
  consumer_context.AddInputs(
      {Packet(), MakePacket<std::string>("15").At(Timestamp(100))});

  simulation_clock->SleepUntil(absl::FromUnixMicros(2000));
  {
    GraphProfiler::Scope profiler_scope(GraphTrace::PROCESS,
                                        consumer_context.get(), &profiler_);
    simulation_clock->Sleep(absl::Microseconds(250));
  }

  profiles = Profiles();
  simulation_clock->ThreadFinish();

  CalculatorProfile consumer_profile =
      GetProfileWithName(profiles, "consumer_calc");

  // process input latency total = 2000 (end) - 1000 (when source started) =
  // 1000 process output latency total = 2000 (end) + 250 - 1000 (when source
  // started) = 1250 For "stream_0" should have not changed since it was empty.
  // For "stream_1" = 2000 (end) - 1250 (when source finished) = 850
  EXPECT_THAT(consumer_profile, Partially(EqualsProto(R"pb(
                name: "consumer_calc"
                process_input_latency { total: 1000 }
                process_output_latency { total: 1250 }
                input_stream_profiles {
                  name: "stream_0"
                  latency { total: 0 }
                }
                input_stream_profiles {
                  name: "stream_1"
                  latency { total: 850 }
                }
              )pb")));

  // Check packets_info_ map for PacketId({"stream_1", 100}) should not yet be
  // garbage collected.
  ASSERT_NE(GetPacketInfo(GetPacketsInfoMap(), {"stream_1", 100}), nullptr);
}

// This test shows that CalculatorGraph::GetCalculatorProfiles and
// GraphProfiler::AddProcessSample() can be called in parallel.
// Without the GraphProfiler::profiler_mutex_ this test should
// fail with --config=tsan with message
// "WARNING: ThreadSanitizer: data race in
// mediapipe::ProcessProfile::set_total"
TEST(GraphProfilerTest, ParallelReads) {
  // A graph that processes a certain number of packets before finishing.
  CalculatorGraphConfig config;
  QCHECK(google::protobuf::TextFormat::ParseFromString(R"(
    profiler_config {
     enable_profiler: true
    }
    node {
      calculator: "RangeCalculator"
      input_side_packet: "range_step"
      output_stream: "out"
      output_stream: "sum"
      output_stream: "mean"
    }
    node {
      calculator: "PassThroughCalculator"
      input_stream: "out"
      input_stream: "sum"
      input_stream: "mean"
      output_stream: "out_1"
      output_stream: "sum_1"
      output_stream: "mean_1"
    }
    output_stream: "OUT:0:the_integers"
    )",
                                                       &config));

  // Start running the graph on its own threads.
  absl::Mutex out_1_mutex;
  std::vector<Packet> out_1_packets;
  CalculatorGraph graph;
  MP_ASSERT_OK(graph.Initialize(config));
  MP_ASSERT_OK(graph.ObserveOutputStream("out_1", [&](const Packet& packet) {
    absl::MutexLock lock(&out_1_mutex);
    out_1_packets.push_back(packet);
    return absl::OkStatus();
  }));
  MP_EXPECT_OK(graph.StartRun(
      {{"range_step", MakePacket<std::pair<uint32_t, uint32_t>>(1000, 1)}}));

  // Repeatedly poll for profile data while the graph runs.
  while (true) {
    std::vector<CalculatorProfile> profiles;
    MP_ASSERT_OK(graph.profiler()->GetCalculatorProfiles(&profiles));
    EXPECT_EQ(2, profiles.size());
    absl::MutexLock lock(&out_1_mutex);
    if (out_1_packets.size() >= 1001) {
      break;
    }
  }
  MP_ASSERT_OK(graph.WaitUntilDone());
  std::vector<CalculatorProfile> profiles;
  MP_ASSERT_OK(graph.profiler()->GetCalculatorProfiles(&profiles));
  // GraphProfiler internally uses map and the profile order is not fixed.
  if (profiles[0].name() == "RangeCalculator") {
    EXPECT_EQ(1000, profiles[0].process_runtime().count(0));
    EXPECT_EQ(1003, profiles[1].process_runtime().count(0));
  } else if (profiles[0].name() == "PassThroughCalculator") {
    EXPECT_EQ(1003, profiles[0].process_runtime().count(0));
    EXPECT_EQ(1000, profiles[1].process_runtime().count(0));
  } else {
    ABSL_LOG(FATAL) << "Unexpected profile name " << profiles[0].name();
  }
  EXPECT_EQ(1001, out_1_packets.size());
}

// Returns the set of calculator names in a GraphProfile captured from
// CalculatorGraph initialized from a certain CalculatorGraphConfig.
std::set<std::string> GetCalculatorNames(const CalculatorGraphConfig& config) {
  std::set<std::string> result;
  CalculatorGraph graph;
  MP_EXPECT_OK(graph.Initialize(config));
  GraphProfile profile;
  MP_EXPECT_OK(graph.profiler()->CaptureProfile(&profile));
  for (auto& p : profile.calculator_profiles()) {
    result.insert(p.name());
  }
  return result;
}

TEST(GraphProfilerTest, CalculatorProfileFilter) {
  CalculatorGraphConfig config;
  QCHECK(google::protobuf::TextFormat::ParseFromString(R"(
    profiler_config {
     enable_profiler: true
    }
    node {
      calculator: "RangeCalculator"
      input_side_packet: "range_step"
      output_stream: "out"
      output_stream: "sum"
      output_stream: "mean"
    }
    node {
      calculator: "PassThroughCalculator"
      input_stream: "out"
      input_stream: "sum"
      input_stream: "mean"
      output_stream: "out_1"
      output_stream: "sum_1"
      output_stream: "mean_1"
    }
    output_stream: "OUT:0:the_integers"
    )",
                                                       &config));

  std::set<std::string> expected_names;
  expected_names = {"RangeCalculator", "PassThroughCalculator"};
  EXPECT_EQ(GetCalculatorNames(config), expected_names);

  *config.mutable_profiler_config()->mutable_calculator_filter() =
      "RangeCalculator";
  expected_names = {"RangeCalculator"};
  EXPECT_EQ(GetCalculatorNames(config), expected_names);

  *config.mutable_profiler_config()->mutable_calculator_filter() = "Range.*";
  expected_names = {"RangeCalculator"};
  EXPECT_EQ(GetCalculatorNames(config), expected_names);

  *config.mutable_profiler_config()->mutable_calculator_filter() =
      ".*Calculator";
  expected_names = {"RangeCalculator", "PassThroughCalculator"};
  EXPECT_EQ(GetCalculatorNames(config), expected_names);

  *config.mutable_profiler_config()->mutable_calculator_filter() = ".*Clock.*";
  expected_names = {};
  EXPECT_EQ(GetCalculatorNames(config), expected_names);
}

TEST(GraphProfilerTest, CaptureProfilePopulateConfig) {
  CalculatorGraphConfig config;
  QCHECK(google::protobuf::TextFormat::ParseFromString(R"(
    profiler_config {
      enable_profiler: true
      trace_enabled: true
    }
    input_stream: "input_stream"
    node {
      calculator: "DummyTestCalculator"
      input_stream: "input_stream"
    }
    node {
      calculator: "DummyTestCalculator"
      input_stream: "input_stream"
    }
    )",
                                                       &config));
  CalculatorGraph graph;
  MP_ASSERT_OK(graph.Initialize(config));
  GraphProfile profile;
  MP_ASSERT_OK(
      graph.profiler()->CaptureProfile(&profile, PopulateGraphConfig::kFull));
  EXPECT_THAT(profile.config(), Partially(EqualsProto(R"pb(
                input_stream: "input_stream"
                node {
                  name: "DummyTestCalculator_1"
                  calculator: "DummyTestCalculator"
                  input_stream: "input_stream"
                }
                node {
                  name: "DummyTestCalculator_2"
                  calculator: "DummyTestCalculator"
                  input_stream: "input_stream"
                }
              )pb")));
  EXPECT_THAT(profile.graph_trace(),
              ElementsAre(Partially(EqualsProto(
                  R"pb(
                    calculator_name: "DummyTestCalculator_1"
                    calculator_name: "DummyTestCalculator_2"
                  )pb"))));
}

}  // namespace
}  // namespace mediapipe
