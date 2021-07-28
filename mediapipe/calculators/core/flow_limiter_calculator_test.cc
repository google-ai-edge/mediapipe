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
#include "mediapipe/calculators/core/flow_limiter_calculator.pb.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/calculator_runner.h"
#include "mediapipe/framework/formats/image_frame.h"
#include "mediapipe/framework/port/gmock.h"
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/framework/port/integral_types.h"
#include "mediapipe/framework/port/parse_text_proto.h"
#include "mediapipe/framework/port/status_matchers.h"
#include "mediapipe/framework/timestamp.h"
#include "mediapipe/framework/tool/simulation_clock.h"
#include "mediapipe/framework/tool/simulation_clock_executor.h"
#include "mediapipe/framework/tool/sink.h"

namespace mediapipe {

namespace {

constexpr char kDropTimestampsTag[] = "DROP_TIMESTAMPS";
constexpr char kClockTag[] = "CLOCK";
constexpr char kWarmupTimeTag[] = "WARMUP_TIME";
constexpr char kSleepTimeTag[] = "SLEEP_TIME";
constexpr char kPacketTag[] = "PACKET";

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
std::vector<int64> TimestampValues(const std::vector<Packet>& packets) {
  std::vector<int64> result;
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

// Tests demonstrating an FlowLimiterCalculator operating in a cyclic graph.
class FlowLimiterCalculatorSemaphoreTest : public testing::Test {
 public:
  FlowLimiterCalculatorSemaphoreTest() : exit_semaphore_(0) {}

  void SetUp() override {
    graph_config_ = InflightGraphConfig();
    tool::AddVectorSink("out_1", &graph_config_, &out_1_packets_);
  }

  void InitializeGraph(int max_in_flight) {
    ProcessFunction semaphore_1_func = [&](const InputStreamShardSet& inputs,
                                           OutputStreamShardSet* outputs) {
      exit_semaphore_.Acquire(1);
      return PassthroughFunction(inputs, outputs);
    };
    FlowLimiterCalculatorOptions options;
    options.set_max_in_flight(max_in_flight);
    options.set_max_in_queue(1);
    MP_ASSERT_OK(graph_.Initialize(
        graph_config_, {
                           {"limiter_options", Adopt(new auto(options))},
                           {"callback_1", Adopt(new auto(semaphore_1_func))},
                       }));

    allow_poller_.reset(
        new OutputStreamPoller(graph_.AddOutputStreamPoller("allow").value()));
  }

  // Adds a packet to a graph input stream.
  void AddPacket(const std::string& input_name, int value) {
    MP_EXPECT_OK(graph_.AddPacketToInputStream(
        input_name, MakePacket<int>(value).At(Timestamp(value))));
  }

  // A calculator graph starting with an FlowLimiterCalculator and
  // ending with a InFlightFinishCalculator.
  // Back-edge "finished" limits processing to one frame in-flight.
  // The LambdaCalculator is used to keep certain frames in flight.
  CalculatorGraphConfig InflightGraphConfig() {
    return ParseTextProtoOrDie<CalculatorGraphConfig>(R"pb(
      input_stream: 'in_1'
      node {
        calculator: 'FlowLimiterCalculator'
        input_side_packet: 'OPTIONS:limiter_options'
        input_stream: 'in_1'
        input_stream: 'FINISHED:out_1'
        input_stream_info: { tag_index: 'FINISHED' back_edge: true }
        output_stream: 'in_1_sampled'
        output_stream: 'ALLOW:allow'
      }
      node {
        calculator: 'LambdaCalculator'
        input_side_packet: 'callback_1'
        input_stream: 'in_1_sampled'
        output_stream: 'out_1'
      }
    )pb");
  }

 protected:
  CalculatorGraphConfig graph_config_;
  CalculatorGraph graph_;
  AtomicSemaphore exit_semaphore_;
  std::vector<Packet> out_1_packets_;
  std::unique_ptr<OutputStreamPoller> allow_poller_;
};

// A test demonstrating an FlowLimiterCalculator operating in a cyclic
// graph. This test shows that:
//
// (1) Frames exceeding the queue size are dropped.
// (2) The "ALLOW" signal is produced.
// (3) Timestamps are passed through unaltered.
//
TEST_F(FlowLimiterCalculatorSemaphoreTest, FramesDropped) {
  InitializeGraph(1);
  MP_ASSERT_OK(graph_.StartRun({}));

  auto send_packet = [this](const std::string& input_name, int64 n) {
    MP_EXPECT_OK(graph_.AddPacketToInputStream(
        input_name, MakePacket<int64>(n).At(Timestamp(n))));
  };

  Packet allow_packet;
  send_packet("in_1", 0);
  for (int i = 0; i < 9; i++) {
    EXPECT_TRUE(allow_poller_->Next(&allow_packet));
    EXPECT_TRUE(allow_packet.Get<bool>());
    // This input should wait in the limiter input queue.
    send_packet("in_1", i * 10 + 5);
    // This input should drop the previous input.
    send_packet("in_1", i * 10 + 10);
    EXPECT_TRUE(allow_poller_->Next(&allow_packet));
    EXPECT_FALSE(allow_packet.Get<bool>());
    exit_semaphore_.Release(1);
  }
  exit_semaphore_.Release(1);
  MP_EXPECT_OK(graph_.CloseInputStream("in_1"));
  MP_EXPECT_OK(graph_.WaitUntilIdle());

  // All output streams are closed and all output packets are delivered,
  // with stream "in_1" closed.
  EXPECT_EQ(10, out_1_packets_.size());

  // Timestamps have not been altered.
  EXPECT_EQ(PacketValues<int64>(out_1_packets_),
            TimestampValues(out_1_packets_));

  // Extra inputs on in_1 have been dropped.
  EXPECT_EQ(TimestampValues(out_1_packets_),
            (std::vector<int64>{0, 10, 20, 30, 40, 50, 60, 70, 80, 90}));
}

// A calculator that sleeps during Process.
class SleepCalculator : public CalculatorBase {
 public:
  static absl::Status GetContract(CalculatorContract* cc) {
    cc->Inputs().Tag(kPacketTag).SetAny();
    cc->Outputs().Tag(kPacketTag).SetSameAs(&cc->Inputs().Tag(kPacketTag));
    cc->InputSidePackets().Tag(kSleepTimeTag).Set<int64>();
    cc->InputSidePackets().Tag(kWarmupTimeTag).Set<int64>();
    cc->InputSidePackets().Tag(kClockTag).Set<mediapipe::Clock*>();
    cc->SetTimestampOffset(0);
    return absl::OkStatus();
  }

  absl::Status Open(CalculatorContext* cc) final {
    clock_ = cc->InputSidePackets().Tag(kClockTag).Get<mediapipe::Clock*>();
    return absl::OkStatus();
  }

  absl::Status Process(CalculatorContext* cc) final {
    ++packet_count;
    absl::Duration sleep_time = absl::Microseconds(
        packet_count == 1
            ? cc->InputSidePackets().Tag(kWarmupTimeTag).Get<int64>()
            : cc->InputSidePackets().Tag(kSleepTimeTag).Get<int64>());
    clock_->Sleep(sleep_time);
    cc->Outputs()
        .Tag(kPacketTag)
        .AddPacket(cc->Inputs().Tag(kPacketTag).Value());
    return absl::OkStatus();
  }

 private:
  ::mediapipe::Clock* clock_ = nullptr;
  int packet_count = 0;
};
REGISTER_CALCULATOR(SleepCalculator);

// A calculator that drops a packet occasionally.
// Drops the 3rd packet, and optionally the corresponding timestamp bound.
class DropCalculator : public CalculatorBase {
 public:
  static absl::Status GetContract(CalculatorContract* cc) {
    cc->Inputs().Tag(kPacketTag).SetAny();
    cc->Outputs().Tag(kPacketTag).SetSameAs(&cc->Inputs().Tag(kPacketTag));
    cc->InputSidePackets().Tag(kDropTimestampsTag).Set<bool>();
    cc->SetProcessTimestampBounds(true);
    return absl::OkStatus();
  }

  absl::Status Process(CalculatorContext* cc) final {
    if (!cc->Inputs().Tag(kPacketTag).Value().IsEmpty()) {
      ++packet_count;
    }
    bool drop = (packet_count == 3);
    if (!drop && !cc->Inputs().Tag(kPacketTag).Value().IsEmpty()) {
      cc->Outputs()
          .Tag(kPacketTag)
          .AddPacket(cc->Inputs().Tag(kPacketTag).Value());
    }
    if (!drop || !cc->InputSidePackets().Tag(kDropTimestampsTag).Get<bool>()) {
      cc->Outputs()
          .Tag(kPacketTag)
          .SetNextTimestampBound(cc->InputTimestamp().NextAllowedInStream());
    }
    return absl::OkStatus();
  }

 private:
  int packet_count = 0;
};
REGISTER_CALCULATOR(DropCalculator);

// Tests demonstrating an FlowLimiterCalculator processing FINISHED timestamps.
class FlowLimiterCalculatorTest : public testing::Test {
 protected:
  CalculatorGraphConfig InflightGraphConfig() {
    return ParseTextProtoOrDie<CalculatorGraphConfig>(R"pb(
      input_stream: 'in_1'
      node {
        calculator: 'FlowLimiterCalculator'
        input_side_packet: 'OPTIONS:limiter_options'
        input_stream: 'in_1'
        input_stream: 'FINISHED:out_1'
        input_stream_info: { tag_index: 'FINISHED' back_edge: true }
        output_stream: 'in_1_sampled'
        output_stream: 'ALLOW:allow'
      }
      node {
        calculator: 'SleepCalculator'
        input_side_packet: 'WARMUP_TIME:warmup_time'
        input_side_packet: 'SLEEP_TIME:sleep_time'
        input_side_packet: 'CLOCK:clock'
        input_stream: 'PACKET:in_1_sampled'
        output_stream: 'PACKET:out_1_sampled'
      }
      node {
        calculator: 'DropCalculator'
        input_side_packet: "DROP_TIMESTAMPS:drop_timesamps"
        input_stream: 'PACKET:out_1_sampled'
        output_stream: 'PACKET:out_1'
      }
    )pb");
  }

  // Parse an absl::Time from RFC3339 format.
  absl::Time ParseTime(const std::string& date_time_str) {
    absl::Time result;
    absl::ParseTime(absl::RFC3339_sec, date_time_str, &result, nullptr);
    return result;
  }

  // The point in simulated time when the test starts.
  absl::Time StartTime() { return ParseTime("2020-11-03T20:00:00Z"); }

  // Initialize the test clock to follow simulated time.
  void SetUpSimulationClock() {
    auto executor = std::make_shared<SimulationClockExecutor>(8);
    simulation_clock_ = executor->GetClock();
    clock_ = simulation_clock_.get();
    simulation_clock_->ThreadStart();
    clock_->SleepUntil(StartTime());
    simulation_clock_->ThreadFinish();
    MP_ASSERT_OK(graph_.SetExecutor("", executor));
  }

  // Initialize the test clock to follow wall time.
  void SetUpRealClock() { clock_ = mediapipe::Clock::RealClock(); }

  // Create a few mediapipe input Packets holding ints.
  void SetUpInputData() {
    for (int i = 0; i < 100; ++i) {
      input_packets_.push_back(MakePacket<int>(i).At(Timestamp(i * 10000)));
    }
  }

 protected:
  CalculatorGraph graph_;
  mediapipe::Clock* clock_;
  std::shared_ptr<SimulationClock> simulation_clock_;
  std::vector<Packet> input_packets_;
  std::vector<Packet> out_1_packets_;
  std::vector<Packet> allow_packets_;
};

// Shows that "FINISHED" can be indicated with either a packet or a timestamp
// bound.  DropCalculator periodically drops one packet but always propagates
// the timestamp bound.  Input packets are released or dropped promptly after
// each "FINISH" packet or a timestamp bound arrives.
TEST_F(FlowLimiterCalculatorTest, FinishedTimestamps) {
  // Configure the test.
  SetUpInputData();
  SetUpSimulationClock();
  CalculatorGraphConfig graph_config = InflightGraphConfig();
  auto limiter_options = ParseTextProtoOrDie<FlowLimiterCalculatorOptions>(R"pb(
    max_in_flight: 1
    max_in_queue: 1
  )pb");
  std::map<std::string, Packet> side_packets = {
      {"limiter_options",
       MakePacket<FlowLimiterCalculatorOptions>(limiter_options)},
      {"warmup_time", MakePacket<int64>(22000)},
      {"sleep_time", MakePacket<int64>(22000)},
      {"drop_timesamps", MakePacket<bool>(false)},
      {"clock", MakePacket<mediapipe::Clock*>(clock_)},
  };

  // Start the graph.
  MP_ASSERT_OK(graph_.Initialize(graph_config));
  MP_EXPECT_OK(graph_.ObserveOutputStream("out_1", [this](Packet p) {
    out_1_packets_.push_back(p);
    return absl::OkStatus();
  }));
  MP_EXPECT_OK(graph_.ObserveOutputStream("allow", [this](Packet p) {
    allow_packets_.push_back(p);
    return absl::OkStatus();
  }));
  simulation_clock_->ThreadStart();
  MP_ASSERT_OK(graph_.StartRun(side_packets));

  // Add 9 input packets.
  // 1. packet-0 is released,
  // 2. packet-1 is queued,
  // 3. packet-2 is queued and packet-1 is dropped,
  // 4. packet-2 is released, and so forth.
  MP_EXPECT_OK(graph_.AddPacketToInputStream("in_1", input_packets_[0]));
  clock_->Sleep(absl::Microseconds(1));
  EXPECT_EQ(allow_packets_.size(), 1);
  EXPECT_EQ(allow_packets_.back().Get<bool>(), true);
  clock_->Sleep(absl::Microseconds(10000));
  for (int i = 1; i < 8; i += 2) {
    MP_EXPECT_OK(graph_.AddPacketToInputStream("in_1", input_packets_[i]));
    clock_->Sleep(absl::Microseconds(10000));
    EXPECT_EQ(allow_packets_.size(), i);
    MP_EXPECT_OK(graph_.AddPacketToInputStream("in_1", input_packets_[i + 1]));
    clock_->Sleep(absl::Microseconds(1));
    EXPECT_EQ(allow_packets_.size(), i + 1);
    EXPECT_EQ(allow_packets_.back().Get<bool>(), false);
    clock_->Sleep(absl::Microseconds(10000));
    EXPECT_EQ(allow_packets_.size(), i + 2);
    EXPECT_EQ(allow_packets_.back().Get<bool>(), true);
  }

  // Finish the graph.
  MP_EXPECT_OK(graph_.CloseAllPacketSources());
  clock_->Sleep(absl::Microseconds(40000));
  MP_EXPECT_OK(graph_.WaitUntilDone());
  simulation_clock_->ThreadFinish();

  // Validate the output.
  // input_packets_[4] is dropped by the DropCalculator.
  std::vector<Packet> expected_output = {input_packets_[0], input_packets_[2],
                                         input_packets_[6], input_packets_[8]};
  EXPECT_EQ(out_1_packets_, expected_output);
}

// Shows that an output packet can be lost completely, and the
// FlowLimiterCalculator will stop waiting for it after in_flight_timeout.
// DropCalculator completely loses one packet including its timestamp bound.
// FlowLimiterCalculator waits 100 ms, and then starts releasing packets again.
TEST_F(FlowLimiterCalculatorTest, FinishedLost) {
  // Configure the test.
  SetUpInputData();
  SetUpSimulationClock();
  CalculatorGraphConfig graph_config = InflightGraphConfig();
  auto limiter_options = ParseTextProtoOrDie<FlowLimiterCalculatorOptions>(R"pb(
    max_in_flight: 1
    max_in_queue: 1
    in_flight_timeout: 100000  # 100 ms
  )pb");
  std::map<std::string, Packet> side_packets = {
      {"limiter_options",
       MakePacket<FlowLimiterCalculatorOptions>(limiter_options)},
      {"warmup_time", MakePacket<int64>(22000)},
      {"sleep_time", MakePacket<int64>(22000)},
      {"drop_timesamps", MakePacket<bool>(true)},
      {"clock", MakePacket<mediapipe::Clock*>(clock_)},
  };

  // Start the graph.
  MP_ASSERT_OK(graph_.Initialize(graph_config));
  MP_EXPECT_OK(graph_.ObserveOutputStream("out_1", [this](Packet p) {
    out_1_packets_.push_back(p);
    return absl::OkStatus();
  }));
  MP_EXPECT_OK(graph_.ObserveOutputStream("allow", [this](Packet p) {
    allow_packets_.push_back(p);
    return absl::OkStatus();
  }));
  simulation_clock_->ThreadStart();
  MP_ASSERT_OK(graph_.StartRun(side_packets));

  // Add 21 input packets.
  // 1. packet-0 is released, packet-1 queued and dropped, and so forth.
  // 2. packet-4 is lost by DropCalculator.
  // 3. packet-5 through 13 are dropped while waiting for packet-4.
  // 4. packet-4 expires and queued packet-14 is released.
  // 5. packet-17, 19, and 20 are released on time.
  MP_EXPECT_OK(graph_.AddPacketToInputStream("in_1", input_packets_[0]));
  clock_->Sleep(absl::Microseconds(10000));
  for (int i = 1; i < 21; ++i) {
    MP_EXPECT_OK(graph_.AddPacketToInputStream("in_1", input_packets_[i]));
    clock_->Sleep(absl::Microseconds(10000));
  }

  // Finish the graph.
  MP_EXPECT_OK(graph_.CloseAllPacketSources());
  clock_->Sleep(absl::Microseconds(40000));
  MP_EXPECT_OK(graph_.WaitUntilDone());
  simulation_clock_->ThreadFinish();

  // Validate the output.
  // input_packets_[4] is lost by the DropCalculator.
  std::vector<Packet> expected_output = {
      input_packets_[0],  input_packets_[2],  input_packets_[14],
      input_packets_[17], input_packets_[19], input_packets_[20],
  };
  EXPECT_EQ(out_1_packets_, expected_output);
}

// Shows what happens when a finish packet is delayed beyond in_flight_timeout.
// After in_flight_timeout, FlowLimiterCalculator continues releasing packets.
// Temporarily, more than max_in_flight frames are in flight.
// Eventually, the number of frames in flight returns to max_in_flight.
TEST_F(FlowLimiterCalculatorTest, FinishedDelayed) {
  // Configure the test.
  SetUpInputData();
  SetUpSimulationClock();
  CalculatorGraphConfig graph_config = InflightGraphConfig();
  auto limiter_options = ParseTextProtoOrDie<FlowLimiterCalculatorOptions>(R"pb(
    max_in_flight: 1
    max_in_queue: 1
    in_flight_timeout: 100000  # 100 ms
  )pb");
  std::map<std::string, Packet> side_packets = {
      {"limiter_options",
       MakePacket<FlowLimiterCalculatorOptions>(limiter_options)},
      {"warmup_time", MakePacket<int64>(500000)},
      {"sleep_time", MakePacket<int64>(22000)},
      {"drop_timesamps", MakePacket<bool>(false)},
      {"clock", MakePacket<mediapipe::Clock*>(clock_)},
  };

  // Start the graph.
  MP_ASSERT_OK(graph_.Initialize(graph_config));
  MP_EXPECT_OK(graph_.ObserveOutputStream("out_1", [this](Packet p) {
    out_1_packets_.push_back(p);
    return absl::OkStatus();
  }));
  MP_EXPECT_OK(graph_.ObserveOutputStream("allow", [this](Packet p) {
    allow_packets_.push_back(p);
    return absl::OkStatus();
  }));
  simulation_clock_->ThreadStart();
  MP_ASSERT_OK(graph_.StartRun(side_packets));

  // Add 71 input packets.
  // 1. During the 500 ms WARMUP_TIME, the in_flight_timeout releases
  //    packets 0, 10, 20, 30, 40, 50, which are queued at the SleepCalculator.
  // 2. During the next 120 ms, these 6 packets are processed.
  // 3. After the graph is finally finished with warmup and the backlog packets,
  //    packets 60 through 70 are released and processed on time.
  MP_EXPECT_OK(graph_.AddPacketToInputStream("in_1", input_packets_[0]));
  clock_->Sleep(absl::Microseconds(10000));
  for (int i = 1; i < 71; ++i) {
    MP_EXPECT_OK(graph_.AddPacketToInputStream("in_1", input_packets_[i]));
    clock_->Sleep(absl::Microseconds(10000));
  }

  // Finish the graph.
  MP_EXPECT_OK(graph_.CloseAllPacketSources());
  clock_->Sleep(absl::Microseconds(40000));
  MP_EXPECT_OK(graph_.WaitUntilDone());
  simulation_clock_->ThreadFinish();

  // Validate the output.
  // The graph is warming up or backlogged until packet 60.
  std::vector<Packet> expected_output = {
      input_packets_[0],  input_packets_[10], input_packets_[30],
      input_packets_[40], input_packets_[50], input_packets_[60],
      input_packets_[63], input_packets_[65], input_packets_[67],
      input_packets_[69], input_packets_[70],
  };
  EXPECT_EQ(out_1_packets_, expected_output);
}

// Shows that packets on auxiliary input streams are relesed for the same
// timestamps as the main input stream, whether the auxiliary packets arrive
// early or late.
TEST_F(FlowLimiterCalculatorTest, TwoInputStreams) {
  // Configure the test.
  SetUpInputData();
  SetUpSimulationClock();
  CalculatorGraphConfig graph_config =
      ParseTextProtoOrDie<CalculatorGraphConfig>(R"pb(
        input_stream: 'in_1'
        input_stream: 'in_2'
        node {
          calculator: 'FlowLimiterCalculator'
          input_side_packet: 'OPTIONS:limiter_options'
          input_stream: 'in_1'
          input_stream: 'in_2'
          input_stream: 'FINISHED:out_1'
          input_stream_info: { tag_index: 'FINISHED' back_edge: true }
          output_stream: 'in_1_sampled'
          output_stream: 'in_2_sampled'
          output_stream: 'ALLOW:allow'
        }
        node {
          calculator: 'SleepCalculator'
          input_side_packet: 'WARMUP_TIME:warmup_time'
          input_side_packet: 'SLEEP_TIME:sleep_time'
          input_side_packet: 'CLOCK:clock'
          input_stream: 'PACKET:in_1_sampled'
          output_stream: 'PACKET:out_1_sampled'
        }
        node {
          calculator: 'DropCalculator'
          input_side_packet: "DROP_TIMESTAMPS:drop_timesamps"
          input_stream: 'PACKET:out_1_sampled'
          output_stream: 'PACKET:out_1'
        }
      )pb");

  auto limiter_options = ParseTextProtoOrDie<FlowLimiterCalculatorOptions>(R"pb(
    max_in_flight: 1
    max_in_queue: 1
    in_flight_timeout: 100000  # 100 ms
  )pb");
  std::map<std::string, Packet> side_packets = {
      {"limiter_options",
       MakePacket<FlowLimiterCalculatorOptions>(limiter_options)},
      {"warmup_time", MakePacket<int64>(22000)},
      {"sleep_time", MakePacket<int64>(22000)},
      {"drop_timesamps", MakePacket<bool>(true)},
      {"clock", MakePacket<mediapipe::Clock*>(clock_)},
  };

  // Start the graph.
  MP_ASSERT_OK(graph_.Initialize(graph_config));
  MP_EXPECT_OK(graph_.ObserveOutputStream("out_1", [this](Packet p) {
    out_1_packets_.push_back(p);
    return absl::OkStatus();
  }));
  std::vector<Packet> out_2_packets;
  MP_EXPECT_OK(graph_.ObserveOutputStream("in_2_sampled", [&](Packet p) {
    out_2_packets.push_back(p);
    return absl::OkStatus();
  }));
  MP_EXPECT_OK(graph_.ObserveOutputStream("allow", [this](Packet p) {
    allow_packets_.push_back(p);
    return absl::OkStatus();
  }));
  simulation_clock_->ThreadStart();
  MP_ASSERT_OK(graph_.StartRun(side_packets));

  // Add packets 0..9 to stream in_1, and packets 0..10 to stream in_2.
  MP_EXPECT_OK(graph_.AddPacketToInputStream("in_1", input_packets_[0]));
  clock_->Sleep(absl::Microseconds(10000));
  for (int i = 1; i < 10; ++i) {
    MP_EXPECT_OK(graph_.AddPacketToInputStream("in_1", input_packets_[i]));
    MP_EXPECT_OK(graph_.AddPacketToInputStream("in_2", input_packets_[i - 1]));
    clock_->Sleep(absl::Microseconds(10000));
  }

  // Add packets 10..20 to stream in_1, and packets 11..21 to stream in_2.
  for (int i = 10; i < 21; ++i) {
    MP_EXPECT_OK(graph_.AddPacketToInputStream("in_2", input_packets_[i + 1]));
    MP_EXPECT_OK(graph_.AddPacketToInputStream("in_1", input_packets_[i]));
    clock_->Sleep(absl::Microseconds(10000));
  }

  // Finish the graph run.
  MP_EXPECT_OK(graph_.CloseAllPacketSources());
  clock_->Sleep(absl::Microseconds(40000));
  MP_EXPECT_OK(graph_.WaitUntilDone());
  simulation_clock_->ThreadFinish();

  // Validate the output.
  // Packet input_packets_[4] is lost by the DropCalculator.
  std::vector<Packet> expected_output = {
      input_packets_[0],  input_packets_[2],  input_packets_[14],
      input_packets_[17], input_packets_[19], input_packets_[20],
  };
  EXPECT_EQ(out_1_packets_, expected_output);
  // Exactly the timestamps released by FlowLimiterCalculator for in_1_sampled.
  std::vector<Packet> expected_output_2 = {
      input_packets_[0],  input_packets_[2],  input_packets_[4],
      input_packets_[14], input_packets_[17], input_packets_[19],
      input_packets_[20],
  };
  EXPECT_EQ(out_2_packets, expected_output_2);
}

// Shows how FlowLimiterCalculator releases packets with max_in_queue 0.
// Shows how auxiliary input streams still work with max_in_queue 0.
// The processing time "sleep_time" is reduced from 22ms to 12ms to create
// the same frame rate as FlowLimiterCalculatorTest::TwoInputStreams.
TEST_F(FlowLimiterCalculatorTest, ZeroQueue) {
  // Configure the test.
  SetUpInputData();
  SetUpSimulationClock();
  CalculatorGraphConfig graph_config =
      ParseTextProtoOrDie<CalculatorGraphConfig>(R"pb(
        input_stream: 'in_1'
        input_stream: 'in_2'
        node {
          calculator: 'FlowLimiterCalculator'
          input_side_packet: 'OPTIONS:limiter_options'
          input_stream: 'in_1'
          input_stream: 'in_2'
          input_stream: 'FINISHED:out_1'
          input_stream_info: { tag_index: 'FINISHED' back_edge: true }
          output_stream: 'in_1_sampled'
          output_stream: 'in_2_sampled'
          output_stream: 'ALLOW:allow'
        }
        node {
          calculator: 'SleepCalculator'
          input_side_packet: 'WARMUP_TIME:warmup_time'
          input_side_packet: 'SLEEP_TIME:sleep_time'
          input_side_packet: 'CLOCK:clock'
          input_stream: 'PACKET:in_1_sampled'
          output_stream: 'PACKET:out_1_sampled'
        }
        node {
          calculator: 'DropCalculator'
          input_side_packet: "DROP_TIMESTAMPS:drop_timesamps"
          input_stream: 'PACKET:out_1_sampled'
          output_stream: 'PACKET:out_1'
        }
      )pb");

  auto limiter_options = ParseTextProtoOrDie<FlowLimiterCalculatorOptions>(R"pb(
    max_in_flight: 1
    max_in_queue: 0
    in_flight_timeout: 100000  # 100 ms
  )pb");
  std::map<std::string, Packet> side_packets = {
      {"limiter_options",
       MakePacket<FlowLimiterCalculatorOptions>(limiter_options)},
      {"warmup_time", MakePacket<int64>(12000)},
      {"sleep_time", MakePacket<int64>(12000)},
      {"drop_timesamps", MakePacket<bool>(true)},
      {"clock", MakePacket<mediapipe::Clock*>(clock_)},
  };

  // Start the graph.
  MP_ASSERT_OK(graph_.Initialize(graph_config));
  MP_EXPECT_OK(graph_.ObserveOutputStream("out_1", [this](Packet p) {
    out_1_packets_.push_back(p);
    return absl::OkStatus();
  }));
  std::vector<Packet> out_2_packets;
  MP_EXPECT_OK(graph_.ObserveOutputStream("in_2_sampled", [&](Packet p) {
    out_2_packets.push_back(p);
    return absl::OkStatus();
  }));
  MP_EXPECT_OK(graph_.ObserveOutputStream("allow", [this](Packet p) {
    allow_packets_.push_back(p);
    return absl::OkStatus();
  }));
  simulation_clock_->ThreadStart();
  MP_ASSERT_OK(graph_.StartRun(side_packets));

  // Add packets 0..9 to stream in_1, and packets 0..10 to stream in_2.
  MP_EXPECT_OK(graph_.AddPacketToInputStream("in_1", input_packets_[0]));
  clock_->Sleep(absl::Microseconds(10000));
  for (int i = 1; i < 10; ++i) {
    MP_EXPECT_OK(graph_.AddPacketToInputStream("in_1", input_packets_[i]));
    MP_EXPECT_OK(graph_.AddPacketToInputStream("in_2", input_packets_[i - 1]));
    clock_->Sleep(absl::Microseconds(10000));
  }

  // Add packets 10..20 to stream in_1, and packets 11..21 to stream in_2.
  for (int i = 10; i < 21; ++i) {
    MP_EXPECT_OK(graph_.AddPacketToInputStream("in_2", input_packets_[i + 1]));
    MP_EXPECT_OK(graph_.AddPacketToInputStream("in_1", input_packets_[i]));
    clock_->Sleep(absl::Microseconds(10000));
  }

  // Finish the graph run.
  MP_EXPECT_OK(graph_.CloseAllPacketSources());
  clock_->Sleep(absl::Microseconds(40000));
  MP_EXPECT_OK(graph_.WaitUntilDone());
  simulation_clock_->ThreadFinish();

  // Validate the output.
  // Packet input_packets_[4] is lost by the DropCalculator.
  std::vector<Packet> expected_output = {
      input_packets_[0],  input_packets_[2],  input_packets_[15],
      input_packets_[17], input_packets_[19],
  };
  EXPECT_EQ(out_1_packets_, expected_output);
  // Exactly the timestamps released by FlowLimiterCalculator for in_1_sampled.
  std::vector<Packet> expected_output_2 = {
      input_packets_[0],  input_packets_[2],  input_packets_[4],
      input_packets_[15], input_packets_[17], input_packets_[19],
  };
  EXPECT_EQ(out_2_packets, expected_output_2);
}

}  // anonymous namespace
}  // namespace mediapipe
