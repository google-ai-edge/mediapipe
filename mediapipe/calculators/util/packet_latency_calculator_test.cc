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

#include "absl/time/time.h"
#include "mediapipe/calculators/util/latency.pb.h"
#include "mediapipe/framework/calculator_runner.h"
#include "mediapipe/framework/deps/clock.h"
#include "mediapipe/framework/deps/message_matchers.h"
#include "mediapipe/framework/port/gmock.h"
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/framework/port/parse_text_proto.h"
#include "mediapipe/framework/port/status_matchers.h"
#include "mediapipe/framework/timestamp.h"
#include "mediapipe/framework/tool/simulation_clock_executor.h"
#include "mediapipe/framework/tool/sink.h"

namespace mediapipe {

namespace {

class PacketLatencyCalculatorTest : public ::testing::Test {
 protected:
  void SetupSimulationClock() {
    auto executor = std::make_shared<SimulationClockExecutor>(4);
    simulation_clock_ = executor->GetClock();
    MP_ASSERT_OK(graph_.SetExecutor("", executor));
  }

  void InitializeSingleStreamGraph() {
    graph_config_ = ParseTextProtoOrDie<CalculatorGraphConfig>(R"(
      input_stream: "delayed_packet_0"
      input_stream: "camera_frames"
      node {
        calculator: "PacketLatencyCalculator"
        input_side_packet: "CLOCK:clock"
        input_stream: "delayed_packet_0"
        input_stream: "REFERENCE_SIGNAL:camera_frames"
        output_stream: "packet_latency_0"
        options {
          [mediapipe.PacketLatencyCalculatorOptions.ext] {
            num_intervals: 3
            interval_size_usec: 4
            reset_duration_usec: 100
            packet_labels: "dummy input 0"
          }
        }
        input_stream_handler {
          input_stream_handler: "ImmediateInputStreamHandler"
        }
      }
    )");

    mediapipe::tool::AddVectorSink("packet_latency_0", &graph_config_,
                                   &out_0_packets_);

    // Create the simulation clock side packet.
    SetupSimulationClock();
    std::map<std::string, ::mediapipe::Packet> side_packet;
    side_packet["clock"] =
        ::mediapipe::MakePacket<std::shared_ptr<::mediapipe::Clock>>(
            simulation_clock_);

    // Start graph run.
    MP_ASSERT_OK(graph_.Initialize(graph_config_, {}));
    MP_ASSERT_OK(graph_.StartRun(side_packet));
    // Let Calculator::Open() calls finish before continuing.
    MP_ASSERT_OK(graph_.WaitUntilIdle());
  }

  void InitializeMultipleStreamGraph() {
    graph_config_ = ParseTextProtoOrDie<CalculatorGraphConfig>(R"(
      input_stream: "delayed_packet_0"
      input_stream: "delayed_packet_1"
      input_stream: "delayed_packet_2"
      input_stream: "camera_frames"
      node {
        calculator: "PacketLatencyCalculator"
        input_side_packet: "CLOCK:clock"
        input_stream: "delayed_packet_0"
        input_stream: "delayed_packet_1"
        input_stream: "delayed_packet_2"
        input_stream: "REFERENCE_SIGNAL:camera_frames"
        output_stream: "packet_latency_0"
        output_stream: "packet_latency_1"
        output_stream: "packet_latency_2"
        options {
          [mediapipe.PacketLatencyCalculatorOptions.ext] {
            num_intervals: 3
            interval_size_usec: 4
            packet_labels: "dummy input 0"
            packet_labels: "dummy input 1"
            packet_labels: "dummy input 2"
          }
        }
        input_stream_handler {
          input_stream_handler: "ImmediateInputStreamHandler"
        }
      }
    )");

    mediapipe::tool::AddVectorSink("packet_latency_0", &graph_config_,
                                   &out_0_packets_);
    mediapipe::tool::AddVectorSink("packet_latency_1", &graph_config_,
                                   &out_1_packets_);
    mediapipe::tool::AddVectorSink("packet_latency_2", &graph_config_,
                                   &out_2_packets_);
    MP_ASSERT_OK(graph_.Initialize(graph_config_, {}));

    // Create the simulation clock side packet.
    simulation_clock_.reset(new SimulationClock());
    std::map<std::string, ::mediapipe::Packet> side_packet;
    side_packet["clock"] =
        ::mediapipe::MakePacket<std::shared_ptr<::mediapipe::Clock>>(
            simulation_clock_);

    // Start graph run.
    MP_ASSERT_OK(graph_.StartRun(side_packet));
    // Let Calculator::Open() calls finish before continuing.
    MP_ASSERT_OK(graph_.WaitUntilIdle());
  }

  void InitializeSingleStreamGraphWithoutClock() {
    graph_config_ = ParseTextProtoOrDie<CalculatorGraphConfig>(R"(
      input_stream: "delayed_packet_0"
      input_stream: "camera_frames"
      node {
        calculator: "PacketLatencyCalculator"
        input_stream: "delayed_packet_0"
        input_stream: "REFERENCE_SIGNAL:camera_frames"
        output_stream: "packet_latency_0"
        options {
          [mediapipe.PacketLatencyCalculatorOptions.ext] {
            num_intervals: 3
            interval_size_usec: 4
            packet_labels: "dummy input 0"
          }
        }
        input_stream_handler {
          input_stream_handler: "ImmediateInputStreamHandler"
        }
      }
    )");

    mediapipe::tool::AddVectorSink("packet_latency_0", &graph_config_,
                                   &out_0_packets_);

    // Create the simulation clock side packet.
    SetupSimulationClock();
    std::map<std::string, ::mediapipe::Packet> side_packet;
    side_packet["clock"] =
        ::mediapipe::MakePacket<std::shared_ptr<::mediapipe::Clock>>(
            simulation_clock_);

    // Start graph run.
    MP_ASSERT_OK(graph_.Initialize(graph_config_, {}));
    MP_ASSERT_OK(graph_.StartRun(side_packet));
    // Let Calculator::Open() calls finish before continuing.
    MP_ASSERT_OK(graph_.WaitUntilIdle());
  }

  PacketLatency CreatePacketLatency(const double latency_usec,
                                    const int64 num_intervals,
                                    const int64 interval_size_usec,
                                    const std::vector<int>& counts,
                                    const int64 avg_latency_usec,
                                    const std::string& label) {
    PacketLatency latency_info;
    latency_info.set_current_latency_usec(latency_usec);
    latency_info.set_num_intervals(num_intervals);
    latency_info.set_interval_size_usec(interval_size_usec);
    int sum_counts = 0;
    for (const int& count : counts) {
      latency_info.add_counts(count);
      sum_counts += count;
    }
    latency_info.set_avg_latency_usec(avg_latency_usec);
    latency_info.set_sum_latency_usec(avg_latency_usec * sum_counts);
    latency_info.set_label(label);
    return latency_info;
  }

  std::shared_ptr<::mediapipe::Clock> simulation_clock_;
  CalculatorGraphConfig graph_config_;
  CalculatorGraph graph_;
  std::vector<Packet> out_0_packets_;
  std::vector<Packet> out_1_packets_;
  std::vector<Packet> out_2_packets_;
};

// Calculator must not output any latency until input packets are received.
TEST_F(PacketLatencyCalculatorTest, DoesNotOutputUntilInputPacketReceived) {
  // Initialize graph_.
  InitializeSingleStreamGraph();
  dynamic_cast<SimulationClock*>(&*simulation_clock_)->ThreadStart();

  // Send reference packets with timestamps 0, 6 and 10 usec.
  MP_ASSERT_OK(graph_.AddPacketToInputStream(
      "camera_frames", Adopt(new double()).At(Timestamp(0))));
  MP_ASSERT_OK(graph_.AddPacketToInputStream(
      "camera_frames", Adopt(new double()).At(Timestamp(6))));
  MP_ASSERT_OK(graph_.AddPacketToInputStream(
      "camera_frames", Adopt(new double()).At(Timestamp(10))));

  dynamic_cast<SimulationClock*>(&*simulation_clock_)->ThreadFinish();
  MP_ASSERT_OK(graph_.CloseAllInputStreams());
  MP_ASSERT_OK(graph_.WaitUntilDone());

  // Expect zero output packets.
  ASSERT_EQ(out_0_packets_.size(), 0);
}

// Calculator must output correct latency for single stream.
TEST_F(PacketLatencyCalculatorTest, OutputsCorrectLatencyForSingleStream) {
  // Initialize graph_.
  InitializeSingleStreamGraph();
  dynamic_cast<SimulationClock*>(&*simulation_clock_)->ThreadStart();

  // Send a reference packet with timestamp 10 usec at time 12 usec.
  simulation_clock_->Sleep(absl::Microseconds(12));
  MP_ASSERT_OK(graph_.AddPacketToInputStream(
      "camera_frames", Adopt(new double()).At(Timestamp(10))));

  // Add two delayed packets with timestamp 1 and 8 resp.
  simulation_clock_->Sleep(absl::Microseconds(1));
  MP_ASSERT_OK(graph_.AddPacketToInputStream(
      "delayed_packet_0", Adopt(new double()).At(Timestamp(1))));
  simulation_clock_->Sleep(absl::Microseconds(1));
  MP_ASSERT_OK(graph_.AddPacketToInputStream(
      "delayed_packet_0", Adopt(new double()).At(Timestamp(8))));

  dynamic_cast<SimulationClock*>(&*simulation_clock_)->ThreadFinish();
  MP_ASSERT_OK(graph_.CloseAllInputStreams());
  MP_ASSERT_OK(graph_.WaitUntilDone());

  // Expect two latency packets with timestamp 1 and 8 resp.
  ASSERT_EQ(out_0_packets_.size(), 2);
  EXPECT_EQ(out_0_packets_[0].Timestamp().Value(), 1);
  EXPECT_EQ(out_0_packets_[1].Timestamp().Value(), 8);

  EXPECT_THAT(
      out_0_packets_[0].Get<PacketLatency>(),
      EqualsProto(CreatePacketLatency(
          /*latency_usec=*/10,
          /*num_intervals=*/3, /*interval_size_usec=*/4,
          /*counts=*/{0, 0, 1}, /*avg_latency_usec=*/10, "dummy input 0")));

  EXPECT_THAT(
      out_0_packets_[1].Get<PacketLatency>(),
      EqualsProto(CreatePacketLatency(
          /*latency_usec=*/4,
          /*num_intervals=*/3, /*interval_size_usec=*/4,
          /*counts=*/{0, 1, 1}, /*avg_latency_usec=*/7, "dummy input 0")));
}

// Calculator must not output latency until reference signal is received.
TEST_F(PacketLatencyCalculatorTest, DoesNotOutputUntilReferencePacketReceived) {
  // Initialize graph_.
  InitializeSingleStreamGraph();
  dynamic_cast<SimulationClock*>(&*simulation_clock_)->ThreadStart();

  // Add two packets with timestamp 1 and 2.
  MP_ASSERT_OK(graph_.AddPacketToInputStream(
      "delayed_packet_0", Adopt(new double()).At(Timestamp(1))));
  MP_ASSERT_OK(graph_.AddPacketToInputStream(
      "delayed_packet_0", Adopt(new double()).At(Timestamp(2))));

  // Send a reference packet with timestamp 10 usec.
  simulation_clock_->Sleep(absl::Microseconds(1));
  MP_ASSERT_OK(graph_.AddPacketToInputStream(
      "camera_frames", Adopt(new double()).At(Timestamp(10))));
  simulation_clock_->Sleep(absl::Microseconds(1));

  // Add two delayed packets with timestamp 7 and 9 resp.
  MP_ASSERT_OK(graph_.AddPacketToInputStream(
      "delayed_packet_0", Adopt(new double()).At(Timestamp(7))));
  MP_ASSERT_OK(graph_.AddPacketToInputStream(
      "delayed_packet_0", Adopt(new double()).At(Timestamp(9))));
  simulation_clock_->Sleep(absl::Microseconds(1));

  dynamic_cast<SimulationClock*>(&*simulation_clock_)->ThreadFinish();
  MP_ASSERT_OK(graph_.CloseAllInputStreams());
  MP_ASSERT_OK(graph_.WaitUntilDone());

  // Expect two latency packets with timestamp 7 and 9 resp. The packets with
  // timestamps 1 and 2 should not have any latency associated with them since
  // reference signal was not sent until then.
  ASSERT_EQ(out_0_packets_.size(), 2);
  EXPECT_EQ(out_0_packets_[0].Timestamp().Value(), 7);
  EXPECT_EQ(out_0_packets_[1].Timestamp().Value(), 9);

  EXPECT_THAT(
      out_0_packets_[0].Get<PacketLatency>(),
      EqualsProto(CreatePacketLatency(
          /*latency_usec=*/4,
          /*num_intervals=*/3, /*interval_size_usec=*/4,
          /*counts=*/{0, 1, 0}, /*avg_latency_usec=*/4, "dummy input 0")));

  EXPECT_THAT(
      out_0_packets_[1].Get<PacketLatency>(),
      EqualsProto(CreatePacketLatency(
          /*latency_usec=*/2, /*num_intervals=*/3,
          /*interval_size_usec=*/4,
          /*counts=*/{1, 1, 0}, /*avg_latency_usec=*/3, "dummy input 0")));
}

// Calculator outputs latency even when a clock is not provided.
TEST_F(PacketLatencyCalculatorTest, OutputsCorrectLatencyWhenNoClock) {
  // Initialize graph_.
  InitializeSingleStreamGraphWithoutClock();
  dynamic_cast<SimulationClock*>(&*simulation_clock_)->ThreadStart();

  // Send a reference packet with timestamp 10 usec.
  MP_ASSERT_OK(graph_.AddPacketToInputStream(
      "camera_frames", Adopt(new double()).At(Timestamp(10))));

  // Add two delayed packets with timestamp 5 and 10 resp.
  MP_ASSERT_OK(graph_.AddPacketToInputStream(
      "delayed_packet_0", Adopt(new double()).At(Timestamp(5))));
  MP_ASSERT_OK(graph_.AddPacketToInputStream(
      "delayed_packet_0", Adopt(new double()).At(Timestamp(10))));

  dynamic_cast<SimulationClock*>(&*simulation_clock_)->ThreadFinish();
  MP_ASSERT_OK(graph_.CloseAllInputStreams());
  MP_ASSERT_OK(graph_.WaitUntilDone());

  // Expect two latency packets with timestamp 5 and 10 resp.
  ASSERT_EQ(out_0_packets_.size(), 2);
  EXPECT_EQ(out_0_packets_[0].Timestamp().Value(), 5);
  EXPECT_EQ(out_0_packets_[1].Timestamp().Value(), 10);
}

// Calculator must output correct histograms counts for the corner bins.
TEST_F(PacketLatencyCalculatorTest,
       OutputsCorrectLatencyStatisticsInTimeWindow) {
  // Initialize graph_.
  InitializeSingleStreamGraph();
  dynamic_cast<SimulationClock*>(&*simulation_clock_)->ThreadStart();

  // Send a reference packet with timestamp 20 usec.
  MP_ASSERT_OK(graph_.AddPacketToInputStream(
      "camera_frames", Adopt(new double()).At(Timestamp(20))));

  // Add two delayed packets with timestamp 0 and 20 resp.
  MP_ASSERT_OK(graph_.AddPacketToInputStream(
      "delayed_packet_0", Adopt(new double()).At(Timestamp(0))));
  MP_ASSERT_OK(graph_.AddPacketToInputStream(
      "delayed_packet_0", Adopt(new double()).At(Timestamp(20))));

  dynamic_cast<SimulationClock*>(&*simulation_clock_)->ThreadFinish();
  MP_ASSERT_OK(graph_.CloseAllInputStreams());
  MP_ASSERT_OK(graph_.WaitUntilDone());

  // Expect two latency packets with timestamp 0 and 20 resp.
  ASSERT_EQ(out_0_packets_.size(), 2);
  EXPECT_EQ(out_0_packets_[0].Timestamp().Value(), 0);
  EXPECT_EQ(out_0_packets_[1].Timestamp().Value(), 20);

  EXPECT_THAT(
      out_0_packets_[0].Get<PacketLatency>(),
      EqualsProto(CreatePacketLatency(
          /*latency_usec=*/20, /*num_intervals=*/3,
          /*interval_size_usec=*/4,
          /*counts=*/{0, 0, 1}, /*avg_latency_usec=*/20, "dummy input 0")));

  EXPECT_THAT(
      out_0_packets_[1].Get<PacketLatency>(),
      EqualsProto(CreatePacketLatency(
          /*latency_usec=*/0, /*num_intervals=*/3,
          /*interval_size_usec=*/4,
          /*counts=*/{1, 0, 1}, /*avg_latency_usec=*/10, "dummy input 0")));
}

// Calculator must reset histogram and average after specified duration.
TEST_F(PacketLatencyCalculatorTest, ResetsHistogramAndAverageCorrectly) {
  // Initialize graph_.
  InitializeSingleStreamGraph();
  dynamic_cast<SimulationClock*>(&*simulation_clock_)->ThreadStart();

  // Send a reference packet with timestamp 0 usec.
  MP_ASSERT_OK(graph_.AddPacketToInputStream(
      "camera_frames", Adopt(new double()).At(Timestamp(0))));

  // Add a delayed packet with timestamp 0 usec at time 20 usec.
  simulation_clock_->Sleep(absl::Microseconds(20));
  MP_ASSERT_OK(graph_.AddPacketToInputStream(
      "delayed_packet_0", Adopt(new double()).At(Timestamp(0))));

  // Do a long sleep so that histogram and average are reset.
  simulation_clock_->Sleep(absl::Microseconds(100));

  // Add a delayed packet with timestamp 115 usec at time 120 usec.
  MP_ASSERT_OK(graph_.AddPacketToInputStream(
      "delayed_packet_0", Adopt(new double()).At(Timestamp(115))));

  dynamic_cast<SimulationClock*>(&*simulation_clock_)->ThreadFinish();
  MP_ASSERT_OK(graph_.CloseAllInputStreams());
  MP_ASSERT_OK(graph_.WaitUntilDone());

  // Expect two latency packets with timestamp 0 and 115 resp.
  ASSERT_EQ(out_0_packets_.size(), 2);
  EXPECT_EQ(out_0_packets_[0].Timestamp().Value(), 0);
  EXPECT_EQ(out_0_packets_[1].Timestamp().Value(), 115);

  EXPECT_THAT(
      out_0_packets_[0].Get<PacketLatency>(),
      EqualsProto(CreatePacketLatency(
          /*latency_usec=*/20, /*num_intervals=*/3,
          /*interval_size_usec=*/4,
          /*counts=*/{0, 0, 1}, /*avg_latency_usec=*/20, "dummy input 0")));

  // The new average and histogram should ignore the previous latency because
  // reset has happened.
  EXPECT_THAT(
      out_0_packets_[1].Get<PacketLatency>(),
      EqualsProto(CreatePacketLatency(
          /*latency_usec=*/5, /*num_intervals=*/3,
          /*interval_size_usec=*/4,
          /*counts=*/{0, 1, 0}, /*avg_latency_usec=*/5, "dummy input 0")));
}

// Calculator must output correct latency for multiple streams.
TEST_F(PacketLatencyCalculatorTest, OutputsCorrectLatencyForMultipleStreams) {
  // Initialize graph.
  InitializeMultipleStreamGraph();
  dynamic_cast<SimulationClock*>(&*simulation_clock_)->ThreadStart();

  // Send a reference packet with timestamp 10 usec.
  MP_ASSERT_OK(graph_.AddPacketToInputStream(
      "camera_frames", Adopt(new double()).At(Timestamp(10))));

  // Add delayed packets on each input stream.

  // Fastest stream.
  MP_ASSERT_OK(graph_.AddPacketToInputStream(
      "delayed_packet_0", Adopt(new double()).At(Timestamp(10))));

  // Slow stream.
  MP_ASSERT_OK(graph_.AddPacketToInputStream(
      "delayed_packet_1", Adopt(new double()).At(Timestamp(5))));

  // Slowest stream.
  MP_ASSERT_OK(graph_.AddPacketToInputStream(
      "delayed_packet_2", Adopt(new double()).At(Timestamp(0))));

  dynamic_cast<SimulationClock*>(&*simulation_clock_)->ThreadFinish();
  MP_ASSERT_OK(graph_.CloseAllInputStreams());
  MP_ASSERT_OK(graph_.WaitUntilDone());

  // Expect one latency packet on each output stream.
  ASSERT_EQ(out_0_packets_.size(), 1);
  ASSERT_EQ(out_1_packets_.size(), 1);
  ASSERT_EQ(out_2_packets_.size(), 1);
  EXPECT_EQ(out_0_packets_[0].Timestamp().Value(), 10);
  EXPECT_EQ(out_1_packets_[0].Timestamp().Value(), 5);
  EXPECT_EQ(out_2_packets_[0].Timestamp().Value(), 0);

  EXPECT_THAT(
      out_0_packets_[0].Get<PacketLatency>(),
      EqualsProto(CreatePacketLatency(
          /*latency_usec=*/0, /*num_intervals=*/3,
          /*interval_size_usec=*/4,
          /*counts=*/{1, 0, 0}, /*avg_latency_usec=*/0, "dummy input 0")));
  EXPECT_THAT(
      out_1_packets_[0].Get<PacketLatency>(),
      EqualsProto(CreatePacketLatency(
          /*latency_usec=*/5, /*num_intervals=*/3,
          /*interval_size_usec=*/4,
          /*counts=*/{0, 1, 0}, /*avg_latency_usec=*/5, "dummy input 1")));
  EXPECT_THAT(
      out_2_packets_[0].Get<PacketLatency>(),
      EqualsProto(CreatePacketLatency(
          /*latency_usec=*/10, /*num_intervals=*/3,
          /*interval_size_usec=*/4,
          /*counts=*/{0, 0, 1}, /*avg_latency_usec=*/10, "dummy input 2")));
}

}  // namespace
}  // namespace mediapipe
