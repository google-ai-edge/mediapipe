// Copyright 2018 The MediaPipe Authors.
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

#include <vector>

#include "absl/synchronization/mutex.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/packet.h"
#include "mediapipe/framework/port/gmock.h"
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/framework/port/integral_types.h"
#include "mediapipe/framework/port/logging.h"
#include "mediapipe/framework/port/parse_text_proto.h"
#include "mediapipe/framework/port/status_matchers.h"
#include "mediapipe/framework/port/threadpool.h"
#include "mediapipe/framework/stream_handler/fixed_size_input_stream_handler.pb.h"

namespace mediapipe {

namespace {

const int64 kMaxPacketId = 100;
const int64 kSlowCalculatorRate = 10;

// Rate limiter for TestSlowCalculator.
ABSL_CONST_INIT absl::Mutex g_source_mutex(absl::kConstInit);
int64 g_source_counter ABSL_GUARDED_BY(g_source_mutex);

// Rate limiter for TestSourceCalculator.
int64 g_slow_counter ABSL_GUARDED_BY(g_source_mutex);

// Flag that indicates that the source is done.
bool g_source_done ABSL_GUARDED_BY(g_source_mutex);

class TestSourceCalculator : public CalculatorBase {
 public:
  TestSourceCalculator() : current_packet_id_(0) {}
  static ::mediapipe::Status GetContract(CalculatorContract* cc) {
    cc->Outputs().Index(0).Set<int64>();
    return ::mediapipe::OkStatus();
  }
  ::mediapipe::Status Open(CalculatorContext* cc) override {
    absl::MutexLock lock(&g_source_mutex);
    g_source_counter = 0;
    g_source_done = false;
    return ::mediapipe::OkStatus();
  }
  ::mediapipe::Status Process(CalculatorContext* cc) override {
    if (current_packet_id_ == kMaxPacketId) {
      absl::MutexLock lock(&g_source_mutex);
      g_source_done = true;
      return tool::StatusStop();
    }
    cc->Outputs().Index(0).Add(new int64(0), Timestamp(current_packet_id_));
    ++current_packet_id_;
    {
      absl::MutexLock lock(&g_source_mutex);
      ++g_source_counter;
      g_source_mutex.Await(
          absl::Condition(this, &TestSourceCalculator::CanProceed));
    }
    return ::mediapipe::OkStatus();
  }

 private:
  bool CanProceed() const ABSL_EXCLUSIVE_LOCKS_REQUIRED(g_source_mutex) {
    return g_source_counter <= kSlowCalculatorRate * g_slow_counter ||
           g_source_counter <= 1;
  }
  int64 current_packet_id_;
};

REGISTER_CALCULATOR(TestSourceCalculator);

class TestSlowCalculator : public CalculatorBase {
 public:
  TestSlowCalculator() = default;
  static ::mediapipe::Status GetContract(CalculatorContract* cc) {
    cc->Inputs().Index(0).Set<int64>();
    cc->Outputs().Index(0).Set<int64>();
    return ::mediapipe::OkStatus();
  }
  ::mediapipe::Status Open(CalculatorContext* cc) override {
    absl::MutexLock lock(&g_source_mutex);
    g_slow_counter = 0;
    return ::mediapipe::OkStatus();
  }
  ::mediapipe::Status Process(CalculatorContext* cc) override {
    cc->Outputs().Index(0).Add(new int64(0),
                               cc->Inputs().Index(0).Value().Timestamp());
    {
      absl::MutexLock lock(&g_source_mutex);
      ++g_slow_counter;
      g_source_mutex.Await(
          absl::Condition(this, &TestSlowCalculator::CanProceed));
    }
    return ::mediapipe::OkStatus();
  }

 private:
  bool CanProceed() const ABSL_EXCLUSIVE_LOCKS_REQUIRED(g_source_mutex) {
    return g_source_counter > kSlowCalculatorRate * g_slow_counter ||
           g_source_done;
  }
};

REGISTER_CALCULATOR(TestSlowCalculator);

// Return the values of the timestamps of a vector of Packets.
static std::vector<int64> TimestampValues(const std::vector<Packet>& packets) {
  std::vector<int64> result;
  for (const Packet& p : packets) {
    result.push_back(p.Timestamp().Value());
  }
  return result;
}

void SetFixedMinSize(CalculatorGraphConfig::Node* node, bool fixed_min_size) {
  node->mutable_input_stream_handler()
      ->mutable_options()
      ->MutableExtension(FixedSizeInputStreamHandlerOptions::ext)
      ->set_fixed_min_size(fixed_min_size);
}

class FixedSizeInputStreamHandlerTest : public ::testing::TestWithParam<bool> {
};
INSTANTIATE_TEST_SUITE_P(InstantiationFixed, FixedSizeInputStreamHandlerTest,
                         ::testing::Values(false, true));

TEST_P(FixedSizeInputStreamHandlerTest, DropsPackets) {
  // Sink consumes roughly 10x slower than source produces output which is
  // simulated with a couple of conditional critical sections.  Both calculators
  // are rate limited in a circular fashion.  The source produces 10 packets,
  // then the TestSlowCalculator consumes one packet, then the source produces
  // the next 10 packets.  One packet is sent initially by the source to start
  // the processing.  The CanProceed conditions for the two calculators are
  // mutally exclusive to avoid race conditions between them.  Queue size is
  // regulated by FixedSizeInputStreamHandler.
  CalculatorGraphConfig graph_config =
      ParseTextProtoOrDie<CalculatorGraphConfig>(
          R"(node {
               calculator: "TestSourceCalculator"
               output_stream: "input_packets"
             }
             node {
               calculator: "TestSlowCalculator"
               input_stream: "input_packets"
               output_stream: "output_packets"
               input_stream_handler {
                 input_stream_handler: "FixedSizeInputStreamHandler"
               }
             })");
  SetFixedMinSize(graph_config.mutable_node(1), GetParam());
  std::vector<Packet> output_packets;
  tool::AddVectorSink("output_packets", &graph_config, &output_packets);
  CalculatorGraph graph;
  MP_ASSERT_OK(graph.Initialize(graph_config));
  MP_ASSERT_OK(graph.Run());

  // The TestSlowCalculator consumes one packet after every tenth packet
  // is sent.  All other packets are dropped by the FixedSizeInputStreamHandler.
  // The TestSourceCalculator sends 10 more packets after each packet is
  // consumed.  In this way, the TestSlowCalculator consumes and outputs only
  // every tenth packet.
  EXPECT_EQ(output_packets.size(), 11);
  std::vector<int64> expected_ts = {0, 9, 19, 29, 39, 49, 59, 69, 79, 89, 99};
  EXPECT_THAT(TimestampValues(output_packets),
              testing::ContainerEq(expected_ts));
}

// A regression test for b/74820963. This test should not time out.
TEST_P(FixedSizeInputStreamHandlerTest, DropsPacketsInFullStream) {
  // CountingSourceCalculator outputs 10 packets at a time. Since max_queue_size
  // is 10, CountingSourceCalculator will fill up the "input_packets" input
  // stream of PassThroughCalculator and cause itself to be throttled. When
  // FixedSizeInputStreamHandler discards the queued packets, it should report
  // the "queue became non-full" event and unthrottle CountingSourceCalculator.
  // If FixedSizeInputStreamHandler fails to report the event,
  // CountingSourceCalculator will stay throttled and the test will time out.
  CalculatorGraphConfig graph_config =
      ParseTextProtoOrDie<CalculatorGraphConfig>(
          R"(max_queue_size: 10
             node {
               calculator: "CountingSourceCalculator"
               input_side_packet: "MAX_COUNT:max_count"
               input_side_packet: "BATCH_SIZE:batch_size"
               output_stream: "input_packets"
             }
             node {
               calculator: "PassThroughCalculator"
               input_stream: "input_packets"
               output_stream: "output_packets"
               input_stream_handler {
                 input_stream_handler: "FixedSizeInputStreamHandler"
               }
             })");
  SetFixedMinSize(graph_config.mutable_node(1), GetParam());
  std::vector<Packet> output_packets;
  tool::AddVectorSink("output_packets", &graph_config, &output_packets);
  CalculatorGraph graph;
  MP_ASSERT_OK(
      graph.Initialize(graph_config, {{"max_count", MakePacket<int>(10)},
                                      {"batch_size", MakePacket<int>(10)}}));
  MP_ASSERT_OK(graph.Run());
}

// Tests FixedSizeInputStreamHandler with several input streams running
// asynchronously in parallel.
TEST_P(FixedSizeInputStreamHandlerTest, ParallelWriteAndRead) {
  CalculatorGraphConfig graph_config =
      ParseTextProtoOrDie<CalculatorGraphConfig>(
          R"(
            input_stream: "in_0"
            input_stream: "in_1"
            input_stream: "in_2"
            node {
              calculator: "PassThroughCalculator"
              input_stream: "in_0"
              input_stream: "in_1"
              input_stream: "in_2"
              output_stream: "out_0"
              output_stream: "out_1"
              output_stream: "out_2"
              input_stream_handler {
                input_stream_handler: "FixedSizeInputStreamHandler"
                options {
                  [mediapipe.FixedSizeInputStreamHandlerOptions.ext] {
                    trigger_queue_size: 3
                    target_queue_size: 1
                  }
                }
              }
            })");
  SetFixedMinSize(graph_config.mutable_node(0), GetParam());
  std::vector<Packet> output_packets[3];
  for (int i = 0; i < 3; ++i) {
    tool::AddVectorSink(absl::StrCat("out_", i), &graph_config,
                        &output_packets[i]);
  }
  CalculatorGraph graph;
  MP_ASSERT_OK(graph.Initialize(graph_config, {}));
  MP_ASSERT_OK(graph.StartRun({}));

  {
    ::mediapipe::ThreadPool pool(3);
    pool.StartWorkers();

    // Start 3 writers.
    for (int w = 0; w < 3; ++w) {
      pool.Schedule([&, w]() {
        std::string stream_name = absl::StrCat("in_", w);
        for (int i = 0; i < 50; ++i) {
          Packet p = MakePacket<int>(i).At(Timestamp(i));
          MP_EXPECT_OK(graph.AddPacketToInputStream(stream_name, p));
          absl::SleepFor(absl::Microseconds(100));
        }
      });
    }
  }

  MP_ASSERT_OK(graph.CloseAllInputStreams());
  MP_ASSERT_OK(graph.WaitUntilDone());
  for (int i = 0; i < 3; ++i) {
    EXPECT_EQ(output_packets[i].size(), output_packets[0].size());
    for (int j = 0; j < output_packets[i].size(); j++) {
      EXPECT_EQ(output_packets[i][j].Get<int>(),
                output_packets[0][j].Get<int>());
    }
  }
}

// Tests dropping of packets that arrive later.
// A: 1 2 3[4 5 6]
// B:     3 4[5 6 7]
// C:
// This should cut at 5. If we then send 4 on C, it should be dropped, just as
// it would have been if it arrived earlier.
TEST_P(FixedSizeInputStreamHandlerTest, LateArrivalDrop) {
  CalculatorGraphConfig graph_config =
      ParseTextProtoOrDie<CalculatorGraphConfig>(
          R"(
            input_stream: "in_0"
            input_stream: "in_1"
            input_stream: "in_2"
            node {
              calculator: "PassThroughCalculator"
              input_stream: "in_0"
              input_stream: "in_1"
              input_stream: "in_2"
              output_stream: "out_0"
              output_stream: "out_1"
              output_stream: "out_2"
              input_stream_handler {
                input_stream_handler: "FixedSizeInputStreamHandler"
                options {
                  [mediapipe.FixedSizeInputStreamHandlerOptions.ext] {
                    trigger_queue_size: 4
                    target_queue_size: 3
                  }
                }
              }
            })");
  SetFixedMinSize(graph_config.mutable_node(0), GetParam());
  std::vector<Packet> output_packets[3];
  std::string in_streams[3];
  for (int i = 0; i < 3; ++i) {
    in_streams[i] = absl::StrCat("in_", i);
    tool::AddVectorSink(absl::StrCat("out_", i), &graph_config,
                        &output_packets[i]);
  }
  CalculatorGraph graph;
  MP_ASSERT_OK(graph.Initialize(graph_config, {}));
  MP_ASSERT_OK(graph.StartRun({}));

  for (int i = 1; i <= 6; i++) {
    MP_EXPECT_OK(graph.AddPacketToInputStream(
        in_streams[0], MakePacket<int>(i).At(Timestamp(i))));
  }
  for (int i = 3; i <= 7; i++) {
    MP_EXPECT_OK(graph.AddPacketToInputStream(
        in_streams[1], MakePacket<int>(i).At(Timestamp(i))));
    MP_ASSERT_OK(graph.WaitUntilIdle());
  }
  // At this point everything before ts 5 should be dropped.
  for (int i = 4; i <= 7; i++) {
    MP_EXPECT_OK(graph.AddPacketToInputStream(
        in_streams[2], MakePacket<int>(i).At(Timestamp(i))));
    MP_ASSERT_OK(graph.WaitUntilIdle());
  }

  MP_ASSERT_OK(graph.CloseAllInputStreams());
  MP_ASSERT_OK(graph.WaitUntilDone());

  if (GetParam()) {
    EXPECT_THAT(TimestampValues(output_packets[0]),
                testing::ContainerEq(std::vector<int64>{1, 2, 3, 4, 5, 6}));
    EXPECT_THAT(TimestampValues(output_packets[1]),
                testing::ContainerEq(std::vector<int64>{3, 4, 5, 6, 7}));
    EXPECT_THAT(TimestampValues(output_packets[2]),
                testing::ContainerEq(std::vector<int64>{4, 5, 6, 7}));
  } else {
    EXPECT_THAT(TimestampValues(output_packets[0]),
                testing::ContainerEq(std::vector<int64>{5, 6}));
    EXPECT_THAT(TimestampValues(output_packets[1]),
                testing::ContainerEq(std::vector<int64>{5, 6, 7}));
    EXPECT_THAT(TimestampValues(output_packets[2]),
                testing::ContainerEq(std::vector<int64>{5, 6, 7}));
  }
}

}  // namespace
}  // namespace mediapipe
