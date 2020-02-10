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
#include "absl/synchronization/mutex.h"
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

namespace mediapipe {

namespace {

class CalculatorGraphEventLoopTest : public testing::Test {
 public:
  void AddThreadSafeVectorSink(const Packet& packet) {
    absl::WriterMutexLock lock(&output_packets_mutex_);
    output_packets_.push_back(packet);
  }

 protected:
  std::vector<Packet> output_packets_ ABSL_GUARDED_BY(output_packets_mutex_);
  absl::Mutex output_packets_mutex_;
};

// Allows blocking of the Process() call by locking the blocking_mutex passed to
// the input side packet. Used to force input stream queues to build up for
// testing.
class BlockingPassThroughCalculator : public CalculatorBase {
 public:
  static ::mediapipe::Status GetContract(CalculatorContract* cc) {
    cc->Inputs().Index(0).SetAny();
    cc->Outputs().Index(0).SetSameAs(&cc->Inputs().Index(0));
    cc->InputSidePackets().Index(0).Set<std::unique_ptr<absl::Mutex>>();

    return ::mediapipe::OkStatus();
  }

  ::mediapipe::Status Open(CalculatorContext* cc) final {
    mutex_ = GetFromUniquePtr<absl::Mutex>(cc->InputSidePackets().Index(0));
    return ::mediapipe::OkStatus();
  }

  ::mediapipe::Status Process(CalculatorContext* cc) final {
    mutex_->Lock();
    cc->Outputs().Index(0).AddPacket(
        cc->Inputs().Index(0).Value().At(cc->InputTimestamp()));
    mutex_->Unlock();
    return ::mediapipe::OkStatus();
  }

 private:
  absl::Mutex* mutex_;
};

REGISTER_CALCULATOR(BlockingPassThroughCalculator);

struct SimpleHeader {
  int width;
  int height;
};

class UsingHeaderCalculator : public CalculatorBase {
 public:
  static ::mediapipe::Status GetContract(CalculatorContract* cc) {
    cc->Inputs().Index(0).SetAny();
    cc->Outputs().Index(0).SetSameAs(&cc->Inputs().Index(0));
    return ::mediapipe::OkStatus();
  }

  ::mediapipe::Status Open(CalculatorContext* cc) final {
    if (cc->Inputs().Index(0).Header().IsEmpty()) {
      return ::mediapipe::UnknownError("No stream header present.");
    }

    const SimpleHeader& header =
        cc->Inputs().Index(0).Header().Get<SimpleHeader>();
    std::unique_ptr<SimpleHeader> output_header(new SimpleHeader);
    output_header->width = header.width;
    output_header->height = header.height;

    cc->Outputs().Index(0).SetHeader(Adopt(output_header.release()));
    return ::mediapipe::OkStatus();
  }

  ::mediapipe::Status Process(CalculatorContext* cc) final {
    cc->Outputs().Index(0).AddPacket(
        cc->Inputs().Index(0).Value().At(cc->InputTimestamp()));
    return ::mediapipe::OkStatus();
  }
};
REGISTER_CALCULATOR(UsingHeaderCalculator);

TEST_F(CalculatorGraphEventLoopTest, WellProvisionedEventLoop) {
  CalculatorGraphConfig graph_config;
  ASSERT_TRUE(proto_ns::TextFormat::ParseFromString(
      R"(
          node {
            calculator: "PassThroughCalculator"
            input_stream: "input_numbers"
            output_stream: "output_numbers"
          }
          node {
            calculator: "CallbackCalculator"
            input_stream: "output_numbers"
            input_side_packet: "CALLBACK:callback"
          }
          input_stream: "input_numbers"
      )",
      &graph_config));

  // Start MediaPipe graph.
  CalculatorGraph graph(graph_config);
  MP_ASSERT_OK(graph.StartRun(
      {{"callback", MakePacket<std::function<void(const Packet&)>>(std::bind(
                        &CalculatorGraphEventLoopTest::AddThreadSafeVectorSink,
                        this, std::placeholders::_1))}}));

  // Insert 100 packets at the rate the calculator can keep up with.
  for (int i = 0; i < 100; ++i) {
    MP_ASSERT_OK(graph.AddPacketToInputStream(
        "input_numbers", Adopt(new int(i)).At(Timestamp(i))));
    // Wait for all packets to be received by the sink.
    while (true) {
      {
        absl::ReaderMutexLock lock(&output_packets_mutex_);
        if (output_packets_.size() > i) {
          break;
        }
      }
      absl::SleepFor(absl::Microseconds(1));
    }
  }
  // Check partial results.
  {
    absl::ReaderMutexLock lock(&output_packets_mutex_);
    ASSERT_EQ(100, output_packets_.size());
    for (int i = 0; i < 100; ++i) {
      EXPECT_EQ(i, output_packets_[i].Get<int>());
    }
  }

  // Insert 100 more packets at rate the graph can't keep up.
  for (int i = 100; i < 200; ++i) {
    MP_ASSERT_OK(graph.AddPacketToInputStream(
        "input_numbers", Adopt(new int(i)).At(Timestamp(i))));
  }
  // Don't wait but just close the input stream.
  MP_ASSERT_OK(graph.CloseInputStream("input_numbers"));
  // Wait properly via the API until the graph is done.
  MP_ASSERT_OK(graph.WaitUntilDone());
  // Check final results.
  {
    absl::ReaderMutexLock lock(&output_packets_mutex_);
    ASSERT_EQ(200, output_packets_.size());
    for (int i = 0; i < 200; ++i) {
      EXPECT_EQ(i, output_packets_[i].Get<int>());
    }
  }
}

// Pass-Through calculator that fails upon receiving the 10th packet.
class FailingPassThroughCalculator : public CalculatorBase {
 public:
  static ::mediapipe::Status GetContract(CalculatorContract* cc) {
    cc->Inputs().Index(0).SetAny();
    cc->Outputs().Index(0).SetSameAs(&cc->Inputs().Index(0));
    return ::mediapipe::OkStatus();
  }

  ::mediapipe::Status Process(CalculatorContext* cc) final {
    Timestamp timestamp = cc->InputTimestamp();
    if (timestamp.Value() == 9) {
      return ::mediapipe::UnknownError(
          "Meant to fail (magicstringincludedhere).");
    }
    cc->Outputs().Index(0).AddPacket(
        cc->Inputs().Index(0).Value().At(timestamp));
    return ::mediapipe::OkStatus();
  }
};
REGISTER_CALCULATOR(FailingPassThroughCalculator);

TEST_F(CalculatorGraphEventLoopTest, FailingEventLoop) {
  CalculatorGraphConfig graph_config;
  ASSERT_TRUE(proto_ns::TextFormat::ParseFromString(
      R"(
          node {
            calculator: "FailingPassThroughCalculator"
            input_stream: "input_numbers"
            output_stream: "output_numbers"
          }
          node {
            calculator: "CallbackCalculator"
            input_stream: "output_numbers"
            input_side_packet: "CALLBACK:callback"
          }
          input_stream: "input_numbers")",
      &graph_config));

  // Start MediaPipe graph.
  CalculatorGraph graph(graph_config);
  MP_ASSERT_OK(graph.StartRun(
      {{"callback", MakePacket<std::function<void(const Packet&)>>(std::bind(
                        &CalculatorGraphEventLoopTest::AddThreadSafeVectorSink,
                        this, std::placeholders::_1))}}));

  // Insert packets.
  ::mediapipe::Status status;
  for (int i = 0; true; ++i) {
    status = graph.AddPacketToInputStream("input_numbers",
                                          Adopt(new int(i)).At(Timestamp(i)));
    if (!status.ok()) {
      ASSERT_TRUE(graph.HasError());  // Graph failed.
      ASSERT_THAT(
          status.message(),
          testing::HasSubstr("Meant to fail (magicstringincludedhere)."));
      break;
    }
  }
  MP_ASSERT_OK(graph.CloseInputStream("input_numbers"));
  status = graph.WaitUntilDone();
  ASSERT_THAT(status.message(),
              testing::HasSubstr("Meant to fail (magicstringincludedhere)."));
}

// Test the step by step mode.
TEST_F(CalculatorGraphEventLoopTest, StepByStepSchedulerLoop) {
  CalculatorGraphConfig graph_config;
  ASSERT_TRUE(proto_ns::TextFormat::ParseFromString(
      R"(
          node {
            calculator: "PassThroughCalculator"
            input_stream: "input_numbers"
            output_stream: "output_numbers"
          }
          node {
            calculator: "CallbackCalculator"
            input_stream: "output_numbers"
            input_side_packet: "CALLBACK:callback"
          }
          input_stream: "input_numbers"
      )",
      &graph_config));

  // Start MediaPipe graph.
  CalculatorGraph graph(graph_config);
  MP_ASSERT_OK(graph.StartRun(
      {{"callback", MakePacket<std::function<void(const Packet&)>>(std::bind(
                        &CalculatorGraphEventLoopTest::AddThreadSafeVectorSink,
                        this, std::placeholders::_1))}}));

  // Add packet one at a time, we should be able to syncrhonize the output for
  // each addition in the step by step mode.
  for (int i = 0; i < 100; ++i) {
    MP_ASSERT_OK(graph.AddPacketToInputStream(
        "input_numbers", Adopt(new int(i)).At(Timestamp(i))));
    MP_ASSERT_OK(graph.WaitUntilIdle());
    absl::ReaderMutexLock lock(&output_packets_mutex_);
    ASSERT_EQ(i + 1, output_packets_.size());
  }
  // Don't wait but just close the input stream.
  MP_ASSERT_OK(graph.CloseInputStream("input_numbers"));
  // Wait properly via the API until the graph is done.
  MP_ASSERT_OK(graph.WaitUntilDone());
}

// Test setting the stream header.
TEST_F(CalculatorGraphEventLoopTest, SetStreamHeader) {
  CalculatorGraphConfig graph_config;
  ASSERT_TRUE(proto_ns::TextFormat::ParseFromString(
      R"(
          node {
            calculator: "UsingHeaderCalculator"
            input_stream: "input_numbers"
            output_stream: "output_numbers"
          }
          node {
            calculator: "CallbackCalculator"
            input_stream: "output_numbers"
            input_side_packet: "CALLBACK:callback"
          }
          input_stream: "input_numbers"
      )",
      &graph_config));

  CalculatorGraph graph(graph_config);
  MP_ASSERT_OK(graph.StartRun(
      {{"callback", MakePacket<std::function<void(const Packet&)>>(std::bind(
                        &CalculatorGraphEventLoopTest::AddThreadSafeVectorSink,
                        this, std::placeholders::_1))}}));

  ::mediapipe::Status status = graph.WaitUntilIdle();
  // Expect to fail if header not set.
  ASSERT_FALSE(status.ok());
  EXPECT_EQ(status.code(), ::mediapipe::StatusCode::kUnknown);
  EXPECT_THAT(status.message(),
              testing::HasSubstr("No stream header present."));

  CalculatorGraph graph2(graph_config);
  std::unique_ptr<SimpleHeader> header(new SimpleHeader);
  header->width = 320;
  header->height = 240;
  // With stream header set, the StartRun should succeed.
  MP_ASSERT_OK(graph2.StartRun(
      {{"callback", MakePacket<std::function<void(const Packet&)>>(std::bind(
                        &CalculatorGraphEventLoopTest::AddThreadSafeVectorSink,
                        this, std::placeholders::_1))}},
      {{"input_numbers", Adopt(header.release())}}));
  // Don't wait but just close the input stream.
  MP_ASSERT_OK(graph2.CloseInputStream("input_numbers"));
  // Wait properly via the API until the graph is done.
  MP_ASSERT_OK(graph2.WaitUntilDone());
}

// Test ADD_IF_NOT_FULL mode for graph input streams (by creating more packets
// than the queue will support). At least some of these attempts should fail.
TEST_F(CalculatorGraphEventLoopTest, TryToAddPacketToInputStream) {
  CalculatorGraphConfig graph_config;
  ASSERT_TRUE(proto_ns::TextFormat::ParseFromString(
      R"(
          node {
            calculator: "BlockingPassThroughCalculator"
            input_stream: "input_numbers"
            output_stream: "output_numbers"
            input_side_packet: "blocking_mutex"
          }
          node {
            calculator: "CallbackCalculator"
            input_stream: "output_numbers"
            input_side_packet: "CALLBACK:callback"
          }
          input_stream: "input_numbers"
          num_threads: 2
          max_queue_size: 1
      )",
      &graph_config));

  absl::Mutex* mutex = new absl::Mutex();
  Packet mutex_side_packet = AdoptAsUniquePtr(mutex);

  CalculatorGraph graph(graph_config);
  graph.SetGraphInputStreamAddMode(
      CalculatorGraph::GraphInputStreamAddMode::ADD_IF_NOT_FULL);

  // Start MediaPipe graph.
  MP_ASSERT_OK(graph.StartRun(
      {{"callback", MakePacket<std::function<void(const Packet&)>>(std::bind(
                        &CalculatorGraphEventLoopTest::AddThreadSafeVectorSink,
                        this, std::placeholders::_1))},
       {"blocking_mutex", mutex_side_packet}}));

  constexpr int kNumInputPackets = 2;
  constexpr int kMaxQueueSize = 1;

  // Lock the mutex so that the BlockingPassThroughCalculator cannot read any of
  // these packets.
  mutex->Lock();
  int fail_count = 0;
  // Expect at least kNumInputPackets - kMaxQueueSize - 1 attempts to add
  // packets to fail since the queue builds up. The -1 is because our throttling
  // mechanism could be off by 1 at most due to the order of acquisition of
  // locks.
  for (int i = 0; i < kNumInputPackets; ++i) {
    ::mediapipe::Status status = graph.AddPacketToInputStream(
        "input_numbers", Adopt(new int(i)).At(Timestamp(i)));
    if (!status.ok()) {
      ++fail_count;
    }
  }
  mutex->Unlock();

  EXPECT_GE(fail_count, kNumInputPackets - kMaxQueueSize - 1);
  // Don't wait but just close the input stream.
  MP_ASSERT_OK(graph.CloseInputStream("input_numbers"));
  // Wait properly via the API until the graph is done.
  MP_ASSERT_OK(graph.WaitUntilDone());
}

// Verify that "max_queue_size: -1" disables throttling of graph-input-streams.
TEST_F(CalculatorGraphEventLoopTest, ThrottlingDisabled) {
  CalculatorGraphConfig graph_config;
  ASSERT_TRUE(proto_ns::TextFormat::ParseFromString(
      R"(
          node {
            calculator: "BlockingPassThroughCalculator"
            input_stream: "input_numbers"
            output_stream: "output_numbers"
            input_side_packet: "blocking_mutex"
          }
          input_stream: "input_numbers"
          max_queue_size: -1
      )",
      &graph_config));

  absl::Mutex* mutex = new absl::Mutex();
  Packet mutex_side_packet = AdoptAsUniquePtr(mutex);

  CalculatorGraph graph(graph_config);
  graph.SetGraphInputStreamAddMode(
      CalculatorGraph::GraphInputStreamAddMode::ADD_IF_NOT_FULL);

  // Start MediaPipe graph.
  MP_ASSERT_OK(graph.StartRun({{"blocking_mutex", mutex_side_packet}}));

  // Lock the mutex so that the BlockingPassThroughCalculator cannot read any
  // of these packets.
  mutex->Lock();
  for (int i = 0; i < 10; ++i) {
    MP_EXPECT_OK(graph.AddPacketToInputStream(
        "input_numbers", Adopt(new int(i)).At(Timestamp(i))));
  }
  mutex->Unlock();
  MP_EXPECT_OK(graph.CloseInputStream("input_numbers"));
  MP_EXPECT_OK(graph.WaitUntilDone());
}

// Verify that the graph input stream throttling code still works if we run the
// graph twice.
TEST_F(CalculatorGraphEventLoopTest, ThrottleGraphInputStreamTwice) {
  CalculatorGraphConfig graph_config;
  ASSERT_TRUE(proto_ns::TextFormat::ParseFromString(
      R"(
          node {
            calculator: "BlockingPassThroughCalculator"
            input_stream: "input_numbers"
            output_stream: "output_numbers"
            input_side_packet: "blocking_mutex"
          }
          input_stream: "input_numbers"
          max_queue_size: 1
      )",
      &graph_config));

  absl::Mutex* mutex = new absl::Mutex();
  Packet mutex_side_packet = AdoptAsUniquePtr(mutex);

  CalculatorGraph graph(graph_config);
  graph.SetGraphInputStreamAddMode(
      CalculatorGraph::GraphInputStreamAddMode::ADD_IF_NOT_FULL);

  // Run the graph twice.
  for (int i = 0; i < 2; ++i) {
    // Start MediaPipe graph.
    MP_ASSERT_OK(graph.StartRun({{"blocking_mutex", mutex_side_packet}}));

    // Lock the mutex so that the BlockingPassThroughCalculator cannot read any
    // of these packets.
    mutex->Lock();
    ::mediapipe::Status status = ::mediapipe::OkStatus();
    for (int i = 0; i < 10; ++i) {
      status = graph.AddPacketToInputStream("input_numbers",
                                            Adopt(new int(i)).At(Timestamp(i)));
      if (!status.ok()) {
        break;
      }
    }
    mutex->Unlock();
    ASSERT_FALSE(status.ok());
    EXPECT_EQ(status.code(), ::mediapipe::StatusCode::kUnavailable);
    EXPECT_THAT(status.message(), testing::HasSubstr("Graph is throttled."));
    MP_ASSERT_OK(graph.CloseInputStream("input_numbers"));
    MP_ASSERT_OK(graph.WaitUntilDone());
  }
}

// Test WAIT_TILL_NOT_FULL mode (default mode) for graph input streams (by
// creating more packets than the queue will support). All packets sent to the
// graph should be processed.
TEST_F(CalculatorGraphEventLoopTest, WaitToAddPacketToInputStream) {
  CalculatorGraphConfig graph_config;
  ASSERT_TRUE(proto_ns::TextFormat::ParseFromString(
      R"(
          node {
            calculator: "PassThroughCalculator"
            input_stream: "input_numbers"
            output_stream: "output_numbers"
          }
          node {
            calculator: "CallbackCalculator"
            input_stream: "output_numbers"
            input_side_packet: "CALLBACK:callback"
          }
          input_stream: "input_numbers"
          num_threads: 2
          max_queue_size: 10
      )",
      &graph_config));

  // Start MediaPipe graph.
  CalculatorGraph graph(graph_config);
  MP_ASSERT_OK(graph.StartRun(
      {{"callback", MakePacket<std::function<void(const Packet&)>>(std::bind(
                        &CalculatorGraphEventLoopTest::AddThreadSafeVectorSink,
                        this, std::placeholders::_1))}}));

  constexpr int kNumInputPackets = 20;
  // All of these packets should be accepted by the graph.
  int fail_count = 0;
  for (int i = 0; i < kNumInputPackets; ++i) {
    ::mediapipe::Status status = graph.AddPacketToInputStream(
        "input_numbers", Adopt(new int(i)).At(Timestamp(i)));
    if (!status.ok()) {
      ++fail_count;
    }
  }

  EXPECT_EQ(0, fail_count);

  // Don't wait but just close the input stream.
  MP_ASSERT_OK(graph.CloseInputStream("input_numbers"));
  // Wait properly via the API until the graph is done.
  MP_ASSERT_OK(graph.WaitUntilDone());

  absl::ReaderMutexLock lock(&output_packets_mutex_);
  ASSERT_EQ(kNumInputPackets, output_packets_.size());
}

// Captures log messages during testing.
class TextMessageLogSink : public LogSink {
 public:
  std::vector<std::string> messages;
  void Send(const LogEntry& entry) {
    messages.push_back(std::string(entry.text_message()));
  }
};

// Verifies that CalculatorGraph::UnthrottleSources does not run repeatedly
// in a "busy-loop" while the graph is throttled due to a graph-output stream.
TEST_F(CalculatorGraphEventLoopTest, UnthrottleSources) {
  CalculatorGraphConfig graph_config;
  ASSERT_TRUE(proto_ns::TextFormat::ParseFromString(
      R"(
          node {
            calculator: "PassThroughCalculator"
            input_stream: "input_numbers"
            output_stream: "output_numbers"
          }
          input_stream: "input_numbers"
          output_stream: "output_numbers"
          num_threads: 2
          max_queue_size: 5
      )",
      &graph_config));
  constexpr int kQueueSize = 5;

  // Initialize and start the mediapipe graph.
  CalculatorGraph graph;
  MP_ASSERT_OK(graph.Initialize(graph_config));
  graph.SetGraphInputStreamAddMode(
      CalculatorGraph::GraphInputStreamAddMode::ADD_IF_NOT_FULL);
  auto poller_status = graph.AddOutputStreamPoller("output_numbers");
  MP_ASSERT_OK(poller_status.status());
  mediapipe::OutputStreamPoller& poller = poller_status.ValueOrDie();
  poller.SetMaxQueueSize(kQueueSize);
  MP_ASSERT_OK(graph.StartRun({}));

  // Lambda that adds a packet to the calculator graph.
  auto add_packet = [&graph](std::string s, int i) {
    return graph.AddPacketToInputStream(s, MakePacket<int>(i).At(Timestamp(i)));
  };

  // Start capturing VLOG messages from the mediapipe::Scheduler.
  TextMessageLogSink log_listener;
  mediapipe::AddLogSink(&log_listener);
  SetVLOGLevel("scheduler", 3);

  // Add just enough packets to fill the output stream queue.
  std::vector<Packet> out_packets;
  for (int i = 0; i < kQueueSize; ++i) {
    MP_EXPECT_OK(add_packet("input_numbers", i));
    MP_EXPECT_OK(graph.WaitUntilIdle());
  }

  // The graph is throttled due to the full output stream.
  EXPECT_FALSE(add_packet("input_numbers", kQueueSize).ok());

  // CalculatorGraph::UnthrottleSources should be called just one time.
  absl::SleepFor(absl::Milliseconds(100));

  // Read all packets from the output stream queue and close the graph.
  for (int i = 0; i < kQueueSize; ++i) {
    Packet packet;
    EXPECT_TRUE(poller.Next(&packet));
    out_packets.push_back(packet);
  }
  MP_EXPECT_OK(graph.CloseAllInputStreams());
  MP_EXPECT_OK(graph.WaitUntilDone());
  EXPECT_EQ(kQueueSize, out_packets.size());

  // Stop capturing VLOG messages.
  SetVLOGLevel("scheduler", 0);
  mediapipe::RemoveLogSink(&log_listener);

  // Count and validate the calls to UnthrottleSources.
  int loop_count = 0;
  for (auto& message : log_listener.messages) {
    loop_count += (message == "HandleIdle: unthrottling") ? 1 : 0;
  }
  EXPECT_GE(loop_count, 1);
  EXPECT_LE(loop_count, 2);
}

}  // namespace
}  // namespace mediapipe
