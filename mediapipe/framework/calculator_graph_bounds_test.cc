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

#include <vector>

#include "absl/strings/str_replace.h"
#include "mediapipe/framework/calculator_context.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/port/canonical_errors.h"
#include "mediapipe/framework/port/gmock.h"
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/framework/port/logging.h"
#include "mediapipe/framework/port/parse_text_proto.h"
#include "mediapipe/framework/port/status.h"
#include "mediapipe/framework/port/status_matchers.h"
#include "mediapipe/framework/thread_pool_executor.h"
#include "mediapipe/framework/timestamp.h"
#include "mediapipe/util/packet_test_util.h"

namespace mediapipe {
namespace {

constexpr int kIntTestValue = 33;

typedef std::function<absl::Status(CalculatorContext* cc)>
    CalculatorContextFunction;

// Returns the contents of a set of Packets.
// The contents must be copyable.
template <typename T>
std::vector<T> GetContents(const std::vector<Packet>& packets) {
  std::vector<T> result;
  for (Packet p : packets) {
    result.push_back(p.Get<T>());
  }
  return result;
}

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

// A mediapipe::Executor that signals the start and finish of each task.
class CountingExecutor : public Executor {
 public:
  CountingExecutor(int num_threads, std::function<void()> start_callback,
                   std::function<void()> finish_callback)
      : thread_pool_(num_threads),
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

// A Calculator that adds the integer values in the packets in all the input
// streams and outputs the sum to the output stream.
class IntAdderCalculator : public CalculatorBase {
 public:
  static absl::Status GetContract(CalculatorContract* cc) {
    for (int i = 0; i < cc->Inputs().NumEntries(); ++i) {
      cc->Inputs().Index(i).Set<int>();
    }
    cc->Outputs().Index(0).Set<int>();
    cc->SetTimestampOffset(TimestampDiff(0));
    return absl::OkStatus();
  }

  absl::Status Open(CalculatorContext* cc) final { return absl::OkStatus(); }

  absl::Status Process(CalculatorContext* cc) final {
    int sum = 0;
    for (int i = 0; i < cc->Inputs().NumEntries(); ++i) {
      sum += cc->Inputs().Index(i).Get<int>();
    }
    cc->Outputs().Index(0).Add(new int(sum), cc->InputTimestamp());
    return absl::OkStatus();
  }
};
REGISTER_CALCULATOR(IntAdderCalculator);

template <typename InputType>
class TypedSinkCalculator : public CalculatorBase {
 public:
  static absl::Status GetContract(CalculatorContract* cc) {
    cc->Inputs().Index(0).Set<InputType>();
    return absl::OkStatus();
  }

  absl::Status Process(CalculatorContext* cc) override {
    return absl::OkStatus();
  }
};
typedef TypedSinkCalculator<std::string> StringSinkCalculator;
typedef TypedSinkCalculator<int> IntSinkCalculator;
REGISTER_CALCULATOR(StringSinkCalculator);
REGISTER_CALCULATOR(IntSinkCalculator);

// A Calculator that passes an input packet through if it contains an even
// integer.
class EvenIntFilterCalculator : public CalculatorBase {
 public:
  static absl::Status GetContract(CalculatorContract* cc) {
    cc->Inputs().Index(0).Set<int>();
    cc->Outputs().Index(0).Set<int>();
    return absl::OkStatus();
  }

  absl::Status Process(CalculatorContext* cc) final {
    int value = cc->Inputs().Index(0).Get<int>();
    if (value % 2 == 0) {
      cc->Outputs().Index(0).AddPacket(cc->Inputs().Index(0).Value());
    } else {
      cc->Outputs().Index(0).SetNextTimestampBound(
          cc->InputTimestamp().NextAllowedInStream());
    }
    return absl::OkStatus();
  }
};
REGISTER_CALCULATOR(EvenIntFilterCalculator);

// A Calculator that passes packets through or not, depending on a second
// input. The first input stream's packets are only propagated if the second
// input stream carries the value true.
class ValveCalculator : public CalculatorBase {
 public:
  static absl::Status GetContract(CalculatorContract* cc) {
    cc->Inputs().Index(0).SetAny();
    cc->Inputs().Index(1).Set<bool>();
    cc->Outputs().Index(0).SetSameAs(&cc->Inputs().Index(0));
    return absl::OkStatus();
  }

  absl::Status Open(CalculatorContext* cc) final {
    cc->Outputs().Index(0).SetHeader(cc->Inputs().Index(0).Header());
    return absl::OkStatus();
  }

  absl::Status Process(CalculatorContext* cc) final {
    if (cc->Inputs().Index(1).Get<bool>()) {
      cc->GetCounter("PassThrough")->Increment();
      cc->Outputs().Index(0).AddPacket(cc->Inputs().Index(0).Value());
    } else {
      cc->GetCounter("Block")->Increment();
      // The next timestamp bound is the minimum timestamp that the next packet
      // can have, so, if we want to inform the downstream that no packet at
      // InputTimestamp() is coming, we need to set it to the next value.
      // We could also just call SetOffset(TimestampDiff(0)) in Open, and then
      // we would not have to call this manually.
      cc->Outputs().Index(0).SetNextTimestampBound(
          cc->InputTimestamp().NextAllowedInStream());
    }
    return absl::OkStatus();
  }
};
REGISTER_CALCULATOR(ValveCalculator);

// A Calculator that simply passes its input Packets and header through,
// but shifts the timestamp.
class TimeShiftCalculator : public CalculatorBase {
 public:
  static absl::Status GetContract(CalculatorContract* cc) {
    cc->Inputs().Index(0).SetAny();
    cc->Outputs().Index(0).SetSameAs(&cc->Inputs().Index(0));
    cc->InputSidePackets().Index(0).Set<TimestampDiff>();
    return absl::OkStatus();
  }

  absl::Status Open(CalculatorContext* cc) final {
    // Input: arbitrary Packets.
    // Output: copy of the input.
    cc->Outputs().Index(0).SetHeader(cc->Inputs().Index(0).Header());
    shift_ = cc->InputSidePackets().Index(0).Get<TimestampDiff>();
    cc->SetOffset(shift_);
    return absl::OkStatus();
  }

  absl::Status Process(CalculatorContext* cc) final {
    cc->GetCounter("PassThrough")->Increment();
    cc->Outputs().Index(0).AddPacket(
        cc->Inputs().Index(0).Value().At(cc->InputTimestamp() + shift_));
    return absl::OkStatus();
  }

 private:
  TimestampDiff shift_;
};
REGISTER_CALCULATOR(TimeShiftCalculator);

// A source calculator that alternates between outputting an integer (0, 1, 2,
// ..., 100) and setting the next timestamp bound. The timestamps of the output
// packets and next timestamp bounds are 0, 10, 20, 30, ...
//
//   T=0    Output 0
//   T=10   Set timestamp bound
//   T=20   Output 1
//   T=30   Set timestamp bound
//   ...
//   T=2000 Output 100
class OutputAndBoundSourceCalculator : public CalculatorBase {
 public:
  static absl::Status GetContract(CalculatorContract* cc) {
    cc->Outputs().Index(0).Set<int>();
    return absl::OkStatus();
  }

  absl::Status Open(CalculatorContext* cc) override {
    counter_ = 0;
    return absl::OkStatus();
  }

  absl::Status Process(CalculatorContext* cc) override {
    Timestamp timestamp(counter_);
    if (counter_ % 20 == 0) {
      cc->Outputs().Index(0).AddPacket(
          MakePacket<int>(counter_ / 20).At(timestamp));
    } else {
      cc->Outputs().Index(0).SetNextTimestampBound(timestamp);
    }
    if (counter_ == 2000) {
      return tool::StatusStop();
    }
    counter_ += 10;
    return absl::OkStatus();
  }

 private:
  int counter_;
};
REGISTER_CALCULATOR(OutputAndBoundSourceCalculator);

// A calculator that outputs an initial packet of value 0 at time 0 in the
// Open() method, and then delays each input packet by 20 time units in the
// Process() method. The input stream and output stream have the integer type.
class Delay20Calculator : public CalculatorBase {
 public:
  static absl::Status GetContract(CalculatorContract* cc) {
    cc->Inputs().Index(0).Set<int>();
    cc->Outputs().Index(0).Set<int>();
    cc->SetTimestampOffset(TimestampDiff(20));
    return absl::OkStatus();
  }

  absl::Status Open(CalculatorContext* cc) final {
    cc->Outputs().Index(0).AddPacket(MakePacket<int>(0).At(Timestamp(0)));
    return absl::OkStatus();
  }

  absl::Status Process(CalculatorContext* cc) final {
    const Packet& packet = cc->Inputs().Index(0).Value();
    Timestamp timestamp = packet.Timestamp() + 20;
    cc->Outputs().Index(0).AddPacket(packet.At(timestamp));
    return absl::OkStatus();
  }
};
REGISTER_CALCULATOR(Delay20Calculator);

class CustomBoundCalculator : public CalculatorBase {
 public:
  static absl::Status GetContract(CalculatorContract* cc) {
    cc->Inputs().Index(0).Set<int>();
    cc->Outputs().Index(0).Set<int>();
    return absl::OkStatus();
  }

  absl::Status Open(CalculatorContext* cc) final { return absl::OkStatus(); }

  absl::Status Process(CalculatorContext* cc) final {
    cc->Outputs().Index(0).SetNextTimestampBound(cc->InputTimestamp() + 1);
    return absl::OkStatus();
  }
};
REGISTER_CALCULATOR(CustomBoundCalculator);

// Test that SetNextTimestampBound propagates.
TEST(CalculatorGraph, SetNextTimestampBoundPropagation) {
  CalculatorGraph graph;
  CalculatorGraphConfig config =
      mediapipe::ParseTextProtoOrDie<CalculatorGraphConfig>(R"pb(
        input_stream: 'in'
        input_stream: 'gate'
        node {
          calculator: 'ValveCalculator'
          input_stream: 'in'
          input_stream: 'gate'
          output_stream: 'gated'
        }
        node {
          calculator: 'PassThroughCalculator'
          input_stream: 'gated'
          output_stream: 'passed'
        }
        node {
          calculator: 'TimeShiftCalculator'
          input_stream: 'passed'
          output_stream: 'shifted'
          input_side_packet: 'shift'
        }
        node {
          calculator: 'MergeCalculator'
          input_stream: 'in'
          input_stream: 'shifted'
          output_stream: 'merged'
        }
        node {
          name: 'merged_output'
          calculator: 'PassThroughCalculator'
          input_stream: 'merged'
          output_stream: 'out'
        }
      )pb");

  Timestamp timestamp = Timestamp(0);
  auto send_inputs = [&graph, &timestamp](int input, bool pass) {
    ++timestamp;
    MP_EXPECT_OK(graph.AddPacketToInputStream(
        "in", MakePacket<int>(input).At(timestamp)));
    MP_EXPECT_OK(graph.AddPacketToInputStream(
        "gate", MakePacket<bool>(pass).At(timestamp)));
  };

  MP_ASSERT_OK(graph.Initialize(config));
  MP_ASSERT_OK(graph.StartRun({{"shift", MakePacket<TimestampDiff>(0)}}));

  auto pass_counter =
      graph.GetCounterFactory()->GetCounter("ValveCalculator-PassThrough");
  auto block_counter =
      graph.GetCounterFactory()->GetCounter("ValveCalculator-Block");
  auto merged_counter =
      graph.GetCounterFactory()->GetCounter("merged_output-PassThrough");

  send_inputs(1, true);
  send_inputs(2, true);
  send_inputs(3, false);
  send_inputs(4, false);
  MP_ASSERT_OK(graph.WaitUntilIdle());

  // Verify that MergeCalculator was able to run even when the gated branch
  // was blocked.
  EXPECT_EQ(2, pass_counter->Get());
  EXPECT_EQ(2, block_counter->Get());
  EXPECT_EQ(4, merged_counter->Get());

  send_inputs(5, true);
  send_inputs(6, false);
  MP_ASSERT_OK(graph.WaitUntilIdle());

  EXPECT_EQ(3, pass_counter->Get());
  EXPECT_EQ(3, block_counter->Get());
  EXPECT_EQ(6, merged_counter->Get());

  MP_ASSERT_OK(graph.CloseAllInputStreams());
  MP_ASSERT_OK(graph.WaitUntilDone());

  // Now test with time shift
  MP_ASSERT_OK(graph.StartRun({{"shift", MakePacket<TimestampDiff>(-1)}}));

  send_inputs(7, true);
  MP_ASSERT_OK(graph.WaitUntilIdle());

  // The merger should have run only once now, at timestamp 6, with inputs
  // <null, 7>. If we do not respect the offset and unblock the merger for
  // timestamp 7 too, then it will have run twice, with 6: <null,7> and
  // 7: <7, null>.
  EXPECT_EQ(4, pass_counter->Get());
  EXPECT_EQ(3, block_counter->Get());
  EXPECT_EQ(7, merged_counter->Get());

  MP_ASSERT_OK(graph.CloseAllInputStreams());
  MP_ASSERT_OK(graph.WaitUntilDone());

  EXPECT_EQ(4, pass_counter->Get());
  EXPECT_EQ(3, block_counter->Get());
  EXPECT_EQ(8, merged_counter->Get());
}

// Both input streams of the calculator node have the same next timestamp
// bound. One input stream has a packet at that timestamp. The other input
// stream is empty. We should not run the Process() method of the node in this
// case.
TEST(CalculatorGraph, NotAllInputPacketsAtNextTimestampBoundAvailable) {
  //
  // in0_unfiltered  in1_to_be_filtered
  //        |             |
  //        |             V
  //        |  +-----------------------+
  //        |  |EvenIntFilterCalculator|
  //        |  +-----------------------+
  //        |             |
  //        \             /
  //         \           / in1_filtered
  //          \         /
  //           |       |
  //           V       V
  //      +------------------+
  //      |IntAdderCalculator|
  //      +------------------+
  //               |
  //               V
  //              out
  //
  CalculatorGraph graph;
  CalculatorGraphConfig config =
      mediapipe::ParseTextProtoOrDie<CalculatorGraphConfig>(R"pb(
        input_stream: 'in0_unfiltered'
        input_stream: 'in1_to_be_filtered'
        node {
          calculator: 'EvenIntFilterCalculator'
          input_stream: 'in1_to_be_filtered'
          output_stream: 'in1_filtered'
        }
        node {
          calculator: 'IntAdderCalculator'
          input_stream: 'in0_unfiltered'
          input_stream: 'in1_filtered'
          output_stream: 'out'
        }
      )pb");
  std::vector<Packet> packet_dump;
  tool::AddVectorSink("out", &config, &packet_dump);

  MP_ASSERT_OK(graph.Initialize(config));
  MP_ASSERT_OK(graph.StartRun({}));

  Timestamp timestamp = Timestamp(0);

  // We send an integer with timestamp 1 to the in0_unfiltered input stream of
  // the IntAdderCalculator. We then send an even integer with timestamp 1 to
  // the EvenIntFilterCalculator. This packet will go through and
  // the IntAdderCalculator will run. The next timestamp bounds of both the
  // input streams of the IntAdderCalculator will become 2.

  ++timestamp;  // Timestamp 1.
  MP_EXPECT_OK(graph.AddPacketToInputStream("in0_unfiltered",
                                            MakePacket<int>(1).At(timestamp)));
  MP_EXPECT_OK(graph.AddPacketToInputStream("in1_to_be_filtered",
                                            MakePacket<int>(2).At(timestamp)));
  MP_ASSERT_OK(graph.WaitUntilIdle());
  ASSERT_EQ(1, packet_dump.size());
  EXPECT_EQ(3, packet_dump[0].Get<int>());

  // We send an odd integer with timestamp 2 to the EvenIntFilterCalculator.
  // This packet will be filtered out and the next timestamp bound of the
  // in1_filtered input stream of the IntAdderCalculator will become 3.

  ++timestamp;  // Timestamp 2.
  MP_EXPECT_OK(graph.AddPacketToInputStream("in1_to_be_filtered",
                                            MakePacket<int>(3).At(timestamp)));

  // We send an integer with timestamp 3 to the in0_unfiltered input stream of
  // the IntAdderCalculator. MediaPipe should propagate the next timestamp bound
  // across the IntAdderCalculator but should not run its Process() method.

  ++timestamp;  // Timestamp 3.
  MP_EXPECT_OK(graph.AddPacketToInputStream("in0_unfiltered",
                                            MakePacket<int>(3).At(timestamp)));
  MP_ASSERT_OK(graph.WaitUntilIdle());
  ASSERT_EQ(1, packet_dump.size());

  // We send an even integer with timestamp 3 to the IntAdderCalculator. This
  // packet will go through and the IntAdderCalculator will run.

  MP_EXPECT_OK(graph.AddPacketToInputStream("in1_to_be_filtered",
                                            MakePacket<int>(4).At(timestamp)));
  MP_ASSERT_OK(graph.WaitUntilIdle());
  ASSERT_EQ(2, packet_dump.size());
  EXPECT_EQ(7, packet_dump[1].Get<int>());

  MP_ASSERT_OK(graph.CloseAllInputStreams());
  MP_ASSERT_OK(graph.WaitUntilDone());
  EXPECT_EQ(2, packet_dump.size());
}

TEST(CalculatorGraph, PropagateBoundLoop) {
  CalculatorGraphConfig config =
      mediapipe::ParseTextProtoOrDie<CalculatorGraphConfig>(R"pb(
        node {
          calculator: 'OutputAndBoundSourceCalculator'
          output_stream: 'integers'
        }
        node {
          calculator: 'IntAdderCalculator'
          input_stream: 'integers'
          input_stream: 'old_sum'
          input_stream_info: {
            tag_index: ':1'  # 'old_sum'
            back_edge: true
          }
          output_stream: 'sum'
          input_stream_handler {
            input_stream_handler: 'EarlyCloseInputStreamHandler'
          }
        }
        node {
          calculator: 'Delay20Calculator'
          input_stream: 'sum'
          output_stream: 'old_sum'
        }
      )pb");
  std::vector<Packet> packet_dump;
  tool::AddVectorSink("sum", &config, &packet_dump);

  CalculatorGraph graph;
  MP_ASSERT_OK(graph.Initialize(config));
  MP_ASSERT_OK(graph.Run());
  ASSERT_EQ(101, packet_dump.size());
  int sum = 0;
  for (int i = 0; i < 101; ++i) {
    sum += i;
    EXPECT_EQ(sum, packet_dump[i].Get<int>());
    EXPECT_EQ(Timestamp(i * 20), packet_dump[i].Timestamp());
  }
}

TEST(CalculatorGraph, CheckBatchProcessingBoundPropagation) {
  // The timestamp bound sent by OutputAndBoundSourceCalculator shouldn't be
  // directly propagated to the output stream when PassThroughCalculator has
  // anything in its default calculator context for batch processing. Otherwise,
  // the sink calculator's input stream should report packet timestamp
  // mismatches.
  CalculatorGraphConfig config =
      mediapipe::ParseTextProtoOrDie<CalculatorGraphConfig>(R"pb(
        node {
          calculator: 'OutputAndBoundSourceCalculator'
          output_stream: 'integers'
        }
        node {
          calculator: 'PassThroughCalculator'
          input_stream: 'integers'
          output_stream: 'output'
          input_stream_handler {
            input_stream_handler: "DefaultInputStreamHandler"
            options: {
              [mediapipe.DefaultInputStreamHandlerOptions.ext]: {
                batch_size: 10
              }
            }
          }
        }
        node { calculator: 'IntSinkCalculator' input_stream: 'output' }
      )pb");
  CalculatorGraph graph;
  MP_ASSERT_OK(graph.Initialize(config));
  MP_ASSERT_OK(graph.Run());
}

// Shows that ImmediateInputStreamHandler allows bounds propagation.
TEST(CalculatorGraphBoundsTest, ImmediateHandlerBounds) {
  // CustomBoundCalculator produces only timestamp bounds.
  // The first PassThroughCalculator propagates bounds using SetOffset(0).
  // The second PassthroughCalculator delivers an output packet whenever the
  // first PassThroughCalculator delivers a timestamp bound.
  CalculatorGraphConfig config =
      mediapipe::ParseTextProtoOrDie<CalculatorGraphConfig>(R"pb(
        input_stream: 'input'
        node {
          calculator: 'CustomBoundCalculator'
          input_stream: 'input'
          output_stream: 'bounds'
        }
        node {
          calculator: 'PassThroughCalculator'
          input_stream: 'bounds'
          output_stream: 'bounds_2'
          input_stream_handler {
            input_stream_handler: "ImmediateInputStreamHandler"
          }
        }
        node {
          calculator: 'PassThroughCalculator'
          input_stream: 'bounds_2'
          input_stream: 'input'
          output_stream: 'bounds_output'
          output_stream: 'output'
        }
      )pb");
  CalculatorGraph graph;
  std::vector<Packet> output_packets;
  MP_ASSERT_OK(graph.Initialize(config));
  MP_ASSERT_OK(graph.ObserveOutputStream("output", [&](const Packet& p) {
    output_packets.push_back(p);
    return absl::OkStatus();
  }));
  MP_ASSERT_OK(graph.StartRun({}));
  MP_ASSERT_OK(graph.WaitUntilIdle());

  // Add four packets into the graph.
  constexpr int kNumInputs = 4;
  for (int i = 0; i < kNumInputs; ++i) {
    Packet p = MakePacket<int>(kIntTestValue).At(Timestamp(i));
    MP_ASSERT_OK(graph.AddPacketToInputStream("input", p));
  }

  // Four packets arrive at the output only if timestamp bounds are propagated.
  MP_ASSERT_OK(graph.WaitUntilIdle());
  EXPECT_EQ(output_packets.size(), 4);

  // Eventually four packets arrive.
  MP_ASSERT_OK(graph.CloseAllPacketSources());
  MP_ASSERT_OK(graph.WaitUntilDone());
  EXPECT_EQ(output_packets.size(), 4);
}

// A Calculator that only sets timestamp bound by SetOffset().
class OffsetBoundCalculator : public CalculatorBase {
 public:
  static absl::Status GetContract(CalculatorContract* cc) {
    cc->Inputs().Index(0).Set<int>();
    cc->Outputs().Index(0).Set<int>();
    cc->SetTimestampOffset(TimestampDiff(0));
    return absl::OkStatus();
  }

  absl::Status Open(CalculatorContext* cc) final { return absl::OkStatus(); }

  absl::Status Process(CalculatorContext* cc) final { return absl::OkStatus(); }
};
REGISTER_CALCULATOR(OffsetBoundCalculator);

// A Calculator that produces a packet for each call to Process.
class BoundToPacketCalculator : public CalculatorBase {
 public:
  static absl::Status GetContract(CalculatorContract* cc) {
    for (int i = 0; i < cc->Inputs().NumEntries(); ++i) {
      cc->Inputs().Index(i).SetAny();
    }
    for (int i = 0; i < cc->Outputs().NumEntries(); ++i) {
      cc->Outputs().Index(i).Set<Timestamp>();
    }
    return absl::OkStatus();
  }

  absl::Status Open(CalculatorContext* cc) final { return absl::OkStatus(); }

  absl::Status Process(CalculatorContext* cc) final {
    for (int i = 0; i < cc->Outputs().NumEntries(); ++i) {
      Timestamp t = cc->Inputs().Index(i).Value().Timestamp();
      cc->Outputs().Index(i).AddPacket(
          mediapipe::MakePacket<Timestamp>(t).At(cc->InputTimestamp()));
    }
    return absl::OkStatus();
  }
};
REGISTER_CALCULATOR(BoundToPacketCalculator);

// A Calculator that produces packets at timestamps beyond the input timestamp.
class FuturePacketCalculator : public CalculatorBase {
 public:
  static constexpr int64_t kOutputFutureMicros = 3;

  static absl::Status GetContract(CalculatorContract* cc) {
    cc->Inputs().Index(0).Set<int>();
    cc->Outputs().Index(0).Set<int>();
    return absl::OkStatus();
  }

  absl::Status Open(CalculatorContext* cc) final { return absl::OkStatus(); }

  absl::Status Process(CalculatorContext* cc) final {
    const Packet& packet = cc->Inputs().Index(0).Value();
    Timestamp timestamp =
        Timestamp(packet.Timestamp().Value() + kOutputFutureMicros);
    cc->Outputs().Index(0).AddPacket(packet.At(timestamp));
    return absl::OkStatus();
  }
};
REGISTER_CALCULATOR(FuturePacketCalculator);

// Verifies that SetOffset still propagates when Process is called and
// produces no output packets.
TEST(CalculatorGraphBoundsTest, OffsetBoundPropagation) {
  // OffsetBoundCalculator produces only timestamp bounds.
  // The PassThroughCalculator delivers an output packet whenever the
  // OffsetBoundCalculator delivers a timestamp bound.
  CalculatorGraphConfig config =
      mediapipe::ParseTextProtoOrDie<CalculatorGraphConfig>(R"pb(
        input_stream: 'input'
        node {
          calculator: 'OffsetBoundCalculator'
          input_stream: 'input'
          output_stream: 'bounds'
        }
        node {
          calculator: 'PassThroughCalculator'
          input_stream: 'bounds'
          input_stream: 'input'
          output_stream: 'bounds_output'
          output_stream: 'output'
        }
      )pb");
  CalculatorGraph graph;
  std::vector<Packet> output_packets;
  MP_ASSERT_OK(graph.Initialize(config));
  MP_ASSERT_OK(graph.ObserveOutputStream("output", [&](const Packet& p) {
    output_packets.push_back(p);
    return absl::OkStatus();
  }));
  MP_ASSERT_OK(graph.StartRun({}));
  MP_ASSERT_OK(graph.WaitUntilIdle());

  // Add four packets into the graph.
  constexpr int kNumInputs = 4;
  for (int i = 0; i < kNumInputs; ++i) {
    Packet p = MakePacket<int>(kIntTestValue).At(Timestamp(i));
    MP_ASSERT_OK(graph.AddPacketToInputStream("input", p));
  }

  // Four packets arrive at the output only if timestamp bounds are propagated.
  MP_ASSERT_OK(graph.WaitUntilIdle());
  EXPECT_EQ(output_packets.size(), kNumInputs);

  // Shutdown the graph.
  MP_ASSERT_OK(graph.CloseAllPacketSources());
  MP_ASSERT_OK(graph.WaitUntilDone());
}

// Shows that timestamp bounds changes alone do not invoke Process,
// without SetProcessTimestampBounds(true).
TEST(CalculatorGraphBoundsTest, BoundWithoutInputPackets) {
  // OffsetBoundCalculator produces only timestamp bounds.
  // The BoundToPacketCalculator delivers an output packet whenever the
  // OffsetBoundCalculator delivers a timestamp bound.
  CalculatorGraphConfig config =
      mediapipe::ParseTextProtoOrDie<CalculatorGraphConfig>(R"pb(
        input_stream: 'input'
        node {
          calculator: 'FuturePacketCalculator'
          input_stream: 'input'
          output_stream: 'input_2'
        }
        node {
          calculator: 'OffsetBoundCalculator'
          input_stream: 'input_2'
          output_stream: 'bounds'
        }
        node {
          calculator: 'BoundToPacketCalculator'
          input_stream: 'bounds'
          output_stream: 'output'
        }
      )pb");
  CalculatorGraph graph;
  std::vector<Packet> output_packets;
  MP_ASSERT_OK(graph.Initialize(config));
  MP_ASSERT_OK(graph.ObserveOutputStream("output", [&](const Packet& p) {
    output_packets.push_back(p);
    return absl::OkStatus();
  }));
  MP_ASSERT_OK(graph.StartRun({}));
  MP_ASSERT_OK(graph.WaitUntilIdle());

  // Add four packets into the graph.
  constexpr int kNumInputs = 4;
  for (int i = 0; i < kNumInputs; ++i) {
    Packet p = MakePacket<int>(kIntTestValue).At(Timestamp(i));
    MP_ASSERT_OK(graph.AddPacketToInputStream("input", p));
    MP_ASSERT_OK(graph.WaitUntilIdle());
  }

  // No packets arrive, because FuturePacketCalculator produces 4 packets but
  // OffsetBoundCalculator relays only the 4 timestamps without any packets, and
  // BoundToPacketCalculator does not process timestamps using
  // SetProcessTimestampBounds. Thus, the graph does not invoke
  // BoundToPacketCalculator::Process.
  MP_ASSERT_OK(graph.WaitUntilIdle());
  EXPECT_EQ(output_packets.size(), 0);

  // Shutdown the graph.
  MP_ASSERT_OK(graph.CloseAllPacketSources());
  MP_ASSERT_OK(graph.WaitUntilDone());
}

// Shows that when fixed-size-input-stream-handler drops packets,
// no timetamp bounds are announced.
TEST(CalculatorGraphBoundsTest, FixedSizeHandlerBounds) {
  // LambdaCalculator with FixedSizeInputStreamHandler will drop packets
  // while it is busy.  Timetamps for the dropped packets are only relevant
  // when SetOffset is active on the LambdaCalculator.
  // The PassthroughCalculator delivers an output packet whenever the
  // LambdaCalculator delivers a timestamp bound.
  CalculatorGraphConfig config =
      mediapipe::ParseTextProtoOrDie<CalculatorGraphConfig>(R"pb(
        input_stream: 'input'
        input_side_packet: 'open_function'
        input_side_packet: 'process_function'
        node {
          calculator: 'LambdaCalculator'
          input_stream: 'input'
          output_stream: 'thinned'
          input_side_packet: 'OPEN:open_fn'
          input_side_packet: 'PROCESS:process_fn'
          input_stream_handler {
            input_stream_handler: "FixedSizeInputStreamHandler"
          }
        }
        node {
          calculator: 'PassThroughCalculator'
          input_stream: 'thinned'
          input_stream: 'input'
          output_stream: 'thinned_output'
          output_stream: 'output'
        }
      )pb");
  CalculatorGraph graph;

  // The task_semaphore counts the number of running tasks.
  constexpr int kTaskSupply = 10;
  AtomicSemaphore task_semaphore(/*supply=*/kTaskSupply);

  // This executor invokes a callback at the start and finish of each task.
  auto executor = std::make_shared<CountingExecutor>(
      4, /*start_callback=*/[&]() { task_semaphore.Acquire(1); },
      /*finish_callback=*/[&]() { task_semaphore.Release(1); });
  MP_ASSERT_OK(graph.SetExecutor(/*name=*/"", executor));

  // Monitor output from the graph.
  MP_ASSERT_OK(graph.Initialize(config));
  std::vector<Packet> outputs;
  MP_ASSERT_OK(graph.ObserveOutputStream("output", [&](const Packet& p) {
    outputs.push_back(p);
    return absl::OkStatus();
  }));
  std::vector<Packet> thinned_outputs;
  MP_ASSERT_OK(
      graph.ObserveOutputStream("thinned_output", [&](const Packet& p) {
        thinned_outputs.push_back(p);
        return absl::OkStatus();
      }));

  // The enter_semaphore is used to wait for LambdaCalculator::Process.
  // The exit_semaphore blocks and unblocks LambdaCalculator::Process.
  AtomicSemaphore enter_semaphore(0);
  AtomicSemaphore exit_semaphore(0);
  CalculatorContextFunction open_fn = [&](CalculatorContext* cc) {
    cc->SetOffset(0);
    return absl::OkStatus();
  };
  CalculatorContextFunction process_fn = [&](CalculatorContext* cc) {
    enter_semaphore.Release(1);
    exit_semaphore.Acquire(1);
    cc->Outputs().Index(0).AddPacket(cc->Inputs().Index(0).Value());
    return absl::OkStatus();
  };
  MP_ASSERT_OK(graph.StartRun({
      {"open_fn", Adopt(new auto(open_fn))},
      {"process_fn", Adopt(new auto(process_fn))},
  }));
  MP_ASSERT_OK(graph.WaitUntilIdle());

  // Add four packets into the graph.
  constexpr int kNumInputs = 4;
  for (int i = 0; i < kNumInputs; ++i) {
    Packet p = MakePacket<int>(33).At(Timestamp(i));
    MP_ASSERT_OK(graph.AddPacketToInputStream("input", p));
  }

  // Wait until only the LambdaCalculator is running,
  // by wating until the task_semaphore has only one token occupied.
  // At this point 2 packets were dropped by the FixedSizeInputStreamHandler.
  task_semaphore.Acquire(kTaskSupply - 1);
  task_semaphore.Release(kTaskSupply - 1);

  // No timestamp bounds and no packets are emitted yet.
  EXPECT_EQ(outputs.size(), 0);
  EXPECT_EQ(thinned_outputs.size(), 0);

  // Allow the first LambdaCalculator::Process call to complete.
  // Wait for the second LambdaCalculator::Process call to begin.
  // Wait until only the LambdaCalculator is running.
  enter_semaphore.Acquire(1);
  exit_semaphore.Release(1);
  enter_semaphore.Acquire(1);
  task_semaphore.Acquire(kTaskSupply - 1);
  task_semaphore.Release(kTaskSupply - 1);

  // Only one timestamp bound and one packet are emitted.
  EXPECT_EQ(outputs.size(), 1);
  EXPECT_EQ(thinned_outputs.size(), 1);

  // Allow the second LambdaCalculator::Process call to complete.
  exit_semaphore.Release(1);
  MP_ASSERT_OK(graph.WaitUntilIdle());

  // Packets 1 and 2 were dropped by the FixedSizeInputStreamHandler.
  EXPECT_EQ(thinned_outputs.size(), 2);
  EXPECT_EQ(thinned_outputs[0].Timestamp(), Timestamp(0));
  EXPECT_EQ(thinned_outputs[1].Timestamp(), Timestamp(kNumInputs - 1));
  EXPECT_EQ(outputs.size(), kNumInputs);
  MP_ASSERT_OK(graph.CloseAllPacketSources());
  MP_ASSERT_OK(graph.WaitUntilDone());
}

// A Calculator that outputs only the last packet from its input stream.
class LastPacketCalculator : public CalculatorBase {
 public:
  static absl::Status GetContract(CalculatorContract* cc) {
    cc->Inputs().Index(0).SetAny();
    cc->Outputs().Index(0).SetAny();
    return absl::OkStatus();
  }
  absl::Status Open(CalculatorContext* cc) final { return absl::OkStatus(); }
  absl::Status Process(CalculatorContext* cc) final {
    cc->Outputs().Index(0).SetNextTimestampBound(cc->InputTimestamp());
    last_packet_ = cc->Inputs().Index(0).Value();
    return absl::OkStatus();
  }
  absl::Status Close(CalculatorContext* cc) final {
    cc->Outputs().Index(0).AddPacket(last_packet_);
    return absl::OkStatus();
  }

 private:
  Packet last_packet_;
};
REGISTER_CALCULATOR(LastPacketCalculator);

// Shows that the last packet in an input stream can be detected.
TEST(CalculatorGraphBoundsTest, LastPacketCheck) {
  // LastPacketCalculator emits only the last input stream packet.
  // It emits a timestamp bound after the arrival of a successor input stream
  // packet or input stream close.  The output "last_output" shows the
  // last packet, and "output" shows the timestamp bounds.
  CalculatorGraphConfig config =
      mediapipe::ParseTextProtoOrDie<CalculatorGraphConfig>(R"pb(
        input_stream: 'input'
        output_stream: 'output'
        output_stream: 'last_output'
        node {
          calculator: 'PassThroughCalculator'
          input_stream: 'input'
          output_stream: 'input_2'
        }
        node {
          calculator: 'LastPacketCalculator'
          input_stream: 'input_2'
          output_stream: 'last_packet'
        }
        node {
          calculator: 'PassThroughCalculator'
          input_stream: 'input'
          input_stream: 'last_packet'
          output_stream: 'output'
          output_stream: 'last_output'
        }
      )pb");
  CalculatorGraph graph;
  std::vector<Packet> output_packets;
  MP_ASSERT_OK(graph.Initialize(config));
  MP_ASSERT_OK(graph.ObserveOutputStream("output", [&](const Packet& p) {
    output_packets.push_back(p);
    return absl::OkStatus();
  }));
  std::vector<Packet> last_output_packets;
  MP_ASSERT_OK(graph.ObserveOutputStream("last_output", [&](const Packet& p) {
    last_output_packets.push_back(p);
    return absl::OkStatus();
  }));
  MP_ASSERT_OK(graph.StartRun({}));
  MP_ASSERT_OK(graph.WaitUntilIdle());

  // Add four packets into the graph.
  constexpr int kNumInputs = 4;
  for (int i = 0; i < kNumInputs; ++i) {
    Packet p = MakePacket<int>(33).At(Timestamp(i));
    MP_ASSERT_OK(graph.AddPacketToInputStream("input", p));
    MP_ASSERT_OK(graph.WaitUntilIdle());
    EXPECT_EQ(i, output_packets.size());
    EXPECT_EQ(0, last_output_packets.size());
  }

  // Shutdown the graph.
  MP_ASSERT_OK(graph.CloseAllPacketSources());
  MP_ASSERT_OK(graph.WaitUntilIdle());
  EXPECT_EQ(kNumInputs, output_packets.size());
  EXPECT_EQ(1, last_output_packets.size());
  MP_ASSERT_OK(graph.WaitUntilDone());
}

// Shows that bounds are indicated for input streams without input packets.
void TestBoundsForEmptyInputs(std::string input_stream_handler) {
  // FuturePacketCalculator and OffsetBoundCalculator produce future ts bounds.
  // BoundToPacketCalculator reports all of its bounds, including empty inputs.
  std::string config_str = R"(
            input_stream: 'input'
            node {
              calculator: 'FuturePacketCalculator'
              input_stream: 'input'
              output_stream: 'futures'
            }
            node {
              calculator: 'OffsetBoundCalculator'
              input_stream: 'futures'
              output_stream: 'bounds'
            }
            node {
              calculator: 'BoundToPacketCalculator'
              input_stream: 'input'
              input_stream: 'bounds'
              output_stream: 'input_ts'
              output_stream: 'bounds_ts'
              input_stream_handler { $input_stream_handler }
            }
          )";
  absl::StrReplaceAll({{"$input_stream_handler", input_stream_handler}},
                      &config_str);
  CalculatorGraphConfig config =
      mediapipe::ParseTextProtoOrDie<CalculatorGraphConfig>(config_str);
  CalculatorGraph graph;
  std::vector<Packet> input_ts_packets;
  std::vector<Packet> bounds_ts_packets;
  MP_ASSERT_OK(graph.Initialize(config));
  MP_ASSERT_OK(graph.ObserveOutputStream("input_ts", [&](const Packet& p) {
    input_ts_packets.push_back(p);
    return absl::OkStatus();
  }));
  MP_ASSERT_OK(graph.ObserveOutputStream("bounds_ts", [&](const Packet& p) {
    bounds_ts_packets.push_back(p);
    return absl::OkStatus();
  }));
  MP_ASSERT_OK(graph.StartRun({}));
  MP_ASSERT_OK(graph.WaitUntilIdle());

  // Add four packets into the graph, with timedtamps 0, 10, 20, 30.
  constexpr int kNumInputs = 4;
  for (int i = 0; i < kNumInputs; ++i) {
    Packet p = MakePacket<int>(33).At(Timestamp(i * 10));
    MP_ASSERT_OK(graph.AddPacketToInputStream("input", p));
    MP_ASSERT_OK(graph.WaitUntilIdle());
  }

  // Packets arrive.  The input packet timestamps are: 0, 10, 20, 30.
  // The corresponding empty packet timestamps are: 3, 13, 23, 33.
  MP_ASSERT_OK(graph.WaitUntilIdle());
  EXPECT_EQ(input_ts_packets.size(), 4);
  EXPECT_EQ(bounds_ts_packets.size(), 4);

  // The timestamp bounds from OffsetBoundCalculator are: 3, 13, 23, 33.
  // Because the process call waits for the input packets and not for
  // the empty packets, the first empty packet timestamp can be
  // either Timestamp::Unstarted() or Timestamp(3).
  std::vector<Timestamp> expected = {Timestamp::Unstarted(), Timestamp(3),
                                     Timestamp(13), Timestamp(23),
                                     Timestamp(33)};
  for (int i = 0; i < bounds_ts_packets.size(); ++i) {
    Timestamp ts = bounds_ts_packets[i].Get<Timestamp>();
    EXPECT_GE(ts, expected[i]);
    EXPECT_LE(ts, expected[i + 1]);
  }

  // Shutdown the graph.
  MP_ASSERT_OK(graph.CloseAllPacketSources());
  MP_ASSERT_OK(graph.WaitUntilDone());
}

// Shows that bounds are indicated for input streams without input packets.
TEST(CalculatorGraphBoundsTest, BoundsForEmptyInputs_Immediate) {
  TestBoundsForEmptyInputs(R"(
      input_stream_handler: "ImmediateInputStreamHandler")");
}

// Shows that bounds are indicated for input streams without input packets.
TEST(CalculatorGraphBoundsTest, BoundsForEmptyInputs_Default) {
  TestBoundsForEmptyInputs(R"(
      input_stream_handler: "DefaultInputStreamHandler")");
}

// Shows that bounds are indicated for input streams without input packets.
TEST(CalculatorGraphBoundsTest, BoundsForEmptyInputs_SyncSet) {
  TestBoundsForEmptyInputs(R"(
     input_stream_handler: "SyncSetInputStreamHandler")");
}

// Shows that bounds are indicated for input streams without input packets.
TEST(CalculatorGraphBoundsTest, BoundsForEmptyInputs_SyncSets) {
  TestBoundsForEmptyInputs(R"(
      input_stream_handler: "SyncSetInputStreamHandler"
      options {
        [mediapipe.SyncSetInputStreamHandlerOptions.ext] {
          sync_set { tag_index: ":0" }
        }
      }
    )");
}

// A Calculator that produces a packet for each timestamp bounds update.
class ProcessBoundToPacketCalculator : public CalculatorBase {
 public:
  static absl::Status GetContract(CalculatorContract* cc) {
    for (int i = 0; i < cc->Inputs().NumEntries(); ++i) {
      cc->Inputs().Index(i).SetAny();
    }
    for (int i = 0; i < cc->Outputs().NumEntries(); ++i) {
      cc->Outputs().Index(i).Set<Timestamp>();
    }
    cc->SetInputStreamHandler("ImmediateInputStreamHandler");
    cc->SetProcessTimestampBounds(true);
    return absl::OkStatus();
  }

  absl::Status Process(CalculatorContext* cc) final {
    for (int i = 0; i < cc->Outputs().NumEntries(); ++i) {
      Timestamp t = cc->Inputs().Index(i).Value().Timestamp();
      // Create a new packet for each input stream with a new timestamp bound,
      // as long as the new timestamp satisfies the output timestamp bound.
      if (t == cc->InputTimestamp() &&
          t >= cc->Outputs().Index(i).NextTimestampBound()) {
        cc->Outputs().Index(i).Add(new auto(t), t);
      }
    }
    return absl::OkStatus();
  }
};
REGISTER_CALCULATOR(ProcessBoundToPacketCalculator);

// A Calculator that passes through each packet and timestamp immediately.
class ImmediatePassthroughCalculator : public CalculatorBase {
 public:
  static absl::Status GetContract(CalculatorContract* cc) {
    for (int i = 0; i < cc->Inputs().NumEntries(); ++i) {
      cc->Inputs().Index(i).SetAny();
    }
    for (int i = 0; i < cc->Outputs().NumEntries(); ++i) {
      cc->Outputs().Index(i).SetSameAs(&cc->Inputs().Index(i));
    }
    cc->SetInputStreamHandler("ImmediateInputStreamHandler");
    cc->SetProcessTimestampBounds(true);
    return absl::OkStatus();
  }

  absl::Status Process(CalculatorContext* cc) final {
    for (int i = 0; i < cc->Outputs().NumEntries(); ++i) {
      if (!cc->Inputs().Index(i).IsEmpty()) {
        cc->Outputs().Index(i).AddPacket(cc->Inputs().Index(i).Value());
      } else {
        // Update the output stream "i" nextTimestampBound to the timestamp at
        // which a packet may next be available in input stream "i".
        Timestamp input_bound =
            cc->Inputs().Index(i).Value().Timestamp().NextAllowedInStream();
        if (cc->Outputs().Index(i).NextTimestampBound() < input_bound) {
          cc->Outputs().Index(i).SetNextTimestampBound(input_bound);
        }
      }
    }
    return absl::OkStatus();
  }
};
REGISTER_CALCULATOR(ImmediatePassthroughCalculator);

// Shows that Process is called for input-sets without input packets.
void TestProcessForEmptyInputs(const std::string& input_stream_handler) {
  // FuturePacketCalculator and OffsetBoundCalculator produce only ts bounds,
  // The ProcessBoundToPacketCalculator has SetProcessTimestampBounds(true),
  // and produces an output packet for every timestamp bound update.
  std::string config_str = R"(
            input_stream: 'input'
            node {
              calculator: 'FuturePacketCalculator'
              input_stream: 'input'
              output_stream: 'futures'
            }
            node {
              calculator: 'OffsetBoundCalculator'
              input_stream: 'futures'
              output_stream: 'bounds'
            }
            node {
              calculator: 'ProcessBoundToPacketCalculator'
              input_stream: 'bounds'
              output_stream: 'bounds_ts'
              input_stream_handler { $input_stream_handler }
            }
          )";
  absl::StrReplaceAll({{"$input_stream_handler", input_stream_handler}},
                      &config_str);
  CalculatorGraphConfig config =
      mediapipe::ParseTextProtoOrDie<CalculatorGraphConfig>(config_str);
  CalculatorGraph graph;
  std::vector<Packet> input_ts_packets;
  std::vector<Packet> bounds_ts_packets;
  MP_ASSERT_OK(graph.Initialize(config));
  MP_ASSERT_OK(graph.ObserveOutputStream("bounds_ts", [&](const Packet& p) {
    bounds_ts_packets.push_back(p);
    return absl::OkStatus();
  }));
  MP_ASSERT_OK(graph.StartRun({}));
  MP_ASSERT_OK(graph.WaitUntilIdle());

  // Add four packets into the graph at ts {0, 10, 20, 30}.
  constexpr int kFutureMicros = FuturePacketCalculator::kOutputFutureMicros;
  constexpr int kNumInputs = 4;
  std::vector<Timestamp> expected;
  for (int i = 0; i < kNumInputs; ++i) {
    const int ts = i * 10;
    Packet p = MakePacket<int>(kIntTestValue).At(Timestamp(ts));
    MP_ASSERT_OK(graph.AddPacketToInputStream("input", p));
    MP_ASSERT_OK(graph.WaitUntilIdle());

    expected.emplace_back(Timestamp(ts + kFutureMicros));
  }

  // Packets arrive.
  MP_ASSERT_OK(graph.WaitUntilIdle());
  EXPECT_EQ(bounds_ts_packets.size(), kNumInputs);

  // Shutdown the graph.
  MP_ASSERT_OK(graph.CloseAllPacketSources());
  MP_ASSERT_OK(graph.WaitUntilDone());
}

// Shows that Process is called for input-sets without input packets
// using an DefaultInputStreamHandler.
TEST(CalculatorGraphBoundsTest, ProcessTimestampBounds_Default) {
  TestProcessForEmptyInputs(R"(
      input_stream_handler: "DefaultInputStreamHandler")");
}

// Shows that Process is called for input-sets without input packets
// using an ImmediateInputStreamHandler.
TEST(CalculatorGraphBoundsTest, ProcessTimestampBounds_Immediate) {
  TestProcessForEmptyInputs(R"(
      input_stream_handler: "ImmediateInputStreamHandler")");
}

// Shows that Process is called for input-sets without input packets
// using a SyncSetInputStreamHandler with a single sync-set.
TEST(CalculatorGraphBoundsTest, ProcessTimestampBounds_SyncSet) {
  TestProcessForEmptyInputs(R"(
      input_stream_handler: "SyncSetInputStreamHandler")");
}

// Shows that Process is called for input-sets without input packets
// using a SyncSetInputStreamHandler with multiple sync-sets.
TEST(CalculatorGraphBoundsTest, ProcessTimestampBounds_SyncSets) {
  TestProcessForEmptyInputs(R"(
      input_stream_handler: "SyncSetInputStreamHandler"
      options {
        [mediapipe.SyncSetInputStreamHandlerOptions.ext] {
          sync_set { tag_index: ":0" }
        }
      }
    )");
}

// Demonstrates the functionality of an "ImmediatePassthroughCalculator".
// The ImmediatePassthroughCalculator simply relays each input packet to
// the corresponding output stream.  ProcessTimestampBounds is needed to
// relay timestamp bounds as well as packets.
TEST(CalculatorGraphBoundsTest, ProcessTimestampBounds_Passthrough) {
  // OffsetBoundCalculator produces timestamp bounds.
  // ImmediatePassthroughCalculator relays packets and bounds.
  // ProcessBoundToPacketCalculator reports packets and bounds as packets.
  std::string config_str = R"(
            input_stream: "input_0"
            input_stream: "input_1"
            node {
              calculator: "OffsetBoundCalculator"
              input_stream: "input_1"
              output_stream: "bound_1"
            }
            node {
              calculator: "ImmediatePassthroughCalculator"
              input_stream: "input_0"
              input_stream: "bound_1"
              output_stream: "same_0"
              output_stream: "same_1"
            }
            node {
              calculator: "ProcessBoundToPacketCalculator"
              input_stream: "same_0"
              input_stream: "same_1"
              output_stream: "output_0"
              output_stream: "output_1"
            }
          )";
  CalculatorGraphConfig config =
      mediapipe::ParseTextProtoOrDie<CalculatorGraphConfig>(config_str);
  CalculatorGraph graph;
  std::vector<Packet> output_0_packets;
  std::vector<Packet> output_1_packets;
  MP_ASSERT_OK(graph.Initialize(config));
  MP_ASSERT_OK(graph.ObserveOutputStream("output_0", [&](const Packet& p) {
    output_0_packets.push_back(p);
    return absl::OkStatus();
  }));
  MP_ASSERT_OK(graph.ObserveOutputStream("output_1", [&](const Packet& p) {
    output_1_packets.push_back(p);
    return absl::OkStatus();
  }));
  MP_ASSERT_OK(graph.StartRun({}));
  MP_ASSERT_OK(graph.WaitUntilIdle());

  // Add four packets to input_0.
  constexpr int kNumInputs0 = 4;
  std::vector<Timestamp> expected_output_0;
  for (int i = 0; i < kNumInputs0; ++i) {
    const int ts = i * 10;
    Packet p = MakePacket<int>(kIntTestValue).At(Timestamp(ts));
    MP_ASSERT_OK(graph.AddPacketToInputStream("input_0", p));
    MP_ASSERT_OK(graph.WaitUntilIdle());

    expected_output_0.emplace_back(Timestamp(ts));
  }

  // Packets arrive.
  MP_ASSERT_OK(graph.WaitUntilIdle());
  EXPECT_EQ(output_0_packets.size(), kNumInputs0);
  // No packets were pushed in "input_1".
  EXPECT_EQ(output_1_packets.size(), 0);
  EXPECT_EQ(GetContents<Timestamp>(output_0_packets), expected_output_0);

  // Add two timestamp bounds to "input_1" and update "bound_1" at {10, 20}.
  constexpr int kNumInputs1 = 2;
  std::vector<Timestamp> expected_output_1;
  for (int i = 0; i < kNumInputs1; ++i) {
    const int ts = 10 + i * 10;
    Packet p = MakePacket<int>(kIntTestValue).At(Timestamp(ts));
    MP_ASSERT_OK(graph.AddPacketToInputStream("input_1", p));
    MP_ASSERT_OK(graph.WaitUntilIdle());

    expected_output_1.emplace_back(Timestamp(ts));
  }

  // Bounds arrive.
  MP_ASSERT_OK(graph.WaitUntilIdle());
  EXPECT_EQ(output_0_packets.size(), kNumInputs0);
  EXPECT_EQ(output_1_packets.size(), kNumInputs1);
  EXPECT_EQ(GetContents<Timestamp>(output_1_packets), expected_output_1);

  // Shutdown the graph.
  MP_ASSERT_OK(graph.CloseAllPacketSources());
  MP_ASSERT_OK(graph.WaitUntilDone());
}

TEST(CalculatorGraphBoundsTest, PostStreamPacketToSetProcessTimestampBound) {
  std::string config_str = R"(
            input_stream: "input_0"
            node {
              calculator: "ProcessBoundToPacketCalculator"
              input_stream: "input_0"
              output_stream: "output_0"
            }
          )";
  CalculatorGraphConfig config =
      mediapipe::ParseTextProtoOrDie<CalculatorGraphConfig>(config_str);
  CalculatorGraph graph;
  std::vector<Packet> output_0_packets;
  MP_ASSERT_OK(graph.Initialize(config));
  MP_ASSERT_OK(graph.ObserveOutputStream("output_0", [&](const Packet& p) {
    output_0_packets.push_back(p);
    return absl::OkStatus();
  }));
  MP_ASSERT_OK(graph.StartRun({}));
  MP_ASSERT_OK(graph.WaitUntilIdle());

  MP_ASSERT_OK(graph.AddPacketToInputStream(
      "input_0", MakePacket<int>(0).At(Timestamp::PostStream())));
  MP_ASSERT_OK(graph.WaitUntilIdle());
  EXPECT_EQ(output_0_packets.size(), 1);
  EXPECT_EQ(output_0_packets[0].Timestamp(), Timestamp::PostStream());

  // Shutdown the graph.
  MP_ASSERT_OK(graph.CloseAllPacketSources());
  MP_ASSERT_OK(graph.WaitUntilDone());
}

// A Calculator that sends a timestamp bound for every other input.
class OccasionalBoundCalculator : public CalculatorBase {
 public:
  static absl::Status GetContract(CalculatorContract* cc) {
    cc->Inputs().Index(0).Set<int>();
    cc->Outputs().Index(0).SetSameAs(&cc->Inputs().Index(0));
    return absl::OkStatus();
  }

  absl::Status Process(CalculatorContext* cc) final {
    absl::SleepFor(absl::Milliseconds(1));
    if (cc->InputTimestamp().Value() % 20 == 0) {
      Timestamp bound = cc->InputTimestamp().NextAllowedInStream();
      cc->Outputs().Index(0).SetNextTimestampBound(
          std::max(bound, cc->Outputs().Index(0).NextTimestampBound()));
    }
    return absl::OkStatus();
  }
};
REGISTER_CALCULATOR(OccasionalBoundCalculator);

// This test fails without the fix in CL/324708313, because
// PropagateUpdatesToMirrors is called with decreasing next_timestamp_bound,
// because each parallel thread in-flight computes next_timestamp_bound using
// a separate OutputStreamShard::NextTimestampBound.
TEST(CalculatorGraphBoundsTest, MaxInFlightWithOccasionalBound) {
  // OccasionalCalculator runs on parallel threads and sends ts occasionally.
  std::string config_str = R"(
            input_stream: "input_0"
            node {
              calculator: "OccasionalBoundCalculator"
              input_stream: "input_0"
              output_stream: "output_0"
              max_in_flight: 5
            }
            num_threads: 4
          )";
  CalculatorGraphConfig config =
      mediapipe::ParseTextProtoOrDie<CalculatorGraphConfig>(config_str);
  CalculatorGraph graph;
  std::vector<Packet> output_0_packets;
  MP_ASSERT_OK(graph.Initialize(config));
  MP_ASSERT_OK(graph.ObserveOutputStream("output_0", [&](const Packet& p) {
    output_0_packets.push_back(p);
    return absl::OkStatus();
  }));
  MP_ASSERT_OK(graph.StartRun({}));
  MP_ASSERT_OK(graph.WaitUntilIdle());

  // Send in packets.
  for (int i = 0; i < 9; ++i) {
    const int ts = 10 + i * 10;
    Packet p = MakePacket<int>(i).At(Timestamp(ts));
    MP_ASSERT_OK(graph.AddPacketToInputStream("input_0", p));
  }

  // Only bounds arrive.
  MP_ASSERT_OK(graph.WaitUntilIdle());
  EXPECT_EQ(output_0_packets.size(), 0);

  // Shutdown the graph.
  MP_ASSERT_OK(graph.CloseAllPacketSources());
  MP_ASSERT_OK(graph.WaitUntilDone());
}

// A Calculator that uses both SetTimestampOffset and SetNextTimestampBound.
class OffsetAndBoundCalculator : public CalculatorBase {
 public:
  static absl::Status GetContract(CalculatorContract* cc) {
    cc->Inputs().Index(0).Set<int>();
    cc->Outputs().Index(0).SetSameAs(&cc->Inputs().Index(0));
    cc->SetTimestampOffset(TimestampDiff(0));
    return absl::OkStatus();
  }
  absl::Status Open(CalculatorContext* cc) final { return absl::OkStatus(); }
  absl::Status Process(CalculatorContext* cc) final {
    if (cc->InputTimestamp().Value() % 20 == 0) {
      cc->Outputs().Index(0).SetNextTimestampBound(Timestamp(10000));
    }
    return absl::OkStatus();
  }
};
REGISTER_CALCULATOR(OffsetAndBoundCalculator);

// This test shows that the bound defined by SetOffset is ignored
// if it is superseded by SetNextTimestampBound.
TEST(CalculatorGraphBoundsTest, OffsetAndBound) {
  // OffsetAndBoundCalculator runs on parallel threads and sends ts
  // occasionally.
  std::string config_str = R"(
            input_stream: "input_0"
            node {
              calculator: "OffsetAndBoundCalculator"
              input_stream: "input_0"
              output_stream: "output_0"
            }
          )";
  CalculatorGraphConfig config =
      mediapipe::ParseTextProtoOrDie<CalculatorGraphConfig>(config_str);
  CalculatorGraph graph;
  std::vector<Packet> output_0_packets;
  MP_ASSERT_OK(graph.Initialize(config));
  MP_ASSERT_OK(graph.ObserveOutputStream("output_0", [&](const Packet& p) {
    output_0_packets.push_back(p);
    return absl::OkStatus();
  }));
  MP_ASSERT_OK(graph.StartRun({}));
  MP_ASSERT_OK(graph.WaitUntilIdle());

  // Send in packets.
  for (int i = 0; i < 9; ++i) {
    const int ts = 10 + i * 10;
    Packet p = MakePacket<int>(i).At(Timestamp(ts));
    MP_ASSERT_OK(graph.AddPacketToInputStream("input_0", p));
  }

  // Only bounds arrive.
  MP_ASSERT_OK(graph.WaitUntilIdle());
  EXPECT_EQ(output_0_packets.size(), 0);

  // Shutdown the graph.
  MP_ASSERT_OK(graph.CloseAllPacketSources());
  MP_ASSERT_OK(graph.WaitUntilDone());
}

// A Calculator that sends empty output stream packets.
class EmptyPacketCalculator : public CalculatorBase {
 public:
  static absl::Status GetContract(CalculatorContract* cc) {
    cc->Inputs().Index(0).Set<int>();
    cc->Outputs().Index(0).SetSameAs(&cc->Inputs().Index(0));
    return absl::OkStatus();
  }
  absl::Status Open(CalculatorContext* cc) final { return absl::OkStatus(); }
  absl::Status Process(CalculatorContext* cc) final {
    if (cc->InputTimestamp().Value() % 2 == 0) {
      cc->Outputs().Index(0).AddPacket(Packet().At(cc->InputTimestamp()));
    }
    return absl::OkStatus();
  }
};
REGISTER_CALCULATOR(EmptyPacketCalculator);

// This test shows that an output timestamp bound can be specified by outputting
// an empty packet with a settled timestamp.
TEST(CalculatorGraphBoundsTest, EmptyPacketOutput) {
  // OffsetAndBoundCalculator runs on parallel threads and sends ts
  // occasionally.
  std::string config_str = R"(
            input_stream: "input_0"
            node {
              calculator: "EmptyPacketCalculator"
              input_stream: "input_0"
              output_stream: "empty_0"
            }
            node {
              calculator: "ProcessBoundToPacketCalculator"
              input_stream: "empty_0"
              output_stream: "output_0"
            }
          )";
  CalculatorGraphConfig config =
      mediapipe::ParseTextProtoOrDie<CalculatorGraphConfig>(config_str);
  CalculatorGraph graph;
  std::vector<Packet> output_0_packets;
  MP_ASSERT_OK(graph.Initialize(config));
  MP_ASSERT_OK(graph.ObserveOutputStream("output_0", [&](const Packet& p) {
    output_0_packets.push_back(p);
    return absl::OkStatus();
  }));
  MP_ASSERT_OK(graph.StartRun({}));
  MP_ASSERT_OK(graph.WaitUntilIdle());

  // Send in packets.
  for (int i = 0; i < 9; ++i) {
    const int ts = 10 + i * 10;
    Packet p = MakePacket<int>(i).At(Timestamp(ts));
    MP_ASSERT_OK(graph.AddPacketToInputStream("input_0", p));
    MP_ASSERT_OK(graph.WaitUntilIdle());
  }

  // 9 empty packets are converted to bounds and then to packets.
  EXPECT_EQ(output_0_packets.size(), 9);
  for (int i = 0; i < 9; ++i) {
    EXPECT_EQ(output_0_packets[i].Timestamp(), Timestamp(10 + i * 10));
  }

  // Shut down the graph.
  MP_ASSERT_OK(graph.CloseAllPacketSources());
  MP_ASSERT_OK(graph.WaitUntilDone());
}

// This test shows that input timestamp bounds can be specified using
// CalculatorGraph::SetInputStreamTimestampBound.
TEST(CalculatorGraphBoundsTest, SetInputStreamTimestampBound) {
  std::string config_str = R"(
            input_stream: "input_0"
            node {
              calculator: "ProcessBoundToPacketCalculator"
              input_stream: "input_0"
              output_stream: "output_0"
            }
          )";
  CalculatorGraphConfig config =
      mediapipe::ParseTextProtoOrDie<CalculatorGraphConfig>(config_str);
  CalculatorGraph graph;
  std::vector<Packet> output_0_packets;
  MP_ASSERT_OK(graph.Initialize(config));
  MP_ASSERT_OK(graph.ObserveOutputStream("output_0", [&](const Packet& p) {
    output_0_packets.push_back(p);
    return absl::OkStatus();
  }));
  MP_ASSERT_OK(graph.StartRun({}));
  MP_ASSERT_OK(graph.WaitUntilIdle());

  // Send in timestamp bounds.
  for (int i = 0; i < 9; ++i) {
    const int ts = 10 + i * 10;
    MP_ASSERT_OK(graph.SetInputStreamTimestampBound(
        "input_0", Timestamp(ts).NextAllowedInStream()));
    MP_ASSERT_OK(graph.WaitUntilIdle());
  }

  // 9 timestamp bounds are converted to packets.
  EXPECT_EQ(output_0_packets.size(), 9);
  for (int i = 0; i < 9; ++i) {
    EXPECT_EQ(output_0_packets[i].Timestamp(), Timestamp(10 + i * 10));
  }

  // Shutdown the graph.
  MP_ASSERT_OK(graph.CloseAllPacketSources());
  MP_ASSERT_OK(graph.WaitUntilDone());
}

// This test shows how an input stream with infrequent packets, such as
// configuration protobufs, can be consumed while processing more frequent
// packets, such as video frames.
TEST(CalculatorGraphBoundsTest, TimestampBoundsForInfrequentInput) {
  // PassThroughCalculator consuming two input streams, with default ISH.
  std::string config_str = R"pb(
    input_stream: "INFREQUENT:config"
    input_stream: "FREQUENT:frame"
    node {
      calculator: "PassThroughCalculator"
      input_stream: "CONFIG:config"
      input_stream: "VIDEO:frame"
      output_stream: "VIDEO:output_frame"
      output_stream: "CONFIG:output_config"
    }
  )pb";

  CalculatorGraphConfig config =
      mediapipe::ParseTextProtoOrDie<CalculatorGraphConfig>(config_str);
  CalculatorGraph graph;
  std::vector<Packet> frame_packets;
  MP_ASSERT_OK(graph.Initialize(config));
  MP_ASSERT_OK(graph.ObserveOutputStream(
      "output_frame",
      [&](const Packet& p) {
        frame_packets.push_back(p);
        return absl::OkStatus();
      },
      /*observe_bound_updates=*/true));
  std::vector<Packet> config_packets;
  MP_ASSERT_OK(graph.ObserveOutputStream(
      "output_config",
      [&](const Packet& p) {
        config_packets.push_back(p);
        return absl::OkStatus();
      },
      /*observe_bound_updates=*/true));
  MP_ASSERT_OK(graph.StartRun({}));
  MP_ASSERT_OK(graph.WaitUntilIdle());

  // Utility functions to send packets or timestamp bounds.
  auto send_fn = [&](std::string stream, std::string value, int ts) {
    MP_ASSERT_OK(graph.AddPacketToInputStream(
        stream,
        MakePacket<std::string>(absl::StrCat(value)).At(Timestamp(ts))));
    MP_ASSERT_OK(graph.WaitUntilIdle());
  };
  auto bound_fn = [&](std::string stream, int ts) {
    MP_ASSERT_OK(graph.SetInputStreamTimestampBound(stream, Timestamp(ts)));
    MP_ASSERT_OK(graph.WaitUntilIdle());
  };

  // Send in a frame packet.
  send_fn("frame", "frame_0", 0);
  // The frame is not processed yet.
  EXPECT_THAT(frame_packets, ElementsAreArray(PacketMatchers<std::string>({})));
  bound_fn("config", 10000);
  // The frame is processed after a fresh config timestamp bound arrives.
  EXPECT_THAT(frame_packets,
              ElementsAreArray(PacketMatchers<std::string>({
                  MakePacket<std::string>("frame_0").At(Timestamp(0)),
              })));

  // Send in a frame packet.
  send_fn("frame", "frame_1", 20000);
  // The frame is not processed yet.
  // The PassThroughCalculator with TimestampOffset 0 now propagates
  // Timestamp bound 10000 to both "output_frame" and "output_config",
  // which appears here as Packet().At(Timestamp(9999).  The timestamp
  // bounds at 29999 and 50000 are propagated similarly.
  EXPECT_THAT(frame_packets,
              ElementsAreArray(PacketMatchers<std::string>({
                  MakePacket<std::string>("frame_0").At(Timestamp(0)),
                  Packet().At(Timestamp(9999)),
              })));
  bound_fn("config", 30000);
  // The frame is processed after a fresh config timestamp bound arrives.
  EXPECT_THAT(frame_packets,
              ElementsAreArray(PacketMatchers<std::string>({
                  MakePacket<std::string>("frame_0").At(Timestamp(0)),
                  Packet().At(Timestamp(9999)),
                  MakePacket<std::string>("frame_1").At(Timestamp(20000)),
              })));

  // Send in a frame packet.
  send_fn("frame", "frame_2", 40000);
  // The frame is not processed yet.
  EXPECT_THAT(frame_packets,
              ElementsAreArray(PacketMatchers<std::string>({
                  MakePacket<std::string>("frame_0").At(Timestamp(0)),
                  Packet().At(Timestamp(9999)),
                  MakePacket<std::string>("frame_1").At(Timestamp(20000)),
                  Packet().At(Timestamp(29999)),
              })));
  send_fn("config", "config_1", 50000);
  // The frame is processed after a fresh config arrives.
  EXPECT_THAT(frame_packets,
              ElementsAreArray(PacketMatchers<std::string>({
                  MakePacket<std::string>("frame_0").At(Timestamp(0)),
                  Packet().At(Timestamp(9999)),
                  MakePacket<std::string>("frame_1").At(Timestamp(20000)),
                  Packet().At(Timestamp(29999)),
                  MakePacket<std::string>("frame_2").At(Timestamp(40000)),
              })));

  // Send in a frame packet.
  send_fn("frame", "frame_3", 60000);
  // The frame is not processed yet.
  EXPECT_THAT(frame_packets,
              ElementsAreArray(PacketMatchers<std::string>({
                  MakePacket<std::string>("frame_0").At(Timestamp(0)),
                  Packet().At(Timestamp(9999)),
                  MakePacket<std::string>("frame_1").At(Timestamp(20000)),
                  Packet().At(Timestamp(29999)),
                  MakePacket<std::string>("frame_2").At(Timestamp(40000)),
                  Packet().At(Timestamp(50000)),
              })));
  bound_fn("config", 70000);
  // The frame is processed after a fresh config timestamp bound arrives.
  EXPECT_THAT(frame_packets,
              ElementsAreArray(PacketMatchers<std::string>({
                  MakePacket<std::string>("frame_0").At(Timestamp(0)),
                  Packet().At(Timestamp(9999)),
                  MakePacket<std::string>("frame_1").At(Timestamp(20000)),
                  Packet().At(Timestamp(29999)),
                  MakePacket<std::string>("frame_2").At(Timestamp(40000)),
                  Packet().At(Timestamp(50000)),
                  MakePacket<std::string>("frame_3").At(Timestamp(60000)),
              })));

  // One config packet is deleivered.
  EXPECT_THAT(config_packets,
              ElementsAreArray(PacketMatchers<std::string>({
                  Packet().At(Timestamp(0)),
                  Packet().At(Timestamp(9999)),
                  Packet().At(Timestamp(20000)),
                  Packet().At(Timestamp(29999)),
                  Packet().At(Timestamp(40000)),
                  MakePacket<std::string>("config_1").At(Timestamp(50000)),
                  Packet().At(Timestamp(60000)),
              })));

  // Shutdown the graph.
  MP_ASSERT_OK(graph.CloseAllPacketSources());
  MP_ASSERT_OK(graph.WaitUntilDone());
}

}  // namespace
}  // namespace mediapipe
