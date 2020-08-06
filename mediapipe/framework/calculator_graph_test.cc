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

#include "mediapipe/framework/calculator_graph.h"

#include <pthread.h>

#include <atomic>
#include <ctime>
#include <deque>
#include <map>
#include <memory>
#include <tuple>
#include <utility>
#include <vector>

#include "absl/container/fixed_array.h"
#include "absl/memory/memory.h"
#include "absl/strings/escaping.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "absl/strings/substitute.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/collection_item_id.h"
#include "mediapipe/framework/counter_factory.h"
#include "mediapipe/framework/executor.h"
#include "mediapipe/framework/input_stream_handler.h"
#include "mediapipe/framework/lifetime_tracker.h"
#include "mediapipe/framework/mediapipe_options.pb.h"
#include "mediapipe/framework/output_stream_poller.h"
#include "mediapipe/framework/packet_set.h"
#include "mediapipe/framework/packet_type.h"
#include "mediapipe/framework/port/canonical_errors.h"
#include "mediapipe/framework/port/gmock.h"
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/framework/port/logging.h"
#include "mediapipe/framework/port/parse_text_proto.h"
#include "mediapipe/framework/port/ret_check.h"
#include "mediapipe/framework/port/status.h"
#include "mediapipe/framework/port/status_matchers.h"
#include "mediapipe/framework/status_handler.h"
#include "mediapipe/framework/subgraph.h"
#include "mediapipe/framework/thread_pool_executor.h"
#include "mediapipe/framework/thread_pool_executor.pb.h"
#include "mediapipe/framework/timestamp.h"
#include "mediapipe/framework/tool/sink.h"
#include "mediapipe/framework/tool/status_util.h"
#include "mediapipe/framework/type_map.h"

namespace mediapipe {

namespace {

using testing::ElementsAre;
using testing::HasSubstr;

// Pass packets through. Note that it calls SetOffset() in Process()
// instead of Open().
class SetOffsetInProcessCalculator : public CalculatorBase {
 public:
  static ::mediapipe::Status GetContract(CalculatorContract* cc) {
    cc->Inputs().Index(0).SetAny();
    cc->Outputs().Index(0).SetSameAs(&cc->Inputs().Index(0));
    return ::mediapipe::OkStatus();
  }

  ::mediapipe::Status Open(CalculatorContext* cc) final {
    // Input: arbitrary Packets.
    // Output: copy of the input.
    cc->Outputs().Index(0).SetHeader(cc->Inputs().Index(0).Header());
    return ::mediapipe::OkStatus();
  }

  ::mediapipe::Status Process(CalculatorContext* cc) final {
    cc->SetOffset(TimestampDiff(0));
    cc->GetCounter("PassThrough")->Increment();
    cc->Outputs().Index(0).AddPacket(cc->Inputs().Index(0).Value());
    return ::mediapipe::OkStatus();
  }
};
REGISTER_CALCULATOR(SetOffsetInProcessCalculator);

// A Calculator that outputs the square of its input packet (an int).
class SquareIntCalculator : public CalculatorBase {
 public:
  static ::mediapipe::Status GetContract(CalculatorContract* cc) {
    cc->Inputs().Index(0).Set<int>();
    cc->Outputs().Index(0).SetSameAs(&cc->Inputs().Index(0));
    return ::mediapipe::OkStatus();
  }

  ::mediapipe::Status Open(CalculatorContext* cc) final {
    cc->SetOffset(TimestampDiff(0));
    return ::mediapipe::OkStatus();
  }

  ::mediapipe::Status Process(CalculatorContext* cc) final {
    int value = cc->Inputs().Index(0).Value().Get<int>();
    cc->Outputs().Index(0).Add(new int(value * value), cc->InputTimestamp());
    return ::mediapipe::OkStatus();
  }
};
REGISTER_CALCULATOR(SquareIntCalculator);

// A Calculator that selects an output stream from "OUTPUT:0", "OUTPUT:1", ...,
// using the integer value (0, 1, ...) in the packet on the "SELECT" input
// stream, and passes the packet on the "INPUT" input stream to the selected
// output stream.
//
// This calculator is called "Timed" because it sets the next timestamp bound on
// the unselected outputs.
class DemuxTimedCalculator : public CalculatorBase {
 public:
  static ::mediapipe::Status GetContract(CalculatorContract* cc) {
    RET_CHECK_EQ(cc->Inputs().NumEntries(), 2);
    cc->Inputs().Tag("SELECT").Set<int>();
    PacketType* data_input = &cc->Inputs().Tag("INPUT");
    data_input->SetAny();
    for (CollectionItemId id = cc->Outputs().BeginId("OUTPUT");
         id < cc->Outputs().EndId("OUTPUT"); ++id) {
      cc->Outputs().Get(id).SetSameAs(data_input);
    }
    return ::mediapipe::OkStatus();
  }

  ::mediapipe::Status Open(CalculatorContext* cc) final {
    select_input_ = cc->Inputs().GetId("SELECT", 0);
    data_input_ = cc->Inputs().GetId("INPUT", 0);
    output_base_ = cc->Outputs().GetId("OUTPUT", 0);
    num_outputs_ = cc->Outputs().NumEntries("OUTPUT");
    cc->SetOffset(TimestampDiff(0));
    return ::mediapipe::OkStatus();
  }

  ::mediapipe::Status Process(CalculatorContext* cc) final {
    int select = cc->Inputs().Get(select_input_).Get<int>();
    RET_CHECK(0 <= select && select < num_outputs_);
    const Timestamp next_timestamp_bound =
        cc->InputTimestamp().NextAllowedInStream();
    for (int i = 0; i < num_outputs_; ++i) {
      if (i == select) {
        cc->Outputs()
            .Get(output_base_ + i)
            .AddPacket(cc->Inputs().Get(data_input_).Value());
      } else {
        cc->Outputs()
            .Get(output_base_ + i)
            .SetNextTimestampBound(next_timestamp_bound);
      }
    }
    return ::mediapipe::OkStatus();
  }

 private:
  CollectionItemId select_input_;
  CollectionItemId data_input_;
  CollectionItemId output_base_;
  int num_outputs_ = 0;
};

REGISTER_CALCULATOR(DemuxTimedCalculator);

// A Calculator that selects an input stream from "INPUT:0", "INPUT:1", ...,
// using the integer value (0, 1, ...) in the packet on the "SELECT" input
// stream, and passes the packet on the selected input stream to the "OUTPUT"
// output stream.
//
// This calculator is called "Timed" because it requires next timestamp bound
// propagation on the unselected inputs.
class MuxTimedCalculator : public CalculatorBase {
 public:
  static ::mediapipe::Status GetContract(CalculatorContract* cc) {
    cc->Inputs().Tag("SELECT").Set<int>();
    CollectionItemId data_input_id = cc->Inputs().BeginId("INPUT");
    PacketType* data_input0 = &cc->Inputs().Get(data_input_id);
    data_input0->SetAny();
    ++data_input_id;
    for (; data_input_id < cc->Inputs().EndId("INPUT"); ++data_input_id) {
      cc->Inputs().Get(data_input_id).SetSameAs(data_input0);
    }
    RET_CHECK_EQ(cc->Outputs().NumEntries(), 1);
    cc->Outputs().Tag("OUTPUT").SetSameAs(data_input0);
    return ::mediapipe::OkStatus();
  }

  ::mediapipe::Status Open(CalculatorContext* cc) final {
    select_input_ = cc->Inputs().GetId("SELECT", 0);
    data_input_base_ = cc->Inputs().GetId("INPUT", 0);
    num_data_inputs_ = cc->Inputs().NumEntries("INPUT");
    output_ = cc->Outputs().GetId("OUTPUT", 0);
    cc->SetOffset(TimestampDiff(0));
    return ::mediapipe::OkStatus();
  }

  ::mediapipe::Status Process(CalculatorContext* cc) final {
    int select = cc->Inputs().Get(select_input_).Get<int>();
    RET_CHECK(0 <= select && select < num_data_inputs_);
    cc->Outputs().Get(output_).AddPacket(
        cc->Inputs().Get(data_input_base_ + select).Value());
    return ::mediapipe::OkStatus();
  }

 private:
  CollectionItemId select_input_;
  CollectionItemId data_input_base_;
  int num_data_inputs_ = 0;
  CollectionItemId output_;
};

REGISTER_CALCULATOR(MuxTimedCalculator);

// A Calculator that adds the integer values in the packets in all the input
// streams and outputs the sum to the output stream.
class IntAdderCalculator : public CalculatorBase {
 public:
  static ::mediapipe::Status GetContract(CalculatorContract* cc) {
    for (int i = 0; i < cc->Inputs().NumEntries(); ++i) {
      cc->Inputs().Index(i).Set<int>();
    }
    cc->Outputs().Index(0).Set<int>();
    return ::mediapipe::OkStatus();
  }

  ::mediapipe::Status Open(CalculatorContext* cc) final {
    cc->SetOffset(TimestampDiff(0));
    return ::mediapipe::OkStatus();
  }

  ::mediapipe::Status Process(CalculatorContext* cc) final {
    int sum = 0;
    for (int i = 0; i < cc->Inputs().NumEntries(); ++i) {
      sum += cc->Inputs().Index(i).Get<int>();
    }
    cc->Outputs().Index(0).Add(new int(sum), cc->InputTimestamp());
    return ::mediapipe::OkStatus();
  }
};
REGISTER_CALCULATOR(IntAdderCalculator);

// A Calculator that adds the float values in the packets in all the input
// streams and outputs the sum to the output stream.
class FloatAdderCalculator : public CalculatorBase {
 public:
  static ::mediapipe::Status GetContract(CalculatorContract* cc) {
    for (int i = 0; i < cc->Inputs().NumEntries(); ++i) {
      cc->Inputs().Index(i).Set<float>();
    }
    cc->Outputs().Index(0).Set<float>();
    return ::mediapipe::OkStatus();
  }

  ::mediapipe::Status Open(CalculatorContext* cc) final {
    cc->SetOffset(TimestampDiff(0));
    return ::mediapipe::OkStatus();
  }

  ::mediapipe::Status Process(CalculatorContext* cc) final {
    float sum = 0.0;
    for (int i = 0; i < cc->Inputs().NumEntries(); ++i) {
      sum += cc->Inputs().Index(i).Get<float>();
    }
    cc->Outputs().Index(0).Add(new float(sum), cc->InputTimestamp());
    return ::mediapipe::OkStatus();
  }
};
REGISTER_CALCULATOR(FloatAdderCalculator);

// A Calculator that multiplies the integer values in the packets in all the
// input streams and outputs the product to the output stream.
class IntMultiplierCalculator : public CalculatorBase {
 public:
  static ::mediapipe::Status GetContract(CalculatorContract* cc) {
    for (int i = 0; i < cc->Inputs().NumEntries(); ++i) {
      cc->Inputs().Index(i).Set<int>();
    }
    cc->Outputs().Index(0).Set<int>();
    return ::mediapipe::OkStatus();
  }

  ::mediapipe::Status Open(CalculatorContext* cc) final {
    cc->SetOffset(TimestampDiff(0));
    return ::mediapipe::OkStatus();
  }

  ::mediapipe::Status Process(CalculatorContext* cc) final {
    int product = 1;
    for (int i = 0; i < cc->Inputs().NumEntries(); ++i) {
      product *= cc->Inputs().Index(i).Get<int>();
    }
    cc->Outputs().Index(0).Add(new int(product), cc->InputTimestamp());
    return ::mediapipe::OkStatus();
  }
};
REGISTER_CALCULATOR(IntMultiplierCalculator);

// A Calculator that multiplies the float value in an input packet by a float
// constant scalar (specified in a side packet) and outputs the product to the
// output stream.
class FloatScalarMultiplierCalculator : public CalculatorBase {
 public:
  static ::mediapipe::Status GetContract(CalculatorContract* cc) {
    cc->Inputs().Index(0).Set<float>();
    cc->Outputs().Index(0).Set<float>();
    cc->InputSidePackets().Index(0).Set<float>();
    return ::mediapipe::OkStatus();
  }

  ::mediapipe::Status Open(CalculatorContext* cc) final {
    scalar_ = cc->InputSidePackets().Index(0).Get<float>();
    cc->SetOffset(TimestampDiff(0));
    return ::mediapipe::OkStatus();
  }

  ::mediapipe::Status Process(CalculatorContext* cc) final {
    float value = cc->Inputs().Index(0).Value().Get<float>();
    cc->Outputs().Index(0).Add(new float(scalar_ * value),
                               cc->InputTimestamp());
    return ::mediapipe::OkStatus();
  }

 private:
  float scalar_;
};
REGISTER_CALCULATOR(FloatScalarMultiplierCalculator);

// A Calculator that converts an integer to a float.
class IntToFloatCalculator : public CalculatorBase {
 public:
  static ::mediapipe::Status GetContract(CalculatorContract* cc) {
    cc->Inputs().Index(0).Set<int>();
    cc->Outputs().Index(0).Set<float>();
    return ::mediapipe::OkStatus();
  }

  ::mediapipe::Status Open(CalculatorContext* cc) final {
    cc->SetOffset(TimestampDiff(0));
    return ::mediapipe::OkStatus();
  }

  ::mediapipe::Status Process(CalculatorContext* cc) final {
    int value = cc->Inputs().Index(0).Value().Get<int>();
    cc->Outputs().Index(0).Add(new float(static_cast<float>(value)),
                               cc->InputTimestamp());
    return ::mediapipe::OkStatus();
  }
};
REGISTER_CALCULATOR(IntToFloatCalculator);

template <typename OutputType>
class TypedEmptySourceCalculator : public CalculatorBase {
 public:
  static ::mediapipe::Status GetContract(CalculatorContract* cc) {
    cc->Outputs().Index(0).SetAny();
    return ::mediapipe::OkStatus();
  }

  ::mediapipe::Status Process(CalculatorContext* cc) override {
    cc->Outputs().Index(0).Add(new OutputType(), Timestamp::PostStream());
    return tool::StatusStop();
  }
};
typedef TypedEmptySourceCalculator<std::string> StringEmptySourceCalculator;
typedef TypedEmptySourceCalculator<int> IntEmptySourceCalculator;
REGISTER_CALCULATOR(StringEmptySourceCalculator);
REGISTER_CALCULATOR(IntEmptySourceCalculator);

template <typename InputType>
class TypedSinkCalculator : public CalculatorBase {
 public:
  static ::mediapipe::Status GetContract(CalculatorContract* cc) {
    cc->Inputs().Index(0).Set<InputType>();
    return ::mediapipe::OkStatus();
  }

  ::mediapipe::Status Process(CalculatorContext* cc) override {
    return ::mediapipe::OkStatus();
  }
};
typedef TypedSinkCalculator<std::string> StringSinkCalculator;
typedef TypedSinkCalculator<int> IntSinkCalculator;
REGISTER_CALCULATOR(StringSinkCalculator);
REGISTER_CALCULATOR(IntSinkCalculator);

// Output kNumOutputPackets packets, the value of each being the next
// value in the counter provided as an input side packet.  An optional
// second input side packet will, if true, cause this calculator to
// output the first of the kNumOutputPackets packets during Open.
class GlobalCountSourceCalculator : public CalculatorBase {
 public:
  static const int kNumOutputPackets;

  static ::mediapipe::Status GetContract(CalculatorContract* cc) {
    cc->InputSidePackets().Index(0).Set<std::atomic<int>*>();
    if (cc->InputSidePackets().NumEntries() >= 2) {
      cc->InputSidePackets().Index(1).Set<bool>();
    }
    cc->Outputs().Index(0).Set<int>();
    return ::mediapipe::OkStatus();
  }

  ::mediapipe::Status Open(CalculatorContext* cc) override {
    if (cc->InputSidePackets().NumEntries() >= 2 &&
        cc->InputSidePackets().Index(1).Get<bool>()) {
      OutputOne(cc);
    }
    return ::mediapipe::OkStatus();
  }

  ::mediapipe::Status Process(CalculatorContext* cc) override {
    OutputOne(cc);
    if (local_count_ >= kNumOutputPackets) {
      return tool::StatusStop();
    } else {
      return ::mediapipe::OkStatus();
    }
  }

 private:
  void OutputOne(CalculatorContext* cc) {
    std::atomic<int>* counter =
        cc->InputSidePackets().Index(0).Get<std::atomic<int>*>();
    int count = counter->fetch_add(1, std::memory_order_relaxed);
    cc->Outputs().Index(0).Add(new int(count), Timestamp(local_count_));
    ++local_count_;
  }

  int64 local_count_ = 0;
};
const int GlobalCountSourceCalculator::kNumOutputPackets = 5;
REGISTER_CALCULATOR(GlobalCountSourceCalculator);

static const int kTestSequenceLength = 15;

// Outputs the integers 0, 1, 2, 3, ..., 14, all with timestamp 0.
class TestSequence1SourceCalculator : public CalculatorBase {
 public:
  static ::mediapipe::Status GetContract(CalculatorContract* cc) {
    cc->Outputs().Index(0).Set<int>();
    return ::mediapipe::OkStatus();
  }

  ::mediapipe::Status Process(CalculatorContext* cc) override {
    cc->Outputs().Index(0).Add(new int(count_), Timestamp(0));
    ++count_;
    ++num_outputs_;
    if (num_outputs_ >= kTestSequenceLength) {
      return tool::StatusStop();
    } else {
      return ::mediapipe::OkStatus();
    }
  }

 private:
  int count_ = 0;
  int num_outputs_ = 0;
};
REGISTER_CALCULATOR(TestSequence1SourceCalculator);

// Outputs the integers 1, 2, 3, 4, ..., 15, with decreasing timestamps
// 100, 99, 98, 97, ....
class TestSequence2SourceCalculator : public CalculatorBase {
 public:
  static ::mediapipe::Status GetContract(CalculatorContract* cc) {
    cc->Outputs().Index(0).Set<int>();
    return ::mediapipe::OkStatus();
  }

  ::mediapipe::Status Process(CalculatorContext* cc) override {
    cc->Outputs().Index(0).Add(new int(count_), Timestamp(timestamp_));
    ++count_;
    ++num_outputs_;
    --timestamp_;
    if (num_outputs_ >= kTestSequenceLength) {
      return tool::StatusStop();
    } else {
      return ::mediapipe::OkStatus();
    }
  }

 private:
  int count_ = 1;
  int num_outputs_ = 0;
  int timestamp_ = 100;
};
REGISTER_CALCULATOR(TestSequence2SourceCalculator);

// Outputs the integers 0, 1, 2 repeatedly for a total of 15 outputs.
class Modulo3SourceCalculator : public CalculatorBase {
 public:
  static ::mediapipe::Status GetContract(CalculatorContract* cc) {
    cc->Outputs().Index(0).Set<int>();
    return ::mediapipe::OkStatus();
  }

  ::mediapipe::Status Process(CalculatorContext* cc) override {
    cc->Outputs().Index(0).Add(new int(count_ % 3), Timestamp(count_ % 3));
    ++count_;
    ++num_outputs_;
    if (num_outputs_ >= kTestSequenceLength) {
      return tool::StatusStop();
    } else {
      return ::mediapipe::OkStatus();
    }
  }

 private:
  int count_ = 0;
  int num_outputs_ = 0;
};
REGISTER_CALCULATOR(Modulo3SourceCalculator);

// A source calculator that outputs 100 packets all at once and stops. The
// number of output packets (100) is deliberately chosen to be equal to
// max_queue_size, which fills the input streams connected to this source
// calculator and causes the MediaPipe scheduler to throttle this source
// calculator.
class OutputAllSourceCalculator : public CalculatorBase {
 public:
  static constexpr int kNumOutputPackets = 100;

  static ::mediapipe::Status GetContract(CalculatorContract* cc) {
    cc->Outputs().Index(0).Set<int>();
    return ::mediapipe::OkStatus();
  }

  ::mediapipe::Status Process(CalculatorContext* cc) override {
    for (int i = 0; i < kNumOutputPackets; ++i) {
      cc->Outputs().Index(0).Add(new int(0), Timestamp(i));
    }
    return tool::StatusStop();
  }
};
REGISTER_CALCULATOR(OutputAllSourceCalculator);

// A source calculator that outputs one packet at a time. The total number of
// output packets needs to be large enough to eventually fill an input stream
// connected to this source calculator and to force the MediaPipe scheduler to
// run this source calculator as a throttled source when the graph cannot make
// progress.
class OutputOneAtATimeSourceCalculator : public CalculatorBase {
 public:
  static constexpr int kNumOutputPackets = 1000;

  static ::mediapipe::Status GetContract(CalculatorContract* cc) {
    cc->Outputs().Index(0).Set<int>();
    return ::mediapipe::OkStatus();
  }

  ::mediapipe::Status Process(CalculatorContext* cc) override {
    if (index_ < kNumOutputPackets) {
      cc->Outputs().Index(0).Add(new int(0), Timestamp(index_));
      ++index_;
      return ::mediapipe::OkStatus();
    }
    return tool::StatusStop();
  }

 private:
  int index_ = 0;
};
REGISTER_CALCULATOR(OutputOneAtATimeSourceCalculator);

// A calculator that passes through one out of every 101 input packets and
// discards the rest. The decimation ratio (101) is carefully chosen to be
// greater than max_queue_size (100) so that an input stream parallel to the
// input stream connected to this calculator can become full.
class DecimatorCalculator : public CalculatorBase {
 public:
  static constexpr int kDecimationRatio = 101;

  static ::mediapipe::Status GetContract(CalculatorContract* cc) {
    cc->Inputs().Index(0).SetAny();
    cc->Outputs().Index(0).SetSameAs(&cc->Inputs().Index(0));
    return ::mediapipe::OkStatus();
  }

  ::mediapipe::Status Process(CalculatorContext* cc) final {
    if (index_ % kDecimationRatio == 0) {
      cc->Outputs().Index(0).AddPacket(cc->Inputs().Index(0).Value());
    }
    ++index_;
    return ::mediapipe::OkStatus();
  }

 private:
  int index_ = 0;
};
REGISTER_CALCULATOR(DecimatorCalculator);

// An error will be produced in Open() if ERROR_ON_OPEN is true. Otherwise,
// this calculator simply passes its input packets through, unchanged.
class ErrorOnOpenCalculator : public CalculatorBase {
 public:
  static ::mediapipe::Status GetContract(CalculatorContract* cc) {
    cc->Inputs().Index(0).SetAny();
    cc->Outputs().Index(0).SetSameAs(&cc->Inputs().Index(0));
    cc->InputSidePackets().Tag("ERROR_ON_OPEN").Set<bool>();
    return ::mediapipe::OkStatus();
  }

  ::mediapipe::Status Open(CalculatorContext* cc) final {
    if (cc->InputSidePackets().Tag("ERROR_ON_OPEN").Get<bool>()) {
      return ::mediapipe::NotFoundError("expected error");
    }
    return ::mediapipe::OkStatus();
  }

  ::mediapipe::Status Process(CalculatorContext* cc) final {
    cc->Outputs().Index(0).AddPacket(cc->Inputs().Index(0).Value());
    return ::mediapipe::OkStatus();
  }
};
REGISTER_CALCULATOR(ErrorOnOpenCalculator);

// A calculator that outputs an initial packet of value 0 at time 0 in the
// Open() method, and then delays each input packet by one time unit in the
// Process() method. The input stream and output stream have the integer type.
class UnitDelayCalculator : public CalculatorBase {
 public:
  static ::mediapipe::Status GetContract(CalculatorContract* cc) {
    cc->Inputs().Index(0).Set<int>();
    cc->Outputs().Index(0).Set<int>();
    return ::mediapipe::OkStatus();
  }

  ::mediapipe::Status Open(CalculatorContext* cc) final {
    cc->Outputs().Index(0).Add(new int(0), Timestamp(0));
    return ::mediapipe::OkStatus();
  }

  ::mediapipe::Status Process(CalculatorContext* cc) final {
    const Packet& packet = cc->Inputs().Index(0).Value();
    cc->Outputs().Index(0).AddPacket(
        packet.At(packet.Timestamp().NextAllowedInStream()));
    return ::mediapipe::OkStatus();
  }
};
REGISTER_CALCULATOR(UnitDelayCalculator);

class UnitDelayUntimedCalculator : public CalculatorBase {
 public:
  static ::mediapipe::Status GetContract(CalculatorContract* cc) {
    cc->Inputs().Index(0).Set<int>();
    cc->Outputs().Index(0).Set<int>();
    return ::mediapipe::OkStatus();
  }

  ::mediapipe::Status Open(CalculatorContext* cc) final {
    cc->Outputs().Index(0).Add(new int(0), Timestamp::Min());
    return ::mediapipe::OkStatus();
  }

  ::mediapipe::Status Process(CalculatorContext* cc) final {
    cc->Outputs().Index(0).AddPacket(cc->Inputs().Index(0).Value());
    return ::mediapipe::OkStatus();
  }
};
REGISTER_CALCULATOR(UnitDelayUntimedCalculator);

class FloatUnitDelayCalculator : public CalculatorBase {
 public:
  static ::mediapipe::Status GetContract(CalculatorContract* cc) {
    cc->Inputs().Index(0).Set<float>();
    cc->Outputs().Index(0).Set<float>();
    return ::mediapipe::OkStatus();
  }

  ::mediapipe::Status Open(CalculatorContext* cc) final {
    cc->Outputs().Index(0).Add(new float(0.0), Timestamp(0));
    return ::mediapipe::OkStatus();
  }

  ::mediapipe::Status Process(CalculatorContext* cc) final {
    const Packet& packet = cc->Inputs().Index(0).Value();
    cc->Outputs().Index(0).AddPacket(
        packet.At(packet.Timestamp().NextAllowedInStream()));
    return ::mediapipe::OkStatus();
  }
};
REGISTER_CALCULATOR(FloatUnitDelayCalculator);

// A sink calculator that asserts its input stream is empty in Open() and
// discards input packets in Process().
class AssertEmptyInputInOpenCalculator : public CalculatorBase {
 public:
  static ::mediapipe::Status GetContract(CalculatorContract* cc) {
    cc->Inputs().Index(0).SetAny();
    return ::mediapipe::OkStatus();
  }

  ::mediapipe::Status Open(CalculatorContext* cc) final {
    RET_CHECK(cc->Inputs().Index(0).Value().IsEmpty());
    RET_CHECK_EQ(cc->Inputs().Index(0).Value().Timestamp(), Timestamp::Unset());
    return ::mediapipe::OkStatus();
  }

  ::mediapipe::Status Process(CalculatorContext* cc) final {
    return ::mediapipe::OkStatus();
  }
};
REGISTER_CALCULATOR(AssertEmptyInputInOpenCalculator);

// A slow sink calculator that expects 10 input integers with the values
// 0, 1, ..., 9.
class SlowCountingSinkCalculator : public CalculatorBase {
 public:
  static ::mediapipe::Status GetContract(CalculatorContract* cc) {
    cc->Inputs().Index(0).Set<int>();
    return ::mediapipe::OkStatus();
  }

  ::mediapipe::Status Process(CalculatorContext* cc) override {
    absl::SleepFor(absl::Milliseconds(10));
    int value = cc->Inputs().Index(0).Get<int>();
    CHECK_EQ(value, counter_);
    ++counter_;
    return ::mediapipe::OkStatus();
  }

  ::mediapipe::Status Close(CalculatorContext* cc) override {
    CHECK_EQ(10, counter_);
    return ::mediapipe::OkStatus();
  }

 private:
  int counter_ = 0;
};
REGISTER_CALCULATOR(SlowCountingSinkCalculator);

template <typename InputType>
class TypedStatusHandler : public StatusHandler {
 public:
  ~TypedStatusHandler() override = 0;
  static ::mediapipe::Status FillExpectations(
      const MediaPipeOptions& extendable_options,
      PacketTypeSet* input_side_packets) {
    input_side_packets->Index(0).Set<InputType>();
    return ::mediapipe::OkStatus();
  }

  static ::mediapipe::Status HandlePreRunStatus(
      const MediaPipeOptions& extendable_options,
      const PacketSet& input_side_packets,  //
      const ::mediapipe::Status& pre_run_status) {
    return ::mediapipe::OkStatus();
  }

  static ::mediapipe::Status HandleStatus(
      const MediaPipeOptions& extendable_options,
      const PacketSet& input_side_packets,  //
      const ::mediapipe::Status& run_status) {
    return ::mediapipe::OkStatus();
  }
};
typedef TypedStatusHandler<std::string> StringStatusHandler;
typedef TypedStatusHandler<uint32> Uint32StatusHandler;
REGISTER_STATUS_HANDLER(StringStatusHandler);
REGISTER_STATUS_HANDLER(Uint32StatusHandler);

// A std::string generator that will succeed.
class StaticCounterStringGenerator : public PacketGenerator {
 public:
  static ::mediapipe::Status FillExpectations(
      const PacketGeneratorOptions& extendable_options,
      PacketTypeSet* input_side_packets, PacketTypeSet* output_side_packets) {
    for (int i = 0; i < input_side_packets->NumEntries(); ++i) {
      input_side_packets->Index(i).SetAny();
    }
    output_side_packets->Index(0).Set<std::string>();
    return ::mediapipe::OkStatus();
  }

  static ::mediapipe::Status Generate(
      const PacketGeneratorOptions& extendable_options,
      const PacketSet& input_side_packets, PacketSet* output_side_packets) {
    output_side_packets->Index(0) = MakePacket<std::string>("fixed_string");
    ++num_packets_generated_;
    return ::mediapipe::OkStatus();
  }

  static int NumPacketsGenerated() { return num_packets_generated_; }

 private:
  static int num_packets_generated_;
};

int StaticCounterStringGenerator::num_packets_generated_ = 0;

REGISTER_PACKET_GENERATOR(StaticCounterStringGenerator);

// A failing PacketGenerator and Calculator to verify that status handlers get
// called. Both claim to output strings but instead always fail.
class FailingPacketGenerator : public PacketGenerator {
 public:
  static ::mediapipe::Status FillExpectations(
      const PacketGeneratorOptions& extendable_options,
      PacketTypeSet* input_side_packets, PacketTypeSet* output_side_packets) {
    for (int i = 0; i < input_side_packets->NumEntries(); ++i) {
      input_side_packets->Index(i).SetAny();
    }
    output_side_packets->Index(0).Set<std::string>();
    return ::mediapipe::OkStatus();
  }

  static ::mediapipe::Status Generate(
      const PacketGeneratorOptions& extendable_options,
      const PacketSet& input_side_packets, PacketSet* output_side_packets) {
    return ::mediapipe::UnknownError("this always fails.");
  }
};
REGISTER_PACKET_GENERATOR(FailingPacketGenerator);

// Passes the integer through if it is positive.
class EnsurePositivePacketGenerator : public PacketGenerator {
 public:
  static ::mediapipe::Status FillExpectations(
      const PacketGeneratorOptions& extendable_options,
      PacketTypeSet* input_side_packets, PacketTypeSet* output_side_packets) {
    for (int i = 0; i < input_side_packets->NumEntries(); ++i) {
      input_side_packets->Index(i).Set<int>();
      output_side_packets->Index(i).Set<int>();
    }
    return ::mediapipe::OkStatus();
  }

  static ::mediapipe::Status Generate(
      const PacketGeneratorOptions& extendable_options,
      const PacketSet& input_side_packets, PacketSet* output_side_packets) {
    for (int i = 0; i < input_side_packets.NumEntries(); ++i) {
      if (input_side_packets.Index(i).Get<int>() > 0) {
        output_side_packets->Index(i) = input_side_packets.Index(i);
      } else {
        return ::mediapipe::UnknownError(
            absl::StrCat("Integer ", i, " was not positive."));
      }
    }
    return ::mediapipe::OkStatus();
  }
};
REGISTER_PACKET_GENERATOR(EnsurePositivePacketGenerator);

// A Status handler which takes an int side packet and fails in pre run
// if that packet is FailableStatusHandler::kFailPreRun and fails post
// run if that value is FailableStatusHandler::kFailPostRun.  If the
// int is any other value then no failures occur.
class FailableStatusHandler : public StatusHandler {
 public:
  enum {
    kOk = 0,
    kFailPreRun = 1,
    kFailPostRun = 2,
  };

  static ::mediapipe::Status FillExpectations(
      const MediaPipeOptions& extendable_options,
      PacketTypeSet* input_side_packets) {
    input_side_packets->Index(0).Set<int>();
    return ::mediapipe::OkStatus();
  }
  static ::mediapipe::Status HandlePreRunStatus(
      const MediaPipeOptions& extendable_options,
      const PacketSet& input_side_packets,
      const ::mediapipe::Status& pre_run_status) {
    if (input_side_packets.Index(0).Get<int>() == kFailPreRun) {
      return ::mediapipe::UnknownError(
          "FailableStatusHandler failing pre run as intended.");
    } else {
      return ::mediapipe::OkStatus();
    }
  }
  static ::mediapipe::Status HandleStatus(
      const MediaPipeOptions& extendable_options,
      const PacketSet& input_side_packets,
      const ::mediapipe::Status& run_status) {
    if (input_side_packets.Index(0).Get<int>() == kFailPostRun) {
      return ::mediapipe::UnknownError(
          "FailableStatusHandler failing post run as intended.");
    } else {
      return ::mediapipe::OkStatus();
    }
  }
};
REGISTER_STATUS_HANDLER(FailableStatusHandler);

class FailingSourceCalculator : public CalculatorBase {
 public:
  static ::mediapipe::Status GetContract(CalculatorContract* cc) {
    cc->Outputs().Index(0).Set<std::string>();
    return ::mediapipe::OkStatus();
  }

  ::mediapipe::Status Process(CalculatorContext* cc) override {
    return ::mediapipe::UnknownError("this always fails.");
  }
};
REGISTER_CALCULATOR(FailingSourceCalculator);

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

// This calculator posts to a semaphore when it starts its Process method,
// and waits on a different semaphore before returning from Process.
// This allows a test to run code when the calculator is running Process
// without having to depend on any specific timing.
class SemaphoreCalculator : public CalculatorBase {
 public:
  using Semaphore = AtomicSemaphore;

  static ::mediapipe::Status GetContract(CalculatorContract* cc) {
    cc->Inputs().Index(0).SetAny();
    cc->Outputs().Index(0).SetSameAs(&cc->Inputs().Index(0));
    cc->InputSidePackets().Tag("POST_SEM").Set<Semaphore*>();
    cc->InputSidePackets().Tag("WAIT_SEM").Set<Semaphore*>();
    return ::mediapipe::OkStatus();
  }

  ::mediapipe::Status Open(CalculatorContext* cc) override {
    cc->SetOffset(TimestampDiff(0));
    return ::mediapipe::OkStatus();
  }

  ::mediapipe::Status Process(CalculatorContext* cc) override {
    cc->InputSidePackets().Tag("POST_SEM").Get<Semaphore*>()->Release(1);
    cc->InputSidePackets().Tag("WAIT_SEM").Get<Semaphore*>()->Acquire(1);
    cc->Outputs().Index(0).AddPacket(cc->Inputs().Index(0).Value());
    return ::mediapipe::OkStatus();
  }
};
REGISTER_CALCULATOR(SemaphoreCalculator);

// A calculator that has no input streams and output streams, runs only once,
// and takes 20 milliseconds to run.
class OneShot20MsCalculator : public CalculatorBase {
 public:
  static ::mediapipe::Status GetContract(CalculatorContract* cc) {
    return ::mediapipe::OkStatus();
  }

  ::mediapipe::Status Process(CalculatorContext* cc) override {
    absl::SleepFor(absl::Milliseconds(20));
    return tool::StatusStop();
  }
};
REGISTER_CALCULATOR(OneShot20MsCalculator);

// A source calculator that outputs a packet containing the return value of
// pthread_self() (the pthread id of the current thread).
class PthreadSelfSourceCalculator : public CalculatorBase {
 public:
  static ::mediapipe::Status GetContract(CalculatorContract* cc) {
    cc->Outputs().Index(0).Set<pthread_t>();
    return ::mediapipe::OkStatus();
  }

  ::mediapipe::Status Process(CalculatorContext* cc) override {
    cc->Outputs().Index(0).AddPacket(
        MakePacket<pthread_t>(pthread_self()).At(Timestamp(0)));
    return tool::StatusStop();
  }
};
REGISTER_CALCULATOR(PthreadSelfSourceCalculator);

// A source calculator for testing the Calculator::InputTimestamp() method.
// It outputs five int packets with timestamps 0, 1, 2, 3, 4.
class CheckInputTimestampSourceCalculator : public CalculatorBase {
 public:
  static ::mediapipe::Status GetContract(CalculatorContract* cc) {
    cc->Outputs().Index(0).Set<int>();
    return ::mediapipe::OkStatus();
  }

  // InputTimestamp() returns Timestamp::Unstarted() in Open() for both source
  // and non-source nodes.
  ::mediapipe::Status Open(CalculatorContext* cc) final {
    RET_CHECK_EQ(cc->InputTimestamp(), Timestamp::Unstarted());
    return ::mediapipe::OkStatus();
  }

  // InputTimestamp() always returns Timestamp(0) in Process() for source
  // nodes.
  ::mediapipe::Status Process(CalculatorContext* cc) final {
    RET_CHECK_EQ(cc->InputTimestamp(), Timestamp(0));
    cc->Outputs().Index(0).Add(new int(count_), Timestamp(count_));
    ++count_;
    if (count_ >= 5) {
      return tool::StatusStop();
    } else {
      return ::mediapipe::OkStatus();
    }
  }

  // InputTimestamp() returns Timestamp::Done() in Close() for both source
  // and non-source nodes.
  ::mediapipe::Status Close(CalculatorContext* cc) final {
    // Must use CHECK instead of RET_CHECK in Close(), because the framework
    // may call the Close() method of a source node with .IgnoreError().
    CHECK_EQ(cc->InputTimestamp(), Timestamp::Done());
    return ::mediapipe::OkStatus();
  }

 private:
  int count_ = 0;
};
REGISTER_CALCULATOR(CheckInputTimestampSourceCalculator);

// A sink calculator for testing the Calculator::InputTimestamp() method.
// It expects to consume the output of a CheckInputTimestampSourceCalculator.
class CheckInputTimestampSinkCalculator : public CalculatorBase {
 public:
  static ::mediapipe::Status GetContract(CalculatorContract* cc) {
    cc->Inputs().Index(0).Set<int>();
    return ::mediapipe::OkStatus();
  }

  // InputTimestamp() returns Timestamp::Unstarted() in Open() for both source
  // and non-source nodes.
  ::mediapipe::Status Open(CalculatorContext* cc) final {
    RET_CHECK_EQ(cc->InputTimestamp(), Timestamp::Unstarted());
    return ::mediapipe::OkStatus();
  }

  // InputTimestamp() returns the timestamp of input packets in Process() for
  // non-source nodes.
  ::mediapipe::Status Process(CalculatorContext* cc) final {
    RET_CHECK_EQ(cc->InputTimestamp(),
                 cc->Inputs().Index(0).Value().Timestamp());
    RET_CHECK_EQ(cc->InputTimestamp(), Timestamp(count_));
    ++count_;
    return ::mediapipe::OkStatus();
  }

  // InputTimestamp() returns Timestamp::Done() in Close() for both source
  // and non-source nodes.
  ::mediapipe::Status Close(CalculatorContext* cc) final {
    RET_CHECK_EQ(cc->InputTimestamp(), Timestamp::Done());
    return ::mediapipe::OkStatus();
  }

 private:
  int count_ = 0;
};
REGISTER_CALCULATOR(CheckInputTimestampSinkCalculator);

// A source calculator for testing the Calculator::InputTimestamp() method.
// It outputs int packets with timestamps 0, 1, 2, ... until being closed by
// the framework.
class CheckInputTimestamp2SourceCalculator : public CalculatorBase {
 public:
  static ::mediapipe::Status GetContract(CalculatorContract* cc) {
    cc->Outputs().Index(0).Set<int>();
    return ::mediapipe::OkStatus();
  }

  // InputTimestamp() returns Timestamp::Unstarted() in Open() for both source
  // and non-source nodes.
  ::mediapipe::Status Open(CalculatorContext* cc) final {
    RET_CHECK_EQ(cc->InputTimestamp(), Timestamp::Unstarted());
    return ::mediapipe::OkStatus();
  }

  // InputTimestamp() always returns Timestamp(0) in Process() for source
  // nodes.
  ::mediapipe::Status Process(CalculatorContext* cc) final {
    RET_CHECK_EQ(cc->InputTimestamp(), Timestamp(0));
    cc->Outputs().Index(0).Add(new int(count_), Timestamp(count_));
    ++count_;
    return ::mediapipe::OkStatus();
  }

  // InputTimestamp() returns Timestamp::Done() in Close() for both source
  // and non-source nodes.
  ::mediapipe::Status Close(CalculatorContext* cc) final {
    // Must use CHECK instead of RET_CHECK in Close(), because the framework
    // may call the Close() method of a source node with .IgnoreError().
    CHECK_EQ(cc->InputTimestamp(), Timestamp::Done());
    return ::mediapipe::OkStatus();
  }

 private:
  int count_ = 0;
};
REGISTER_CALCULATOR(CheckInputTimestamp2SourceCalculator);

// A sink calculator for testing the Calculator::InputTimestamp() method.
// It expects to consume the output of a CheckInputTimestamp2SourceCalculator.
// It returns tool::StatusStop() after consuming five input packets.
class CheckInputTimestamp2SinkCalculator : public CalculatorBase {
 public:
  static ::mediapipe::Status GetContract(CalculatorContract* cc) {
    cc->Inputs().Index(0).Set<int>();
    return ::mediapipe::OkStatus();
  }

  // InputTimestamp() returns Timestamp::Unstarted() in Open() for both source
  // and non-source nodes.
  ::mediapipe::Status Open(CalculatorContext* cc) final {
    RET_CHECK_EQ(cc->InputTimestamp(), Timestamp::Unstarted());
    return ::mediapipe::OkStatus();
  }

  // InputTimestamp() returns the timestamp of input packets in Process() for
  // non-source nodes.
  ::mediapipe::Status Process(CalculatorContext* cc) final {
    RET_CHECK_EQ(cc->InputTimestamp(),
                 cc->Inputs().Index(0).Value().Timestamp());
    RET_CHECK_EQ(cc->InputTimestamp(), Timestamp(count_));
    ++count_;
    if (count_ >= 5) {
      return tool::StatusStop();
    } else {
      return ::mediapipe::OkStatus();
    }
  }

  // InputTimestamp() returns Timestamp::Done() in Close() for both source
  // and non-source nodes.
  ::mediapipe::Status Close(CalculatorContext* cc) final {
    RET_CHECK_EQ(cc->InputTimestamp(), Timestamp::Done());
    return ::mediapipe::OkStatus();
  }

 private:
  int count_ = 0;
};
REGISTER_CALCULATOR(CheckInputTimestamp2SinkCalculator);

// Takes an input stream packet and passes it (with timestamp removed) as an
// output side packet.
class OutputSidePacketInProcessCalculator : public CalculatorBase {
 public:
  static ::mediapipe::Status GetContract(CalculatorContract* cc) {
    cc->Inputs().Index(0).SetAny();
    cc->OutputSidePackets().Index(0).SetSameAs(&cc->Inputs().Index(0));
    return ::mediapipe::OkStatus();
  }

  ::mediapipe::Status Process(CalculatorContext* cc) final {
    cc->OutputSidePackets().Index(0).Set(
        cc->Inputs().Index(0).Value().At(Timestamp::Unset()));
    return ::mediapipe::OkStatus();
  }
};
REGISTER_CALCULATOR(OutputSidePacketInProcessCalculator);

// A calculator checks if either of two input streams contains a packet and
// sends the packet to the single output stream with the same timestamp.
class SimpleMuxCalculator : public CalculatorBase {
 public:
  static ::mediapipe::Status GetContract(CalculatorContract* cc) {
    RET_CHECK_EQ(cc->Inputs().NumEntries(), 2);
    cc->Inputs().Index(0).SetAny();
    cc->Inputs().Index(1).SetSameAs(&cc->Inputs().Index(0));
    RET_CHECK_EQ(cc->Outputs().NumEntries(), 1);
    cc->Outputs().Index(0).SetSameAs(&cc->Inputs().Index(0));
    return ::mediapipe::OkStatus();
  }

  ::mediapipe::Status Open(CalculatorContext* cc) final {
    data_input_base_ = cc->Inputs().BeginId();
    return ::mediapipe::OkStatus();
  }

  ::mediapipe::Status Process(CalculatorContext* cc) final {
    int select_packet_index = -1;
    if (!cc->Inputs().Index(0).IsEmpty()) {
      select_packet_index = 0;
    } else if (!cc->Inputs().Index(1).IsEmpty()) {
      select_packet_index = 1;
    }
    if (select_packet_index != -1) {
      cc->Outputs().Index(0).AddPacket(
          cc->Inputs().Get(data_input_base_ + select_packet_index).Value());
    }
    return ::mediapipe::OkStatus();
  }

 private:
  CollectionItemId data_input_base_;
};
REGISTER_CALCULATOR(SimpleMuxCalculator);

// Mock status handler that reports the number of times HandleStatus was called
// by modifying the int in its input side packet.
class IncrementingStatusHandler : public StatusHandler {
 public:
  static ::mediapipe::Status FillExpectations(
      const MediaPipeOptions& extendable_options,
      PacketTypeSet* input_side_packets) {
    input_side_packets->Tag("EXTRA").SetAny().Optional();
    input_side_packets->Tag("COUNTER1").Set<std::unique_ptr<int>>();
    input_side_packets->Tag("COUNTER2").Set<std::unique_ptr<int>>();
    return ::mediapipe::OkStatus();
  }

  static ::mediapipe::Status HandlePreRunStatus(
      const MediaPipeOptions& extendable_options,
      const PacketSet& input_side_packets,  //
      const ::mediapipe::Status& pre_run_status) {
    int* counter = GetFromUniquePtr<int>(input_side_packets.Tag("COUNTER1"));
    (*counter)++;
    return pre_run_status_result_;
  }

  static ::mediapipe::Status HandleStatus(
      const MediaPipeOptions& extendable_options,
      const PacketSet& input_side_packets,  //
      const ::mediapipe::Status& run_status) {
    int* counter = GetFromUniquePtr<int>(input_side_packets.Tag("COUNTER2"));
    (*counter)++;
    return post_run_status_result_;
  }

  static void SetPreRunStatusResult(const ::mediapipe::Status& status) {
    pre_run_status_result_ = status;
  }

  static void SetPostRunStatusResult(const ::mediapipe::Status& status) {
    post_run_status_result_ = status;
  }

 private:
  // Return values of HandlePreRunStatus() and HandleSTatus(), respectively.
  static ::mediapipe::Status pre_run_status_result_;
  static ::mediapipe::Status post_run_status_result_;
};

::mediapipe::Status IncrementingStatusHandler::pre_run_status_result_ =
    ::mediapipe::OkStatus();
::mediapipe::Status IncrementingStatusHandler::post_run_status_result_ =
    ::mediapipe::OkStatus();

REGISTER_STATUS_HANDLER(IncrementingStatusHandler);

// A simple executor that runs tasks directly on the current thread.
// NOTE: If CurrentThreadExecutor is used, some CalculatorGraph methods may
// behave differently. For example, CalculatorGraph::StartRun will run the
// graph rather than returning immediately after starting the graph.
// Similarly, CalculatorGraph::AddPacketToInputStream will run the graph
// (until it's idle) rather than returning immediately after adding the packet
// to the graph input stream.
class CurrentThreadExecutor : public Executor {
 public:
  ~CurrentThreadExecutor() override {
    CHECK(!executing_);
    CHECK(tasks_.empty());
  }

  void Schedule(std::function<void()> task) override {
    if (executing_) {
      // Queue the task for later execution (after the currently-running task
      // returns) rather than running the task immediately. This is especially
      // important for a source node (which can be rescheduled immediately after
      // running) to avoid an indefinitely-deep call stack.
      tasks_.emplace_back(std::move(task));
    } else {
      CHECK(tasks_.empty());
      executing_ = true;
      task();
      while (!tasks_.empty()) {
        task = tasks_.front();
        tasks_.pop_front();
        task();
      }
      executing_ = false;
    }
  }

 private:
  // True if the executor is executing tasks.
  bool executing_ = false;
  // The tasks to execute.
  std::deque<std::function<void()>> tasks_;
};

// Returns a CalculatorGraphConfig used by tests.
CalculatorGraphConfig GetConfig() {
  CalculatorGraphConfig config =
      ::mediapipe::ParseTextProtoOrDie<CalculatorGraphConfig>(R"(
        # The graph configuration. We list the nodes in an arbitrary (not
        # topologically-sorted) order to verify that CalculatorGraph can
        # handle such configurations.
        node {
          calculator: "RangeCalculator"
          output_stream: "range3"
          output_stream: "range3_sum"
          output_stream: "range3_mean"
          input_side_packet: "node_3_converted"
        }
        node {
          calculator: "RangeCalculator"
          output_stream: "range5"
          output_stream: "range5_sum"
          output_stream: "range5_mean"
          input_side_packet: "node_5_converted"
        }
        node {
          calculator: "MergeCalculator"
          input_stream: "range3"
          input_stream: "range5_copy"
          output_stream: "merge"
        }
        node {
          calculator: "MergeCalculator"
          input_stream: "range3_sum"
          input_stream: "range5_sum"
          output_stream: "merge_sum"
        }
        node {
          calculator: "PassThroughCalculator"
          input_stream: "range3_stddev"
          input_stream: "range5_stddev"
          output_stream: "range3_stddev_2"
          output_stream: "range5_stddev_2"
        }
        node {
          calculator: "PassThroughCalculator"
          input_stream: "A:range3_stddev_2"
          input_stream: "range5_stddev_2"
          output_stream: "A:range3_stddev_3"
          output_stream: "range5_stddev_3"
        }
        node {
          calculator: "PassThroughCalculator"
          input_stream: "B:range3_stddev_3"
          input_stream: "B:1:range5_stddev_3"
          output_stream: "B:range3_stddev_4"
          output_stream: "B:1:range5_stddev_4"
        }
        node {
          calculator: "MergeCalculator"
          input_stream: "range3_stddev_4"
          input_stream: "range5_stddev_4"
          output_stream: "merge_stddev"
        }
        node {
          calculator: "StdDevCalculator"
          input_stream: "DATA:range3"
          input_stream: "MEAN:range3_mean"
          output_stream: "range3_stddev"
        }
        node {
          calculator: "StdDevCalculator"
          input_stream: "DATA:range5"
          input_stream: "MEAN:range5_mean"
          output_stream: "range5_stddev"
        }
        node {
          name: "copy_range5"
          calculator: "PassThroughCalculator"
          input_stream: "range5"
          output_stream: "range5_copy"
        }
        node {
          calculator: "SaverCalculator"
          input_stream: "merge"
          output_stream: "final"
        }
        node {
          calculator: "SaverCalculator"
          input_stream: "merge_sum"
          output_stream: "final_sum"
        }
        node {
          calculator: "SaverCalculator"
          input_stream: "merge_stddev"
          output_stream: "final_stddev"
        }
        packet_generator {
          packet_generator: "IntSplitterPacketGenerator"
          input_side_packet: "node_3"
          output_side_packet: "node_3_converted"
        }
        packet_generator {
          packet_generator: "TaggedIntSplitterPacketGenerator"
          input_side_packet: "node_5"
          output_side_packet: "HIGH:unused_high"
          output_side_packet: "LOW:unused_low"
          output_side_packet: "PAIR:node_5_converted"
        }
      )");
  return config;
}

// |graph| points to an empty CalculatorGraph object created by the default
// constructor, before CalculatorGraph::Initialize() is called.
void RunComprehensiveTest(CalculatorGraph* graph,
                          const CalculatorGraphConfig& the_config,
                          bool define_node_5) {
  CalculatorGraphConfig proto(the_config);
  Packet dumped_final_sum_packet;
  Packet dumped_final_packet;
  Packet dumped_final_stddev_packet;
  tool::AddPostStreamPacketSink("final", &proto, &dumped_final_packet);
  tool::AddPostStreamPacketSink("final_sum", &proto, &dumped_final_sum_packet);
  tool::AddPostStreamPacketSink("final_stddev", &proto,
                                &dumped_final_stddev_packet);
  MP_ASSERT_OK(graph->Initialize(proto));

  std::map<std::string, Packet> extra_side_packets;
  extra_side_packets.emplace("node_3", Adopt(new uint64((15LL << 32) | 3)));
  if (define_node_5) {
    extra_side_packets.emplace("node_5", Adopt(new uint64((15LL << 32) | 5)));
  }

  // Call graph->Run() several times, to make sure that the appropriate
  // cleanup happens between iterations.
  for (int iteration = 0; iteration < 2; ++iteration) {
    LOG(INFO) << "Loop iteration " << iteration;
    dumped_final_sum_packet = Packet();
    dumped_final_stddev_packet = Packet();
    dumped_final_packet = Packet();
    MP_ASSERT_OK(graph->Run(extra_side_packets));
    // The merger will output the timestamp and all ints output from
    // the range calculators.  The saver will concatenate together the
    // strings with a '/' deliminator.
    EXPECT_EQ(
        "Timestamp(0) 300 500/"
        "Timestamp(3) 301 empty/"
        "Timestamp(5) empty 501/"
        "Timestamp(6) 302 empty/"
        "Timestamp(9) 303 empty/"
        "Timestamp(10) empty 502/"
        "Timestamp(12) 304 empty/"
        "Timestamp(15) 305 503",
        dumped_final_packet.Get<std::string>());
    // Verify that the headers got set correctly.
    EXPECT_EQ(
        "RangeCalculator3 RangeCalculator5",
        graph->FindOutputStreamManager("merge")->Header().Get<std::string>());
    // Verify that sum packets get correctly processed.
    // (The first is a sum of all the 3's output and the second of all
    // the 5's).
    EXPECT_EQ(absl::StrCat(Timestamp::PostStream().DebugString(), " 1815 2006"),
              dumped_final_sum_packet.Get<std::string>());
    EXPECT_EQ(4 * (iteration + 1), graph->GetCounterFactory()
                                       ->GetCounter("copy_range5-PassThrough")
                                       ->Get());
    // Verify that stddev packets get correctly processed.
    // The standard deviation computed as:
    // sqrt(sum((x-mean(x))**2 / length(x)))
    // for x = 300:305 is 1.707825 (multiplied by 100 and rounded it is 171)
    // for x = 500:503 is 1.118034 (multiplied by 100 and rounded it is 112)
    EXPECT_EQ(absl::StrCat(Timestamp::PostStream().DebugString(), " 171 112"),
              dumped_final_stddev_packet.Get<std::string>());

    EXPECT_EQ(4 * (iteration + 1), graph->GetCounterFactory()
                                       ->GetCounter("copy_range5-PassThrough")
                                       ->Get());
  }
  LOG(INFO) << "After Loop Runs.";
  // Verify that the graph can still run (but not successfully) when
  // one of the nodes is caused to fail.
  extra_side_packets.clear();
  extra_side_packets.emplace("node_3", Adopt(new uint64((15LL << 32) | 0)));
  if (define_node_5) {
    extra_side_packets.emplace("node_5", Adopt(new uint64((15LL << 32) | 5)));
  }
  dumped_final_sum_packet = Packet();
  dumped_final_stddev_packet = Packet();
  dumped_final_packet = Packet();
  LOG(INFO) << "Expect an error to be logged here.";
  ASSERT_FALSE(graph->Run(extra_side_packets).ok());
  LOG(INFO) << "Error should have been logged.";
}

TEST(CalculatorGraph, BadInitialization) {
  CalculatorGraphConfig proto = GetConfig();
  CalculatorGraph graph;
  // Force the config to have a missing Calculator.
  proto.mutable_node(1)->clear_calculator();
  ASSERT_FALSE(graph.Initialize(proto).ok());
}

TEST(CalculatorGraph, BadRun) {
  CalculatorGraphConfig proto = GetConfig();
  CalculatorGraph graph;
  MP_ASSERT_OK(graph.Initialize(proto));
  // Don't set the input side packets.
  EXPECT_FALSE(graph.Run().ok());
}

TEST(CalculatorGraph, RunsCorrectly) {
  CalculatorGraph graph;
  CalculatorGraphConfig proto = GetConfig();
  RunComprehensiveTest(&graph, proto, /*define_node_5=*/true);
}

TEST(CalculatorGraph, RunsCorrectlyOnApplicationThread) {
  CalculatorGraph graph;
  CalculatorGraphConfig proto = GetConfig();
  // Force application thread to be used.
  proto.set_num_threads(0);
  RunComprehensiveTest(&graph, proto, /*define_node_5=*/true);
}

TEST(CalculatorGraph, RunsCorrectlyWithExternalExecutor) {
  CalculatorGraph graph;
  MP_ASSERT_OK(graph.SetExecutor("", std::make_shared<ThreadPoolExecutor>(1)));
  CalculatorGraphConfig proto = GetConfig();
  RunComprehensiveTest(&graph, proto, /*define_node_5=*/true);
}

// This test verifies that the MediaPipe framework calls Executor::AddTask()
// without holding any mutex, because CurrentThreadExecutor::AddTask() may
// result in a recursive call to itself.
TEST(CalculatorGraph, RunsCorrectlyWithCurrentThreadExecutor) {
  CalculatorGraph graph;
  MP_ASSERT_OK(
      graph.SetExecutor("", std::make_shared<CurrentThreadExecutor>()));
  CalculatorGraphConfig proto = GetConfig();
  RunComprehensiveTest(&graph, proto, /*define_node_5=*/true);
}

TEST(CalculatorGraph, RunsCorrectlyWithNonDefaultExecutors) {
  CalculatorGraph graph;
  // Add executors "second" and "third".
  MP_ASSERT_OK(
      graph.SetExecutor("second", std::make_shared<ThreadPoolExecutor>(1)));
  MP_ASSERT_OK(
      graph.SetExecutor("third", std::make_shared<ThreadPoolExecutor>(1)));
  CalculatorGraphConfig proto = GetConfig();
  ExecutorConfig* executor = proto.add_executor();
  executor->set_name("second");
  executor = proto.add_executor();
  executor->set_name("third");
  for (int i = 0; i < proto.node_size(); ++i) {
    switch (i % 3) {
      case 0:
        // Use the default executor.
        break;
      case 1:
        proto.mutable_node(i)->set_executor("second");
        break;
      case 2:
        proto.mutable_node(i)->set_executor("third");
        break;
    }
  }
  RunComprehensiveTest(&graph, proto, /*define_node_5=*/true);
}

TEST(CalculatorGraph, RunsCorrectlyWithMultipleExecutors) {
  CalculatorGraph graph;
  // Add executors "second" and "third".
  CalculatorGraphConfig proto = GetConfig();
  ExecutorConfig* executor = proto.add_executor();
  executor->set_name("second");
  executor->set_type("ThreadPoolExecutor");
  MediaPipeOptions* options = executor->mutable_options();
  ThreadPoolExecutorOptions* extension =
      options->MutableExtension(ThreadPoolExecutorOptions::ext);
  extension->set_num_threads(1);
  executor = proto.add_executor();
  executor->set_name("third");
  executor->set_type("ThreadPoolExecutor");
  options = executor->mutable_options();
  extension = options->MutableExtension(ThreadPoolExecutorOptions::ext);
  extension->set_num_threads(1);
  for (int i = 0; i < proto.node_size(); ++i) {
    switch (i % 3) {
      case 0:
        // Use the default executor.
        break;
      case 1:
        proto.mutable_node(i)->set_executor("second");
        break;
      case 2:
        proto.mutable_node(i)->set_executor("third");
        break;
    }
  }
  RunComprehensiveTest(&graph, proto, /*define_node_5=*/true);
}

// Packet generator for an arbitrary unit64 packet.
class Uint64PacketGenerator : public PacketGenerator {
 public:
  static ::mediapipe::Status FillExpectations(
      const PacketGeneratorOptions& extendable_options,
      PacketTypeSet* input_side_packets, PacketTypeSet* output_side_packets) {
    output_side_packets->Index(0).Set<uint64>();
    return ::mediapipe::OkStatus();
  }

  static ::mediapipe::Status Generate(
      const PacketGeneratorOptions& extendable_options,
      const PacketSet& input_side_packets, PacketSet* output_side_packets) {
    output_side_packets->Index(0) = Adopt(new uint64(15LL << 32 | 5));
    return ::mediapipe::OkStatus();
  }
};
REGISTER_PACKET_GENERATOR(Uint64PacketGenerator);

TEST(CalculatorGraph, GeneratePacket) {
  CalculatorGraph graph;
  CalculatorGraphConfig proto = GetConfig();
  PacketGeneratorConfig* generator = proto.add_packet_generator();
  generator->set_packet_generator("Uint64PacketGenerator");
  generator->add_output_side_packet("node_5");
  RunComprehensiveTest(&graph, proto, false);
}

TEST(CalculatorGraph, TypeMismatch) {
  CalculatorGraphConfig config;
  CalculatorGraphConfig::Node* node = config.add_node();
  node->add_output_stream("stream_a");
  node = config.add_node();
  node->add_input_stream("stream_a");
  std::unique_ptr<CalculatorGraph> graph;

  // Type matches, expect success.
  config.mutable_node(0)->set_calculator("StringEmptySourceCalculator");
  config.mutable_node(1)->set_calculator("StringSinkCalculator");
  graph.reset(new CalculatorGraph());
  MP_ASSERT_OK(graph->Initialize(config));
  MP_EXPECT_OK(graph->Run());
  graph.reset(nullptr);

  // Type matches, expect success.
  config.mutable_node(0)->set_calculator("IntEmptySourceCalculator");
  config.mutable_node(1)->set_calculator("IntSinkCalculator");
  graph.reset(new CalculatorGraph());
  MP_ASSERT_OK(graph->Initialize(config));
  MP_EXPECT_OK(graph->Run());
  graph.reset(nullptr);

  // Type mismatch, expect non-crashing failure.
  config.mutable_node(0)->set_calculator("StringEmptySourceCalculator");
  config.mutable_node(1)->set_calculator("IntSinkCalculator");
  graph.reset(new CalculatorGraph());
  MP_ASSERT_OK(graph->Initialize(config));
  EXPECT_FALSE(graph->Run().ok());
  graph.reset(nullptr);

  // Type mismatch, expect non-crashing failure.
  config.mutable_node(0)->set_calculator("IntEmptySourceCalculator");
  config.mutable_node(1)->set_calculator("StringSinkCalculator");
  graph.reset(new CalculatorGraph());
  MP_ASSERT_OK(graph->Initialize(config));
  EXPECT_FALSE(graph->Run().ok());
  graph.reset(nullptr);
}

TEST(CalculatorGraph, LayerOrdering) {
  CalculatorGraphConfig config;
  CalculatorGraphConfig::Node* node;
  node = config.add_node();
  node->set_calculator("GlobalCountSourceCalculator");
  node->add_input_side_packet("global_counter");
  node->add_output_stream("count_layer_0_node_0");
  node->set_source_layer(0);
  node = config.add_node();
  node->set_calculator("GlobalCountSourceCalculator");
  node->add_input_side_packet("global_counter");
  node->add_output_stream("count_layer_1_node_0");
  node->set_source_layer(1);
  node = config.add_node();
  node->set_calculator("GlobalCountSourceCalculator");
  node->add_input_side_packet("global_counter");
  node->add_output_stream("count_layer_1_node_1");
  node->set_source_layer(1);
  node = config.add_node();
  node->set_calculator("GlobalCountSourceCalculator");
  node->add_input_side_packet("global_counter");
  node->add_output_stream("count_layer_2_node_0");
  node->set_source_layer(2);

  // Set num threads to 1 because we rely on sequential execution for this test.
  config.set_num_threads(1);

  std::vector<Packet> dump_layer_0_node_0;
  std::vector<Packet> dump_layer_1_node_0;
  std::vector<Packet> dump_layer_1_node_1;
  std::vector<Packet> dump_layer_2_node_0;
  tool::AddVectorSink("count_layer_0_node_0", &config, &dump_layer_0_node_0);
  tool::AddVectorSink("count_layer_1_node_0", &config, &dump_layer_1_node_0);
  tool::AddVectorSink("count_layer_1_node_1", &config, &dump_layer_1_node_1);
  tool::AddVectorSink("count_layer_2_node_0", &config, &dump_layer_2_node_0);

  auto graph = absl::make_unique<CalculatorGraph>();

  std::atomic<int> global_counter(0);
  std::map<std::string, Packet> input_side_packets;
  input_side_packets["global_counter"] = Adopt(new auto(&global_counter));

  MP_ASSERT_OK(graph->Initialize(config));
  MP_ASSERT_OK(graph->Run(input_side_packets));
  graph.reset(nullptr);

  ASSERT_EQ(GlobalCountSourceCalculator::kNumOutputPackets,
            dump_layer_0_node_0.size());
  ASSERT_EQ(GlobalCountSourceCalculator::kNumOutputPackets,
            dump_layer_1_node_0.size());
  ASSERT_EQ(GlobalCountSourceCalculator::kNumOutputPackets,
            dump_layer_1_node_1.size());
  ASSERT_EQ(GlobalCountSourceCalculator::kNumOutputPackets,
            dump_layer_2_node_0.size());

  // Check layer 0.
  for (int i = 0; i < GlobalCountSourceCalculator::kNumOutputPackets; ++i) {
    EXPECT_EQ(i, dump_layer_0_node_0[i].Get<int>());
    EXPECT_EQ(Timestamp(i), dump_layer_0_node_0[i].Timestamp());
  }
  // Check layer 1 is interleaved (arbitrarily).
  for (int i = 0; i < GlobalCountSourceCalculator::kNumOutputPackets; ++i) {
    EXPECT_TRUE(GlobalCountSourceCalculator::kNumOutputPackets + i * 2 ==
                    dump_layer_1_node_0[i].Get<int>() ||
                GlobalCountSourceCalculator::kNumOutputPackets + i * 2 + 1 ==
                    dump_layer_1_node_0[i].Get<int>());
    EXPECT_TRUE(GlobalCountSourceCalculator::kNumOutputPackets + i * 2 ==
                    dump_layer_1_node_1[i].Get<int>() ||
                GlobalCountSourceCalculator::kNumOutputPackets + i * 2 + 1 ==
                    dump_layer_1_node_1[i].Get<int>());
    EXPECT_EQ(Timestamp(i), dump_layer_1_node_0[i].Timestamp());
    EXPECT_EQ(Timestamp(i), dump_layer_1_node_1[i].Timestamp());
  }
  // Check layer 2.
  for (int i = 0; i < GlobalCountSourceCalculator::kNumOutputPackets; ++i) {
    EXPECT_EQ(3 * GlobalCountSourceCalculator::kNumOutputPackets + i,
              dump_layer_2_node_0[i].Get<int>());
    EXPECT_EQ(Timestamp(i), dump_layer_2_node_0[i].Timestamp());
  }

  EXPECT_EQ(
      20,
      input_side_packets["global_counter"].Get<std::atomic<int>*>()->load());
}

// Tests for status handler input verification.
TEST(CalculatorGraph, StatusHandlerInputVerification) {
  // Status handlers with all inputs present should be OK.
  auto graph = absl::make_unique<CalculatorGraph>();
  CalculatorGraphConfig config =
      ::mediapipe::ParseTextProtoOrDie<CalculatorGraphConfig>(R"(
        packet_generator {
          packet_generator: "StaticCounterStringGenerator"
          output_side_packet: "created_by_factory"
        }
        packet_generator {
          packet_generator: "TaggedIntSplitterPacketGenerator"
          input_side_packet: "a_uint64"
          output_side_packet: "HIGH:generated_by_generator"
          output_side_packet: "LOW:unused_low"
          output_side_packet: "PAIR:unused_pair"
        }
        status_handler {
          status_handler: "Uint32StatusHandler"
          input_side_packet: "generated_by_generator"
        }
        status_handler {
          status_handler: "StringStatusHandler"
          input_side_packet: "created_by_factory"
        }
        status_handler {
          status_handler: "StringStatusHandler"
          input_side_packet: "extra_string"
        }
      )");
  MP_ASSERT_OK(graph->Initialize(config));
  Packet extra_string = Adopt(new std::string("foo"));
  Packet a_uint64 = Adopt(new uint64(0));
  MP_EXPECT_OK(
      graph->Run({{"extra_string", extra_string}, {"a_uint64", a_uint64}}));

  // Should fail verification when missing a required input. The generator is
  // OK, but the StringStatusHandler is missing its input.
  EXPECT_FALSE(graph->Run({{"a_uint64", a_uint64}}).ok());

  // Should fail verification when the type of an already created packet is
  // wrong. Here we give the uint64 packet instead of the std::string packet to
  // the StringStatusHandler.
  EXPECT_FALSE(
      graph->Run({{"extra_string", a_uint64}, {"a_uint64", a_uint64}}).ok());

  // Should fail verification when the type of a packet generated by a base
  // packet factory is wrong. Everything is correct except we add a status
  // handler expecting a uint32 but give it the std::string from the packet
  // factory.
  auto* invalid_handler = config.add_status_handler();
  invalid_handler->set_status_handler("Uint32StatusHandler");
  invalid_handler->add_input_side_packet("created_by_factory");
  graph.reset(new CalculatorGraph());
  ::mediapipe::Status status = graph->Initialize(config);
  EXPECT_THAT(status.message(),
              testing::AllOf(testing::HasSubstr("Uint32StatusHandler"),
                             // The problematic input side packet.
                             testing::HasSubstr("created_by_factory"),
                             // Actual type.
                             testing::HasSubstr("string"),
                             // Expected type.
                             testing::HasSubstr(
                                 MediaPipeTypeStringOrDemangled<uint32>())));

  // Should fail verification when the type of a to-be-generated packet is
  // wrong. The added handler now expects a std::string but will receive the
  // uint32 generated by the existing generator.
  invalid_handler->set_status_handler("StringStatusHandler");
  invalid_handler->set_input_side_packet(0, "generated_by_generator");
  graph.reset(new CalculatorGraph());
  // This is caught earlier, when the type of the PacketGenerator output
  // is compared to the type of the StatusHandler input.

  status = graph->Initialize(config);
  EXPECT_THAT(status.message(),
              testing::AllOf(
                  testing::HasSubstr("StringStatusHandler"),
                  // The problematic input side packet.
                  testing::HasSubstr("generated_by_generator"),
                  // Actual type.
                  testing::HasSubstr(MediaPipeTypeStringOrDemangled<uint32>()),
                  // Expected type.
                  testing::HasSubstr("string")));
}

TEST(CalculatorGraph, GenerateInInitialize) {
  CalculatorGraph graph;
  CalculatorGraphConfig config =
      ::mediapipe::ParseTextProtoOrDie<CalculatorGraphConfig>(R"(
        packet_generator {
          packet_generator: "StaticCounterStringGenerator"
          input_side_packet: "created_by_factory"
          input_side_packet: "input_in_initialize"
          output_side_packet: "foo1"
        }
        packet_generator {
          packet_generator: "StaticCounterStringGenerator"
          input_side_packet: "created_by_factory"
          input_side_packet: "input_in_initialize"
          input_side_packet: "foo1"
          output_side_packet: "foo2"
        }
        packet_generator {
          packet_generator: "StaticCounterStringGenerator"
          input_side_packet: "created_by_factory"
          input_side_packet: "input_in_initialize"
          input_side_packet: "input_in_run"
          output_side_packet: "foo3"
        }
        packet_generator {
          packet_generator: "StaticCounterStringGenerator"
          input_side_packet: "created_by_factory"
          input_side_packet: "input_in_initialize"
          input_side_packet: "input_in_run"
          input_side_packet: "foo3"
          output_side_packet: "foo4"
        }
      )");
  int initial_count = StaticCounterStringGenerator::NumPacketsGenerated();
  MP_ASSERT_OK(graph.Initialize(
      config,
      {{"created_by_factory", MakePacket<std::string>("default string")},
       {"input_in_initialize", MakePacket<int>(10)}}));
  EXPECT_EQ(initial_count + 2,
            StaticCounterStringGenerator::NumPacketsGenerated());
  MP_ASSERT_OK(graph.Run({{"input_in_run", MakePacket<int>(11)}}));
  EXPECT_EQ(initial_count + 4,
            StaticCounterStringGenerator::NumPacketsGenerated());
  MP_ASSERT_OK(graph.Run({{"input_in_run", MakePacket<int>(12)}}));
  EXPECT_EQ(initial_count + 6,
            StaticCounterStringGenerator::NumPacketsGenerated());
}

// Resets the counters in the input side packets used in the HandlersRun test.
// The value of all these counters will be set to the integer zero, as required
// at the beginning of the test.
void ResetCounters(std::map<std::string, Packet>* input_side_packets) {
  (*input_side_packets)["no_input_counter1"] = AdoptAsUniquePtr(new int(0));
  (*input_side_packets)["no_input_counter2"] = AdoptAsUniquePtr(new int(0));
  (*input_side_packets)["available_input_counter1"] =
      AdoptAsUniquePtr(new int(0));
  (*input_side_packets)["available_input_counter2"] =
      AdoptAsUniquePtr(new int(0));
  (*input_side_packets)["unavailable_input_counter1"] =
      AdoptAsUniquePtr(new int(0));
  (*input_side_packets)["unavailable_input_counter2"] =
      AdoptAsUniquePtr(new int(0));
}

// Tests that status handlers run.
// - We specify three status handlers: one taking no input side packets, one
// taking
//   an input side packet that is always provided in the call to Run(), and one
//   that takes the input side packet that will not be produced by the
//   FailingPacketGenerator. The first two should proccess their PRE-RUN status
//   but not their POST-RUN status, the third one should not process either of
//   them since the graph execution fails before the PRE-RUN step.
// - We then replace the FailingPacketGenerator with a non-failing generator,
//   and should have all three handlers running both PRE and POST-RUN (after the
//   FailingSourceCalculator fails).
// - We test that all three status handlers run (with both status) at the end of
//   a successful graph run.
// - Finally, we verify that when the status handler fails (either on PRE or
//   POST run), but the calculators don't, we still receive errors from the
//   calculator run.
TEST(CalculatorGraph, HandlersRun) {
  std::unique_ptr<CalculatorGraph> graph(new CalculatorGraph());
  CalculatorGraphConfig config =
      ::mediapipe::ParseTextProtoOrDie<CalculatorGraphConfig>(R"(
        packet_generator {
          packet_generator: "FailingPacketGenerator"
          output_side_packet: "unavailable"
        }
        node { calculator: "FailingSourceCalculator" output_stream: "output" }
        status_handler {
          status_handler: "IncrementingStatusHandler"
          input_side_packet: "COUNTER1:no_input_counter1"
          input_side_packet: "COUNTER2:no_input_counter2"
        }
        status_handler {
          status_handler: "IncrementingStatusHandler"
          input_side_packet: "COUNTER1:available_input_counter1"
          input_side_packet: "COUNTER2:available_input_counter2"
          input_side_packet: "EXTRA:available_string"
        }
        status_handler {
          status_handler: "IncrementingStatusHandler"
          input_side_packet: "COUNTER1:unavailable_input_counter1"
          input_side_packet: "COUNTER2:unavailable_input_counter2"
          input_side_packet: "EXTRA:unavailable"
        }
      )");
  std::map<std::string, Packet> input_side_packets(
      {{"unused_input", AdoptAsUniquePtr(new int(0))},
       {"no_input_counter1", AdoptAsUniquePtr(new int(0))},
       {"no_input_counter2", AdoptAsUniquePtr(new int(0))},
       {"available_input_counter1", AdoptAsUniquePtr(new int(0))},
       {"available_input_counter2", AdoptAsUniquePtr(new int(0))},
       {"unavailable_input_counter1", AdoptAsUniquePtr(new int(0))},
       {"unavailable_input_counter2", AdoptAsUniquePtr(new int(0))},
       {"available_string", Adopt(new std::string("foo"))}});

  // When the graph fails in initialize (even because of a PacketGenerator
  // returning an error), status handlers should not be run.
  ASSERT_THAT(graph->Initialize(config).ToString(),
              testing::HasSubstr("FailingPacketGenerator"));
  EXPECT_EQ(0,
            *GetFromUniquePtr<int>(input_side_packets.at("no_input_counter1")));
  EXPECT_EQ(0,
            *GetFromUniquePtr<int>(input_side_packets.at("no_input_counter2")));
  EXPECT_EQ(0, *GetFromUniquePtr<int>(
                   input_side_packets.at("available_input_counter1")));
  EXPECT_EQ(0, *GetFromUniquePtr<int>(
                   input_side_packets.at("available_input_counter2")));
  EXPECT_EQ(0, *GetFromUniquePtr<int>(
                   input_side_packets.at("unavailable_input_counter1")));
  EXPECT_EQ(0, *GetFromUniquePtr<int>(
                   input_side_packets.at("unavailable_input_counter2")));

  // Add an input side packet to the packet generator so that it doesn't
  // run at initialize time.
  config.mutable_packet_generator(0)->add_input_side_packet("unused_input");
  graph.reset(new CalculatorGraph());
  MP_ASSERT_OK(graph->Initialize(config));
  ResetCounters(&input_side_packets);
  EXPECT_THAT(graph->Run(input_side_packets).ToString(),
              testing::HasSubstr("FailingPacketGenerator"));
  EXPECT_EQ(1,
            *GetFromUniquePtr<int>(input_side_packets.at("no_input_counter1")));
  EXPECT_EQ(0,
            *GetFromUniquePtr<int>(input_side_packets.at("no_input_counter2")));
  EXPECT_EQ(1, *GetFromUniquePtr<int>(
                   input_side_packets.at("available_input_counter1")));
  EXPECT_EQ(0, *GetFromUniquePtr<int>(
                   input_side_packets.at("available_input_counter2")));
  EXPECT_EQ(0, *GetFromUniquePtr<int>(
                   input_side_packets.at("unavailable_input_counter1")));
  EXPECT_EQ(0, *GetFromUniquePtr<int>(
                   input_side_packets.at("unavailable_input_counter2")));

  // Replace the failing packet generator with something that works. All three
  // status handlers should now process both the PRE and POST-RUN status.
  config.mutable_packet_generator(0)->set_packet_generator(
      "StaticCounterStringGenerator");
  graph.reset(new CalculatorGraph());
  MP_ASSERT_OK(graph->Initialize(config));
  ResetCounters(&input_side_packets);
  // The entire graph should still fail because of the FailingSourceCalculator.
  EXPECT_THAT(graph->Run(input_side_packets).ToString(),
              testing::HasSubstr("FailingSourceCalculator"));
  EXPECT_EQ(1,
            *GetFromUniquePtr<int>(input_side_packets.at("no_input_counter1")));
  EXPECT_EQ(1,
            *GetFromUniquePtr<int>(input_side_packets.at("no_input_counter2")));
  EXPECT_EQ(1, *GetFromUniquePtr<int>(
                   input_side_packets.at("available_input_counter1")));
  EXPECT_EQ(1, *GetFromUniquePtr<int>(
                   input_side_packets.at("available_input_counter2")));
  EXPECT_EQ(1, *GetFromUniquePtr<int>(
                   input_side_packets.at("unavailable_input_counter1")));
  EXPECT_EQ(1, *GetFromUniquePtr<int>(
                   input_side_packets.at("unavailable_input_counter2")));

  // Replace the failing calculator with something that works. All three
  // status handlers should still process both PRE and POST-RUN status as part
  // of the successful graph run.
  config.mutable_node(0)->set_calculator("StringEmptySourceCalculator");
  graph.reset(new CalculatorGraph());
  MP_ASSERT_OK(graph->Initialize(config));
  ResetCounters(&input_side_packets);
  MP_EXPECT_OK(graph->Run(input_side_packets));
  EXPECT_EQ(1,
            *GetFromUniquePtr<int>(input_side_packets.at("no_input_counter1")));
  EXPECT_EQ(1,
            *GetFromUniquePtr<int>(input_side_packets.at("no_input_counter2")));
  EXPECT_EQ(1, *GetFromUniquePtr<int>(
                   input_side_packets.at("available_input_counter1")));
  EXPECT_EQ(1, *GetFromUniquePtr<int>(
                   input_side_packets.at("available_input_counter2")));
  EXPECT_EQ(1, *GetFromUniquePtr<int>(
                   input_side_packets.at("unavailable_input_counter1")));
  EXPECT_EQ(1, *GetFromUniquePtr<int>(
                   input_side_packets.at("unavailable_input_counter2")));

  ::mediapipe::Status run_status;

  // Make status handlers fail. The graph should fail.
  // First, when the PRE_run fails
  IncrementingStatusHandler::SetPreRunStatusResult(
      ::mediapipe::InternalError("Fail at pre-run"));
  graph.reset(new CalculatorGraph());
  MP_ASSERT_OK(graph->Initialize(config));
  ResetCounters(&input_side_packets);
  run_status = graph->Run(input_side_packets);
  EXPECT_TRUE(run_status.code() == ::mediapipe::StatusCode::kInternal);
  EXPECT_THAT(run_status.ToString(), testing::HasSubstr("Fail at pre-run"));
  EXPECT_EQ(1,
            *GetFromUniquePtr<int>(input_side_packets.at("no_input_counter1")));
  EXPECT_EQ(0,
            *GetFromUniquePtr<int>(input_side_packets.at("no_input_counter2")));
  EXPECT_EQ(1, *GetFromUniquePtr<int>(
                   input_side_packets.at("available_input_counter1")));
  EXPECT_EQ(0, *GetFromUniquePtr<int>(
                   input_side_packets.at("available_input_counter2")));
  EXPECT_EQ(1, *GetFromUniquePtr<int>(
                   input_side_packets.at("unavailable_input_counter1")));
  EXPECT_EQ(0, *GetFromUniquePtr<int>(
                   input_side_packets.at("unavailable_input_counter2")));

  // Second, when the POST_run fails
  IncrementingStatusHandler::SetPreRunStatusResult(::mediapipe::OkStatus());
  IncrementingStatusHandler::SetPostRunStatusResult(
      ::mediapipe::InternalError("Fail at post-run"));
  graph.reset(new CalculatorGraph());
  MP_ASSERT_OK(graph->Initialize(config));
  ResetCounters(&input_side_packets);
  run_status = graph->Run(input_side_packets);
  EXPECT_TRUE(run_status.code() == ::mediapipe::StatusCode::kInternal);
  EXPECT_THAT(run_status.ToString(), testing::HasSubstr("Fail at post-run"));
  EXPECT_EQ(1,
            *GetFromUniquePtr<int>(input_side_packets.at("no_input_counter1")));
  EXPECT_EQ(1,
            *GetFromUniquePtr<int>(input_side_packets.at("no_input_counter2")));
  EXPECT_EQ(1, *GetFromUniquePtr<int>(
                   input_side_packets.at("available_input_counter1")));
  EXPECT_EQ(1, *GetFromUniquePtr<int>(
                   input_side_packets.at("available_input_counter2")));
  EXPECT_EQ(1, *GetFromUniquePtr<int>(
                   input_side_packets.at("unavailable_input_counter1")));
  EXPECT_EQ(1, *GetFromUniquePtr<int>(
                   input_side_packets.at("unavailable_input_counter2")));
}

// Test that calling SetOffset() in Calculator::Process() results in the
// ::mediapipe::StatusCode::kFailedPrecondition error.
TEST(CalculatorGraph, SetOffsetInProcess) {
  CalculatorGraph graph;
  CalculatorGraphConfig config =
      ::mediapipe::ParseTextProtoOrDie<CalculatorGraphConfig>(R"(
        input_stream: 'in'
        node {
          calculator: 'SetOffsetInProcessCalculator'
          input_stream: 'in'
          output_stream: 'out'
        }
      )");

  MP_ASSERT_OK(graph.Initialize(config));
  MP_EXPECT_OK(graph.StartRun({}));
  MP_EXPECT_OK(
      graph.AddPacketToInputStream("in", MakePacket<int>(0).At(Timestamp(0))));
  ::mediapipe::Status status = graph.WaitUntilIdle();
  EXPECT_FALSE(status.ok());
  EXPECT_EQ(::mediapipe::StatusCode::kFailedPrecondition, status.code());
}

// Test that MediaPipe releases input packets when it is done with them.
TEST(CalculatorGraph, InputPacketLifetime) {
  CalculatorGraph graph;
  CalculatorGraphConfig config =
      ::mediapipe::ParseTextProtoOrDie<CalculatorGraphConfig>(R"(
        input_stream: 'in'
        node {
          calculator: 'PassThroughCalculator'
          input_stream: 'in'
          output_stream: 'mid'
        }
        node {
          calculator: 'PassThroughCalculator'
          input_stream: 'mid'
          output_stream: 'out'
        }
      )");

  LifetimeTracker tracker;
  Timestamp timestamp = Timestamp(0);
  auto new_packet = [&timestamp, &tracker] {
    return Adopt(tracker.MakeObject().release()).At(++timestamp);
  };

  MP_ASSERT_OK(graph.Initialize(config));
  MP_EXPECT_OK(graph.StartRun({}));
  MP_EXPECT_OK(graph.AddPacketToInputStream("in", new_packet()));
  MP_EXPECT_OK(graph.WaitUntilIdle());
  EXPECT_EQ(0, tracker.live_count());
  MP_EXPECT_OK(graph.AddPacketToInputStream("in", new_packet()));
  MP_EXPECT_OK(graph.AddPacketToInputStream("in", new_packet()));
  MP_EXPECT_OK(graph.AddPacketToInputStream("in", new_packet()));
  MP_EXPECT_OK(graph.WaitUntilIdle());
  EXPECT_EQ(0, tracker.live_count());
  MP_EXPECT_OK(graph.CloseInputStream("in"));
  MP_EXPECT_OK(graph.WaitUntilDone());
}

// Demonstrate an if-then-else graph.
TEST(CalculatorGraph, IfThenElse) {
  // This graph has an if-then-else structure. The left branch, selected by the
  // select value 0, applies a double (multiply by 2) operation. The right
  // branch, selected by the select value 1, applies a square operation. The
  // left branch also has some no-op PassThroughCalculators to make the lengths
  // of the two branches different.
  CalculatorGraph graph;
  CalculatorGraphConfig config =
      ::mediapipe::ParseTextProtoOrDie<CalculatorGraphConfig>(R"(
        input_stream: 'in'
        input_stream: 'select'
        node {
          calculator: 'DemuxTimedCalculator'
          input_stream: 'INPUT:in'
          input_stream: 'SELECT:select'
          output_stream: 'OUTPUT:0:left'
          output_stream: 'OUTPUT:1:right'
        }
        node {
          calculator: 'PassThroughCalculator'
          input_stream: 'left'
          output_stream: 'left1'
        }
        node {
          calculator: 'DoubleIntCalculator'
          input_stream: 'left1'
          output_stream: 'left2'
        }
        node {
          calculator: 'PassThroughCalculator'
          input_stream: 'left2'
          output_stream: 'left3'
        }
        node {
          calculator: 'SquareIntCalculator'
          input_stream: 'right'
          output_stream: 'right1'
        }
        node {
          calculator: 'MuxTimedCalculator'
          input_stream: 'INPUT:0:left3'
          input_stream: 'INPUT:1:right1'
          input_stream: 'SELECT:select'
          output_stream: 'OUTPUT:out'
        }
      )");
  std::vector<Packet> packet_dump;
  tool::AddVectorSink("out", &config, &packet_dump);

  Timestamp timestamp = Timestamp(0);
  auto send_inputs = [&graph, &timestamp](int input, int select) {
    ++timestamp;
    MP_EXPECT_OK(graph.AddPacketToInputStream(
        "in", MakePacket<int>(input).At(timestamp)));
    MP_EXPECT_OK(graph.AddPacketToInputStream(
        "select", MakePacket<int>(select).At(timestamp)));
  };

  MP_ASSERT_OK(graph.Initialize(config));
  MP_ASSERT_OK(graph.StartRun({}));

  // If the "select" input is 0, we apply a double operation. If "select" is 1,
  // we apply a square operation. To make the code easier to understand, define
  // symbolic names for the select values.
  const int kApplyDouble = 0;
  const int kApplySquare = 1;

  send_inputs(1, kApplyDouble);
  MP_ASSERT_OK(graph.WaitUntilIdle());
  ASSERT_EQ(1, packet_dump.size());
  EXPECT_EQ(2, packet_dump[0].Get<int>());

  send_inputs(2, kApplySquare);
  MP_ASSERT_OK(graph.WaitUntilIdle());
  ASSERT_EQ(2, packet_dump.size());
  EXPECT_EQ(4, packet_dump[1].Get<int>());

  send_inputs(3, kApplyDouble);
  send_inputs(4, kApplyDouble);
  MP_ASSERT_OK(graph.WaitUntilIdle());
  EXPECT_EQ(4, packet_dump.size());
  EXPECT_EQ(6, packet_dump[2].Get<int>());
  EXPECT_EQ(8, packet_dump[3].Get<int>());

  send_inputs(5, kApplySquare);
  MP_ASSERT_OK(graph.WaitUntilIdle());
  ASSERT_EQ(5, packet_dump.size());
  EXPECT_EQ(25, packet_dump[4].Get<int>());

  send_inputs(6, kApplyDouble);
  MP_ASSERT_OK(graph.WaitUntilIdle());
  ASSERT_EQ(6, packet_dump.size());
  EXPECT_EQ(12, packet_dump[5].Get<int>());

  send_inputs(7, kApplySquare);
  send_inputs(8, kApplySquare);
  send_inputs(9, kApplySquare);
  send_inputs(10, kApplyDouble);
  MP_ASSERT_OK(graph.WaitUntilIdle());
  ASSERT_EQ(10, packet_dump.size());
  EXPECT_EQ(49, packet_dump[6].Get<int>());
  EXPECT_EQ(64, packet_dump[7].Get<int>());
  EXPECT_EQ(81, packet_dump[8].Get<int>());
  EXPECT_EQ(20, packet_dump[9].Get<int>());

  MP_ASSERT_OK(graph.CloseAllInputStreams());
  MP_ASSERT_OK(graph.WaitUntilDone());
  EXPECT_EQ(10, packet_dump.size());
}

// A simple output selecting test calculator, which omits timestamp bounds
// for the unselected outputs.
class DemuxUntimedCalculator : public CalculatorBase {
 public:
  static ::mediapipe::Status GetContract(CalculatorContract* cc) {
    RET_CHECK_EQ(cc->Inputs().NumEntries(), 2);
    cc->Inputs().Tag("INPUT").SetAny();
    cc->Inputs().Tag("SELECT").Set<int>();
    for (CollectionItemId id = cc->Outputs().BeginId("OUTPUT");
         id < cc->Outputs().EndId("OUTPUT"); ++id) {
      cc->Outputs().Get(id).SetSameAs(&cc->Inputs().Tag("INPUT"));
    }
    return ::mediapipe::OkStatus();
  }
  ::mediapipe::Status Process(CalculatorContext* cc) final {
    int index = cc->Inputs().Tag("SELECT").Get<int>();
    if (!cc->Inputs().Tag("INPUT").IsEmpty()) {
      cc->Outputs()
          .Get("OUTPUT", index)
          .AddPacket(cc->Inputs().Tag("INPUT").Value());
    } else {
      cc->Outputs()
          .Get("OUTPUT", index)
          .SetNextTimestampBound(cc->InputTimestamp() + 1);
    }
    return ::mediapipe::OkStatus();
  }
};
REGISTER_CALCULATOR(DemuxUntimedCalculator);

// Demonstrate an if-then-else graph. This test differs from the IfThenElse test
// in that it uses optional input streams instead of next timestamp bound
// propagation.
TEST(CalculatorGraph, IfThenElse2) {
  // This graph has an if-then-else structure. The left branch, selected by the
  // select value 0, applies a double (multiply by 2) operation. The right
  // branch, selected by the select value 1, applies a square operation. The
  // left branch also has some no-op PassThroughCalculators to make the lengths
  // of the two branches different.
  CalculatorGraph graph;
  CalculatorGraphConfig config =
      ::mediapipe::ParseTextProtoOrDie<CalculatorGraphConfig>(R"(
        input_stream: 'in'
        input_stream: 'select'
        node {
          calculator: 'DemuxUntimedCalculator'
          input_stream: 'INPUT:in'
          input_stream: 'SELECT:select'
          output_stream: 'OUTPUT:0:left'
          output_stream: 'OUTPUT:1:right'
        }
        node {
          calculator: 'PassThroughCalculator'
          input_stream: 'left'
          output_stream: 'left1'
        }
        node {
          calculator: 'DoubleIntCalculator'
          input_stream: 'left1'
          output_stream: 'left2'
        }
        node {
          calculator: 'PassThroughCalculator'
          input_stream: 'left2'
          output_stream: 'left3'
        }
        node {
          calculator: 'SquareIntCalculator'
          input_stream: 'right'
          output_stream: 'right1'
        }
        node {
          calculator: 'MuxCalculator'
          input_stream: 'INPUT:0:left3'
          input_stream: 'INPUT:1:right1'
          input_stream: 'SELECT:select'
          output_stream: 'OUTPUT:out'
          input_stream_handler { input_stream_handler: 'MuxInputStreamHandler' }
        }
      )");
  std::vector<Packet> packet_dump;
  tool::AddVectorSink("out", &config, &packet_dump);

  Timestamp timestamp = Timestamp(0);
  auto send_inputs = [&graph, &timestamp](int input, int select) {
    ++timestamp;
    MP_EXPECT_OK(graph.AddPacketToInputStream(
        "in", MakePacket<int>(input).At(timestamp)));
    MP_EXPECT_OK(graph.AddPacketToInputStream(
        "select", MakePacket<int>(select).At(timestamp)));
  };

  MP_ASSERT_OK(graph.Initialize(config));
  MP_ASSERT_OK(graph.StartRun({}));

  // If the "select" input is 0, we apply a double operation. If "select" is 1,
  // we apply a square operation. To make the code easier to understand, define
  // symbolic names for the select values.
  const int kApplyDouble = 0;
  const int kApplySquare = 1;

  send_inputs(1, kApplyDouble);
  MP_ASSERT_OK(graph.WaitUntilIdle());
  ASSERT_EQ(1, packet_dump.size());
  EXPECT_EQ(2, packet_dump[0].Get<int>());

  send_inputs(2, kApplySquare);
  MP_ASSERT_OK(graph.WaitUntilIdle());
  ASSERT_EQ(2, packet_dump.size());
  EXPECT_EQ(4, packet_dump[1].Get<int>());

  send_inputs(3, kApplyDouble);
  send_inputs(4, kApplyDouble);
  MP_ASSERT_OK(graph.WaitUntilIdle());
  EXPECT_EQ(4, packet_dump.size());
  EXPECT_EQ(6, packet_dump[2].Get<int>());
  EXPECT_EQ(8, packet_dump[3].Get<int>());

  send_inputs(5, kApplySquare);
  MP_ASSERT_OK(graph.WaitUntilIdle());
  ASSERT_EQ(5, packet_dump.size());
  EXPECT_EQ(25, packet_dump[4].Get<int>());

  send_inputs(6, kApplyDouble);
  MP_ASSERT_OK(graph.WaitUntilIdle());
  ASSERT_EQ(6, packet_dump.size());
  EXPECT_EQ(12, packet_dump[5].Get<int>());

  send_inputs(7, kApplySquare);
  send_inputs(8, kApplySquare);
  send_inputs(9, kApplySquare);
  send_inputs(10, kApplyDouble);
  MP_ASSERT_OK(graph.WaitUntilIdle());
  ASSERT_EQ(10, packet_dump.size());
  EXPECT_EQ(49, packet_dump[6].Get<int>());
  EXPECT_EQ(64, packet_dump[7].Get<int>());
  EXPECT_EQ(81, packet_dump[8].Get<int>());
  EXPECT_EQ(20, packet_dump[9].Get<int>());

  MP_ASSERT_OK(graph.CloseAllInputStreams());
  MP_ASSERT_OK(graph.WaitUntilDone());
  EXPECT_EQ(10, packet_dump.size());
}

// A regression test for bug 28321551. The scheduler should be able to run
// the calculator graph to completion without hanging. The test merely checks
// that CalculatorGraph::Run() returns.
TEST(CalculatorGraph, ClosedSourceNodeShouldNotBeUnthrottled) {
  // This calculator graph has two source nodes. The first source node,
  // OutputAllSourceCalculator, outputs a lot of packets in one shot and stops.
  // The second source node, OutputOneAtATimeSourceCalculator, outputs one
  // packet at a time. But it is connected to a node, DecimatorCalculator,
  // that discards most of its input packets and only rarely outputs a packet.
  // The sink node, MergeCalculator, receives three input streams, two from
  // the two source nodes and one from DecimatorCalculator. The two input
  // streams connected to the two source nodes will become full, and the
  // MediaPipe scheduler will throttle the source nodes.
  //
  // The MediaPipe scheduler should not schedule a closed source node, even if
  // the source node filled an input stream and the input stream changes from
  // being "full" to "not full".
  CalculatorGraphConfig config =
      ::mediapipe::ParseTextProtoOrDie<CalculatorGraphConfig>(R"(
        num_threads: 1
        max_queue_size: 100
        node {
          calculator: 'OutputAllSourceCalculator'
          output_stream: 'first_stream'
        }
        node {
          calculator: 'OutputOneAtATimeSourceCalculator'
          output_stream: 'second_stream'
        }
        node {
          calculator: 'DecimatorCalculator'
          input_stream: 'second_stream'
          output_stream: 'decimated_second_stream'
        }
        node {
          calculator: 'MergeCalculator'
          input_stream: 'first_stream'
          input_stream: 'second_stream'
          input_stream: 'decimated_second_stream'
          output_stream: 'output'
        }
      )");

  CalculatorGraph graph;
  MP_ASSERT_OK(graph.Initialize(config));
  MP_ASSERT_OK(graph.Run());
}

// Tests that a calculator can output a packet in the Open() method.
//
// The initial output packet generated by UnitDelayCalculator::Open() causes
// the following to happen before the scheduler starts to run:
// - The downstream PassThroughCalculator becomes ready and is added to the
//   scheduler queue.
// - Since max_queue_size is set to 1, the GlobalCountSourceCalculator is
//   throttled.
// The scheduler should be able to run the graph from this initial state.
TEST(CalculatorGraph, OutputPacketInOpen) {
  CalculatorGraphConfig config =
      ::mediapipe::ParseTextProtoOrDie<CalculatorGraphConfig>(R"(
        max_queue_size: 1
        node {
          calculator: 'GlobalCountSourceCalculator'
          input_side_packet: 'global_counter'
          output_stream: 'integers'
        }
        node {
          calculator: 'UnitDelayCalculator'
          input_stream: 'integers'
          output_stream: 'delayed_integers'
        }
        node {
          calculator: 'PassThroughCalculator'
          input_stream: 'delayed_integers'
          output_stream: 'output'
        }
      )");
  std::vector<Packet> packet_dump;
  tool::AddVectorSink("output", &config, &packet_dump);

  std::atomic<int> global_counter(1);
  std::map<std::string, Packet> input_side_packets;
  input_side_packets["global_counter"] = Adopt(new auto(&global_counter));

  CalculatorGraph graph;
  MP_ASSERT_OK(graph.Initialize(config));
  MP_ASSERT_OK(graph.Run(input_side_packets));
  ASSERT_EQ(GlobalCountSourceCalculator::kNumOutputPackets + 1,
            packet_dump.size());
  EXPECT_EQ(0, packet_dump[0].Get<int>());
  EXPECT_EQ(Timestamp(0), packet_dump[0].Timestamp());
  for (int i = 1; i <= GlobalCountSourceCalculator::kNumOutputPackets; ++i) {
    EXPECT_EQ(i, packet_dump[i].Get<int>());
    EXPECT_EQ(Timestamp(i), packet_dump[i].Timestamp());
  }
}

// Tests that a calculator can output a packet in the Open() method.
//
// The initial output packet generated by UnitDelayCalculator::Open() causes
// the following to happen before the scheduler starts to run:
// - The downstream MergeCalculator does not become ready because its second
//   input stream has no packet.
// - Since max_queue_size is set to 1, the GlobalCountSourceCalculator is
//   throttled.
// The scheduler must schedule a throttled source node from the beginning.
TEST(CalculatorGraph, OutputPacketInOpen2) {
  CalculatorGraphConfig config =
      ::mediapipe::ParseTextProtoOrDie<CalculatorGraphConfig>(R"(
        max_queue_size: 1
        node {
          calculator: 'GlobalCountSourceCalculator'
          input_side_packet: 'global_counter'
          output_stream: 'integers'
        }
        node {
          calculator: 'UnitDelayCalculator'
          input_stream: 'integers'
          output_stream: 'delayed_integers'
        }
        node {
          calculator: 'MergeCalculator'
          input_stream: 'delayed_integers'
          input_stream: 'integers'
          output_stream: 'output'
        }
      )");
  std::vector<Packet> packet_dump;
  tool::AddVectorSink("output", &config, &packet_dump);

  std::atomic<int> global_counter(1);
  std::map<std::string, Packet> input_side_packets;
  input_side_packets["global_counter"] = Adopt(new auto(&global_counter));

  CalculatorGraph graph;
  MP_ASSERT_OK(graph.Initialize(config));
  MP_ASSERT_OK(graph.Run(input_side_packets));
  ASSERT_EQ(GlobalCountSourceCalculator::kNumOutputPackets + 1,
            packet_dump.size());
  int i;
  for (i = 0; i < GlobalCountSourceCalculator::kNumOutputPackets; ++i) {
    std::string expected =
        absl::Substitute("Timestamp($0) $1 $2",
                         packet_dump[i].Timestamp().DebugString(), i, i + 1);
    EXPECT_EQ(expected, packet_dump[i].Get<std::string>());
    EXPECT_EQ(Timestamp(i), packet_dump[i].Timestamp());
  }
  std::string expected = absl::Substitute(
      "Timestamp($0) $1 empty", packet_dump[i].Timestamp().DebugString(), i);
  EXPECT_EQ(expected, packet_dump[i].Get<std::string>());
  EXPECT_EQ(Timestamp(i), packet_dump[i].Timestamp());
}

// Tests that no packets are available on input streams in Open(), even if the
// upstream calculator outputs a packet in Open().
TEST(CalculatorGraph, EmptyInputInOpen) {
  CalculatorGraphConfig config =
      ::mediapipe::ParseTextProtoOrDie<CalculatorGraphConfig>(R"(
        max_queue_size: 1
        node {
          calculator: 'GlobalCountSourceCalculator'
          input_side_packet: 'global_counter'
          output_stream: 'integers'
        }
        # UnitDelayCalculator outputs a packet during Open().
        node {
          calculator: 'UnitDelayCalculator'
          input_stream: 'integers'
          output_stream: 'delayed_integers'
        }
        node {
          calculator: 'AssertEmptyInputInOpenCalculator'
          input_stream: 'delayed_integers'
        }
        node {
          calculator: 'AssertEmptyInputInOpenCalculator'
          input_stream: 'integers'
        }
      )");

  std::atomic<int> global_counter(1);
  std::map<std::string, Packet> input_side_packets;
  input_side_packets["global_counter"] = Adopt(new auto(&global_counter));

  CalculatorGraph graph;
  MP_ASSERT_OK(graph.Initialize(config));
  MP_EXPECT_OK(graph.Run(input_side_packets));
}

// Test for b/33568859.
TEST(CalculatorGraph, UnthrottleRespectsLayers) {
  CalculatorGraphConfig config =
      ::mediapipe::ParseTextProtoOrDie<CalculatorGraphConfig>(R"(
        max_queue_size: 1
        node {
          calculator: 'GlobalCountSourceCalculator'
          input_side_packet: 'global_counter'
          output_stream: 'integers0'
          source_layer: 0
        }
        node {
          calculator: 'GlobalCountSourceCalculator'
          input_side_packet: 'global_counter'
          input_side_packet: 'output_in_open'
          output_stream: 'integers1'
          source_layer: 1
        }
        node {
          calculator: 'PassThroughCalculator'
          input_stream: 'integers1'
          output_stream: 'integers1passthrough'
        }
      )");

  std::vector<Packet> layer0_packets;
  std::vector<Packet> layer1_packets;
  tool::AddVectorSink("integers0", &config, &layer0_packets);
  tool::AddVectorSink("integers1passthrough", &config, &layer1_packets);

  std::atomic<int> global_counter(0);
  std::map<std::string, Packet> input_side_packets;
  input_side_packets["global_counter"] = Adopt(new auto(&global_counter));
  // TODO: Set this value to true. When the calculator outputs a
  // packet in Open, it will trigget b/33568859, and the test will fail. Use
  // this test to verify that b/33568859 is fixed.
  constexpr bool kOutputInOpen = true;
  input_side_packets["output_in_open"] = MakePacket<bool>(kOutputInOpen);

  CalculatorGraph graph;
  MP_ASSERT_OK(graph.Initialize(config));
  MP_ASSERT_OK(graph.Run(input_side_packets));
  ASSERT_EQ(GlobalCountSourceCalculator::kNumOutputPackets,
            layer0_packets.size());
  ASSERT_EQ(GlobalCountSourceCalculator::kNumOutputPackets,
            layer1_packets.size());
  // Check that we ran things in the expected order.
  int count = 0;
  if (kOutputInOpen) {
    EXPECT_EQ(count, layer1_packets[0].Get<int>());
    EXPECT_EQ(Timestamp(0), layer1_packets[0].Timestamp());
    ++count;
  }
  for (int i = 0; i < GlobalCountSourceCalculator::kNumOutputPackets;
       ++i, ++count) {
    EXPECT_EQ(count, layer0_packets[i].Get<int>());
    EXPECT_EQ(Timestamp(i), layer0_packets[i].Timestamp());
  }
  for (int i = kOutputInOpen ? 1 : 0;
       i < GlobalCountSourceCalculator::kNumOutputPackets; ++i, ++count) {
    EXPECT_EQ(count, layer1_packets[i].Get<int>());
    EXPECT_EQ(Timestamp(i), layer1_packets[i].Timestamp());
  }
}

// The graph calculates the sum of all the integers output by the source node
// so far. The graph has one cycle.
TEST(CalculatorGraph, Cycle) {
  CalculatorGraphConfig config =
      ::mediapipe::ParseTextProtoOrDie<CalculatorGraphConfig>(R"(
        node {
          calculator: 'GlobalCountSourceCalculator'
          input_side_packet: 'global_counter'
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
          calculator: 'UnitDelayCalculator'
          input_stream: 'sum'
          output_stream: 'old_sum'
        }
      )");
  std::vector<Packet> packet_dump;
  tool::AddVectorSink("sum", &config, &packet_dump);

  std::atomic<int> global_counter(1);
  std::map<std::string, Packet> input_side_packets;
  input_side_packets["global_counter"] = Adopt(new auto(&global_counter));

  CalculatorGraph graph;
  MP_ASSERT_OK(graph.Initialize(config));
  MP_ASSERT_OK(graph.Run(input_side_packets));
  ASSERT_EQ(GlobalCountSourceCalculator::kNumOutputPackets, packet_dump.size());
  int sum = 0;
  for (int i = 0; i < GlobalCountSourceCalculator::kNumOutputPackets; ++i) {
    sum += i + 1;
    EXPECT_EQ(sum, packet_dump[i].Get<int>());
    EXPECT_EQ(Timestamp(i), packet_dump[i].Timestamp());
  }
}

// The graph calculates the sum of all the integers output by the source node
// so far. The graph has one cycle.
//
// The difference from the "Cycle" test is that the graph is scheduled with
// packet timestamps ignored.
TEST(CalculatorGraph, CycleUntimed) {
  CalculatorGraphConfig config =
      ::mediapipe::ParseTextProtoOrDie<CalculatorGraphConfig>(R"(
        input_stream_handler {
          input_stream_handler: 'BarrierInputStreamHandler'
        }
        node {
          calculator: 'GlobalCountSourceCalculator'
          input_side_packet: 'global_counter'
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
        }
        node {
          calculator: 'UnitDelayUntimedCalculator'
          input_stream: 'sum'
          output_stream: 'old_sum'
        }
      )");
  std::vector<Packet> packet_dump;
  tool::AddVectorSink("sum", &config, &packet_dump);

  std::atomic<int> global_counter(1);
  std::map<std::string, Packet> input_side_packets;
  input_side_packets["global_counter"] = Adopt(new auto(&global_counter));

  CalculatorGraph graph;
  MP_ASSERT_OK(graph.Initialize(config));
  MP_ASSERT_OK(graph.Run(input_side_packets));
  ASSERT_EQ(GlobalCountSourceCalculator::kNumOutputPackets, packet_dump.size());
  int sum = 0;
  for (int i = 0; i < GlobalCountSourceCalculator::kNumOutputPackets; ++i) {
    sum += i + 1;
    EXPECT_EQ(sum, packet_dump[i].Get<int>());
  }
}

// This unit test is a direct form I implementation of Example 6.2 of
// Discrete-Time Signal Processing, 3rd Ed., shown in Figure 6.6. The system
// function of the linear time-invariant (LTI) system is
//     H(z) = (1 + 2 * z^-1) / (1 - 1.5 * z^-1 + 0.9 * z^-2)
// The graph has two cycles.
TEST(CalculatorGraph, DirectFormI) {
  CalculatorGraphConfig config =
      ::mediapipe::ParseTextProtoOrDie<CalculatorGraphConfig>(R"(
        node {
          calculator: 'GlobalCountSourceCalculator'
          input_side_packet: 'global_counter'
          output_stream: 'integers'
        }
        node {
          calculator: 'IntToFloatCalculator'
          input_stream: 'integers'
          output_stream: 'x'
        }
        node {
          calculator: 'FloatUnitDelayCalculator'
          input_stream: 'x'
          output_stream: 'a'
        }
        node {
          calculator: 'FloatScalarMultiplierCalculator'
          input_stream: 'a'
          output_stream: 'b'
          input_side_packet: 'b1'
        }
        node {
          calculator: 'FloatAdderCalculator'
          input_stream: 'x'
          input_stream: 'b'
          output_stream: 'c'
          input_stream_handler {
            input_stream_handler: 'EarlyCloseInputStreamHandler'
          }
        }
        node {
          calculator: 'FloatAdderCalculator'
          input_stream: 'c'
          input_stream: 'f'
          input_stream_info: {
            tag_index: ':1'  # 'f'
            back_edge: true
          }
          output_stream: 'y'
          input_stream_handler {
            input_stream_handler: 'EarlyCloseInputStreamHandler'
          }
        }
        node {
          calculator: 'FloatUnitDelayCalculator'
          input_stream: 'y'
          output_stream: 'd'
        }
        node {
          calculator: 'FloatScalarMultiplierCalculator'
          input_stream: 'd'
          output_stream: 'e'
          input_side_packet: 'a1'
        }
        node {
          calculator: 'FloatUnitDelayCalculator'
          input_stream: 'd'
          output_stream: 'g'
        }
        node {
          calculator: 'FloatScalarMultiplierCalculator'
          input_stream: 'g'
          output_stream: 'h'
          input_side_packet: 'a2'
        }
        node {
          calculator: 'FloatAdderCalculator'
          input_stream: 'e'
          input_stream: 'h'
          output_stream: 'f'
          input_stream_handler {
            input_stream_handler: 'EarlyCloseInputStreamHandler'
          }
        }
      )");
  std::vector<Packet> packet_dump;
  tool::AddVectorSink("y", &config, &packet_dump);

  std::atomic<int> global_counter(1);
  std::map<std::string, Packet> input_side_packets;
  input_side_packets["global_counter"] = Adopt(new auto(&global_counter));
  input_side_packets["a2"] = Adopt(new float(-0.9));
  input_side_packets["a1"] = Adopt(new float(1.5));
  input_side_packets["b1"] = Adopt(new float(2.0));
  CalculatorGraph graph;
  MP_ASSERT_OK(graph.Initialize(config));
  MP_ASSERT_OK(graph.Run(input_side_packets));
  ASSERT_EQ(GlobalCountSourceCalculator::kNumOutputPackets, packet_dump.size());
  ASSERT_EQ(GlobalCountSourceCalculator::kNumOutputPackets, 5);
  EXPECT_FLOAT_EQ(1.0, packet_dump[0].Get<float>());
  EXPECT_FLOAT_EQ(5.5, packet_dump[1].Get<float>());
  EXPECT_FLOAT_EQ(14.35, packet_dump[2].Get<float>());
  EXPECT_FLOAT_EQ(26.575, packet_dump[3].Get<float>());
  EXPECT_FLOAT_EQ(39.9475, packet_dump[4].Get<float>());
  for (int i = 0; i < 5; ++i) {
    EXPECT_EQ(Timestamp(i), packet_dump[i].Timestamp());
  }
}

// This unit test is a direct form II implementation of Example 6.2 of
// Discrete-Time Signal Processing, 3rd Ed., shown in Figure 6.7. The system
// function of the linear time-invariant (LTI) system is
//     H(z) = (1 + 2 * z^-1) / (1 - 1.5 * z^-1 + 0.9 * z^-2)
// The graph has two cycles.
TEST(CalculatorGraph, DirectFormII) {
  CalculatorGraphConfig config =
      ::mediapipe::ParseTextProtoOrDie<CalculatorGraphConfig>(R"(
        node {
          calculator: 'GlobalCountSourceCalculator'
          input_side_packet: 'global_counter'
          output_stream: 'integers'
        }
        node {
          calculator: 'IntToFloatCalculator'
          input_stream: 'integers'
          output_stream: 'x'
        }
        node {
          calculator: 'FloatAdderCalculator'
          input_stream: 'x'
          input_stream: 'f'
          input_stream_info: {
            tag_index: ':1'  # 'f'
            back_edge: true
          }
          output_stream: 'a'
          input_stream_handler {
            input_stream_handler: 'EarlyCloseInputStreamHandler'
          }
        }
        node {
          calculator: 'FloatUnitDelayCalculator'
          input_stream: 'a'
          output_stream: 'b'
        }
        node {
          calculator: 'FloatScalarMultiplierCalculator'
          input_stream: 'b'
          output_stream: 'd'
          input_side_packet: 'a1'
        }
        node {
          calculator: 'FloatUnitDelayCalculator'
          input_stream: 'b'
          output_stream: 'c'
        }
        node {
          calculator: 'FloatScalarMultiplierCalculator'
          input_stream: 'c'
          output_stream: 'e'
          input_side_packet: 'a2'
        }
        node {
          calculator: 'FloatAdderCalculator'
          input_stream: 'd'
          input_stream: 'e'
          output_stream: 'f'
          input_stream_handler {
            input_stream_handler: 'EarlyCloseInputStreamHandler'
          }
        }
        node {
          calculator: 'FloatScalarMultiplierCalculator'
          input_stream: 'b'
          output_stream: 'g'
          input_side_packet: 'b1'
        }
        node {
          calculator: 'FloatAdderCalculator'
          input_stream: 'a'
          input_stream: 'g'
          output_stream: 'y'
          input_stream_handler {
            input_stream_handler: 'EarlyCloseInputStreamHandler'
          }
        }
      )");
  std::vector<Packet> packet_dump;
  tool::AddVectorSink("y", &config, &packet_dump);

  std::atomic<int> global_counter(1);
  std::map<std::string, Packet> input_side_packets;
  input_side_packets["global_counter"] = Adopt(new auto(&global_counter));
  input_side_packets["a2"] = Adopt(new float(-0.9));
  input_side_packets["a1"] = Adopt(new float(1.5));
  input_side_packets["b1"] = Adopt(new float(2.0));

  CalculatorGraph graph;
  MP_ASSERT_OK(graph.Initialize(config));
  MP_ASSERT_OK(graph.Run(input_side_packets));
  ASSERT_EQ(GlobalCountSourceCalculator::kNumOutputPackets, packet_dump.size());
  ASSERT_EQ(GlobalCountSourceCalculator::kNumOutputPackets, 5);
  EXPECT_FLOAT_EQ(1.0, packet_dump[0].Get<float>());
  EXPECT_FLOAT_EQ(5.5, packet_dump[1].Get<float>());
  EXPECT_FLOAT_EQ(14.35, packet_dump[2].Get<float>());
  EXPECT_FLOAT_EQ(26.575, packet_dump[3].Get<float>());
  EXPECT_FLOAT_EQ(39.9475, packet_dump[4].Get<float>());
  for (int i = 0; i < 5; ++i) {
    EXPECT_EQ(Timestamp(i), packet_dump[i].Timestamp());
  }
}

// Calculates the dot products of two streams of three-dimensional vectors.
TEST(CalculatorGraph, DotProduct) {
  // The use of BarrierInputStreamHandler in this graph aligns the input
  // packets to a calculator by arrival order rather than by timestamp.
  CalculatorGraphConfig config =
      ::mediapipe::ParseTextProtoOrDie<CalculatorGraphConfig>(R"(
        input_stream_handler {
          input_stream_handler: 'BarrierInputStreamHandler'
        }
        node {
          calculator: 'TestSequence1SourceCalculator'
          output_stream: 'test_sequence_1'
        }
        node {
          calculator: 'TestSequence2SourceCalculator'
          output_stream: 'test_sequence_2'
        }
        node {
          calculator: 'Modulo3SourceCalculator'
          output_stream: 'select_0_1_2'
        }
        node {
          calculator: 'DemuxUntimedCalculator'
          input_stream: 'INPUT:test_sequence_1'
          input_stream: 'SELECT:select_0_1_2'
          output_stream: 'OUTPUT:0:x_1'
          output_stream: 'OUTPUT:1:y_1'
          output_stream: 'OUTPUT:2:z_1'
        }
        node {
          calculator: 'DemuxUntimedCalculator'
          input_stream: 'INPUT:test_sequence_2'
          input_stream: 'SELECT:select_0_1_2'
          output_stream: 'OUTPUT:0:x_2'
          output_stream: 'OUTPUT:1:y_2'
          output_stream: 'OUTPUT:2:z_2'
        }
        node {
          calculator: 'IntMultiplierCalculator'
          input_stream: 'x_1'
          input_stream: 'x_2'
          output_stream: 'x_product'
        }
        node {
          calculator: 'IntMultiplierCalculator'
          input_stream: 'y_1'
          input_stream: 'y_2'
          output_stream: 'y_product'
        }
        node {
          calculator: 'IntMultiplierCalculator'
          input_stream: 'z_1'
          input_stream: 'z_2'
          output_stream: 'z_product'
        }
        node {
          calculator: 'IntAdderCalculator'
          input_stream: 'x_product'
          input_stream: 'y_product'
          input_stream: 'z_product'
          output_stream: 'dot_product'
        }
      )");
  std::vector<Packet> packet_dump;
  tool::AddVectorSink("dot_product", &config, &packet_dump);

  CalculatorGraph graph;
  MP_ASSERT_OK(graph.Initialize(config));
  MP_ASSERT_OK(graph.Run());

  // The calculator graph performs the following computation:
  //   test_sequence_1 is split into x_1, y_1, z_1.
  //   test_sequence_2 is split into x_2, y_2, z_2.
  //   x_product = x_1 * x_2
  //   y_product = y_1 * y_2
  //   z_product = z_1 * z_2
  //   dot_product = x_product + y_product + z_product
  //
  // The values in these streams are:
  //   test_sequence_1: 0, 1, 2, 3, 4, 5, 6, 7, 8,  9, 10, 11, 12, 13, 14
  //   test_sequence_2: 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15
  //   x_1:         0,  3,   6,   9,  12
  //   x_2:         1,  4,   7,  10,  13
  //   x_product:   0, 12,  42,  90, 156
  //   y_1:         1,  4,   7,  10,  13
  //   y_2:         2,  5,   8,  11,  14
  //   y_product:   2, 20,  56, 110, 182
  //   z_1:         2,  5,   8,  11,  14
  //   z_2:         3,  6,   9,  12,  15
  //   z_product:   6, 30,  72, 132, 210
  //   dot_product: 8, 62, 170, 332, 548

  ASSERT_EQ(kTestSequenceLength / 3, packet_dump.size());
  const int expected[] = {8, 62, 170, 332, 548};
  for (int i = 0; i < packet_dump.size(); ++i) {
    EXPECT_EQ(expected[i], packet_dump[i].Get<int>());
  }
}

TEST(CalculatorGraph, TerminatesOnCancelWithOpenGraphInputStreams) {
  CalculatorGraphConfig config =
      ::mediapipe::ParseTextProtoOrDie<CalculatorGraphConfig>(R"(
        node {
          calculator: 'PassThroughCalculator'
          input_stream: 'in_a'
          input_stream: 'in_b'
          output_stream: 'out_a'
          output_stream: 'out_b'
        }
        input_stream: 'in_a'
        input_stream: 'in_b'
      )");
  CalculatorGraph graph;
  MP_ASSERT_OK(graph.Initialize(config));
  MP_ASSERT_OK(graph.StartRun({}));
  MP_EXPECT_OK(graph.AddPacketToInputStream(
      "in_a", MakePacket<int>(1).At(Timestamp(1))));
  MP_EXPECT_OK(graph.CloseInputStream("in_a"));
  MP_EXPECT_OK(graph.AddPacketToInputStream(
      "in_b", MakePacket<int>(2).At(Timestamp(2))));
  MP_EXPECT_OK(graph.WaitUntilIdle());
  graph.Cancel();
  // This tests that the graph doesn't deadlock on WaitUntilDone (because
  // the scheduler thread is sleeping).
  ::mediapipe::Status status = graph.WaitUntilDone();
  EXPECT_EQ(status.code(), ::mediapipe::StatusCode::kCancelled);
}

TEST(CalculatorGraph, TerminatesOnCancelAfterPause) {
  CalculatorGraphConfig config =
      ::mediapipe::ParseTextProtoOrDie<CalculatorGraphConfig>(R"(
        node {
          calculator: 'PassThroughCalculator'
          input_stream: 'in'
          output_stream: 'out'
        }
        input_stream: 'in'
      )");
  CalculatorGraph graph;
  MP_ASSERT_OK(graph.Initialize(config));
  MP_ASSERT_OK(graph.StartRun({}));
  graph.Pause();
  // Make the PassThroughCalculator runnable while the scheduler is paused.
  MP_EXPECT_OK(
      graph.AddPacketToInputStream("in", MakePacket<int>(1).At(Timestamp(1))));
  // Now cancel the graph run. A non-empty scheduler queue should not prevent
  // the scheduler from terminating.
  graph.Cancel();
  // Any attempt to pause the scheduler after the graph run is cancelled should
  // be ignored.
  graph.Pause();
  // This tests that the graph doesn't deadlock on WaitUntilDone (because
  // the scheduler thread is sleeping).
  ::mediapipe::Status status = graph.WaitUntilDone();
  EXPECT_EQ(status.code(), ::mediapipe::StatusCode::kCancelled);
}

// A PacketGenerator that simply passes its input Packets through
// unchanged.  The inputs may be specified by tag or index.  The outputs
// must match the inputs exactly.  Any options may be specified and will
// also be ignored.
class PassThroughGenerator : public PacketGenerator {
 public:
  static ::mediapipe::Status FillExpectations(
      const PacketGeneratorOptions& extendable_options, PacketTypeSet* inputs,
      PacketTypeSet* outputs) {
    if (!inputs->TagMap()->SameAs(*outputs->TagMap())) {
      return ::mediapipe::InvalidArgumentError(
          "Input and outputs to PassThroughGenerator must use the same tags "
          "and indexes.");
    }
    for (CollectionItemId id = inputs->BeginId(); id < inputs->EndId(); ++id) {
      inputs->Get(id).SetAny();
      outputs->Get(id).SetSameAs(&inputs->Get(id));
    }
    return ::mediapipe::OkStatus();
  }

  static ::mediapipe::Status Generate(
      const PacketGeneratorOptions& extendable_options,
      const PacketSet& input_side_packets, PacketSet* output_side_packets) {
    for (CollectionItemId id = input_side_packets.BeginId();
         id < input_side_packets.EndId(); ++id) {
      output_side_packets->Get(id) = input_side_packets.Get(id);
    }
    return ::mediapipe::OkStatus();
  }
};
REGISTER_PACKET_GENERATOR(PassThroughGenerator);
TEST(CalculatorGraph, RecoverAfterRunError) {
  PacketGeneratorGraph generator_graph;
  CalculatorGraphConfig config =
      ::mediapipe::ParseTextProtoOrDie<CalculatorGraphConfig>(R"(
        node {
          name: 'calculator1'
          calculator: 'CountingSourceCalculator'
          output_stream: 'count1'
          input_side_packet: 'MAX_COUNT:max_count2'
          input_side_packet: 'ERROR_COUNT:max_error2'
        }
        packet_generator {
          packet_generator: 'EnsurePositivePacketGenerator'
          input_side_packet: 'max_count1'
          output_side_packet: 'max_count2'
          input_side_packet: 'max_error1'
          output_side_packet: 'max_error2'
        }
        status_handler {
          status_handler: 'FailableStatusHandler'
          input_side_packet: 'status_handler_command'
        }
      )");

  int packet_count = 0;
  CalculatorGraph graph;
  MP_ASSERT_OK(graph.Initialize(config, {}));
  MP_ASSERT_OK(graph.ObserveOutputStream("count1",
                                         [&packet_count](const Packet& packet) {
                                           ++packet_count;
                                           return ::mediapipe::OkStatus();
                                         }));
  // Set ERROR_COUNT higher than MAX_COUNT and hence the calculator will
  // finish successfully.
  packet_count = 0;
  MP_ASSERT_OK(graph.Run({{"max_count1", MakePacket<int>(10)},
                          {"max_error1", MakePacket<int>(20)},
                          {"status_handler_command",
                           MakePacket<int>(FailableStatusHandler::kOk)}}));
  EXPECT_EQ(packet_count, 10);
  // Fail in PacketGenerator::Generate().
  // Negative max_count1 will cause EnsurePositivePacketGenerator to fail.
  ASSERT_FALSE(graph
                   .Run({{"max_count1", MakePacket<int>(-1)},
                         {"max_error1", MakePacket<int>(20)},
                         {"status_handler_command",
                          MakePacket<int>(FailableStatusHandler::kOk)}})
                   .ok());
  packet_count = 0;
  MP_ASSERT_OK(graph.Run({{"max_count1", MakePacket<int>(10)},
                          {"max_error1", MakePacket<int>(20)},
                          {"status_handler_command",
                           MakePacket<int>(FailableStatusHandler::kOk)}}));
  EXPECT_EQ(packet_count, 10);
  // Fail in PacketGenerator::Generate() also fail in StatusHandler.
  ASSERT_FALSE(graph
                   .Run({{"max_count1", MakePacket<int>(-1)},
                         {"max_error1", MakePacket<int>(20)},
                         {"status_handler_command",
                          MakePacket<int>(FailableStatusHandler::kFailPreRun)}})
                   .ok());
  packet_count = 0;
  MP_ASSERT_OK(graph.Run({{"max_count1", MakePacket<int>(10)},
                          {"max_error1", MakePacket<int>(20)},
                          {"status_handler_command",
                           MakePacket<int>(FailableStatusHandler::kOk)}}));
  EXPECT_EQ(packet_count, 10);
  ASSERT_FALSE(
      graph
          .Run({{"max_count1", MakePacket<int>(-1)},
                {"max_error1", MakePacket<int>(20)},
                {"status_handler_command",
                 MakePacket<int>(FailableStatusHandler::kFailPostRun)}})
          .ok());
  packet_count = 0;
  MP_ASSERT_OK(graph.Run({{"max_count1", MakePacket<int>(10)},
                          {"max_error1", MakePacket<int>(20)},
                          {"status_handler_command",
                           MakePacket<int>(FailableStatusHandler::kOk)}}));
  EXPECT_EQ(packet_count, 10);
  // Fail in Calculator::Process().
  ASSERT_FALSE(graph
                   .Run({{"max_count1", MakePacket<int>(1000)},
                         {"max_error1", MakePacket<int>(10)},
                         {"status_handler_command",
                          MakePacket<int>(FailableStatusHandler::kOk)}})
                   .ok());
  packet_count = 0;
  MP_ASSERT_OK(graph.Run({{"max_count1", MakePacket<int>(10)},
                          {"max_error1", MakePacket<int>(20)},
                          {"status_handler_command",
                           MakePacket<int>(FailableStatusHandler::kOk)}}));
  EXPECT_EQ(packet_count, 10);
  // Fail in Calculator::Process() also fail in StatusHandler.
  ASSERT_FALSE(graph
                   .Run({{"max_count1", MakePacket<int>(1000)},
                         {"max_error1", MakePacket<int>(10)},
                         {"status_handler_command",
                          MakePacket<int>(FailableStatusHandler::kFailPreRun)}})
                   .ok());
  packet_count = 0;
  MP_ASSERT_OK(graph.Run({{"max_count1", MakePacket<int>(10)},
                          {"max_error1", MakePacket<int>(20)},
                          {"status_handler_command",
                           MakePacket<int>(FailableStatusHandler::kOk)}}));
  EXPECT_EQ(packet_count, 10);
  ASSERT_FALSE(
      graph
          .Run({{"max_count1", MakePacket<int>(1000)},
                {"max_error1", MakePacket<int>(10)},
                {"status_handler_command",
                 MakePacket<int>(FailableStatusHandler::kFailPostRun)}})
          .ok());
  packet_count = 0;
  MP_ASSERT_OK(graph.Run({{"max_count1", MakePacket<int>(10)},
                          {"max_error1", MakePacket<int>(20)},
                          {"status_handler_command",
                           MakePacket<int>(FailableStatusHandler::kOk)}}));
  EXPECT_EQ(packet_count, 10);
}

TEST(CalculatorGraph, SetInputStreamMaxQueueSizeWorksSlowCalculator) {
  using Semaphore = SemaphoreCalculator::Semaphore;
  CalculatorGraphConfig config =
      ::mediapipe::ParseTextProtoOrDie<CalculatorGraphConfig>(R"(
        node {
          calculator: 'SemaphoreCalculator'
          input_stream: 'in'
          output_stream: 'out'
          input_side_packet: 'POST_SEM:post_sem'
          input_side_packet: 'WAIT_SEM:wait_sem'
        }
        node {
          calculator: 'SemaphoreCalculator'
          input_stream: 'in_2'
          output_stream: 'out_2'
          input_side_packet: 'POST_SEM:post_sem_busy'
          input_side_packet: 'WAIT_SEM:wait_sem_busy'
        }
        input_stream: 'in'
        input_stream: 'in_2'
        max_queue_size: 100
      )");
  CalculatorGraph graph;
  MP_ASSERT_OK(graph.Initialize(config));
  graph.SetGraphInputStreamAddMode(
      CalculatorGraph::GraphInputStreamAddMode::ADD_IF_NOT_FULL);
  MP_ASSERT_OK(graph.SetInputStreamMaxQueueSize("in", 1));

  Semaphore calc_entered_process(0);
  Semaphore calc_can_exit_process(0);
  Semaphore calc_entered_process_busy(0);
  Semaphore calc_can_exit_process_busy(0);
  MP_ASSERT_OK(graph.StartRun({
      {"post_sem", MakePacket<Semaphore*>(&calc_entered_process)},
      {"wait_sem", MakePacket<Semaphore*>(&calc_can_exit_process)},
      {"post_sem_busy", MakePacket<Semaphore*>(&calc_entered_process_busy)},
      {"wait_sem_busy", MakePacket<Semaphore*>(&calc_can_exit_process_busy)},
  }));

  Timestamp timestamp(0);
  // Prevent deadlock resolution by running the "busy" SemaphoreCalculator
  // for the duration of the test.
  MP_EXPECT_OK(
      graph.AddPacketToInputStream("in_2", MakePacket<int>(0).At(timestamp)));
  MP_EXPECT_OK(
      graph.AddPacketToInputStream("in", MakePacket<int>(0).At(timestamp++)));
  for (int i = 1; i < 20; ++i, ++timestamp) {
    // Wait for the calculator to begin its Process call.
    calc_entered_process.Acquire(1);
    // Now the calculator is stuck processing a packet. We can queue up
    // another one.
    MP_EXPECT_OK(
        graph.AddPacketToInputStream("in", MakePacket<int>(i).At(timestamp)));
    // We should be prevented from adding another, since the queue is now full.
    ::mediapipe::Status status = graph.AddPacketToInputStream(
        "in", MakePacket<int>(i).At(timestamp + 1));
    EXPECT_EQ(status.code(), ::mediapipe::StatusCode::kUnavailable);
    // Allow calculator to complete its Process call.
    calc_can_exit_process.Release(1);
  }
  // Allow the final Process call to complete.
  calc_can_exit_process.Release(1);
  calc_can_exit_process_busy.Release(1);

  MP_ASSERT_OK(graph.CloseAllInputStreams());
  MP_ASSERT_OK(graph.WaitUntilDone());
}

// Verify the scheduler unthrottles the graph input stream to avoid a deadlock,
// and won't enter a busy loop.
TEST(CalculatorGraph, AddPacketNoBusyLoop) {
  // The DecimatorCalculator ouputs 1 out of every 101 input packets and drops
  // the rest, without setting the next timestamp bound on its output. As a
  // result, the MergeCalculator is not runnable in between and packets on its
  // "in" input stream will be queued and exceed the max queue size.
  //
  //               in
  //               |
  //              / \
  //             /   \
  //            /     \
  //            |      \
  //            v      |
  //       +---------+ |
  // 101:1 |Decimator| | <== Packet buildup
  //       +---------+ |
  //            |      |
  //            v      v
  //          +----------+
  //          |  Merge   |
  //          +----------+
  //               |
  //               v
  //              out
  //
  CalculatorGraphConfig config =
      ::mediapipe::ParseTextProtoOrDie<CalculatorGraphConfig>(R"(
        input_stream: 'in'
        max_queue_size: 1
        node {
          calculator: 'DecimatorCalculator'
          input_stream: 'in'
          output_stream: 'decimated_in'
        }
        node {
          calculator: 'MergeCalculator'
          input_stream: 'decimated_in'
          input_stream: 'in'
          output_stream: 'out'
        }
      )");
  CalculatorGraph graph;
  MP_ASSERT_OK(graph.Initialize(config));
  graph.SetGraphInputStreamAddMode(
      CalculatorGraph::GraphInputStreamAddMode::WAIT_TILL_NOT_FULL);
  std::vector<Packet> out_packets;  // Packets from the output stream "out".
  MP_ASSERT_OK(
      graph.ObserveOutputStream("out", [&out_packets](const Packet& packet) {
        out_packets.push_back(packet);
        return ::mediapipe::OkStatus();
      }));

  MP_ASSERT_OK(graph.StartRun({}));

  const int kDecimationRatio = DecimatorCalculator::kDecimationRatio;
  // To leave the graph input stream "in" in the throttled state, kNumPackets
  // can be any value other than a multiple of kDecimationRatio plus one.
  const int kNumPackets = 2 * kDecimationRatio;
  for (int i = 0; i < kNumPackets; ++i) {
    MP_EXPECT_OK(graph.AddPacketToInputStream(
        "in", MakePacket<int>(i).At(Timestamp(i))));
  }

  // The graph input stream "in" is throttled. Wait until the graph is idle.
  MP_ASSERT_OK(graph.WaitUntilIdle());
  // Check that Pause() does not block forever trying to acquire a mutex.
  // This is a regression test for an old bug.
  graph.Pause();
  graph.Resume();

  MP_ASSERT_OK(graph.CloseAllInputStreams());
  MP_ASSERT_OK(graph.WaitUntilDone());

  // The expected output packets are:
  //   "Timestamp(0) 0 0"
  //   "Timestamp(1) empty 1"
  //   ...
  //   "Timestamp(100) empty 100"
  //   "Timestamp(101) 101 101"
  //   "Timestamp(102) empty 102"
  //   ...
  //   "Timestamp(201) empty 201"
  ASSERT_EQ(kNumPackets, out_packets.size());
  for (int i = 0; i < out_packets.size(); ++i) {
    std::string format = (i % kDecimationRatio == 0) ? "Timestamp($0) $0 $0"
                                                     : "Timestamp($0) empty $0";
    std::string expected = absl::Substitute(format, i);
    EXPECT_EQ(expected, out_packets[i].Get<std::string>());
    EXPECT_EQ(Timestamp(i), out_packets[i].Timestamp());
  }
}

namespace nested_ns {

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
REGISTER_CALCULATOR(::mediapipe::nested_ns::ProcessCallbackCalculator);

}  // namespace nested_ns

TEST(CalculatorGraph, CalculatorInNamepsace) {
  CalculatorGraphConfig config;
  CHECK(proto_ns::TextFormat::ParseFromString(R"(
      input_stream: 'in_a'
      node {
        calculator: 'mediapipe.nested_ns.ProcessCallbackCalculator'
        input_stream: 'in_a'
        output_stream: 'out_a'
        input_side_packet: 'callback_1'
      }
      )",
                                              &config));
  CalculatorGraph graph;
  MP_ASSERT_OK(graph.Initialize(config));
  nested_ns::ProcessFunction callback_1;
  MP_ASSERT_OK(
      graph.StartRun({{"callback_1", AdoptAsUniquePtr(new auto(callback_1))}}));
  MP_EXPECT_OK(graph.WaitUntilIdle());
}

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

TEST(CalculatorGraph, ObserveOutputStream) {
  const int max_count = 10;
  CalculatorGraphConfig config =
      ::mediapipe::ParseTextProtoOrDie<CalculatorGraphConfig>(R"(
        node {
          calculator: 'CountingSourceCalculator'
          output_stream: 'count'
          input_side_packet: 'MAX_COUNT:max_count'
        }
        node {
          calculator: 'PassThroughCalculator'
          input_stream: 'count'
          output_stream: 'mid'
        }
        node {
          calculator: 'PassThroughCalculator'
          input_stream: 'mid'
          output_stream: 'out'
        }
      )");
  CalculatorGraph graph;
  MP_ASSERT_OK(
      graph.Initialize(config, {{"max_count", MakePacket<int>(max_count)}}));
  // Observe the internal output stream "count" and the unconnected output
  // stream "out".
  std::vector<Packet> count_packets;  // Packets from the output stream "count".
  std::vector<Packet> out_packets;    // Packets from the output stream "out".
  MP_ASSERT_OK(graph.ObserveOutputStream(
      "count", [&count_packets](const Packet& packet) {
        count_packets.push_back(packet);
        return ::mediapipe::OkStatus();
      }));
  MP_ASSERT_OK(
      graph.ObserveOutputStream("out", [&out_packets](const Packet& packet) {
        out_packets.push_back(packet);
        return ::mediapipe::OkStatus();
      }));
  MP_ASSERT_OK(graph.Run());
  ASSERT_EQ(max_count, count_packets.size());
  for (int i = 0; i < count_packets.size(); ++i) {
    EXPECT_EQ(i, count_packets[i].Get<int>());
    EXPECT_EQ(Timestamp(i), count_packets[i].Timestamp());
  }
  ASSERT_EQ(max_count, out_packets.size());
  for (int i = 0; i < out_packets.size(); ++i) {
    EXPECT_EQ(i, out_packets[i].Get<int>());
    EXPECT_EQ(Timestamp(i), out_packets[i].Timestamp());
  }
}

class PassThroughSubgraph : public Subgraph {
 public:
  ::mediapipe::StatusOr<CalculatorGraphConfig> GetConfig(
      const SubgraphOptions& options) override {
    CalculatorGraphConfig config =
        ::mediapipe::ParseTextProtoOrDie<CalculatorGraphConfig>(R"(
          input_stream: 'INPUT:input'
          output_stream: 'OUTPUT:output'
          node {
            calculator: 'PassThroughCalculator'
            input_stream: 'input'
            output_stream: 'output'
          }
        )");
    return config;
  }
};
REGISTER_MEDIAPIPE_GRAPH(PassThroughSubgraph);

TEST(CalculatorGraph, ObserveOutputStreamSubgraph) {
  const int max_count = 10;
  CalculatorGraphConfig config =
      ::mediapipe::ParseTextProtoOrDie<CalculatorGraphConfig>(R"(
        node {
          calculator: 'CountingSourceCalculator'
          output_stream: 'count'
          input_side_packet: 'MAX_COUNT:max_count'
        }
        node {
          calculator: 'PassThroughSubgraph'
          input_stream: 'INPUT:count'
          output_stream: 'OUTPUT:out'
        }
      )");
  CalculatorGraph graph;
  MP_ASSERT_OK(
      graph.Initialize(config, {{"max_count", MakePacket<int>(max_count)}}));
  // Observe the unconnected output stream "out".
  std::vector<Packet> out_packets;  // Packets from the output stream "out".
  MP_ASSERT_OK(
      graph.ObserveOutputStream("out", [&out_packets](const Packet& packet) {
        out_packets.push_back(packet);
        return ::mediapipe::OkStatus();
      }));
  MP_ASSERT_OK(graph.Run());
  ASSERT_EQ(max_count, out_packets.size());
  for (int i = 0; i < out_packets.size(); ++i) {
    EXPECT_EQ(i, out_packets[i].Get<int>());
    EXPECT_EQ(Timestamp(i), out_packets[i].Timestamp());
  }
}

TEST(CalculatorGraph, ObserveOutputStreamError) {
  const int max_count = 10;
  const int fail_count = 6;
  CalculatorGraphConfig config =
      ::mediapipe::ParseTextProtoOrDie<CalculatorGraphConfig>(R"(
        node {
          calculator: 'CountingSourceCalculator'
          output_stream: 'count'
          input_side_packet: 'MAX_COUNT:max_count'
        }
        node {
          calculator: 'PassThroughCalculator'
          input_stream: 'count'
          output_stream: 'mid'
        }
        node {
          calculator: 'PassThroughCalculator'
          input_stream: 'mid'
          output_stream: 'out'
        }
      )");
  CalculatorGraph graph;
  MP_ASSERT_OK(
      graph.Initialize(config, {{"max_count", MakePacket<int>(max_count)}}));
  // Observe the internal output stream "count" and the unconnected output
  // stream "out".
  std::vector<Packet> count_packets;  // Packets from the output stream "count".
  std::vector<Packet> out_packets;    // Packets from the output stream "out".
  MP_ASSERT_OK(graph.ObserveOutputStream(
      "count", [&count_packets](const Packet& packet) {
        count_packets.push_back(packet);
        if (count_packets.size() >= fail_count) {
          return ::mediapipe::UnknownError("Expected. MagicString-eatnhuea");
        } else {
          return ::mediapipe::OkStatus();
        }
      }));
  MP_ASSERT_OK(
      graph.ObserveOutputStream("out", [&out_packets](const Packet& packet) {
        out_packets.push_back(packet);
        return ::mediapipe::OkStatus();
      }));
  ::mediapipe::Status status = graph.Run();
  ASSERT_THAT(status.message(), testing::HasSubstr("MagicString-eatnhuea"));
  ASSERT_EQ(fail_count, count_packets.size());
  for (int i = 0; i < count_packets.size(); ++i) {
    EXPECT_EQ(i, count_packets[i].Get<int>());
    EXPECT_EQ(Timestamp(i), count_packets[i].Timestamp());
  }
}

TEST(CalculatorGraph, ObserveOutputStreamNonexistent) {
  const int max_count = 10;
  CalculatorGraphConfig config =
      ::mediapipe::ParseTextProtoOrDie<CalculatorGraphConfig>(R"(
        node {
          calculator: 'CountingSourceCalculator'
          output_stream: 'count'
          input_side_packet: 'MAX_COUNT:max_count'
        }
        node {
          calculator: 'PassThroughCalculator'
          input_stream: 'count'
          output_stream: 'mid'
        }
        node {
          calculator: 'PassThroughCalculator'
          input_stream: 'mid'
          output_stream: 'out'
        }
      )");
  CalculatorGraph graph;
  MP_ASSERT_OK(
      graph.Initialize(config, {{"max_count", MakePacket<int>(max_count)}}));
  // Observe the internal output stream "count".
  std::vector<Packet> count_packets;  // Packets from the output stream "count".
  ::mediapipe::Status status = graph.ObserveOutputStream(
      "not_found", [&count_packets](const Packet& packet) {
        count_packets.push_back(packet);
        return ::mediapipe::OkStatus();
      });
  EXPECT_EQ(status.code(), ::mediapipe::StatusCode::kNotFound);
  EXPECT_THAT(status.message(), testing::HasSubstr("not_found"));
}

// Verify that after a fast source node is closed, a slow sink node can
// consume all the accumulated input packets. In other words, closing an
// output stream still allows its mirrors to process all the received packets.
TEST(CalculatorGraph, FastSourceSlowSink) {
  const int max_count = 10;
  CalculatorGraphConfig config =
      ::mediapipe::ParseTextProtoOrDie<CalculatorGraphConfig>(R"(
        num_threads: 2
        max_queue_size: 100
        node {
          calculator: 'CountingSourceCalculator'
          output_stream: 'out'
          input_side_packet: 'MAX_COUNT:max_count'
        }
        node { calculator: 'SlowCountingSinkCalculator' input_stream: 'out' }
      )");
  CalculatorGraph graph;
  MP_ASSERT_OK(
      graph.Initialize(config, {{"max_count", MakePacket<int>(max_count)}}));
  MP_EXPECT_OK(graph.Run());
}

TEST(CalculatorGraph, GraphFinishesWhilePaused) {
  // The graph contains only one node, and the node runs only once. This test
  // sets up the following sequence of events (all times in milliseconds):
  //
  //         Application thread    Worker thread
  //
  // T=0     graph.StartRun        OneShot20MsCalculator::Process starts
  // T=10    graph.Pause
  // T=20                          OneShot20MsCalculator::Process ends.
  //                               So graph finishes running while paused.
  // T=30    graph.Resume
  //
  // graph.WaitUntilDone must not block forever.
  CalculatorGraphConfig config =
      ::mediapipe::ParseTextProtoOrDie<CalculatorGraphConfig>(R"(
        node { calculator: 'OneShot20MsCalculator' }
      )");
  CalculatorGraph graph;
  MP_ASSERT_OK(graph.Initialize(config));
  MP_EXPECT_OK(graph.StartRun({}));
  absl::SleepFor(absl::Milliseconds(10));
  graph.Pause();
  absl::SleepFor(absl::Milliseconds(20));
  graph.Resume();
  MP_EXPECT_OK(graph.WaitUntilDone());
}

// There should be no memory leaks, no error messages (requires manual
// inspection of the test log), etc.
TEST(CalculatorGraph, ConstructAndDestruct) { CalculatorGraph graph; }

// A regression test for b/36364314. UnitDelayCalculator outputs a packet in
// Open(). ErrorOnOpenCalculator fails in Open() if ERROR_ON_OPEN is true.
TEST(CalculatorGraph, RecoverAfterPreviousFailInOpen) {
  const int max_count = 10;
  CalculatorGraphConfig config =
      ::mediapipe::ParseTextProtoOrDie<CalculatorGraphConfig>(R"(
        node {
          calculator: 'CountingSourceCalculator'
          output_stream: 'a'
          input_side_packet: 'MAX_COUNT:max_count'
        }
        node {
          calculator: 'UnitDelayCalculator'
          input_stream: 'a'
          output_stream: 'b'
        }
        node {
          calculator: 'PassThroughCalculator'
          input_stream: 'b'
          output_stream: 'c'
        }
        node {
          calculator: 'ErrorOnOpenCalculator'
          input_stream: 'c'
          output_stream: 'd'
          input_side_packet: 'ERROR_ON_OPEN:fail'
        }
        node { calculator: 'IntSinkCalculator' input_stream: 'd' }
      )");
  CalculatorGraph graph;
  MP_ASSERT_OK(
      graph.Initialize(config, {{"max_count", MakePacket<int>(max_count)}}));
  for (int i = 0; i < 2; ++i) {
    EXPECT_FALSE(graph.Run({{"fail", MakePacket<bool>(true)}}).ok());
    MP_EXPECT_OK(graph.Run({{"fail", MakePacket<bool>(false)}}));
  }
}

TEST(CalculatorGraph, ReuseValidatedGraphConfig) {
  CalculatorGraphConfig config =
      ::mediapipe::ParseTextProtoOrDie<CalculatorGraphConfig>(R"(
        packet_generator {
          packet_generator: "StaticCounterStringGenerator"
          input_side_packet: "created_by_factory"
          input_side_packet: "input_in_initialize"
          output_side_packet: "foo1"
        }
        packet_generator {
          packet_generator: "StaticCounterStringGenerator"
          input_side_packet: "created_by_factory"
          input_side_packet: "input_in_initialize"
          input_side_packet: "foo1"
          output_side_packet: "foo2"
        }
        packet_generator {
          packet_generator: "StaticCounterStringGenerator"
          input_side_packet: "created_by_factory"
          input_side_packet: "input_in_initialize"
          input_side_packet: "input_in_run"
          output_side_packet: "foo3"
        }
        packet_generator {
          packet_generator: "StaticCounterStringGenerator"
          input_side_packet: "created_by_factory"
          input_side_packet: "input_in_initialize"
          input_side_packet: "input_in_run"
          input_side_packet: "foo3"
          output_side_packet: "foo4"
        }
        node {
          calculator: "GlobalCountSourceCalculator"
          input_side_packet: "global_counter"
          output_stream: "unused"
        }
      )");
  ValidatedGraphConfig validated_graph;
  MP_ASSERT_OK(validated_graph.Initialize(config));

  std::atomic<int> global_counter(0);
  Packet global_counter_packet = Adopt(new auto(&global_counter));

  absl::FixedArray<CalculatorGraph> graphs(30);
  for (int i = 0; i < graphs.size(); ++i) {
    CalculatorGraph& graph = graphs[i];
    int initial_generator_count =
        StaticCounterStringGenerator::NumPacketsGenerated();
    int initial_calculator_count = global_counter.load();
    MP_ASSERT_OK(graph.Initialize(
        config,
        {{"created_by_factory", MakePacket<std::string>("default string")},
         {"input_in_initialize", MakePacket<int>(10)},
         {"global_counter", global_counter_packet}}));
    EXPECT_EQ(initial_generator_count + 2,
              StaticCounterStringGenerator::NumPacketsGenerated());
    EXPECT_EQ(initial_calculator_count, global_counter.load());
  }
  for (int k = 0; k < 10; ++k) {
    for (int i = 0; i < graphs.size(); ++i) {
      CalculatorGraph& graph = graphs[i];
      int initial_generator_count =
          StaticCounterStringGenerator::NumPacketsGenerated();
      int initial_calculator_count = global_counter.load();
      MP_ASSERT_OK(graph.Run({{"input_in_run", MakePacket<int>(11)}}));
      EXPECT_EQ(initial_generator_count + 2,
                StaticCounterStringGenerator::NumPacketsGenerated());
      EXPECT_EQ(initial_calculator_count +
                    GlobalCountSourceCalculator::kNumOutputPackets,
                global_counter.load());
    }
  }
}

class TestRangeStdDevSubgraph : public Subgraph {
 public:
  ::mediapipe::StatusOr<CalculatorGraphConfig> GetConfig(
      const SubgraphOptions& options) override {
    CalculatorGraphConfig config =
        ::mediapipe::ParseTextProtoOrDie<CalculatorGraphConfig>(R"(
          input_side_packet: 'node_converted'
          output_stream: 'DATA:range'
          output_stream: 'SUM:range_sum'
          output_stream: 'MEAN:range_mean'
          output_stream: 'STDDEV:range_stddev'
          node {
            calculator: 'RangeCalculator'
            output_stream: 'range'
            output_stream: 'range_sum'
            output_stream: 'range_mean'
            input_side_packet: 'node_converted'
          }
          node {
            calculator: 'StdDevCalculator'
            input_stream: 'DATA:range'
            input_stream: 'MEAN:range_mean'
            output_stream: 'range_stddev'
          }
        )");
    return config;
  }
};
REGISTER_MEDIAPIPE_GRAPH(TestRangeStdDevSubgraph);

class TestMergeSaverSubgraph : public Subgraph {
 public:
  ::mediapipe::StatusOr<CalculatorGraphConfig> GetConfig(
      const SubgraphOptions& options) override {
    CalculatorGraphConfig config =
        ::mediapipe::ParseTextProtoOrDie<CalculatorGraphConfig>(R"(
          input_stream: 'DATA1:range1'
          input_stream: 'DATA2:range2'
          output_stream: 'MERGE:merge'
          output_stream: 'FINAL:final'
          node {
            name: 'merger'
            calculator: 'MergeCalculator'
            input_stream: 'range1'
            input_stream: 'range2'
            output_stream: 'merge'
          }
          node {
            calculator: 'SaverCalculator'
            input_stream: 'merge'
            output_stream: 'final'
          }
        )");
    return config;
  }
};
REGISTER_MEDIAPIPE_GRAPH(TestMergeSaverSubgraph);

CalculatorGraphConfig GetConfigWithSubgraphs() {
  CalculatorGraphConfig proto =
      ::mediapipe::ParseTextProtoOrDie<CalculatorGraphConfig>(R"(
        # Ensure stream name for FindOutputStreamManager
        output_stream: 'MERGE:merge'
        packet_generator {
          packet_generator: 'IntSplitterPacketGenerator'
          input_side_packet: 'node_3'
          output_side_packet: 'node_3_converted'
        }
        packet_generator {
          packet_generator: 'TaggedIntSplitterPacketGenerator'
          input_side_packet: 'node_5'
          output_side_packet: 'HIGH:unused_high'
          output_side_packet: 'LOW:unused_low'
          output_side_packet: 'PAIR:node_5_converted'
        }
        node {
          calculator: 'TestRangeStdDevSubgraph'
          input_side_packet: 'node_3_converted'
          output_stream: 'DATA:range3'
          output_stream: 'SUM:range3_sum'
          output_stream: 'MEAN:range3_mean'
          output_stream: 'STDDEV:range3_stddev'
        }
        node {
          calculator: 'TestRangeStdDevSubgraph'
          input_side_packet: 'node_5_converted'
          output_stream: 'DATA:range5'
          output_stream: 'SUM:range5_sum'
          output_stream: 'MEAN:range5_mean'
          output_stream: 'STDDEV:range5_stddev'
        }
        node {
          name: 'copy_range5'
          calculator: 'PassThroughCalculator'
          input_stream: 'range5'
          output_stream: 'range5_copy'
        }
        node {
          calculator: 'TestMergeSaverSubgraph'
          input_stream: 'DATA1:range3'
          input_stream: 'DATA2:range5_copy'
          output_stream: 'MERGE:merge'
          output_stream: 'FINAL:final'
        }
        node {
          calculator: 'TestMergeSaverSubgraph'
          input_stream: 'DATA1:range3_sum'
          input_stream: 'DATA2:range5_sum'
          output_stream: 'FINAL:final_sum'
        }
        node {
          calculator: 'TestMergeSaverSubgraph'
          input_stream: 'DATA1:range3_stddev'
          input_stream: 'DATA2:range5_stddev'
          output_stream: 'FINAL:final_stddev'
        }
      )");
  return proto;
}

TEST(CalculatorGraph, RunsCorrectlyWithSubgraphs) {
  CalculatorGraph graph;
  CalculatorGraphConfig proto = GetConfigWithSubgraphs();
  RunComprehensiveTest(&graph, proto, /*define_node_5=*/true);
}

TEST(CalculatorGraph, SetExecutorTwice) {
  // SetExecutor must not be called more than once for the same executor name.
  CalculatorGraph graph;
  MP_EXPECT_OK(
      graph.SetExecutor("xyz", std::make_shared<ThreadPoolExecutor>(1)));
  MP_EXPECT_OK(
      graph.SetExecutor("abc", std::make_shared<ThreadPoolExecutor>(1)));
  ::mediapipe::Status status =
      graph.SetExecutor("xyz", std::make_shared<ThreadPoolExecutor>(1));
  EXPECT_EQ(status.code(), ::mediapipe::StatusCode::kAlreadyExists);
  EXPECT_THAT(status.message(), testing::HasSubstr("xyz"));
}

TEST(CalculatorGraph, ReservedNameSetExecutor) {
  // A reserved executor name such as "__gpu" must not be used.
  CalculatorGraph graph;
  ::mediapipe::Status status =
      graph.SetExecutor("__gpu", std::make_shared<ThreadPoolExecutor>(1));
  EXPECT_EQ(status.code(), ::mediapipe::StatusCode::kInvalidArgument);
  EXPECT_THAT(status.message(), testing::AllOf(testing::HasSubstr("__gpu"),
                                               testing::HasSubstr("reserved")));
}

TEST(CalculatorGraph, ReservedNameExecutorConfig) {
  // A reserved executor name such as "__gpu" must not be used.
  CalculatorGraph graph;
  CalculatorGraphConfig config =
      ::mediapipe::ParseTextProtoOrDie<CalculatorGraphConfig>(R"(
        input_stream: 'in'
        executor {
          name: '__gpu'
          type: 'ThreadPoolExecutor'
          options {
            [mediapipe.ThreadPoolExecutorOptions.ext] { num_threads: 1 }
          }
        }
        node {
          calculator: 'PassThroughCalculator'
          input_stream: 'in'
          output_stream: 'out'
        }
      )");
  ::mediapipe::Status status = graph.Initialize(config);
  EXPECT_EQ(status.code(), ::mediapipe::StatusCode::kInvalidArgument);
  EXPECT_THAT(status.message(), testing::AllOf(testing::HasSubstr("__gpu"),
                                               testing::HasSubstr("reserved")));
}

TEST(CalculatorGraph, ReservedNameNodeExecutor) {
  // A reserved executor name such as "__gpu" must not be used.
  CalculatorGraph graph;
  CalculatorGraphConfig config =
      ::mediapipe::ParseTextProtoOrDie<CalculatorGraphConfig>(R"(
        input_stream: 'in'
        node {
          calculator: 'PassThroughCalculator'
          executor: '__gpu'
          input_stream: 'in'
          output_stream: 'out'
        }
      )");
  ::mediapipe::Status status = graph.Initialize(config);
  EXPECT_EQ(status.code(), ::mediapipe::StatusCode::kInvalidArgument);
  EXPECT_THAT(status.message(), testing::AllOf(testing::HasSubstr("__gpu"),
                                               testing::HasSubstr("reserved")));
}

TEST(CalculatorGraph, NonExistentExecutor) {
  // Any executor used by a calculator node must either be created by the
  // graph (which requires an ExecutorConfig with a "type" field) or be
  // provided to the graph with a CalculatorGraph::SetExecutor() call.
  CalculatorGraph graph;
  CalculatorGraphConfig config =
      ::mediapipe::ParseTextProtoOrDie<CalculatorGraphConfig>(R"(
        input_stream: 'in'
        node {
          calculator: 'PassThroughCalculator'
          executor: 'xyz'
          input_stream: 'in'
          output_stream: 'out'
        }
      )");
  ::mediapipe::Status status = graph.Initialize(config);
  EXPECT_EQ(status.code(), ::mediapipe::StatusCode::kInvalidArgument);
  EXPECT_THAT(status.message(),
              testing::AllOf(testing::HasSubstr("xyz"),
                             testing::HasSubstr("not declared")));
}

TEST(CalculatorGraph, UndeclaredExecutor) {
  // Any executor used by a calculator node must be declared in an
  // ExecutorConfig, even if the executor is provided to the graph with a
  // CalculatorGraph::SetExecutor() call.
  CalculatorGraph graph;
  MP_ASSERT_OK(
      graph.SetExecutor("xyz", std::make_shared<ThreadPoolExecutor>(1)));
  CalculatorGraphConfig config =
      ::mediapipe::ParseTextProtoOrDie<CalculatorGraphConfig>(R"(
        input_stream: 'in'
        node {
          calculator: 'PassThroughCalculator'
          executor: 'xyz'
          input_stream: 'in'
          output_stream: 'out'
        }
      )");
  ::mediapipe::Status status = graph.Initialize(config);
  EXPECT_EQ(status.code(), ::mediapipe::StatusCode::kInvalidArgument);
  EXPECT_THAT(status.message(),
              testing::AllOf(testing::HasSubstr("xyz"),
                             testing::HasSubstr("not declared")));
}

TEST(CalculatorGraph, UntypedExecutorDeclaredButNotSet) {
  // If an executor is declared without a "type" field, it must be provided to
  // the graph with a CalculatorGraph::SetExecutor() call.
  CalculatorGraph graph;
  CalculatorGraphConfig config =
      ::mediapipe::ParseTextProtoOrDie<CalculatorGraphConfig>(R"(
        input_stream: 'in'
        executor { name: 'xyz' }
        node {
          calculator: 'PassThroughCalculator'
          executor: 'xyz'
          input_stream: 'in'
          output_stream: 'out'
        }
      )");
  ::mediapipe::Status status = graph.Initialize(config);
  EXPECT_EQ(status.code(), ::mediapipe::StatusCode::kInvalidArgument);
  EXPECT_THAT(status.message(),
              testing::AllOf(testing::HasSubstr("xyz"),
                             testing::HasSubstr("SetExecutor")));
}

TEST(CalculatorGraph, DuplicateExecutorConfig) {
  // More than one ExecutorConfig cannot have the same name.
  CalculatorGraph graph;
  MP_ASSERT_OK(
      graph.SetExecutor("xyz", std::make_shared<ThreadPoolExecutor>(1)));
  CalculatorGraphConfig config =
      ::mediapipe::ParseTextProtoOrDie<CalculatorGraphConfig>(R"(
        input_stream: 'in'
        executor { name: 'xyz' }
        executor { name: 'xyz' }
        node {
          calculator: 'PassThroughCalculator'
          executor: 'xyz'
          input_stream: 'in'
          output_stream: 'out'
        }
      )");
  ::mediapipe::Status status = graph.Initialize(config);
  EXPECT_EQ(status.code(), ::mediapipe::StatusCode::kInvalidArgument);
  EXPECT_THAT(status.message(),
              testing::AllOf(testing::HasSubstr("xyz"),
                             testing::HasSubstr("duplicate")));
}

TEST(CalculatorGraph, TypedExecutorDeclaredAndSet) {
  // If an executor is declared with a "type" field, it must not be provided
  // to the graph with a CalculatorGraph::SetExecutor() call.
  CalculatorGraph graph;
  MP_ASSERT_OK(
      graph.SetExecutor("xyz", std::make_shared<ThreadPoolExecutor>(1)));
  CalculatorGraphConfig config =
      ::mediapipe::ParseTextProtoOrDie<CalculatorGraphConfig>(R"(
        input_stream: 'in'
        executor {
          name: 'xyz'
          type: 'ThreadPoolExecutor'
          options {
            [mediapipe.ThreadPoolExecutorOptions.ext] { num_threads: 1 }
          }
        }
        node {
          calculator: 'PassThroughCalculator'
          executor: 'xyz'
          input_stream: 'in'
          output_stream: 'out'
        }
      )");
  ::mediapipe::Status status = graph.Initialize(config);
  EXPECT_EQ(status.code(), ::mediapipe::StatusCode::kInvalidArgument);
  EXPECT_THAT(status.message(),
              testing::AllOf(testing::HasSubstr("xyz"),
                             testing::HasSubstr("SetExecutor")));
}

// The graph-level num_threads field and the ExecutorConfig for the default
// executor must not both be specified.
TEST(CalculatorGraph, NumThreadsAndDefaultExecutorConfig) {
  CalculatorGraph graph;
  CalculatorGraphConfig config =
      ::mediapipe::ParseTextProtoOrDie<CalculatorGraphConfig>(R"(
        input_stream: 'in'
        num_threads: 1
        executor {
          type: 'ThreadPoolExecutor'
          options {
            [mediapipe.ThreadPoolExecutorOptions.ext] { num_threads: 1 }
          }
        }
        node {
          calculator: 'PassThroughCalculator'
          input_stream: 'in'
          output_stream: 'mid'
        }
        node {
          calculator: 'PassThroughCalculator'
          input_stream: 'mid'
          output_stream: 'out'
        }
      )");
  ::mediapipe::Status status = graph.Initialize(config);
  EXPECT_EQ(status.code(), ::mediapipe::StatusCode::kInvalidArgument);
  EXPECT_THAT(status.message(),
              testing::AllOf(testing::HasSubstr("num_threads"),
                             testing::HasSubstr("default executor")));
}

// The graph-level num_threads field and the ExecutorConfig for a non-default
// executor may coexist.
TEST(CalculatorGraph, NumThreadsAndNonDefaultExecutorConfig) {
  CalculatorGraph graph;
  CalculatorGraphConfig config =
      ::mediapipe::ParseTextProtoOrDie<CalculatorGraphConfig>(R"(
        input_stream: 'in'
        num_threads: 1
        executor {
          name: 'xyz'
          type: 'ThreadPoolExecutor'
          options {
            [mediapipe.ThreadPoolExecutorOptions.ext] { num_threads: 1 }
          }
        }
        node {
          calculator: 'PassThroughCalculator'
          input_stream: 'in'
          output_stream: 'mid'
        }
        node {
          calculator: 'PassThroughCalculator'
          executor: 'xyz'
          input_stream: 'mid'
          output_stream: 'out'
        }
      )");
  MP_EXPECT_OK(graph.Initialize(config));
}

// Verifies that the application thread is used only when
// "ApplicationThreadExecutor" is specified.  In this test
// "ApplicationThreadExecutor" is specified in the ExecutorConfig for the
// default executor.
TEST(CalculatorGraph, RunWithNumThreadsInExecutorConfig) {
  const struct {
    std::string executor_type;
    int num_threads;
    bool use_app_thread_is_expected;
  } cases[] = {{"ApplicationThreadExecutor", 0, true},
               {"<None>", 0, false},
               {"ThreadPoolExecutor", 1, false}};

  CalculatorGraphConfig config =
      ::mediapipe::ParseTextProtoOrDie<CalculatorGraphConfig>(R"(
        executor {
          options {
            [mediapipe.ThreadPoolExecutorOptions.ext] { num_threads: 0 }
          }
        }
        node { calculator: 'PthreadSelfSourceCalculator' output_stream: 'out' }
      )");
  ThreadPoolExecutorOptions* default_executor_options =
      config.mutable_executor(0)->mutable_options()->MutableExtension(
          ThreadPoolExecutorOptions::ext);
  for (int i = 0; i < ABSL_ARRAYSIZE(cases); ++i) {
    default_executor_options->set_num_threads(cases[i].num_threads);
    config.mutable_executor(0)->clear_type();
    if (cases[i].executor_type != "<None>") {
      config.mutable_executor(0)->set_type(cases[i].executor_type);
    }
    CalculatorGraph graph;
    MP_ASSERT_OK(graph.Initialize(config));
    Packet out_packet;
    MP_ASSERT_OK(
        graph.ObserveOutputStream("out", [&out_packet](const Packet& packet) {
          out_packet = packet;
          return ::mediapipe::OkStatus();
        }));
    MP_ASSERT_OK(graph.Run());
    EXPECT_EQ(cases[i].use_app_thread_is_expected,
              out_packet.Get<pthread_t>() == pthread_self())
        << "for case " << i;
  }
}

TEST(CalculatorGraph, CalculatorGraphNotInitialized) {
  CalculatorGraph graph;
  EXPECT_FALSE(graph.Run().ok());
}

TEST(CalculatorGraph, SimulateAssertFailure) {
  CalculatorGraphConfig config =
      ::mediapipe::ParseTextProtoOrDie<CalculatorGraphConfig>(R"(
        num_threads: 2
        node {
          calculator: 'PassThroughCalculator'
          input_stream: 'in_a'
          input_stream: 'in_b'
          output_stream: 'out_a'
          output_stream: 'out_b'
        }
        input_stream: 'in_a'
        input_stream: 'in_b'
      )");
  CalculatorGraph graph;
  MP_ASSERT_OK(graph.Initialize(config));
  MP_ASSERT_OK(graph.StartRun({}));
  MP_EXPECT_OK(graph.WaitUntilIdle());

  // End the test here to simulate an ASSERT_ failure, which will skip the
  // rest of the test and exit the test function immediately. The test should
  // not hang in the CalculatorGraph destructor.
}

// Verifies Calculator::InputTimestamp() returns the expected value in Open(),
// Process(), and Close() for both source and non-source nodes. In this test
// the source node stops the graph.
TEST(CalculatorGraph, CheckInputTimestamp) {
  CalculatorGraphConfig config =
      ::mediapipe::ParseTextProtoOrDie<CalculatorGraphConfig>(R"(
        node {
          calculator: 'CheckInputTimestampSourceCalculator'
          output_stream: 'integer'
        }
        node {
          calculator: 'CheckInputTimestampSinkCalculator'
          input_stream: 'integer'
        }
      )");
  CalculatorGraph graph;
  MP_ASSERT_OK(graph.Initialize(config));
  MP_ASSERT_OK(graph.Run());
}

// Verifies Calculator::InputTimestamp() returns the expected value in Open(),
// Process(), and Close() for both source and non-source nodes. In this test
// the sink node stops the graph, which causes the framework to close the
// source node.
TEST(CalculatorGraph, CheckInputTimestamp2) {
  CalculatorGraphConfig config =
      ::mediapipe::ParseTextProtoOrDie<CalculatorGraphConfig>(R"(
        node {
          calculator: 'CheckInputTimestamp2SourceCalculator'
          output_stream: 'integer'
        }
        node {
          calculator: 'CheckInputTimestamp2SinkCalculator'
          input_stream: 'integer'
        }
      )");
  CalculatorGraph graph;
  MP_ASSERT_OK(graph.Initialize(config));
  MP_ASSERT_OK(graph.Run());
}

TEST(CalculatorGraph, GraphInputStreamWithTag) {
  CalculatorGraphConfig config =
      ::mediapipe::ParseTextProtoOrDie<CalculatorGraphConfig>(R"(
        input_stream: "VIDEO_METADATA:video_metadata"
        input_stream: "max_count"
        node {
          calculator: "PassThroughCalculator"
          input_stream: "FIRST_INPUT:video_metadata"
          input_stream: "max_count"
          output_stream: "FIRST_INPUT:output_0"
          output_stream: "output_1"
        }
      )");
  std::vector<Packet> packet_dump;
  tool::AddVectorSink("output_0", &config, &packet_dump);
  CalculatorGraph graph;
  MP_ASSERT_OK(graph.Initialize(config));
  MP_ASSERT_OK(graph.StartRun({}));
  for (int i = 0; i < 5; ++i) {
    MP_ASSERT_OK(graph.AddPacketToInputStream(
        "video_metadata", MakePacket<int>(i).At(Timestamp(i))));
  }
  MP_ASSERT_OK(graph.CloseAllPacketSources());
  MP_ASSERT_OK(graph.WaitUntilDone());
  ASSERT_EQ(5, packet_dump.size());
}

// Returns the first packet of the input stream.
class FirstPacketFilterCalculator : public CalculatorBase {
 public:
  FirstPacketFilterCalculator() {}
  ~FirstPacketFilterCalculator() override {}

  static ::mediapipe::Status GetContract(CalculatorContract* cc) {
    cc->Inputs().Index(0).SetAny();
    cc->Outputs().Index(0).SetSameAs(&cc->Inputs().Index(0));
    return ::mediapipe::OkStatus();
  }

  ::mediapipe::Status Process(CalculatorContext* cc) override {
    if (!seen_first_packet_) {
      cc->Outputs().Index(0).AddPacket(cc->Inputs().Index(0).Value());
      cc->Outputs().Index(0).Close();
      seen_first_packet_ = true;
    }
    return ::mediapipe::OkStatus();
  }

 private:
  bool seen_first_packet_ = false;
};
REGISTER_CALCULATOR(FirstPacketFilterCalculator);
constexpr int kDefaultMaxCount = 1000;

TEST(CalculatorGraph, TestPollPacket) {
  CalculatorGraphConfig config;
  CalculatorGraphConfig::Node* node = config.add_node();
  node->set_calculator("CountingSourceCalculator");
  node->add_output_stream("output");
  node->add_input_side_packet("MAX_COUNT:max_count");

  CalculatorGraph graph;
  MP_ASSERT_OK(graph.Initialize(config));
  auto status_or_poller = graph.AddOutputStreamPoller("output");
  ASSERT_TRUE(status_or_poller.ok());
  OutputStreamPoller poller = std::move(status_or_poller.ValueOrDie());
  MP_ASSERT_OK(
      graph.StartRun({{"max_count", MakePacket<int>(kDefaultMaxCount)}}));
  Packet packet;
  int num_packets = 0;
  while (poller.Next(&packet)) {
    EXPECT_EQ(num_packets, packet.Get<int>());
    ++num_packets;
  }
  MP_ASSERT_OK(graph.CloseAllPacketSources());
  MP_ASSERT_OK(graph.WaitUntilDone());
  EXPECT_FALSE(poller.Next(&packet));
  EXPECT_EQ(kDefaultMaxCount, num_packets);
}

TEST(CalculatorGraph, TestOutputStreamPollerDesiredQueueSize) {
  CalculatorGraphConfig config;
  CalculatorGraphConfig::Node* node = config.add_node();
  node->set_calculator("CountingSourceCalculator");
  node->add_output_stream("output");
  node->add_input_side_packet("MAX_COUNT:max_count");

  for (int queue_size = 1; queue_size < 10; ++queue_size) {
    CalculatorGraph graph;
    MP_ASSERT_OK(graph.Initialize(config));
    auto status_or_poller = graph.AddOutputStreamPoller("output");
    ASSERT_TRUE(status_or_poller.ok());
    OutputStreamPoller poller = std::move(status_or_poller.ValueOrDie());
    poller.SetMaxQueueSize(queue_size);
    MP_ASSERT_OK(
        graph.StartRun({{"max_count", MakePacket<int>(kDefaultMaxCount)}}));
    Packet packet;
    int num_packets = 0;
    while (poller.Next(&packet)) {
      EXPECT_EQ(num_packets, packet.Get<int>());
      ++num_packets;
    }
    MP_ASSERT_OK(graph.CloseAllPacketSources());
    MP_ASSERT_OK(graph.WaitUntilDone());
    EXPECT_FALSE(poller.Next(&packet));
    EXPECT_EQ(kDefaultMaxCount, num_packets);
  }
}

TEST(CalculatorGraph, TestPollPacketsFromMultipleStreams) {
  CalculatorGraphConfig config;
  CalculatorGraphConfig::Node* node1 = config.add_node();
  node1->set_calculator("CountingSourceCalculator");
  node1->add_output_stream("stream1");
  node1->add_input_side_packet("MAX_COUNT:max_count");
  CalculatorGraphConfig::Node* node2 = config.add_node();
  node2->set_calculator("PassThroughCalculator");
  node2->add_input_stream("stream1");
  node2->add_output_stream("stream2");

  CalculatorGraph graph;
  MP_ASSERT_OK(graph.Initialize(config));
  auto status_or_poller1 = graph.AddOutputStreamPoller("stream1");
  ASSERT_TRUE(status_or_poller1.ok());
  OutputStreamPoller poller1 = std::move(status_or_poller1.ValueOrDie());
  auto status_or_poller2 = graph.AddOutputStreamPoller("stream2");
  ASSERT_TRUE(status_or_poller2.ok());
  OutputStreamPoller poller2 = std::move(status_or_poller2.ValueOrDie());
  MP_ASSERT_OK(
      graph.StartRun({{"max_count", MakePacket<int>(kDefaultMaxCount)}}));
  Packet packet1;
  Packet packet2;
  int num_packets1 = 0;
  int num_packets2 = 0;
  int running_pollers = 2;
  while (running_pollers > 0) {
    if (poller1.Next(&packet1)) {
      EXPECT_EQ(num_packets1++, packet1.Get<int>());
    } else {
      --running_pollers;
    }
    if (poller2.Next(&packet2)) {
      EXPECT_EQ(num_packets2++, packet2.Get<int>());
    } else {
      --running_pollers;
    }
  }
  MP_ASSERT_OK(graph.CloseAllPacketSources());
  MP_ASSERT_OK(graph.WaitUntilDone());
  EXPECT_FALSE(poller1.Next(&packet1));
  EXPECT_FALSE(poller2.Next(&packet2));
  EXPECT_EQ(kDefaultMaxCount, num_packets1);
  EXPECT_EQ(kDefaultMaxCount, num_packets2);
}

// Ensure that when a custom input stream handler is used to handle packets from
// input streams, an error message is outputted with the appropriate link to
// resolve the issue when the calculator doesn't handle inputs in monotonically
// increasing order of timestamps.
TEST(CalculatorGraph, SimpleMuxCalculatorWithCustomInputStreamHandler) {
  CalculatorGraph graph;
  CalculatorGraphConfig config =
      ::mediapipe::ParseTextProtoOrDie<CalculatorGraphConfig>(R"(
        input_stream: 'input0'
        input_stream: 'input1'
        node {
          calculator: 'SimpleMuxCalculator'
          input_stream: 'input0'
          input_stream: 'input1'
          input_stream_handler {
            input_stream_handler: "ImmediateInputStreamHandler"
          }
          output_stream: 'output'
        }
      )");
  std::vector<Packet> packet_dump;
  tool::AddVectorSink("output", &config, &packet_dump);

  MP_ASSERT_OK(graph.Initialize(config));
  MP_ASSERT_OK(graph.StartRun({}));

  // Send packets to input stream "input0" at timestamps 0 and 1 consecutively.
  Timestamp input0_timestamp = Timestamp(0);
  MP_EXPECT_OK(graph.AddPacketToInputStream(
      "input0", MakePacket<int>(1).At(input0_timestamp)));
  MP_ASSERT_OK(graph.WaitUntilIdle());
  ASSERT_EQ(1, packet_dump.size());
  EXPECT_EQ(1, packet_dump[0].Get<int>());

  ++input0_timestamp;
  MP_EXPECT_OK(graph.AddPacketToInputStream(
      "input0", MakePacket<int>(3).At(input0_timestamp)));
  MP_ASSERT_OK(graph.WaitUntilIdle());
  ASSERT_EQ(2, packet_dump.size());
  EXPECT_EQ(3, packet_dump[1].Get<int>());

  // Send a packet to input stream "input1" at timestamp 0 after sending two
  // packets at timestamps 0 and 1 to input stream "input0". This will result
  // in a mismatch in timestamps as the SimpleMuxCalculator doesn't handle
  // inputs from all streams in monotonically increasing order of timestamps.
  Timestamp input1_timestamp = Timestamp(0);
  MP_EXPECT_OK(graph.AddPacketToInputStream(
      "input1", MakePacket<int>(2).At(input1_timestamp)));
  ::mediapipe::Status run_status = graph.WaitUntilIdle();
  EXPECT_THAT(
      run_status.ToString(),
      testing::AllOf(
          // The core problem.
          testing::HasSubstr("timestamp mismatch on a calculator"),
          testing::HasSubstr(
              "timestamps that are not strictly monotonically increasing"),
          // Link to the possible solution.
          testing::HasSubstr("ImmediateInputStreamHandler class comment")));
}

void DoTestMultipleGraphRuns(absl::string_view input_stream_handler,
                             bool select_packet) {
  std::string graph_proto = absl::StrFormat(R"(
    input_stream: 'input'
    input_stream: 'select'
    node {
      calculator: 'PassThroughCalculator'
      input_stream: 'input'
      input_stream: 'select'
      input_stream_handler {
        input_stream_handler: "%s"
      }
      output_stream: 'output'
      output_stream: 'select_out'
    }
  )",
                                            input_stream_handler.data());
  CalculatorGraphConfig config =
      ::mediapipe::ParseTextProtoOrDie<CalculatorGraphConfig>(graph_proto);
  std::vector<Packet> packet_dump;
  tool::AddVectorSink("output", &config, &packet_dump);

  CalculatorGraph graph;
  MP_ASSERT_OK(graph.Initialize(config));

  struct Run {
    Timestamp timestamp;
    int value;
  };
  std::vector<Run> runs = {{.timestamp = Timestamp(2000), .value = 2},
                           {.timestamp = Timestamp(1000), .value = 1}};
  for (const Run& run : runs) {
    MP_ASSERT_OK(graph.StartRun({}));

    if (select_packet) {
      MP_EXPECT_OK(graph.AddPacketToInputStream(
          "select", MakePacket<int>(0).At(run.timestamp)));
    }
    MP_EXPECT_OK(graph.AddPacketToInputStream(
        "input", MakePacket<int>(run.value).At(run.timestamp)));
    MP_ASSERT_OK(graph.WaitUntilIdle());
    ASSERT_EQ(1, packet_dump.size());
    EXPECT_EQ(run.value, packet_dump[0].Get<int>());
    EXPECT_EQ(run.timestamp, packet_dump[0].Timestamp());

    MP_ASSERT_OK(graph.CloseAllPacketSources());
    MP_ASSERT_OK(graph.WaitUntilDone());

    packet_dump.clear();
  }
}

TEST(CalculatorGraph, MultipleRunsWithDifferentInputStreamHandlers) {
  DoTestMultipleGraphRuns("BarrierInputStreamHandler", true);
  DoTestMultipleGraphRuns("DefaultInputStreamHandler", true);
  DoTestMultipleGraphRuns("EarlyCloseInputStreamHandler", true);
  DoTestMultipleGraphRuns("FixedSizeInputStreamHandler", true);
  DoTestMultipleGraphRuns("ImmediateInputStreamHandler", false);
  DoTestMultipleGraphRuns("MuxInputStreamHandler", true);
  DoTestMultipleGraphRuns("SyncSetInputStreamHandler", true);
  DoTestMultipleGraphRuns("TimestampAlignInputStreamHandler", true);
}

}  // namespace
}  // namespace mediapipe
