// Copyright 2020 The MediaPipe Authors.
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
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/deps/clock.h"
#include "mediapipe/framework/deps/monotonic_clock.h"
#include "mediapipe/framework/port/logging.h"
#include "mediapipe/framework/port/ret_check.h"
#include "mediapipe/framework/port/status.h"

namespace mediapipe {
namespace {
// Tag name for clock side packet.
constexpr char kClockTag[] = "CLOCK";
}  // namespace

// A calculator that outputs the current clock time at which it receives input
// packets. Use a separate instance of this calculator for each input stream
// you wish to output a clock time for.
//
// InputSidePacket (Optional):
// CLOCK: A clock to use for querying the current time.
//
// Inputs:
//   A single packet stream we wish to get the current clocktime for

// Outputs:
//   A single stream of absl::Time packets, representing the clock time at which
//   we received the input stream's packets.

// Example config:
// node {
//   calculator: "ClockTimestampCalculator"
//   input_side_packet: "CLOCK:monotonic_clock"
//   input_stream: "packet_stream"
//   output_stream: "packet_clocktime_stream"
// }
//
class ClockTimestampCalculator : public CalculatorBase {
 public:
  ClockTimestampCalculator() {}

  static ::mediapipe::Status GetContract(CalculatorContract* cc);

  ::mediapipe::Status Open(CalculatorContext* cc) override;
  ::mediapipe::Status Process(CalculatorContext* cc) override;

 private:
  // Clock object.
  std::shared_ptr<::mediapipe::Clock> clock_;
};
REGISTER_CALCULATOR(ClockTimestampCalculator);

::mediapipe::Status ClockTimestampCalculator::GetContract(
    CalculatorContract* cc) {
  RET_CHECK_EQ(cc->Inputs().NumEntries(), 1);
  RET_CHECK_EQ(cc->Outputs().NumEntries(), 1);

  cc->Inputs().Index(0).SetAny();
  cc->Outputs().Index(0).Set<absl::Time>();

  // Optional Clock input side packet.
  if (cc->InputSidePackets().HasTag(kClockTag)) {
    cc->InputSidePackets()
        .Tag(kClockTag)
        .Set<std::shared_ptr<::mediapipe::Clock>>();
  }

  return ::mediapipe::OkStatus();
}

::mediapipe::Status ClockTimestampCalculator::Open(CalculatorContext* cc) {
  // Direct passthrough, as far as timestamp and bounds are concerned.
  cc->SetOffset(TimestampDiff(0));

  // Initialize the clock.
  if (cc->InputSidePackets().HasTag(kClockTag)) {
    clock_ = cc->InputSidePackets()
                 .Tag("CLOCK")
                 .Get<std::shared_ptr<::mediapipe::Clock>>();
  } else {
    clock_.reset(
        ::mediapipe::MonotonicClock::CreateSynchronizedMonotonicClock());
  }

  return ::mediapipe::OkStatus();
}

::mediapipe::Status ClockTimestampCalculator::Process(CalculatorContext* cc) {
  // Push the Time packet to output.
  auto timestamp_packet = MakePacket<absl::Time>(clock_->TimeNow());
  cc->Outputs().Index(0).AddPacket(timestamp_packet.At(cc->InputTimestamp()));
  return ::mediapipe::OkStatus();
}

}  // namespace mediapipe
