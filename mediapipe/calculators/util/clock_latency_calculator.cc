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
#include "mediapipe/framework/port/logging.h"
#include "mediapipe/framework/port/ret_check.h"
#include "mediapipe/framework/port/status.h"

namespace mediapipe {
namespace {
// Tag name for reference signal.
constexpr char kReferenceTag[] = "REFERENCE";
}  // namespace

// A calculator that diffs multiple input absl::Time streams against a
// reference Time stream, and outputs the resulting absl::Duration's. Useful
// in combination with ClockTimestampCalculator to be able to determine the
// latency between two different points in a graph.
//
// Inputs:  At least one non-reference Time stream is required.
//   0- Time stream 0
//   1- Time stream 1
//   ...
//   N- Time stream N
//   REFERENCE_SIGNAL (required): The Time stream by which all others are
//     compared. Should be the stream from which our other streams were
//     computed, in order to provide meaningful latency results.
//
// Outputs:
//   0- Duration from REFERENCE_SIGNAL to input stream 0
//   1- Duration from REFERENCE_SIGNAL to input stream 1
//   ...
//   N- Duration from REFERENCE_SIGNAL to input stream N
//
// Example config:
// node {
//   calculator: "ClockLatencyCalculator"
//   input_stream: "packet_clocktime_stream_0"
//   input_stream: "packet_clocktime_stream_1"
//   input_stream: "packet_clocktime_stream_2"
//   input_stream: "REFERENCE_SIGNAL: packet_clocktime_stream_reference"
//   output_stream: "packet_latency_stream_0"
//   output_stream: "packet_latency_stream_1"
//   output_stream: "packet_latency_stream_2"
// }
//
class ClockLatencyCalculator : public CalculatorBase {
 public:
  ClockLatencyCalculator() {}

  static ::mediapipe::Status GetContract(CalculatorContract* cc);

  ::mediapipe::Status Open(CalculatorContext* cc) override;
  ::mediapipe::Status Process(CalculatorContext* cc) override;

 private:
  int64 num_packet_streams_ = -1;
};
REGISTER_CALCULATOR(ClockLatencyCalculator);

::mediapipe::Status ClockLatencyCalculator::GetContract(
    CalculatorContract* cc) {
  RET_CHECK_GT(cc->Inputs().NumEntries(), 1);

  int64 num_packet_streams = cc->Inputs().NumEntries() - 1;
  RET_CHECK_EQ(cc->Outputs().NumEntries(), num_packet_streams);

  for (int64 i = 0; i < num_packet_streams; ++i) {
    cc->Inputs().Index(i).Set<absl::Time>();
    cc->Outputs().Index(i).Set<absl::Duration>();
  }
  cc->Inputs().Tag(kReferenceTag).Set<absl::Time>();

  return ::mediapipe::OkStatus();
}

::mediapipe::Status ClockLatencyCalculator::Open(CalculatorContext* cc) {
  // Direct passthrough, as far as timestamp and bounds are concerned.
  cc->SetOffset(TimestampDiff(0));
  num_packet_streams_ = cc->Inputs().NumEntries() - 1;
  return ::mediapipe::OkStatus();
}

::mediapipe::Status ClockLatencyCalculator::Process(CalculatorContext* cc) {
  // Get reference time.
  RET_CHECK(!cc->Inputs().Tag(kReferenceTag).IsEmpty());
  const absl::Time& reference_time =
      cc->Inputs().Tag(kReferenceTag).Get<absl::Time>();

  // Push Duration packets for every input stream we have.
  for (int64 i = 0; i < num_packet_streams_; ++i) {
    if (!cc->Inputs().Index(i).IsEmpty()) {
      const absl::Time& input_stream_time =
          cc->Inputs().Index(i).Get<absl::Time>();
      cc->Outputs().Index(i).AddPacket(
          MakePacket<absl::Duration>(input_stream_time - reference_time)
              .At(cc->InputTimestamp()));
    }
  }

  return ::mediapipe::OkStatus();
}

}  // namespace mediapipe
