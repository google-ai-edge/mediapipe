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

#include "absl/log/absl_log.h"
#include "absl/status/status.h"
#include "mediapipe/calculators/core/immediate_mux_calculator.pb.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/port/ret_check.h"

namespace mediapipe {

// This Calculator multiplexes several input streams into a single output
// stream, dropping input packets with timestamps older than the last output
// packet. In case two packets arrive with the same timestamp, the packet with
// the lower stream index will be output and the rest will be dropped.
//
// This Calculator optionally produces a finish indicator as its second output
// stream.  One indicator packet is produced for each input packet received.
//
// This Calculator can be used with an ImmediateInputStreamHandler or with the
// default ISH.
//
// This Calculator is designed to work with a Demux calculator such as the
// RoundRobinDemuxCalculator.  Therefore, packets from different input streams
// are normally not expected to have the same timestamp.
//
// NOTE: this calculator can drop packets non-deterministically, depending on
// how fast the input streams are fed. In most cases, MuxCalculator should be
// preferred. In particular, dropping packets can interfere with rate limiting
// mechanisms.
//
// The user can set the `process_timestamp_bounds` option to true to maintain a
// more stable behavior with timestamp bound updates, that the calculator will
// propagate the timestamp bound update inputs downstream and increase the
// input timestamp bound correspondingly, and drop later packets with smaller
// input timestamps.
class ImmediateMuxCalculator : public CalculatorBase {
 public:
  // This calculator combines any set of input streams into a single output
  // stream.  All input stream types must match the output stream type.
  static absl::Status GetContract(CalculatorContract* cc);

  // Passes any input packet to the output stream immediately, unless the packet
  // timestamp is lower than a previously passed packet.
  absl::Status Process(CalculatorContext* cc) override;
  absl::Status Open(CalculatorContext* cc) override;

 private:
  bool process_timestamp_bounds_ = false;
  Timestamp current_timestamp_bound_ = Timestamp::PreStream();
};
REGISTER_CALCULATOR(ImmediateMuxCalculator);

absl::Status ImmediateMuxCalculator::GetContract(CalculatorContract* cc) {
  RET_CHECK(cc->Outputs().NumEntries() >= 1 && cc->Outputs().NumEntries() <= 2)
      << "This calculator produces only one or two output streams.";
  cc->Outputs().Index(0).SetAny();
  if (cc->Outputs().NumEntries() >= 2) {
    cc->Outputs().Index(1).Set<bool>();
  }
  for (int i = 0; i < cc->Inputs().NumEntries(); ++i) {
    cc->Inputs().Index(i).SetSameAs(&cc->Outputs().Index(0));
  }
  const auto& options = cc->Options<mediapipe::ImmediateMuxCalculatorOptions>();
  if (options.process_timestamp_bounds()) {
    cc->SetProcessTimestampBounds(true);
  }
  return absl::OkStatus();
}

absl::Status ImmediateMuxCalculator::Open(CalculatorContext* cc) {
  const auto& options = cc->Options<mediapipe::ImmediateMuxCalculatorOptions>();
  process_timestamp_bounds_ = options.process_timestamp_bounds();
  if (!process_timestamp_bounds_) {
    cc->SetOffset(TimestampDiff(0));
  }
  return absl::OkStatus();
}

absl::Status ImmediateMuxCalculator::Process(CalculatorContext* cc) {
  bool is_timestamp_bound_update = true;
  // Pass along the first packet if it's timestamp exceeds the current timestamp
  // bound. Otherwise, drop the packet.
  for (int i = 0; i < cc->Inputs().NumEntries(); ++i) {
    const Packet& packet = cc->Inputs().Index(i).Value();
    if (!packet.IsEmpty()) {
      is_timestamp_bound_update = false;
      if (packet.Timestamp() >= current_timestamp_bound_) {
        cc->Outputs().Index(0).AddPacket(packet);
        current_timestamp_bound_ = packet.Timestamp().NextAllowedInStream();
      } else {
        ABSL_LOG_FIRST_N(WARNING, 5)
            << "Dropping a packet with timestamp " << packet.Timestamp();
      }
      if (cc->Outputs().NumEntries() >= 2 &&
          !cc->Outputs().Index(1).IsClosed()) {
        Timestamp output_timestamp = std::max(
            cc->InputTimestamp(), cc->Outputs().Index(1).NextTimestampBound());
        cc->Outputs().Index(1).Add(new bool(true), output_timestamp);
      }
    }
  }

  // Optionally propagate timestamp bound updates if enabled.
  // If the calculator is configured with ImmediateInputStreamHandler, and has
  // more than one input streams, it is possible that a later packet has a
  // timestamp bound update call with timestamp smaller than
  // current_timestamp_bound_. In this case, the update should not occur.
  if (process_timestamp_bounds_ && is_timestamp_bound_update &&
      cc->InputTimestamp() >= current_timestamp_bound_) {
    if (!cc->InputTimestamp().HasNextAllowedInStream()) {
      // The calculator receives a close input stream call.
      // Close all output streams.
      for (int i = 0; i < cc->Outputs().NumEntries(); ++i) {
        cc->Outputs().Index(i).Close();
      }
    } else {
      current_timestamp_bound_ = cc->InputTimestamp().NextAllowedInStream();
      for (int i = 0; i < cc->Outputs().NumEntries(); ++i) {
        cc->Outputs().Index(i).SetNextTimestampBound(current_timestamp_bound_);
      }
    }
  }
  return absl::OkStatus();
}

}  // namespace mediapipe
