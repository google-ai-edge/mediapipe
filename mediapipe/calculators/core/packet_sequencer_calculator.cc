// Copyright 2022 The MediaPipe Authors.
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

#include "mediapipe/framework/api2/contract.h"
#include "mediapipe/framework/api2/node.h"
#include "mediapipe/framework/api2/packet.h"
#include "mediapipe/framework/api2/port.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/port/status.h"

namespace mediapipe {
namespace api2 {

// This calculator assigns a timestamp to each "INPUT" packet reflecting
// the most recent "TICK" timestamp.
//
// Each "TICK" timestamp is propagated as a settled "OUTPUT" timestamp.
// This allows "TICK" packets to be processed right away.
// When an "INPUT" packet arrives, it is sent to the "OUTPUT" stream with
// the next unsettled "OUTPUT" timestamp, which is normally one greater than
// the most recent "TICK" timestamp.
//
// If a "TICK" packet and an "INPUT" packet arrive together, the "OUTPUT"
// packet timestamp is derived from the previous "TICK" timestamp,
// and the new "OUTPUT" bound is derived from the current "TICK" timestamp.
// This allows the current "INPUT" packet to cover the current "TICK" timestamp.
//
// Example config:
// node {
//   calculator: "PacketSequencerCalculator"
//   input_stream: "INPUT:switch_selection"
//   input_stream: "TICK:input_image"
//   input_stream: "TICK:input_audio"
//   output_stream: "OUTPUT:switch_selection_timed"
// }
//
class PacketSequencerCalculator : public Node {
 public:
  static constexpr Input<AnyType>::Multiple kInput{"INPUT"};
  static constexpr Input<AnyType>::Multiple kTick{"TICK"};
  static constexpr Output<AnyType>::Multiple kOutput{"OUTPUT"};

  MEDIAPIPE_NODE_CONTRACT(kInput, kTick, kOutput,
                          StreamHandler("ImmediateInputStreamHandler"),
                          TimestampChange::Arbitrary());

  static absl::Status UpdateContract(CalculatorContract* cc) {
    RET_CHECK_EQ(kInput(cc).Count(), kOutput(cc).Count());
    return absl::OkStatus();
  }

  absl::Status Process(CalculatorContext* cc) final {
    // Pass through any input packets at the output stream bound.
    for (int i = 0; i < kInput(cc).Count(); ++i) {
      Timestamp stream_bound = kOutput(cc)[i].NextTimestampBound();
      const PacketBase input_packet = kInput(cc)[i].packet();
      if (!input_packet.IsEmpty()) {
        Timestamp output_ts = std::max(Timestamp::Min(), stream_bound);
        kOutput(cc)[i].Send(input_packet.At(output_ts));
      }
    }

    // Find the new tick timestamp, if any.
    Timestamp tick_ts = Timestamp::Min();
    for (int i = 0; i < kTick(cc).Count(); ++i) {
      const PacketBase& tick_packet = kTick(cc)[i].packet();
      // For either an input packet or an empty input stream,
      // the packet timestamp indicates the latest "settled timestamp",
      // and when it arrives it equals the InputTimestamp().
      if (tick_packet.timestamp() == cc->InputTimestamp()) {
        tick_ts = std::max(tick_ts, tick_packet.timestamp());
        break;
      }
    }

    // Advance all output stream bounds past the tick timestamp.
    for (int i = 0; i < kInput(cc).Count(); ++i) {
      Timestamp stream_bound = kOutput(cc)[i].NextTimestampBound();
      if (tick_ts >= stream_bound) {
        kOutput(cc)[i].SetNextTimestampBound(tick_ts.NextAllowedInStream());
      }
    }
    return absl::OkStatus();
  }
};

MEDIAPIPE_REGISTER_NODE(PacketSequencerCalculator);
}  // namespace api2
}  // namespace mediapipe
