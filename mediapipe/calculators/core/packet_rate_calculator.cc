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

#include <utility>
#include <vector>

#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/port/ret_check.h"
#include "mediapipe/framework/port/status.h"

namespace {
using namespace mediapipe;
/***
 * PacketRateCalculator allows to calculate rate of incoming packages.
 * E.g. when you want to extract FPS, or any other processing rate.
 * - As input it accepts any packets stream.
 * - As output it emits rate (floating point scalar), namely amount of packets per second based
 * calculation of current packet and previous packet.
 * - For very first packet it emits empty packet.
 *
 * Usage example:
 * node {
 *   calculator: "PacketRateCalculator"
 *   input_stream: "image"
 *   output_stream: "image_fps"
 * }
 *
 */
class PacketRateCalculator : public CalculatorBase {
public:
  static absl::Status GetContract(CalculatorContract *cc) {
    auto &side_inputs = cc->InputSidePackets();
    RET_CHECK_EQ(cc->Inputs().NumEntries(), 1);
    RET_CHECK_EQ(cc->Outputs().NumEntries(), 1);

    cc->Inputs().Index(0).SetAny();

    cc->Outputs().Index(0).Set<float>();
    return absl::OkStatus();
  }

  absl::Status Open(CalculatorContext *cc) final { return absl::OkStatus(); }

  // Releases input packets allowed by the max_in_flight constraint.
  absl::Status Process(CalculatorContext *cc) final {
    Timestamp latest_ts = cc->Inputs().Index(0).Value().Timestamp();

    if (prev_timestamp_.Value() != kint64min) {
      float seconds_since_last_packet = (latest_ts - prev_timestamp_).Seconds();
      auto rate = std::make_unique<float>(1. / seconds_since_last_packet);
      cc->Outputs().Index(0).Add(rate.release(), latest_ts);
    } else {
      cc->Outputs().Index(0).AddPacket(Packet().At(latest_ts));
    }

    prev_timestamp_ = latest_ts;

    return absl::OkStatus();
  }

private:
  Timestamp prev_timestamp_;
};
} // namespace

namespace mediapipe {
REGISTER_CALCULATOR(PacketRateCalculator);
} // namespace mediapipe
