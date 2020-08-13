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

#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/port/status.h"

namespace mediapipe {

// For each non empty input packet, emits a single output packet containing a
// boolean value "true", "false" in response to empty packets (a.k.a. timestamp
// bound updates) This can be used to "flag" the presence of an arbitrary packet
// type as input into a downstream calculator.
//
// Inputs:
//   PACKET - any type.
//
// Outputs:
//   PRESENCE - bool.
//     "true" if packet is not empty, "false" if there's timestamp bound update
//     instead.
//
// Examples:
// node: {
//   calculator: "PacketPresenceCalculator"
//   input_stream: "PACKET:packet"
//   output_stream: "PRESENCE:presence"
// }
//
// This calculator can be used in conjuction with GateCalculator in order to
// allow/disallow processing. For instance:
// node: {
//   calculator: "PacketPresenceCalculator"
//   input_stream: "PACKET:value"
//   output_stream: "PRESENCE:disallow_if_present"
// }
// node {
//   calculator: "GateCalculator"
//   input_stream: "image"
//   input_stream: "DISALLOW:disallow_if_present"
//   output_stream: "image_for_processing"
//   options: {
//     [mediapipe.GateCalculatorOptions.ext] {
//       empty_packets_as_allow: true
//     }
//   }
// }
class PacketPresenceCalculator : public CalculatorBase {
 public:
  static ::mediapipe::Status GetContract(CalculatorContract* cc) {
    cc->Inputs().Tag("PACKET").SetAny();
    cc->Outputs().Tag("PRESENCE").Set<bool>();
    // Process() function is invoked in response to input stream timestamp
    // bound updates.
    cc->SetProcessTimestampBounds(true);
    return ::mediapipe::OkStatus();
  }

  ::mediapipe::Status Open(CalculatorContext* cc) override {
    cc->SetOffset(TimestampDiff(0));
    return ::mediapipe::OkStatus();
  }

  ::mediapipe::Status Process(CalculatorContext* cc) final {
    cc->Outputs()
        .Tag("PRESENCE")
        .AddPacket(MakePacket<bool>(!cc->Inputs().Tag("PACKET").IsEmpty())
                       .At(cc->InputTimestamp()));
    return ::mediapipe::OkStatus();
  }
};
REGISTER_CALCULATOR(PacketPresenceCalculator);

}  // namespace mediapipe
