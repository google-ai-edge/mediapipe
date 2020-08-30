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

#include <string>

#include "mediapipe/calculators/core/constant_side_packet_calculator.pb.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/collection_item_id.h"
#include "mediapipe/framework/port/canonical_errors.h"
#include "mediapipe/framework/port/integral_types.h"
#include "mediapipe/framework/port/ret_check.h"
#include "mediapipe/framework/port/status.h"

namespace mediapipe {

// Generates an output side packet or multiple output side packets according to
// the specified options.
//
// Example configs:
// node {
//   calculator: "ConstantSidePacketCalculator"
//   output_side_packet: "PACKET:packet"
//   options: {
//     [mediapipe.ConstantSidePacketCalculatorOptions.ext]: {
//       packet { int_value: 2 }
//     }
//   }
// }
//
// node {
//   calculator: "ConstantSidePacketCalculator"
//   output_side_packet: "PACKET:0:int_packet"
//   output_side_packet: "PACKET:1:bool_packet"
//   options: {
//     [mediapipe.ConstantSidePacketCalculatorOptions.ext]: {
//       packet { int_value: 2 }
//       packet { bool_value: true }
//     }
//   }
// }
class ConstantSidePacketCalculator : public CalculatorBase {
 public:
  static ::mediapipe::Status GetContract(CalculatorContract* cc) {
    const auto& options =
        cc->Options<::mediapipe::ConstantSidePacketCalculatorOptions>();
    RET_CHECK_EQ(cc->OutputSidePackets().NumEntries(kPacketTag),
                 options.packet_size())
        << "Number of output side packets has to be same as number of packets "
           "configured in options.";

    int index = 0;
    for (CollectionItemId id = cc->OutputSidePackets().BeginId(kPacketTag);
         id != cc->OutputSidePackets().EndId(kPacketTag); ++id, ++index) {
      const auto& packet_options = options.packet(index);
      auto& packet = cc->OutputSidePackets().Get(id);
      if (packet_options.has_int_value()) {
        packet.Set<int>();
      } else if (packet_options.has_float_value()) {
        packet.Set<float>();
      } else if (packet_options.has_bool_value()) {
        packet.Set<bool>();
      } else if (packet_options.has_string_value()) {
        packet.Set<std::string>();
      } else if (packet_options.has_uint64_value()) {
        packet.Set<uint64>();
      } else {
        return ::mediapipe::InvalidArgumentError(
            "None of supported values were specified in options.");
      }
    }
    return ::mediapipe::OkStatus();
  }

  ::mediapipe::Status Open(CalculatorContext* cc) override {
    const auto& options =
        cc->Options<::mediapipe::ConstantSidePacketCalculatorOptions>();
    int index = 0;
    for (CollectionItemId id = cc->OutputSidePackets().BeginId(kPacketTag);
         id != cc->OutputSidePackets().EndId(kPacketTag); ++id, ++index) {
      auto& packet = cc->OutputSidePackets().Get(id);
      const auto& packet_options = options.packet(index);
      if (packet_options.has_int_value()) {
        packet.Set(MakePacket<int>(packet_options.int_value()));
      } else if (packet_options.has_float_value()) {
        packet.Set(MakePacket<float>(packet_options.float_value()));
      } else if (packet_options.has_bool_value()) {
        packet.Set(MakePacket<bool>(packet_options.bool_value()));
      } else if (packet_options.has_string_value()) {
        packet.Set(MakePacket<std::string>(packet_options.string_value()));
      } else if (packet_options.has_uint64_value()) {
        packet.Set(MakePacket<uint64>(packet_options.uint64_value()));
      } else {
        return ::mediapipe::InvalidArgumentError(
            "None of supported values were specified in options.");
      }
    }
    return ::mediapipe::OkStatus();
  }

  ::mediapipe::Status Process(CalculatorContext* cc) override {
    return ::mediapipe::OkStatus();
  }

 private:
  static constexpr const char* kPacketTag = "PACKET";
};

REGISTER_CALCULATOR(ConstantSidePacketCalculator);

}  // namespace mediapipe
