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

#include <memory>
#include <vector>

#include "absl/base/macros.h"
#include "absl/strings/string_view.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/port/canonical_errors.h"
#include "mediapipe/framework/port/status.h"
#include "mediapipe/framework/tool/source.pb.h"

namespace mediapipe {

namespace tool {

// A calculator which takes N input side packets and passes them as
// N outputs. Each input side packet contains a vector of Packets, or a single
// Packet, as given in the options. The elements of the vector contained in
// the i-th input side packet are output as individual packets to the i-th
// output stream. Optionally, the packets can be timestamped, with either their
// index within the vector, or with Timestamp::PostStream(). No type
// checking is performed. It is only checked that the calculator receives 0
// inputs and the number of outputs equals the number of input side packets.
class SidePacketsToStreamsCalculator : public CalculatorBase {
 public:
  SidePacketsToStreamsCalculator() {}
  SidePacketsToStreamsCalculator(const SidePacketsToStreamsCalculator&) =
      delete;
  SidePacketsToStreamsCalculator& operator=(
      const SidePacketsToStreamsCalculator&) = delete;
  ~SidePacketsToStreamsCalculator() override {}

  static absl::Status GetContract(CalculatorContract* cc) {
    auto& options = cc->Options<SidePacketsToStreamsCalculatorOptions>();
    if (options.has_num_inputs() &&
        (options.num_inputs() != cc->InputSidePackets().NumEntries() ||
         options.num_inputs() != cc->Outputs().NumEntries())) {
      return absl::InvalidArgumentError(
          "If num_inputs is specified it must be equal to the number of "
          "input side packets and output streams.");
    }
    if (!options.vectors_of_packets() &&
        options.set_timestamp() ==
            SidePacketsToStreamsCalculatorOptions::NONE) {
      return absl::InvalidArgumentError(
          "If set_timestamp is NONE, vectors_of_packets must not be false.");
    }
    for (int i = 0; i < cc->InputSidePackets().NumEntries(); ++i) {
      if (options.vectors_of_packets()) {
        cc->InputSidePackets().Index(i).Set<std::vector<Packet>>();
      } else {
        cc->InputSidePackets().Index(i).SetAny();
      }
    }
    for (int i = 0; i < cc->InputSidePackets().NumEntries(); ++i) {
      if (options.vectors_of_packets()) {
        cc->Outputs().Index(i).SetAny();
      } else {
        cc->Outputs().Index(i).SetSameAs(&cc->InputSidePackets().Index(i));
      }
    }
    return absl::OkStatus();
  }

  absl::Status Process(CalculatorContext* cc) final {
    const auto& options = cc->Options<SidePacketsToStreamsCalculatorOptions>();
    // The i-th input side packet contains a vector of packets corresponding
    // to the values of this input for all batch elements.

    int batch_size = -1;
    if (options.vectors_of_packets()) {
      // Verify the batch size is consistent.
      for (const Packet& input_side_packet : cc->InputSidePackets()) {
        const auto& packets = input_side_packet.Get<std::vector<Packet>>();
        if (batch_size >= 0) {
          if (packets.size() != batch_size) {
            return absl::InvalidArgumentError(
                "The specified input side packets contain vectors of different "
                "sizes.");
          }
        } else {
          batch_size = packets.size();
        }
      }
    } else {
      batch_size = 1;
    }

    for (int b = 0; b < batch_size; ++b) {
      for (int i = 0; i < cc->InputSidePackets().NumEntries(); ++i) {
        Packet packet;
        if (options.vectors_of_packets()) {
          const auto& packets =
              cc->InputSidePackets().Index(i).Get<std::vector<Packet>>();
          packet = packets[b];
        } else {
          packet = cc->InputSidePackets().Index(i);
        }
        switch (options.set_timestamp()) {
          case SidePacketsToStreamsCalculatorOptions::VECTOR_INDEX:
            cc->Outputs().Index(i).AddPacket(packet.At(Timestamp(b)));
            break;
          case SidePacketsToStreamsCalculatorOptions::WHOLE_STREAM:
            cc->Outputs().Index(i).AddPacket(
                packet.At(Timestamp::PostStream()));
            break;
          case SidePacketsToStreamsCalculatorOptions::PRE_STREAM:
            cc->Outputs().Index(i).AddPacket(packet.At(Timestamp::PreStream()));
            break;
          default:
            // SidePacketsToStreamsCalculatorOptions::NONE
            cc->Outputs().Index(i).AddPacket(packet);
        }
      }
    }
    return tool::StatusStop();
  }
};

REGISTER_CALCULATOR(SidePacketsToStreamsCalculator);

}  // namespace tool
}  // namespace mediapipe
