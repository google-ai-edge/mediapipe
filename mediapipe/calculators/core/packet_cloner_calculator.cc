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
//
// This takes packets from N+1 streams, A_1, A_2, ..., A_N, B.
// For every packet that appears in B, outputs the most recent packet from each
// of the A_i on a separate stream.

#include <string_view>
#include <vector>

#include "absl/strings/string_view.h"
#include "mediapipe/calculators/core/packet_cloner_calculator.pb.h"
#include "mediapipe/framework/calculator_framework.h"

namespace mediapipe {

// For every packet received on the last stream, output the latest packet
// obtained on all other streams. Therefore, if the last stream outputs at a
// higher rate than the others, this effectively clones the packets from the
// other streams to match the last.
//
// Example config:
// node {
//   calculator: "PacketClonerCalculator"
//   input_stream: "first_base_signal"
//   input_stream: "second_base_signal"
//   input_stream: "tick_signal"  # or input_stream: "TICK:tick_signal"
//   output_stream: "cloned_first_base_signal"
//   output_stream: "cloned_second_base_signal"
// }
//
// Or you can use "TICK" tag and put corresponding input stream at any location,
// for example at the very beginning:
// node {
//   calculator: "PacketClonerCalculator"
//   input_stream: "TICK:tick_signal"
//   input_stream: "first_base_signal"
//   input_stream: "second_base_signal"
//   output_stream: "cloned_first_base_signal"
//   output_stream: "cloned_second_base_signal"
// }
//
// Related:
//   packet_cloner_calculator.proto: Options for this calculator.
//   merge_input_streams_calculator.cc: One output stream.
//   packet_inner_join_calculator.cc: Don't output unless all inputs are new.
class PacketClonerCalculator : public CalculatorBase {
 public:
  static absl::Status GetContract(CalculatorContract* cc) {
    const Ids ids = GetIds(*cc);
    for (const auto& in_out : ids.inputs_outputs) {
      auto& input = cc->Inputs().Get(in_out.in);
      input.SetAny();
      cc->Outputs().Get(in_out.out).SetSameAs(&input);
    }
    cc->Inputs().Get(ids.tick_id).SetAny();
    return absl::OkStatus();
  }

  absl::Status Open(CalculatorContext* cc) final {
    // Load options.
    const auto calculator_options =
        cc->Options<mediapipe::PacketClonerCalculatorOptions>();
    output_only_when_all_inputs_received_ =
        calculator_options.output_only_when_all_inputs_received() ||
        calculator_options.output_packets_only_when_all_inputs_received();
    output_empty_packets_before_all_inputs_received_ =
        calculator_options.output_packets_only_when_all_inputs_received();

    // Prepare input and output ids.
    ids_ = GetIds(*cc);
    current_.resize(ids_.inputs_outputs.size());

    // Pass along the header for each stream if present.
    for (const auto& in_out : ids_.inputs_outputs) {
      auto& input = cc->Inputs().Get(in_out.in);
      if (!input.Header().IsEmpty()) {
        cc->Outputs().Get(in_out.out).SetHeader(input.Header());
      }
    }
    return absl::OkStatus();
  }

  absl::Status Process(CalculatorContext* cc) final {
    // Store input signals.
    for (int i = 0; i < ids_.inputs_outputs.size(); ++i) {
      const auto& input = cc->Inputs().Get(ids_.inputs_outputs[i].in);
      if (!input.IsEmpty()) {
        current_[i] = input.Value();
      }
    }

    // Output according to the TICK signal.
    if (!cc->Inputs().Get(ids_.tick_id).IsEmpty()) {
      if (output_only_when_all_inputs_received_) {
        // Return if one of the input is null.
        for (int i = 0; i < ids_.inputs_outputs.size(); ++i) {
          if (current_[i].IsEmpty()) {
            if (output_empty_packets_before_all_inputs_received_) {
              SetAllNextTimestampBounds(cc);
            }
            return absl::OkStatus();
          }
        }
      }
      // Output each stream.
      for (int i = 0; i < ids_.inputs_outputs.size(); ++i) {
        auto& output = cc->Outputs().Get(ids_.inputs_outputs[i].out);
        if (!current_[i].IsEmpty()) {
          output.AddPacket(current_[i].At(cc->InputTimestamp()));
        } else {
          output.SetNextTimestampBound(
              cc->InputTimestamp().NextAllowedInStream());
        }
      }
    }
    return absl::OkStatus();
  }

 private:
  struct Ids {
    struct InputOutput {
      CollectionItemId in;
      CollectionItemId out;
    };
    CollectionItemId tick_id;
    std::vector<InputOutput> inputs_outputs;
  };

  template <typename CC>
  static Ids GetIds(CC& cc) {
    Ids ids;
    static constexpr absl::string_view kEmptyTag = "";
    int num_inputs_to_clone = cc.Inputs().NumEntries(kEmptyTag);
    static constexpr absl::string_view kTickTag = "TICK";
    if (cc.Inputs().HasTag(kTickTag)) {
      ids.tick_id = cc.Inputs().GetId(kTickTag, 0);
    } else {
      --num_inputs_to_clone;
      ids.tick_id = cc.Inputs().GetId(kEmptyTag, num_inputs_to_clone);
    }
    for (int i = 0; i < num_inputs_to_clone; ++i) {
      ids.inputs_outputs.push_back({.in = cc.Inputs().GetId(kEmptyTag, i),
                                    .out = cc.Outputs().GetId(kEmptyTag, i)});
    }
    return ids;
  }

  void SetAllNextTimestampBounds(CalculatorContext* cc) {
    for (const auto& in_out : ids_.inputs_outputs) {
      cc->Outputs()
          .Get(in_out.out)
          .SetNextTimestampBound(cc->InputTimestamp().NextAllowedInStream());
    }
  }

  std::vector<Packet> current_;
  Ids ids_;
  bool output_only_when_all_inputs_received_;
  bool output_empty_packets_before_all_inputs_received_;
};

REGISTER_CALCULATOR(PacketClonerCalculator);

}  // namespace mediapipe
