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

#include "mediapipe/calculators/core/gate_calculator.pb.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/port/ret_check.h"
#include "mediapipe/framework/port/status.h"
#include "mediapipe/util/header_util.h"

namespace mediapipe {

namespace {
enum GateState {
  GATE_UNINITIALIZED,
  GATE_ALLOW,
  GATE_DISALLOW,
};

std::string ToString(GateState state) {
  switch (state) {
    case GATE_UNINITIALIZED:
      return "UNINITIALIZED";
    case GATE_ALLOW:
      return "ALLOW";
    case GATE_DISALLOW:
      return "DISALLOW";
  }
  DLOG(FATAL) << "Unknown GateState";
  return "UNKNOWN";
}
}  // namespace

// Controls whether or not the input packets are passed further along the graph.
// Takes multiple data input streams and either an ALLOW or a DISALLOW control
// input stream. It outputs an output stream for each input stream that is not
// ALLOW or DISALLOW as well as an optional STATE_CHANGE stream which downstream
// calculators can use to respond to state-change events.
//
// If the current ALLOW packet is set to true, the input packets are passed to
// their corresponding output stream unchanged. If the ALLOW packet is set to
// false, the current input packet is NOT passed to the output stream. If using
// DISALLOW, the behavior is opposite of ALLOW.
//
// By default, an empty packet in the ALLOW or DISALLOW input stream indicates
// disallowing the corresponding packets in other input streams. The behavior
// can be inverted with a calculator option.
//
// Intended to be used with the default input stream handler, which synchronizes
// all data input streams with the ALLOW/DISALLOW control input stream.
//
// Example config:
// node {
//   calculator: "GateCalculator"
//   input_stream: "input_stream0"
//   input_stream: "input_stream1"
//   input_stream: "input_streamN"
//   input_stream: "ALLOW:allow" or "DISALLOW:disallow"
//   output_stream: "STATE_CHANGE:state_change"
//   output_stream: "output_stream0"
//   output_stream: "output_stream1"
//   output_stream: "output_streamN"
// }
class GateCalculator : public CalculatorBase {
 public:
  GateCalculator() {}

  static ::mediapipe::Status GetContract(CalculatorContract* cc) {
    // Assume that input streams do not have a tag and that gating signal is
    // tagged either ALLOW or DISALLOW.
    RET_CHECK(cc->Inputs().HasTag("ALLOW") ^ cc->Inputs().HasTag("DISALLOW"));
    const int num_data_streams = cc->Inputs().NumEntries("");
    RET_CHECK_GE(num_data_streams, 1);
    RET_CHECK_EQ(cc->Outputs().NumEntries(""), num_data_streams)
        << "Number of data output streams must match with data input streams.";

    for (int i = 0; i < num_data_streams; ++i) {
      cc->Inputs().Get("", i).SetAny();
      cc->Outputs().Get("", i).SetSameAs(&cc->Inputs().Get("", i));
    }
    if (cc->Inputs().HasTag("ALLOW")) {
      cc->Inputs().Tag("ALLOW").Set<bool>();
    } else {
      cc->Inputs().Tag("DISALLOW").Set<bool>();
    }

    if (cc->Outputs().HasTag("STATE_CHANGE")) {
      cc->Outputs().Tag("STATE_CHANGE").Set<bool>();
    }

    return ::mediapipe::OkStatus();
  }

  ::mediapipe::Status Open(CalculatorContext* cc) final {
    cc->SetOffset(TimestampDiff(0));
    num_data_streams_ = cc->Inputs().NumEntries("");
    last_gate_state_ = GATE_UNINITIALIZED;
    RET_CHECK_OK(CopyInputHeadersToOutputs(cc->Inputs(), &cc->Outputs()));

    const auto& options = cc->Options<::mediapipe::GateCalculatorOptions>();
    empty_packets_as_allow_ = options.empty_packets_as_allow();

    return ::mediapipe::OkStatus();
  }

  ::mediapipe::Status Process(CalculatorContext* cc) final {
    bool allow = empty_packets_as_allow_;
    if (cc->Inputs().HasTag("ALLOW") && !cc->Inputs().Tag("ALLOW").IsEmpty()) {
      allow = cc->Inputs().Tag("ALLOW").Get<bool>();
    }
    if (cc->Inputs().HasTag("DISALLOW") &&
        !cc->Inputs().Tag("DISALLOW").IsEmpty()) {
      allow = !cc->Inputs().Tag("DISALLOW").Get<bool>();
    }

    const GateState new_gate_state = allow ? GATE_ALLOW : GATE_DISALLOW;

    if (cc->Outputs().HasTag("STATE_CHANGE")) {
      if (last_gate_state_ != GATE_UNINITIALIZED &&
          last_gate_state_ != new_gate_state) {
        VLOG(2) << "State transition in " << cc->NodeName() << " @ "
                << cc->InputTimestamp().Value() << " from "
                << ToString(last_gate_state_) << " to "
                << ToString(new_gate_state);
        cc->Outputs()
            .Tag("STATE_CHANGE")
            .AddPacket(MakePacket<bool>(allow).At(cc->InputTimestamp()));
      }
    }
    last_gate_state_ = new_gate_state;

    if (!allow) {
      return ::mediapipe::OkStatus();
    }

    // Process data streams.
    for (int i = 0; i < num_data_streams_; ++i) {
      if (!cc->Inputs().Get("", i).IsEmpty()) {
        cc->Outputs().Get("", i).AddPacket(cc->Inputs().Get("", i).Value());
      }
    }

    return ::mediapipe::OkStatus();
  }

 private:
  GateState last_gate_state_ = GATE_UNINITIALIZED;
  int num_data_streams_;
  bool empty_packets_as_allow_;
};
REGISTER_CALCULATOR(GateCalculator);

}  // namespace mediapipe
