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

#include <memory>

#include "mediapipe/calculators/util/logic_calculator.pb.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/port/status.h"

namespace mediapipe {
using mediapipe::LogicCalculatorOptions;

// A calculator to compute logical functions of bool inputs.
// With just one input, the output equals the input as expected.
//
// Inputs: One or more bool inputs, which may be input-stream-packets,
// input-side-packets, or options input-values.
//
// Outputs: One bool stream.
//
// Example config:
// node {
//   calculator: "LogicCalculator"
//   input_stream: "has_data"
//   input_side_packet: "enable"
//   input_stream: "is_valid"
//   output_stream: "process_data"
//   options {
//     [mediapipe.LogicCalculatorOptions.ext] {
//       op: AND
//       input_value: true
//     }
//   }
// }
class LogicCalculator : public CalculatorBase {
 public:
  static absl::Status GetContract(CalculatorContract* cc) {
    for (int k = 0; k < cc->Inputs().NumEntries(""); ++k) {
      cc->Inputs().Index(k).Set<bool>();
    }
    for (int k = 0; k < cc->InputSidePackets().NumEntries(""); ++k) {
      cc->InputSidePackets().Index(k).Set<bool>();
    }
    RET_CHECK_GE(cc->Inputs().NumEntries("") +
                     cc->InputSidePackets().NumEntries("") +
                     cc->Options<LogicCalculatorOptions>().input_value_size(),
                 1);
    RET_CHECK_EQ(cc->Outputs().NumEntries(""), 1);
    cc->Outputs().Index(0).Set<bool>();
    return absl::OkStatus();
  }

  absl::Status Open(CalculatorContext* cc) override {
    options_ = cc->Options<LogicCalculatorOptions>();
    cc->SetOffset(TimestampDiff(0));
    return absl::OkStatus();
  }

  bool LogicalOp(bool b1, bool b2) {
    switch (options_.op()) {
      case LogicCalculatorOptions::AND:
        return b1 && b2;
      case LogicCalculatorOptions::OR:
        return b1 || b2;
      case LogicCalculatorOptions::XOR:
        return b1 ^ b2;
    }
    return false;
  }

  absl::Status Process(CalculatorContext* cc) override {
    bool result = options_.op() == LogicCalculatorOptions::AND ? true : false;
    for (int k = 0; k < options_.input_value_size(); ++k) {
      result = LogicalOp(result, options_.input_value(k));
    }
    for (int k = 0; k < cc->Inputs().NumEntries(""); ++k) {
      result = LogicalOp(result, cc->Inputs().Index(k).Value().Get<bool>());
    }
    for (int k = 0; k < cc->InputSidePackets().NumEntries(""); ++k) {
      result = LogicalOp(result, cc->InputSidePackets().Index(k).Get<bool>());
    }
    if (options_.negate()) {
      result = !result;
    }
    cc->Outputs().Index(0).Add(new bool(result), cc->InputTimestamp());
    return absl::OkStatus();
  }

 private:
  LogicCalculatorOptions options_;
};
REGISTER_CALCULATOR(LogicCalculator);

}  // namespace mediapipe
