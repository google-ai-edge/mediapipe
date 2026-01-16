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

#include "mediapipe/calculators/core/merge_calculator.h"

#include "absl/log/absl_log.h"
#include "absl/status/status.h"
#include "mediapipe/framework/api3/calculator.h"
#include "mediapipe/framework/api3/calculator_context.h"
#include "mediapipe/framework/api3/calculator_contract.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/port/ret_check.h"

namespace mediapipe::api3 {

class MergeNodeImpl : public Calculator<MergeNode, MergeNodeImpl> {
 public:
  static absl::Status UpdateContract(CalculatorContract<MergeNode>& cc) {
    RET_CHECK_GT(cc.in.Count(), 0) << "Needs at least one input stream";
    if (cc.in.Count() == 1) {
      ABSL_LOG(WARNING)
          << "MergeCalculator expects multiple input streams to merge but is "
             "receiving only one. Make sure the calculator is configured "
             "correctly or consider removing this calculator to reduce "
             "unnecessary overhead.";
    }
    return absl::OkStatus();
  }

  absl::Status Process(CalculatorContext<MergeNode>& cc) final {
    // Output the packet from the first input stream with a packet ready at this
    // timestamp.
    for (const auto& input : cc.in) {
      if (input) {
        cc.out.Send(input.Packet());
        return absl::OkStatus();
      }
    }

    ABSL_LOG(WARNING) << "Empty input packets at timestamp "
                      << cc.InputTimestamp().Value();

    return absl::OkStatus();
  }
};

}  // namespace mediapipe::api3
