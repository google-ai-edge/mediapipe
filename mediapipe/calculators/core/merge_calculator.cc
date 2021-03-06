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

#include "mediapipe/framework/api2/node.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/port/ret_check.h"
#include "mediapipe/framework/port/status.h"

namespace mediapipe {
namespace api2 {

// This calculator takes a set of input streams and combines them into a single
// output stream. The packets from different streams do not need to contain the
// same type. If there are packets arriving at the same time from two or more
// input streams, the packet corresponding to the input stream with the smallest
// index is passed to the output and the rest are ignored.
//
// Example use-case:
// Suppose we have two (or more) different algorithms for detecting shot
// boundaries and we need to merge their packets into a single stream. The
// algorithms may emit shot boundaries at the same time and their output types
// may not be compatible. Subsequent calculators that process the merged stream
// may be interested only in the timestamps of the shot boundary packets and so
// it may not even need to inspect the values stored inside the packets.
//
// Example config:
// node {
//   calculator: "MergeCalculator"
//   input_stream: "shot_info1"
//   input_stream: "shot_info2"
//   input_stream: "shot_info3"
//   output_stream: "merged_shot_infos"
// }
//
class MergeCalculator : public Node {
 public:
  static constexpr Input<AnyType>::Multiple kIn{""};
  static constexpr Output<AnyType> kOut{""};

  MEDIAPIPE_NODE_CONTRACT(kIn, kOut);

  static absl::Status UpdateContract(CalculatorContract* cc) {
    RET_CHECK_GT(kIn(cc).Count(), 0) << "Needs at least one input stream";
    if (kIn(cc).Count() == 1) {
      LOG(WARNING)
          << "MergeCalculator expects multiple input streams to merge but is "
             "receiving only one. Make sure the calculator is configured "
             "correctly or consider removing this calculator to reduce "
             "unnecessary overhead.";
    }
    return absl::OkStatus();
  }

  absl::Status Process(CalculatorContext* cc) final {
    // Output the packet from the first input stream with a packet ready at this
    // timestamp.
    for (const auto& input : kIn(cc)) {
      if (!input.IsEmpty()) {
        kOut(cc).Send(input.packet());
        return absl::OkStatus();
      }
    }

    LOG(WARNING) << "Empty input packets at timestamp "
                 << cc->InputTimestamp().Value();

    return absl::OkStatus();
  }
};

MEDIAPIPE_REGISTER_NODE(MergeCalculator);

}  // namespace api2
}  // namespace mediapipe
