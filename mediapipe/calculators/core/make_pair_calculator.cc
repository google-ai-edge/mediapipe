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

#include <utility>
#include <vector>

#include "mediapipe/framework/api2/node.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/port/status.h"

namespace mediapipe {
namespace api2 {

// Given two input streams (A, B), output a single stream containing a pair<A,
// B>.
//
// Example config:
// node {
//   calculator: "MakePairCalculator"
//   input_stream: "packet_a"
//   input_stream: "packet_b"
//   output_stream: "output_pair_a_b"
// }
class MakePairCalculator : public Node {
 public:
  static constexpr Input<AnyType>::Multiple kIn{""};
  // Note that currently api2::Packet is a different type from mediapipe::Packet
  static constexpr Output<std::pair<mediapipe::Packet, mediapipe::Packet>>
      kPair{""};

  MEDIAPIPE_NODE_CONTRACT(kIn, kPair);

  static absl::Status UpdateContract(CalculatorContract* cc) {
    RET_CHECK_EQ(kIn(cc).Count(), 2);
    return absl::OkStatus();
  }

  absl::Status Process(CalculatorContext* cc) override {
    kPair(cc).Send({kIn(cc)[0].packet(), kIn(cc)[1].packet()});
    return absl::OkStatus();
  }
};

MEDIAPIPE_REGISTER_NODE(MakePairCalculator);

}  // namespace api2
}  // namespace mediapipe
