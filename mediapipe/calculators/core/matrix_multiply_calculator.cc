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

#include "Eigen/Core"
#include "mediapipe/framework/api2/node.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/matrix.h"
#include "mediapipe/framework/port/status.h"

namespace mediapipe {
namespace api2 {
// Perform a (left) matrix multiply.  Meaning (output = A * input)
// where A is the matrix which is provided as an input side packet.
//
// Example config:
// node {
//   calculator: "MatrixMultiplyCalculator"
//   input_stream: "samples"
//   output_stream: "multiplied_samples"
//   input_side_packet: "multiplication_matrix"
// }
class MatrixMultiplyCalculator : public Node {
 public:
  static constexpr Input<Matrix> kIn{""};
  static constexpr Output<Matrix> kOut{""};
  static constexpr SideInput<Matrix> kSide{""};

  MEDIAPIPE_NODE_CONTRACT(kIn, kOut, kSide);

  absl::Status Process(CalculatorContext* cc) override;
};
MEDIAPIPE_REGISTER_NODE(MatrixMultiplyCalculator);

absl::Status MatrixMultiplyCalculator::Process(CalculatorContext* cc) {
  kOut(cc).Send(*kSide(cc) * *kIn(cc));
  return absl::OkStatus();
}

}  // namespace api2
}  // namespace mediapipe
