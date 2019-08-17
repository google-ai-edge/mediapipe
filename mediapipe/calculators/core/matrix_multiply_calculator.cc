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
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/matrix.h"
#include "mediapipe/framework/port/status.h"

namespace mediapipe {
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
class MatrixMultiplyCalculator : public CalculatorBase {
 public:
  MatrixMultiplyCalculator() {}
  ~MatrixMultiplyCalculator() override {}

  static ::mediapipe::Status GetContract(CalculatorContract* cc);

  ::mediapipe::Status Open(CalculatorContext* cc) override;
  ::mediapipe::Status Process(CalculatorContext* cc) override;
};
REGISTER_CALCULATOR(MatrixMultiplyCalculator);

// static
::mediapipe::Status MatrixMultiplyCalculator::GetContract(
    CalculatorContract* cc) {
  cc->Inputs().Index(0).Set<Matrix>();
  cc->Outputs().Index(0).Set<Matrix>();
  cc->InputSidePackets().Index(0).Set<Matrix>();
  return ::mediapipe::OkStatus();
}

::mediapipe::Status MatrixMultiplyCalculator::Open(CalculatorContext* cc) {
  // The output is at the same timestamp as the input.
  cc->SetOffset(TimestampDiff(0));
  return ::mediapipe::OkStatus();
}

::mediapipe::Status MatrixMultiplyCalculator::Process(CalculatorContext* cc) {
  Matrix* multiplied = new Matrix();
  *multiplied = cc->InputSidePackets().Index(0).Get<Matrix>() *
                cc->Inputs().Index(0).Get<Matrix>();
  cc->Outputs().Index(0).Add(multiplied, cc->InputTimestamp());
  return ::mediapipe::OkStatus();
}

}  // namespace mediapipe
