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
// Defines MatrixToVectorCalculator.
#include <math.h>

#include <deque>
#include <memory>
#include <string>

#include "Eigen/Core"
#include "absl/memory/memory.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/matrix.h"
#include "mediapipe/framework/port/integral_types.h"
#include "mediapipe/framework/port/logging.h"
#include "mediapipe/framework/port/parse_text_proto.h"
#include "mediapipe/framework/tool/status_util.h"
#include "mediapipe/util/time_series_util.h"

namespace mediapipe {

// A calculator that converts a Matrix M to a vector containing all the
// entries of M in column-major order.
//
// Example config:
// node {
//   calculator: "MatrixToVectorCalculator"
//   input_stream: "input_matrix"
//   output_stream: "column_major_vector"
// }
class MatrixToVectorCalculator : public CalculatorBase {
 public:
  static ::mediapipe::Status GetContract(CalculatorContract* cc) {
    cc->Inputs().Index(0).Set<Matrix>(
        // Input Packet containing a Matrix.
    );
    cc->Outputs().Index(0).Set<std::vector<float>>(
        // Output Packet containing a vector, one for each input Packet.
    );
    return ::mediapipe::OkStatus();
  }

  ::mediapipe::Status Open(CalculatorContext* cc) override;

  // Outputs a packet containing a vector for each input packet.
  ::mediapipe::Status Process(CalculatorContext* cc) override;
};
REGISTER_CALCULATOR(MatrixToVectorCalculator);

::mediapipe::Status MatrixToVectorCalculator::Open(CalculatorContext* cc) {
  // Inform the framework that we don't alter timestamps.
  cc->SetOffset(mediapipe::TimestampDiff(0));
  return ::mediapipe::OkStatus();
}

::mediapipe::Status MatrixToVectorCalculator::Process(CalculatorContext* cc) {
  const Matrix& input = cc->Inputs().Index(0).Get<Matrix>();
  auto output = absl::make_unique<std::vector<float>>();

  // The following lines work to convert the Matrix to a vector because Matrix
  // is an Eigen::MatrixXf and Eigen uses column-major layout by default.
  output->resize(input.rows() * input.cols());
  auto output_as_matrix =
      Eigen::Map<Matrix>(output->data(), input.rows(), input.cols());
  output_as_matrix = input;

  cc->Outputs().Index(0).Add(output.release(), cc->InputTimestamp());
  return ::mediapipe::OkStatus();
}

}  // namespace mediapipe
