// Copyright 2021 The MediaPipe Authors.
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

#ifndef MEDIAPIPE_CALCULATORS_UTIL_INVERSE_MATRIX_CALCULATOR_H_
#define MEDIAPIPE_CALCULATORS_UTIL_INVERSE_MATRIX_CALCULATOR_H_

#include <array>

#include "absl/strings/string_view.h"
#include "mediapipe/framework/api3/contract.h"
#include "mediapipe/framework/api3/node.h"

namespace mediapipe::api3 {

inline constexpr absl::string_view kInverseMatrixNodeName =
    "InverseMatrixCalculator";

// Inverses a row-major 4x4 matrix.
//
// Proto usage example:
//   node {
//     calculator: "InverseMatrixCalculator"
//     input_stream: "MATRIX:input_matrix"
//     output_stream: "MATRIX:output_matrix"
//   }
struct InverseMatrixNode : Node<kInverseMatrixNodeName> {
  template <typename S>
  struct Contract {
    // Row major 4x4 matrix to inverse.
    Input<S, std::array<float, 16>> input_matrix{"MATRIX"};
    // Row major 4x4 inversed matrix.
    Output<S, std::array<float, 16>> output_matrix{"MATRIX"};
  };
};

}  // namespace mediapipe::api3

#endif  // MEDIAPIPE_CALCULATORS_UTIL_INVERSE_MATRIX_CALCULATOR_H_
