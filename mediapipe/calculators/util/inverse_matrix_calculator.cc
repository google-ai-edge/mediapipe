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

#include "mediapipe/calculators/util/inverse_matrix_calculator.h"

#include <array>
#include <utility>

#include "Eigen/Core"
#include "Eigen/Geometry"  // IWYU pragma: keep(for computeInverseWithCheck)
#include "absl/status/status.h"
#include "mediapipe/framework/api3/calculator.h"
#include "mediapipe/framework/api3/calculator_context.h"
#include "mediapipe/framework/port/ret_check.h"

namespace mediapipe {
namespace api3 {

class InverseMatrixNodeImpl
    : public Calculator<InverseMatrixNode, InverseMatrixNodeImpl> {
  absl::Status Process(CalculatorContext<InverseMatrixNode>& cc) override {
    if (!cc.input_matrix) {
      return absl::OkStatus();
    }
    Eigen::Matrix<float, 4, 4, Eigen::RowMajor> matrix(
        cc.input_matrix.GetOrDie().data());

    Eigen::Matrix<float, 4, 4, Eigen::RowMajor> inverse_matrix;
    bool inverse_check = false;
    // The matrix is invertible if the absolute value of its determinant is
    // greater than this threshold. Quite small threshold is selected to enable
    // inverting valid matrices containing relatively small values resulting in
    // a small determinant.
    constexpr double kAbsDeterminantThreshold =
        Eigen::NumTraits<double>::epsilon();
    matrix.computeInverseWithCheck(inverse_matrix, inverse_check,
                                   kAbsDeterminantThreshold);
    RET_CHECK(inverse_check)
        << "Inverse matrix cannot be calculated for: " << matrix;

    std::array<float, 16> output;
    Eigen::Map<Eigen::Matrix<float, 4, 4, Eigen::RowMajor>>(
        output.data(), 4, 4) = inverse_matrix.matrix();
    cc.output_matrix.Send(std::move(output));

    return absl::OkStatus();
  }
};

}  // namespace api3
}  // namespace mediapipe
