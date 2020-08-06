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
// Converts vector<float> (or vector<vector<float>>) to 1D (or 2D) tf::Tensor.

#include "mediapipe/calculators/tensorflow/vector_float_to_tensor_calculator_options.pb.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/port/ret_check.h"
#include "mediapipe/framework/port/status.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/types.h"

namespace mediapipe {

namespace {
auto& INPUT_1D = VectorFloatToTensorCalculatorOptions::INPUT_1D;
auto& INPUT_2D = VectorFloatToTensorCalculatorOptions::INPUT_2D;
}  // namespace

namespace tf = ::tensorflow;

// The calculator expects one input (a packet containing a vector<float> or
// vector<vector<float>>) and generates one output (a packet containing a
// tf::Tensor containing the same data). The output tensor will be either
// 1D or 2D with dimensions corresponding to the input vector float.
// It will hold DT_FLOAT values.
//
// Example config:
// node {
//   calculator: "VectorFloatToTensorCalculator"
//   input_stream: "vector_float_features"
//   output_stream: "tensor_features"
// }
class VectorFloatToTensorCalculator : public CalculatorBase {
 public:
  static ::mediapipe::Status GetContract(CalculatorContract* cc);

  ::mediapipe::Status Open(CalculatorContext* cc) override;
  ::mediapipe::Status Process(CalculatorContext* cc) override;

 private:
  VectorFloatToTensorCalculatorOptions options_;
};
REGISTER_CALCULATOR(VectorFloatToTensorCalculator);

::mediapipe::Status VectorFloatToTensorCalculator::GetContract(
    CalculatorContract* cc) {
  const auto& options = cc->Options<VectorFloatToTensorCalculatorOptions>();
  // Start with only one input packet.
  RET_CHECK_EQ(cc->Inputs().NumEntries(), 1)
      << "Only one input stream is supported.";
  if (options.input_size() == INPUT_2D) {
    cc->Inputs().Index(0).Set<std::vector<std::vector<float>>>(
        /* "Input vector<vector<float>>." */);
  } else if (options.input_size() == INPUT_1D) {
    cc->Inputs().Index(0).Set<std::vector<float>>(
        // Output vector<float>.
    );
  } else {
    LOG(FATAL) << "input size not supported";
  }
  RET_CHECK_EQ(cc->Outputs().NumEntries(), 1)
      << "Only one output stream is supported.";
  cc->Outputs().Index(0).Set<tf::Tensor>(
      // Output stream with data as tf::Tensor and the same TimeSeriesHeader.
  );
  return ::mediapipe::OkStatus();
}

::mediapipe::Status VectorFloatToTensorCalculator::Open(CalculatorContext* cc) {
  options_ = cc->Options<VectorFloatToTensorCalculatorOptions>();
  return ::mediapipe::OkStatus();
}

::mediapipe::Status VectorFloatToTensorCalculator::Process(
    CalculatorContext* cc) {
  tf::TensorShape tensor_shape;
  if (options_.input_size() == INPUT_2D) {
    const std::vector<std::vector<float>>& input =
        cc->Inputs().Index(0).Value().Get<std::vector<std::vector<float>>>();

    const int32 rows = input.size();
    RET_CHECK_GE(rows, 1);
    const int32 cols = input[0].size();
    RET_CHECK_GE(cols, 1);
    for (int i = 1; i < rows; ++i) {
      RET_CHECK_EQ(input[i].size(), cols);
    }
    if (options_.transpose()) {
      tensor_shape = tf::TensorShape({cols, rows});
    } else {
      tensor_shape = tf::TensorShape({rows, cols});
    }
    auto output = ::absl::make_unique<tf::Tensor>(tf::DT_FLOAT, tensor_shape);
    for (int r = 0; r < rows; ++r) {
      for (int c = 0; c < cols; ++c) {
        if (options_.transpose()) {
          output->tensor<float, 2>()(c, r) = input[r][c];
        } else {
          output->tensor<float, 2>()(r, c) = input[r][c];
        }
      }
    }
    cc->Outputs().Index(0).Add(output.release(), cc->InputTimestamp());
  } else if (options_.input_size() == INPUT_1D) {
    const std::vector<float>& input =
        cc->Inputs().Index(0).Value().Get<std::vector<float>>();
    RET_CHECK_GE(input.size(), 1);
    const int32 length = input.size();
    tensor_shape = tf::TensorShape({length});
    auto output = ::absl::make_unique<tf::Tensor>(tf::DT_FLOAT, tensor_shape);
    for (int i = 0; i < length; ++i) {
      output->tensor<float, 1>()(i) = input.at(i);
    }
    cc->Outputs().Index(0).Add(output.release(), cc->InputTimestamp());
  } else {
    LOG(FATAL) << "input size not supported";
  }
  return ::mediapipe::OkStatus();
}

}  // namespace mediapipe
