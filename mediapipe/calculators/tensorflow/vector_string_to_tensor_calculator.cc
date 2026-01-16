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
// Converts vector<std::string> (or vector<vector<std::string>>) to 1D (or 2D)
// tf::Tensor.

#include "absl/log/absl_log.h"
#include "mediapipe/calculators/tensorflow/vector_string_to_tensor_calculator_options.pb.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/port/ret_check.h"
#include "mediapipe/framework/port/status.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/types.h"

namespace mediapipe {

namespace {
auto& INPUT_1D = VectorStringToTensorCalculatorOptions::INPUT_1D;
auto& INPUT_2D = VectorStringToTensorCalculatorOptions::INPUT_2D;
}  // namespace

namespace tf = ::tensorflow;

// The calculator expects one input (a packet containing a vector<std::string>
// or vector<vector<std::string>>) and generates one output (a packet containing
// a tf::Tensor containing the same data). The output tensor will be either 1D
// or 2D with dimensions corresponding to the input vector std::string. It will
// hold DT_STRING values.
//
// Example config:
// node {
//   calculator: "VectorStringToTensorCalculator"
//   input_stream: "vector_string_features"
//   output_stream: "tensor_features"
// }
class VectorStringToTensorCalculator : public CalculatorBase {
 public:
  static absl::Status GetContract(CalculatorContract* cc);

  absl::Status Open(CalculatorContext* cc) override;
  absl::Status Process(CalculatorContext* cc) override;

 private:
  VectorStringToTensorCalculatorOptions options_;
};
REGISTER_CALCULATOR(VectorStringToTensorCalculator);

absl::Status VectorStringToTensorCalculator::GetContract(
    CalculatorContract* cc) {
  const auto& options = cc->Options<VectorStringToTensorCalculatorOptions>();
  // Start with only one input packet.
  RET_CHECK_EQ(cc->Inputs().NumEntries(), 1)
      << "Only one input stream is supported.";
  if (options.input_size() == INPUT_2D) {
    cc->Inputs().Index(0).Set<std::vector<std::vector<std::string>>>(
        /* "Input vector<vector<std::string>>." */);
  } else if (options.input_size() == INPUT_1D) {
    cc->Inputs().Index(0).Set<std::vector<std::string>>(
        // Input vector<std::string>.
    );
  } else {
    ABSL_LOG(FATAL) << "input size not supported";
  }
  RET_CHECK_EQ(cc->Outputs().NumEntries(), 1)
      << "Only one output stream is supported.";
  cc->Outputs().Index(0).Set<tf::Tensor>(
      // Output stream with data as tf::Tensor and the same TimeSeriesHeader.
  );
  return absl::OkStatus();
}

absl::Status VectorStringToTensorCalculator::Open(CalculatorContext* cc) {
  options_ = cc->Options<VectorStringToTensorCalculatorOptions>();
  cc->SetOffset(0);
  return absl::OkStatus();
}

absl::Status VectorStringToTensorCalculator::Process(CalculatorContext* cc) {
  tf::TensorShape tensor_shape;
  if (options_.input_size() == INPUT_2D) {
    const std::vector<std::vector<std::string>>& input =
        cc->Inputs()
            .Index(0)
            .Value()
            .Get<std::vector<std::vector<std::string>>>();

    const int32_t rows = input.size();
    RET_CHECK_GE(rows, 1);
    const int32_t cols = input[0].size();
    RET_CHECK_GE(cols, 1);
    for (int i = 1; i < rows; ++i) {
      RET_CHECK_EQ(input[i].size(), cols);
    }
    if (options_.transpose()) {
      tensor_shape = tf::TensorShape({cols, rows});
    } else {
      tensor_shape = tf::TensorShape({rows, cols});
    }
    auto output = ::absl::make_unique<tf::Tensor>(tf::DT_STRING, tensor_shape);
    for (int r = 0; r < rows; ++r) {
      for (int c = 0; c < cols; ++c) {
        if (options_.transpose()) {
          output->tensor<tensorflow::tstring, 2>()(c, r) = input[r][c];
        } else {
          output->tensor<tensorflow::tstring, 2>()(r, c) = input[r][c];
        }
      }
    }
    cc->Outputs().Index(0).Add(output.release(), cc->InputTimestamp());
  } else if (options_.input_size() == INPUT_1D) {
    const std::vector<std::string>& input =
        cc->Inputs().Index(0).Value().Get<std::vector<std::string>>();
    RET_CHECK_GE(input.size(), 1);
    const int32_t length = input.size();
    tensor_shape = tf::TensorShape({length});
    auto output = ::absl::make_unique<tf::Tensor>(tf::DT_STRING, tensor_shape);
    for (int i = 0; i < length; ++i) {
      output->tensor<tensorflow::tstring, 1>()(i) = input.at(i);
    }
    cc->Outputs().Index(0).Add(output.release(), cc->InputTimestamp());
  } else {
    ABSL_LOG(FATAL) << "input size not supported";
  }
  return absl::OkStatus();
}

}  // namespace mediapipe
