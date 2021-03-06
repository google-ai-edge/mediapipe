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
// Converts a single int or vector<int> or vector<vector<int>> to 1D (or 2D)
// tf::Tensor.

#include "mediapipe/calculators/tensorflow/vector_int_to_tensor_calculator_options.pb.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/port/ret_check.h"
#include "mediapipe/framework/port/status.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/types.h"

namespace mediapipe {

const char kVectorInt[] = "VECTOR_INT";
const char kSingleInt[] = "SINGLE_INT";
const char kTensorOut[] = "TENSOR_OUT";

namespace {
auto& INPUT_1D = VectorIntToTensorCalculatorOptions::INPUT_1D;
auto& INPUT_2D = VectorIntToTensorCalculatorOptions::INPUT_2D;
}  // namespace

namespace tf = ::tensorflow;

template <typename TensorType>
void AssignMatrixValue(int r, int c, int value, tf::Tensor* output_tensor) {
  output_tensor->tensor<TensorType, 2>()(r, c) = value;
}

// The calculator expects one input (a packet containing a single int or
// vector<int> or vector<vector<int>>) and generates one output (a packet
// containing a tf::Tensor containing the same data). The output tensor will be
// either 1D or 2D with dimensions corresponding to the input vector int. It
// will hold DT_INT32 or DT_UINT8 or DT_INT64 values.
//
// Example config:
// node {
//  calculator: "VectorIntToTensorCalculator"
//   input_stream: "SINGLE_INT:segment_size_int_stream"
//   output_stream: "TENSOR_OUT:segment_size_tensor"
// }
//
// or
//
// node {
//   calculator: "VectorIntToTensorCalculator"
//   input_stream: "VECTOR_INT:vector_int_features"
//   output_stream: "TENSOR_OUT:tensor_features"
// }
class VectorIntToTensorCalculator : public CalculatorBase {
 public:
  static absl::Status GetContract(CalculatorContract* cc);

  absl::Status Open(CalculatorContext* cc) override;
  absl::Status Process(CalculatorContext* cc) override;

 private:
  VectorIntToTensorCalculatorOptions options_;
};
REGISTER_CALCULATOR(VectorIntToTensorCalculator);

absl::Status VectorIntToTensorCalculator::GetContract(CalculatorContract* cc) {
  const auto& options = cc->Options<VectorIntToTensorCalculatorOptions>();
  // Start with only one input packet.
  RET_CHECK_EQ(cc->Inputs().NumEntries(), 1)
      << "Only one input stream is supported.";
  if (options.input_size() == INPUT_2D) {
    cc->Inputs().Tag(kVectorInt).Set<std::vector<std::vector<int>>>();
  } else if (options.input_size() == INPUT_1D) {
    if (cc->Inputs().HasTag(kSingleInt)) {
      cc->Inputs().Tag(kSingleInt).Set<int>();
    } else {
      cc->Inputs().Tag(kVectorInt).Set<std::vector<int>>();
    }
  } else {
    LOG(FATAL) << "input size not supported";
  }
  RET_CHECK_EQ(cc->Outputs().NumEntries(), 1)
      << "Only one output stream is supported.";
  cc->Outputs().Tag(kTensorOut).Set<tf::Tensor>();
  return absl::OkStatus();
}

absl::Status VectorIntToTensorCalculator::Open(CalculatorContext* cc) {
  options_ = cc->Options<VectorIntToTensorCalculatorOptions>();
  RET_CHECK(options_.tensor_data_type() == tf::DT_UINT8 ||
            options_.tensor_data_type() == tf::DT_INT32 ||
            options_.tensor_data_type() == tf::DT_INT64)
      << "Output tensor data type is not supported.";
  return absl::OkStatus();
}

absl::Status VectorIntToTensorCalculator::Process(CalculatorContext* cc) {
  tf::TensorShape tensor_shape;
  if (options_.input_size() == INPUT_2D) {
    const std::vector<std::vector<int>>& input =
        cc->Inputs()
            .Tag(kVectorInt)
            .Value()
            .Get<std::vector<std::vector<int>>>();

    const int32 rows = input.size();
    CHECK_GE(rows, 1);
    const int32 cols = input[0].size();
    CHECK_GE(cols, 1);
    for (int i = 1; i < rows; ++i) {
      CHECK_EQ(input[i].size(), cols);
    }
    if (options_.transpose()) {
      tensor_shape = tf::TensorShape({cols, rows});
    } else {
      tensor_shape = tf::TensorShape({rows, cols});
    }
    auto output = ::absl::make_unique<tf::Tensor>(options_.tensor_data_type(),
                                                  tensor_shape);
    if (options_.transpose()) {
      for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
          switch (options_.tensor_data_type()) {
            case tf::DT_INT64:
              AssignMatrixValue<tf::int64>(c, r, input[r][c], output.get());
              break;
            case tf::DT_UINT8:
              AssignMatrixValue<uint8>(c, r, input[r][c], output.get());
              break;
            case tf::DT_INT32:
              AssignMatrixValue<int>(c, r, input[r][c], output.get());
              break;
            default:
              LOG(FATAL) << "tensor data type is not supported.";
          }
        }
      }
    } else {
      for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
          switch (options_.tensor_data_type()) {
            case tf::DT_INT64:
              AssignMatrixValue<tf::int64>(r, c, input[r][c], output.get());
              break;
            case tf::DT_UINT8:
              AssignMatrixValue<uint8>(r, c, input[r][c], output.get());
              break;
            case tf::DT_INT32:
              AssignMatrixValue<int>(r, c, input[r][c], output.get());
              break;
            default:
              LOG(FATAL) << "tensor data type is not supported.";
          }
        }
      }
    }
    cc->Outputs().Tag(kTensorOut).Add(output.release(), cc->InputTimestamp());
  } else if (options_.input_size() == INPUT_1D) {
    std::vector<int> input;
    if (cc->Inputs().HasTag(kSingleInt)) {
      input.push_back(cc->Inputs().Tag(kSingleInt).Get<int>());
    } else {
      input = cc->Inputs().Tag(kVectorInt).Value().Get<std::vector<int>>();
    }
    CHECK_GE(input.size(), 1);
    const int32 length = input.size();
    tensor_shape = tf::TensorShape({length});
    auto output = ::absl::make_unique<tf::Tensor>(options_.tensor_data_type(),
                                                  tensor_shape);
    for (int i = 0; i < length; ++i) {
      switch (options_.tensor_data_type()) {
        case tf::DT_INT64:
          output->tensor<tf::int64, 1>()(i) = input.at(i);
          break;
        case tf::DT_UINT8:
          output->tensor<uint8, 1>()(i) = input.at(i);
          break;
        case tf::DT_INT32:
          output->tensor<int, 1>()(i) = input.at(i);
          break;
        default:
          LOG(FATAL) << "tensor data type is not supported.";
      }
    }
    cc->Outputs().Tag(kTensorOut).Add(output.release(), cc->InputTimestamp());
  } else {
    LOG(FATAL) << "input size not supported";
  }
  return absl::OkStatus();
}

}  // namespace mediapipe
