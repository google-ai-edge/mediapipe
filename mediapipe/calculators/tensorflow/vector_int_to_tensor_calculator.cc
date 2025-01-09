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

#include <cstdint>
#include <memory>
#include <vector>

#include "absl/log/absl_check.h"
#include "absl/log/absl_log.h"
#include "absl/status/status.h"
#include "mediapipe/calculators/tensorflow/vector_int_to_tensor_calculator_options.pb.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/port/ret_check.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"

namespace mediapipe {

const char kVectorInt[] = "VECTOR_INT";
const char kSingleInt[] = "SINGLE_INT";
const char kTensorOut[] = "TENSOR_OUT";

namespace {
auto& INPUT_1D = VectorIntToTensorCalculatorOptions::INPUT_1D;
auto& INPUT_2D = VectorIntToTensorCalculatorOptions::INPUT_2D;

bool RequiresUnsignedType(tensorflow::DataType data_type) {
  return data_type == tensorflow::DT_UINT32;
}

namespace tf = ::tensorflow;

template <typename TensorType>
void AssignMatrixValue(int r, int c, int value, tf::Tensor* output_tensor) {
  output_tensor->tensor<TensorType, 2>()(r, c) = value;
}

template <typename InputType, typename DataType>
absl::Status ProcessVectorIntToTensor(
    const VectorIntToTensorCalculatorOptions& options, CalculatorContext* cc) {
  tf::TensorShape tensor_shape;
  if (options.input_size() == INPUT_2D) {
    const std::vector<std::vector<InputType>>& input =
        cc->Inputs()
            .Tag(kVectorInt)
            .Value()
            .Get<std::vector<std::vector<InputType>>>();

    const int32_t rows = input.size();
    ABSL_CHECK_GE(rows, 1);
    const int32_t cols = input[0].size();
    ABSL_CHECK_GE(cols, 1);
    for (int i = 1; i < rows; ++i) {
      ABSL_CHECK_EQ(input[i].size(), cols);
    }
    if (options.transpose()) {
      tensor_shape = tf::TensorShape({cols, rows});
    } else {
      tensor_shape = tf::TensorShape({rows, cols});
    }
    auto output =
        std::make_unique<tf::Tensor>(options.tensor_data_type(), tensor_shape);
    if (options.transpose()) {
      for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
          AssignMatrixValue<DataType>(c, r, input[r][c], output.get());
        }
      }
    } else {
      for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
          AssignMatrixValue<DataType>(r, c, input[r][c], output.get());
        }
      }
    }
    cc->Outputs().Tag(kTensorOut).Add(output.release(), cc->InputTimestamp());
  } else if (options.input_size() == INPUT_1D) {
    std::vector<InputType> input;
    if (cc->Inputs().HasTag(kSingleInt)) {
      input.push_back(cc->Inputs().Tag(kSingleInt).Get<InputType>());
    } else {
      input =
          cc->Inputs().Tag(kVectorInt).Value().Get<std::vector<InputType>>();
    }

    if (options.scalar_output()) {
      ABSL_CHECK_EQ(input.size(), 1);
      tensor_shape = tf::TensorShape({});
      auto output = std::make_unique<tf::Tensor>(options.tensor_data_type(),
                                                 tensor_shape);
      output->scalar<DataType>()() = input.at(0);
      cc->Outputs().Tag(kTensorOut).Add(output.release(), cc->InputTimestamp());
    } else {
      ABSL_CHECK_GE(input.size(), 1);
      const int32_t length = input.size();
      tensor_shape = tf::TensorShape({length});
      auto output = std::make_unique<tf::Tensor>(options.tensor_data_type(),
                                                 tensor_shape);
      for (int i = 0; i < length; ++i) {
        output->tensor<DataType, 1>()(i) = input.at(i);
      }
      cc->Outputs().Tag(kTensorOut).Add(output.release(), cc->InputTimestamp());
    }
  } else {
    ABSL_LOG(FATAL) << "input size not supported";
  }
  return absl::OkStatus();
}

}  // namespace

// The calculator expects one input (a packet containing a single int or
// vector<int> or vector<vector<int>>) and generates one output (a packet
// containing a tf::Tensor containing the same data). The output tensor will be
// either 1D or 2D with dimensions corresponding to the input vector int. It
// will hold DT_INT32 or DT_UINT32 or DT_UINT8 or DT_INT64 values.
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
      if (RequiresUnsignedType(options.tensor_data_type())) {
        cc->Inputs().Tag(kSingleInt).Set<uint32_t>();
      } else {
        cc->Inputs().Tag(kSingleInt).Set<int>();
      }
    } else {
      if (RequiresUnsignedType(options.tensor_data_type())) {
        cc->Inputs().Tag(kVectorInt).Set<std::vector<uint32_t>>();
      } else {
        cc->Inputs().Tag(kVectorInt).Set<std::vector<int>>();
      }
    }
  } else {
    ABSL_LOG(FATAL) << "input size not supported";
  }
  if (options.scalar_output() && options.input_size() != INPUT_1D) {
    return absl::InvalidArgumentError(
        "scalar_output is only supported for input_size = INPUT_1D.");
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
            options_.tensor_data_type() == tf::DT_UINT32 ||
            options_.tensor_data_type() == tf::DT_INT64)
      << "Output tensor data type is not supported.";
  return absl::OkStatus();
}

absl::Status VectorIntToTensorCalculator::Process(CalculatorContext* cc) {
  switch (options_.tensor_data_type()) {
    case tf::DT_INT64:
      return ProcessVectorIntToTensor<int, int64_t>(options_, cc);
    case tf::DT_UINT8:
      return ProcessVectorIntToTensor<int, uint8_t>(options_, cc);
    case tf::DT_INT32:
      return ProcessVectorIntToTensor<int, int>(options_, cc);
    case tf::DT_UINT32:
      return ProcessVectorIntToTensor<uint32_t, uint32_t>(options_, cc);
    default:
      ABSL_LOG(FATAL) << "tensor data type is not supported.";
  }
}

}  // namespace mediapipe
