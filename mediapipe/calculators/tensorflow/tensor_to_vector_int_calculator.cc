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
// Calculator converts from one-dimensional Tensor of DT_FLOAT to vector<float>
// OR from (batched) two-dimensional Tensor of DT_FLOAT to vector<vector<float>.

#include <memory>

#include "absl/base/integral_types.h"
#include "mediapipe/calculators/tensorflow/tensor_to_vector_int_calculator_options.pb.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/port/status.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/types.h"

namespace mediapipe {

namespace tf = ::tensorflow;

class TensorToVectorIntCalculator : public CalculatorBase {
 public:
  static absl::Status GetContract(CalculatorContract* cc);

  absl::Status Open(CalculatorContext* cc) override;
  absl::Status Process(CalculatorContext* cc) override;

 private:
  void TokenizeVector(std::vector<int64>* vector) const;

  TensorToVectorIntCalculatorOptions options_;
};
REGISTER_CALCULATOR(TensorToVectorIntCalculator);

absl::Status TensorToVectorIntCalculator::GetContract(CalculatorContract* cc) {
  // Start with only one input packet.
  RET_CHECK_EQ(cc->Inputs().NumEntries(), 1)
      << "Only one input stream is supported.";
  cc->Inputs().Index(0).Set<tf::Tensor>(
      // Input Tensor
  );
  RET_CHECK_EQ(cc->Outputs().NumEntries(), 1)
      << "Only one output stream is supported.";
  const auto& options = cc->Options<TensorToVectorIntCalculatorOptions>();
  if (options.tensor_is_2d()) {
    RET_CHECK(!options.flatten_nd());
    cc->Outputs().Index(0).Set<std::vector<std::vector<int64>>>(
        /* "Output vector<vector<float>>." */);
  } else {
    cc->Outputs().Index(0).Set<std::vector<int64>>(
        // Output vector<float>.
    );
  }
  return absl::OkStatus();
}

absl::Status TensorToVectorIntCalculator::Open(CalculatorContext* cc) {
  options_ = cc->Options<TensorToVectorIntCalculatorOptions>();

  // Inform mediapipe that this calculator produces an output at time t for
  // each input received at time t (i.e. this calculator does not buffer
  // inputs). This enables mediapipe to propagate time of arrival estimates in
  // mediapipe graphs through this calculator.
  cc->SetOffset(/*offset=*/0);

  return absl::OkStatus();
}

absl::Status TensorToVectorIntCalculator::Process(CalculatorContext* cc) {
  const tf::Tensor& input_tensor =
      cc->Inputs().Index(0).Value().Get<tf::Tensor>();
  RET_CHECK(tf::DT_INT32 == input_tensor.dtype() ||
            tf::DT_INT64 == input_tensor.dtype())
      << "expected DT_INT32 or DT_INT64 input but got "
      << tensorflow::DataTypeString(input_tensor.dtype());

  if (options_.tensor_is_2d()) {
    RET_CHECK(2 == input_tensor.dims())
        << "Expected 2-dimensional Tensor, but the tensor shape is: "
        << input_tensor.shape().DebugString();
    auto output = absl::make_unique<std::vector<std::vector<int64>>>(
        input_tensor.dim_size(0), std::vector<int64>(input_tensor.dim_size(1)));
    for (int i = 0; i < input_tensor.dim_size(0); ++i) {
      auto& instance_output = output->at(i);
      if (tf::DT_INT32 == input_tensor.dtype()) {
        const auto& slice =
            input_tensor.Slice(i, i + 1).unaligned_flat<int32>();
        for (int j = 0; j < input_tensor.dim_size(1); ++j) {
          instance_output.at(j) = slice(j);
        }
      } else {
        const auto& slice =
            input_tensor.Slice(i, i + 1).unaligned_flat<int64>();
        for (int j = 0; j < input_tensor.dim_size(1); ++j) {
          instance_output.at(j) = slice(j);
        }
      }
      TokenizeVector(&instance_output);
    }
    cc->Outputs().Index(0).Add(output.release(), cc->InputTimestamp());
  } else {
    if (!options_.flatten_nd()) {
      RET_CHECK(1 == input_tensor.dims())
          << "`flatten_nd` is not set. Expected 1-dimensional Tensor, but the "
          << "tensor shape is: " << input_tensor.shape().DebugString();
    }
    auto output =
        absl::make_unique<std::vector<int64>>(input_tensor.NumElements());
    if (tf::DT_INT32 == input_tensor.dtype()) {
      const auto& tensor_values = input_tensor.flat<int32>();
      for (int i = 0; i < input_tensor.NumElements(); ++i) {
        output->at(i) = tensor_values(i);
      }
    } else {
      const auto& tensor_values = input_tensor.flat<int64>();
      for (int i = 0; i < input_tensor.NumElements(); ++i) {
        output->at(i) = tensor_values(i);
      }
    }
    TokenizeVector(output.get());
    cc->Outputs().Index(0).Add(output.release(), cc->InputTimestamp());
  }

  return absl::OkStatus();
}

void TensorToVectorIntCalculator::TokenizeVector(
    std::vector<int64>* vector) const {
  if (!options_.tensor_is_token()) {
    return;
  }
  std::vector<int64> tokens;
  for (int i = 0; i < vector->size(); ++i) {
    if (vector->at(i) > options_.token_threshold()) {
      tokens.push_back(i + 1);
    }
  }
  vector->swap(tokens);
}

}  // namespace mediapipe
