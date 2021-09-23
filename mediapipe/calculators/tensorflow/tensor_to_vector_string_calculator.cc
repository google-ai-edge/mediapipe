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
// Calculator converts from one-dimensional Tensor of DT_STRING to
// vector<std::string> OR from (batched) two-dimensional Tensor of DT_STRING to
// vector<vector<std::string>.

#include "mediapipe/calculators/tensorflow/tensor_to_vector_string_calculator_options.pb.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/port/ret_check.h"
#include "mediapipe/framework/port/status.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/types.h"

namespace mediapipe {

namespace tf = ::tensorflow;

class TensorToVectorStringCalculator : public CalculatorBase {
 public:
  static absl::Status GetContract(CalculatorContract* cc);

  absl::Status Open(CalculatorContext* cc) override;
  absl::Status Process(CalculatorContext* cc) override;

 private:
  TensorToVectorStringCalculatorOptions options_;
};
REGISTER_CALCULATOR(TensorToVectorStringCalculator);

absl::Status TensorToVectorStringCalculator::GetContract(
    CalculatorContract* cc) {
  // Start with only one input packet.
  RET_CHECK_EQ(cc->Inputs().NumEntries(), 1)
      << "Only one input stream is supported.";
  cc->Inputs().Index(0).Set<tf::Tensor>(
      // Input Tensor
  );
  RET_CHECK_EQ(cc->Outputs().NumEntries(), 1)
      << "Only one output stream is supported.";
  const auto& options = cc->Options<TensorToVectorStringCalculatorOptions>();
  if (options.tensor_is_2d()) {
    RET_CHECK(!options.flatten_nd());
    cc->Outputs().Index(0).Set<std::vector<std::vector<std::string>>>(
        /* "Output vector<vector<std::string>>." */);
  } else {
    cc->Outputs().Index(0).Set<std::vector<std::string>>(
        // Output vector<std::string>.
    );
  }
  return absl::OkStatus();
}

absl::Status TensorToVectorStringCalculator::Open(CalculatorContext* cc) {
  options_ = cc->Options<TensorToVectorStringCalculatorOptions>();

  // Inform mediapipe that this calculator produces an output at time t for
  // each input received at time t (i.e. this calculator does not buffer
  // inputs). This enables mediapipe to propagate time of arrival estimates in
  // mediapipe graphs through this calculator.
  cc->SetOffset(/*offset=*/0);

  return absl::OkStatus();
}

absl::Status TensorToVectorStringCalculator::Process(CalculatorContext* cc) {
  const tf::Tensor& input_tensor =
      cc->Inputs().Index(0).Value().Get<tf::Tensor>();
  RET_CHECK(tf::DT_STRING == input_tensor.dtype())
      << "expected DT_STRING input but got "
      << tensorflow::DataTypeString(input_tensor.dtype());

  if (options_.tensor_is_2d()) {
    RET_CHECK(2 == input_tensor.dims())
        << "Expected 2-dimensional Tensor, but the tensor shape is: "
        << input_tensor.shape().DebugString();
    auto output = absl::make_unique<std::vector<std::vector<std::string>>>(
        input_tensor.dim_size(0),
        std::vector<std::string>(input_tensor.dim_size(1)));
    for (int i = 0; i < input_tensor.dim_size(0); ++i) {
      auto& instance_output = output->at(i);
      const auto& slice =
          input_tensor.Slice(i, i + 1).unaligned_flat<tensorflow::tstring>();
      for (int j = 0; j < input_tensor.dim_size(1); ++j) {
        instance_output.at(j) = slice(j);
      }
    }
    cc->Outputs().Index(0).Add(output.release(), cc->InputTimestamp());
  } else {
    if (!options_.flatten_nd()) {
      RET_CHECK(1 == input_tensor.dims())
          << "`flatten_nd` is not set. Expected 1-dimensional Tensor, but the "
          << "tensor shape is: " << input_tensor.shape().DebugString();
    }
    auto output =
        absl::make_unique<std::vector<std::string>>(input_tensor.NumElements());
    const auto& tensor_values = input_tensor.flat<tensorflow::tstring>();
    for (int i = 0; i < input_tensor.NumElements(); ++i) {
      output->at(i) = tensor_values(i);
    }
    cc->Outputs().Index(0).Add(output.release(), cc->InputTimestamp());
  }

  return absl::OkStatus();
}

}  // namespace mediapipe
