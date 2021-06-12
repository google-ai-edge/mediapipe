// Copyright 2018 The MediaPipe Authors.
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

#include "mediapipe/calculators/tensorflow/tensor_squeeze_dimensions_calculator.pb.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/port/ret_check.h"
#include "mediapipe/framework/port/status.h"
#include "tensorflow/core/framework/tensor.h"

namespace mediapipe {

namespace tf = ::tensorflow;

// Given an input Tensor (example dimensions [1, 1024, 1, 5]), it squeezes all
// dimensions with size 1, or dimensions at specific indices, producing a tensor
// containing identical data (example output dimensions [1024, 5]).
class TensorSqueezeDimensionsCalculator : public CalculatorBase {
 public:
  static absl::Status GetContract(CalculatorContract* cc) {
    RET_CHECK_EQ(cc->Inputs().NumEntries(), 1) << "Need one input";
    cc->Inputs().Index(0).Set<tf::Tensor>(
        // Input Tensor
    );
    RET_CHECK_EQ(cc->Outputs().NumEntries(), 1) << "Need one output";
    cc->Outputs().Index(0).Set<tf::Tensor>(
        // Output Tensor Reduced Dimensions
    );
    return absl::OkStatus();
  }

  absl::Status Open(CalculatorContext* cc) override {
    options_ = cc->Options<TensorSqueezeDimensionsCalculatorOptions>();
    RET_CHECK(options_.squeeze_all_single_dims() ^ (options_.dim_size() > 0))
        << "Must specify dimensions to remove, or set squeeze_all_single_dims, "
           "but not both. Received options: "
        << options_.DebugString();
    if (options_.dim_size() > 0) {
      remove_dims_ =
          std::vector<int32>(options_.dim().begin(), options_.dim().end());
      std::sort(remove_dims_.rbegin(), remove_dims_.rend());
      remove_dims_initialized_ = true;
    }
    cc->SetOffset(0);
    return absl::OkStatus();
  }

  absl::Status Process(CalculatorContext* cc) override {
    const tf::Tensor& input_tensor = cc->Inputs().Index(0).Get<tf::Tensor>();
    tf::TensorShape tensor_shape = input_tensor.shape();
    if (!remove_dims_initialized_) {
      // Happens iff options.squeeze_all_single_dims is set.
      // Initialize remove_dims_ to all dimensions with size 1.
      InitializeToRemoveAllSingletonDimensions(tensor_shape);
      remove_dims_initialized_ = true;
    }
    for (const int dim : remove_dims_) {
      RET_CHECK_GT(tensor_shape.dims(), dim)
          << "Dimension " << dim
          << " does not exist in input tensor with num dimensions "
          << input_tensor.dims();
      RET_CHECK_EQ(tensor_shape.dim_size(dim), 1)
          << "Cannot remove dimension " << dim << " with size "
          << tensor_shape.dim_size(dim);
      tensor_shape.RemoveDim(dim);
    }

    std::unique_ptr<tf::Tensor> output_tensor(new tf::Tensor);
    RET_CHECK(output_tensor->CopyFrom(input_tensor, tensor_shape));
    cc->Outputs().Index(0).Add(output_tensor.release(), cc->InputTimestamp());
    return absl::OkStatus();
  }

  absl::Status Close(CalculatorContext* cc) override {
    return absl::OkStatus();
  }

 private:
  TensorSqueezeDimensionsCalculatorOptions options_;
  std::vector<int32> remove_dims_;
  bool remove_dims_initialized_;

  void InitializeToRemoveAllSingletonDimensions(
      const tf::TensorShape& tensor_shape) {
    const int dims = tensor_shape.dims();
    for (int i = dims - 1; i >= 0; --i) {
      if (tensor_shape.dim_size(i) == 1) {
        remove_dims_.push_back(i);
      }
    }
    if (remove_dims_.empty()) {
      LOG(ERROR) << "TensorSqueezeDimensionsCalculator is squeezing input with "
                    "no single-dimensions. Calculator will be a no-op.";
      LOG(ERROR) << "Input to TensorSqueezeDimensionsCalculator has shape "
                 << tensor_shape.DebugString();
    }
  }
};
REGISTER_CALCULATOR(TensorSqueezeDimensionsCalculator);

}  // namespace mediapipe
