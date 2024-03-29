/* Copyright 2024 The MediaPipe Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include <algorithm>
#include <cstdint>
#include <utility>
#include <vector>

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "mediapipe/calculators/tensor/vector_to_tensor_calculator.pb.h"
#include "mediapipe/framework/api2/node.h"
#include "mediapipe/framework/api2/packet.h"
#include "mediapipe/framework/api2/port.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/tensor.h"
#include "mediapipe/framework/memory_manager.h"
#include "mediapipe/framework/memory_manager_service.h"
#include "mediapipe/framework/port/ret_check.h"
#include "mediapipe/framework/port/status_macros.h"

namespace mediapipe {
namespace {

using ::mediapipe::CalculatorContext;
using ::mediapipe::Tensor;
using ::mediapipe::api2::Input;
using ::mediapipe::api2::Node;
using ::mediapipe::api2::OneOf;
using ::mediapipe::api2::Output;

}  // namespace

// Copies a vector of type (float, uint8_t, int8_t, int32_t, char, bool) into
// the CPU memory of a Tensor.
// Note that an additional copy can occur when a GPU view is requested from the
// output Tensor. For top performance, calculators should use platform-specific
// buffers which can be wrapped by Tensors.
class VectorToTensorCalculator : public Node {
 public:
  using SupportedInputVectors =
      OneOf<std::vector<float>, std::vector<uint8_t>, std::vector<int8_t>,
            std::vector<int32_t>, std::vector<char>, std::vector<bool>>;
  static constexpr Input<SupportedInputVectors> kVectorIn{"VECTOR"};
  static constexpr Output<Tensor> kOutTensor{"TENSOR"};
  MEDIAPIPE_NODE_CONTRACT(kVectorIn, kOutTensor);

  absl::Status Open(CalculatorContext* cc) override;
  absl::Status Process(CalculatorContext* cc) override;

  static absl::Status UpdateContract(CalculatorContract* cc);

 private:
  absl::StatusOr<Tensor> ConvertVectorToTensor(
      const api2::Packet<SupportedInputVectors>& input,
      bool output_dynamic_tensor_shape);

  template <typename VectorT, Tensor::ElementType TensorT>
  absl::StatusOr<Tensor> CopyVectorToNewTensor(
      const std::vector<VectorT>& input, bool output_dynamic_tensor_shape);

  // Enable pooling of AHWBs in Tensor instances.
  MemoryManager* memory_manager_ = nullptr;
};

absl::Status VectorToTensorCalculator::Open(CalculatorContext* cc) {
  if (cc->Service(kMemoryManagerService).IsAvailable()) {
    memory_manager_ = &cc->Service(kMemoryManagerService).GetObject();
  }
  return absl::OkStatus();
}

template <typename VectorT, Tensor::ElementType TensorT>
absl::StatusOr<Tensor> VectorToTensorCalculator::CopyVectorToNewTensor(
    const std::vector<VectorT>& input, bool output_dynamic_tensor_shape) {
  RET_CHECK_GT(input.size(), 0) << "Input vector is empty";
  std::vector<int> dimensions = {1, static_cast<int>(input.size())};
  Tensor tensor(TensorT, Tensor::Shape(dimensions, output_dynamic_tensor_shape),
                memory_manager_);
  const auto cpu_write_view = tensor.GetCpuWriteView();
  std::copy(input.begin(), input.end(), cpu_write_view.buffer<VectorT>());
  return tensor;
}

absl::StatusOr<Tensor> VectorToTensorCalculator::ConvertVectorToTensor(
    const api2::Packet<SupportedInputVectors>& input,
    bool output_dynamic_tensor_shape) {
  if (input.Has<std::vector<float>>()) {
    return CopyVectorToNewTensor<float, Tensor::ElementType::kFloat32>(
        input.Get<std::vector<float>>(), output_dynamic_tensor_shape);
  }
  if (input.Has<std::vector<uint8_t>>()) {
    return CopyVectorToNewTensor<uint8_t, Tensor::ElementType::kUInt8>(
        input.Get<std::vector<uint8_t>>(), output_dynamic_tensor_shape);
  }
  if (input.Has<std::vector<int8_t>>()) {
    return CopyVectorToNewTensor<int8_t, Tensor::ElementType::kInt8>(
        input.Get<std::vector<int8_t>>(), output_dynamic_tensor_shape);
  }
  if (input.Has<std::vector<int32_t>>()) {
    return CopyVectorToNewTensor<int32_t, Tensor::ElementType::kInt32>(
        input.Get<std::vector<int32_t>>(), output_dynamic_tensor_shape);
  }
  if (input.Has<std::vector<char>>()) {
    return CopyVectorToNewTensor<char, Tensor::ElementType::kChar>(
        input.Get<std::vector<char>>(), output_dynamic_tensor_shape);
  }
  if (input.Has<std::vector<bool>>()) {
    return CopyVectorToNewTensor<bool, Tensor::ElementType::kBool>(
        input.Get<std::vector<bool>>(), output_dynamic_tensor_shape);
  }
  return absl::InvalidArgumentError("Unsupported type");
}

absl::Status VectorToTensorCalculator::Process(CalculatorContext* cc) {
  const VectorToTensorCalculatorOptions& options =
      cc->Options<VectorToTensorCalculatorOptions>();
  MP_ASSIGN_OR_RETURN(
      Tensor tensor, ConvertVectorToTensor(
                         kVectorIn(cc), options.output_dynamic_tensor_shape()));
  kOutTensor(cc).Send(std::move(tensor));
  return absl::OkStatus();
}

absl::Status VectorToTensorCalculator::UpdateContract(CalculatorContract* cc) {
  cc->UseService(kMemoryManagerService).Optional();
  return absl::OkStatus();
}

MEDIAPIPE_REGISTER_NODE(VectorToTensorCalculator);

}  // namespace mediapipe
