// Copyright 2022 The MediaPipe Authors.
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

#include <memory>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "mediapipe/framework/api2/node.h"
#include "mediapipe/framework/api2/port.h"
#include "mediapipe/framework/calculator_context.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/tensor.h"
#include "mediapipe/framework/memory_manager.h"
#include "mediapipe/framework/memory_manager_service.h"
#include "mediapipe/framework/port/ret_check.h"

namespace mediapipe {
namespace api2 {
namespace {

template <typename T>
void Dequantize(const Tensor& input, Tensor* output) {
  auto input_view = input.GetCpuReadView();
  auto input_buffer = input_view.buffer<T>();
  auto output_view = output->GetCpuWriteView();
  auto output_buffer = output_view.buffer<float>();
  for (int i = 0; i < input.shape().num_elements(); ++i) {
    output_buffer[i] = input.quantization_parameters().scale *
                       (static_cast<int>(input_buffer[i]) -
                        input.quantization_parameters().zero_point);
  }
}

}  // namespace

// Performs dequantization using the quantization parameters from the input
// UInt8 or Int8 tensors. Each element of the input tensors is converted using:
//
//   output = quantization_parameters.scale *
//     (input - quantization_parameters.zero_point)
//
// Input:
//  TENSORS - Vector of quantized Tensors of type kUint8 or kInt8.
// Output:
//  TENSORS - Vector of dequantized Tensors of type kFloat32.
//
// Usage example:
// node {
//   calculator: "TensorsDequantizationCalculator"
//   input_stream: "TENSORS:quantized_tensors"
//   output_stream: "TENSORS:dequantized_tensors"
// }
class TensorsDequantizationCalculator : public Node {
 public:
  static constexpr Input<std::vector<Tensor>> kInTensors{"TENSORS"};
  static constexpr Output<std::vector<Tensor>> kOutTensors{"TENSORS"};
  MEDIAPIPE_NODE_CONTRACT(kInTensors, kOutTensors);

  absl::Status Open(CalculatorContext* cc) override;
  absl::Status Process(CalculatorContext* cc) override;

  static absl::Status UpdateContract(CalculatorContract* cc);

 private:
  // Enable pooling of AHWBs in Tensor instances.
  MemoryManager* memory_manager_ = nullptr;
};

absl::Status TensorsDequantizationCalculator::Open(CalculatorContext* cc) {
  if (cc->Service(kMemoryManagerService).IsAvailable()) {
    memory_manager_ = &cc->Service(kMemoryManagerService).GetObject();
  }
  return absl::OkStatus();
}

absl::Status TensorsDequantizationCalculator::Process(CalculatorContext* cc) {
  if (kInTensors(cc).IsEmpty()) {
    return absl::OkStatus();
  }
  const auto& input_tensors = *kInTensors(cc);
  RET_CHECK(!input_tensors.empty());
  auto output_tensors = std::make_unique<std::vector<Tensor>>();
  output_tensors->reserve(input_tensors.size());
  for (const auto& input_tensor : input_tensors) {
    output_tensors->emplace_back(Tensor::ElementType::kFloat32,
                                 input_tensor.shape(), memory_manager_);
    switch (input_tensor.element_type()) {
      case Tensor::ElementType::kUInt8:
        Dequantize<uint8_t>(input_tensor, &output_tensors->back());
        break;
      case Tensor::ElementType::kInt8:
        Dequantize<int8_t>(input_tensor, &output_tensors->back());
        break;
      case Tensor::ElementType::kBool:
        Dequantize<bool>(input_tensor, &output_tensors->back());
        break;
      default:
        return absl::InvalidArgumentError(absl::StrCat(
            "Unsupported input tensor type: ", input_tensor.element_type()));
    }
  }
  kOutTensors(cc).Send(std::move(output_tensors));
  return absl::OkStatus();
}

// static
absl::Status TensorsDequantizationCalculator::UpdateContract(
    CalculatorContract* cc) {
  cc->UseService(kMemoryManagerService).Optional();
  return absl::OkStatus();
}

MEDIAPIPE_REGISTER_NODE(TensorsDequantizationCalculator);

}  // namespace api2
}  // namespace mediapipe
