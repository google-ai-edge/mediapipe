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

#include <cstring>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "mediapipe/framework/api2/node.h"
#include "mediapipe/framework/api2/port.h"
#include "mediapipe/framework/calculator_context.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/tensor.h"
#include "mediapipe/framework/memory_manager.h"
#include "mediapipe/framework/memory_manager_service.h"

namespace mediapipe {
namespace api2 {

// Trivially converts an input string into a Tensor that stores a copy of
// the string.
//
// Inputs:
//   TEXT - std::string
//
// Outputs:
//   TENSORS - std::vector<Tensor>
//     Vector containing a single Tensor storing a copy of the input string.
//     Note that the underlying buffer of the Tensor is not necessarily
//     null-terminated. It is the graph writer's responsibility to copy the
//     correct number of characters when copying from this Tensor's buffer.
//
// Example:
//   node {
//     calculator: "TextToTensorCalculator"
//     input_stream: "TEXT:text"
//     output_stream: "TENSORS:tensors"
//   }
class TextToTensorCalculator : public Node {
 public:
  static constexpr Input<std::string> kTextIn{"TEXT"};
  static constexpr Output<std::vector<Tensor>> kTensorsOut{"TENSORS"};

  MEDIAPIPE_NODE_CONTRACT(kTextIn, kTensorsOut);

  absl::Status Open(CalculatorContext* cc) final;
  absl::Status Process(CalculatorContext* cc) override;

  static absl::Status UpdateContract(CalculatorContract* cc);

 private:
  // Enable pooling of AHWBs in Tensor instances.
  MemoryManager* memory_manager_ = nullptr;
};

absl::Status TextToTensorCalculator::Open(CalculatorContext* cc) {
  if (cc->Service(kMemoryManagerService).IsAvailable()) {
    memory_manager_ = &cc->Service(kMemoryManagerService).GetObject();
  }
  return absl::OkStatus();
}

absl::Status TextToTensorCalculator::Process(CalculatorContext* cc) {
  absl::string_view text = kTextIn(cc).Get();
  int input_len = static_cast<int>(text.length());

  std::vector<Tensor> result;
  result.push_back({Tensor::ElementType::kChar, Tensor::Shape({input_len}),
                    memory_manager_});
  std::memcpy(result[0].GetCpuWriteView().buffer<char>(), text.data(),
              input_len * sizeof(char));
  kTensorsOut(cc).Send(std::move(result));
  return absl::OkStatus();
}

// static
absl::Status TextToTensorCalculator::UpdateContract(CalculatorContract* cc) {
  cc->UseService(kMemoryManagerService).Optional();
  return absl::OkStatus();
}

MEDIAPIPE_REGISTER_NODE(TextToTensorCalculator);

}  // namespace api2
}  // namespace mediapipe
