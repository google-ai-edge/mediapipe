/* Copyright 2023 The MediaPipe Authors.

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

#include <memory>
#include <utility>
#include <vector>

#include "absl/log/absl_check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "mediapipe/framework/api2/node.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/tensor.h"

namespace mediapipe {
namespace api2 {

// In iteration mode, output the image guidance tensors at the current timestamp
// and advance the output stream timestamp bound by the number of steps.
// Otherwise, output the image guidance tensors at the current timestamp only.
class DiffusionPluginsOutputCalculator : public Node {
 public:
  static constexpr Input<std::vector<Tensor>> kTensorsIn{"TENSORS"};
  static constexpr Input<int> kStepsIn{"STEPS"};
  static constexpr Input<int>::Optional kIterationIn{"ITERATION"};
  static constexpr Output<std::vector<Tensor>> kTensorsOut{"TENSORS"};
  MEDIAPIPE_NODE_CONTRACT(kTensorsIn, kStepsIn, kIterationIn, kTensorsOut);

  absl::Status Process(CalculatorContext* cc) override {
    if (kTensorsIn(cc).IsEmpty()) {
      return absl::OkStatus();
    }
    // Consumes the tensor vector to avoid data copy.
    absl::StatusOr<std::unique_ptr<std::vector<Tensor>>> status_or_tensor =
        cc->Inputs().Tag("TENSORS").Value().Consume<std::vector<Tensor>>();
    if (!status_or_tensor.ok()) {
      return absl::InternalError("Input tensor vector is not consumable.");
    }
    if (kIterationIn(cc).IsConnected()) {
      ABSL_CHECK_EQ(kIterationIn(cc).Get(), 0);
      kTensorsOut(cc).Send(std::move(*status_or_tensor.value()));
      kTensorsOut(cc).SetNextTimestampBound(cc->InputTimestamp() +
                                            kStepsIn(cc).Get());
    } else {
      kTensorsOut(cc).Send(std::move(*status_or_tensor.value()));
    }
    return absl::OkStatus();
  }
};

MEDIAPIPE_REGISTER_NODE(DiffusionPluginsOutputCalculator);

}  // namespace api2
}  // namespace mediapipe
