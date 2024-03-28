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

#include <algorithm>
#include <memory>
#include <utility>

#include "absl/status/status.h"
#include "mediapipe/calculators/tensor/feedback_tensors_calculator.pb.h"
#include "mediapipe/framework/api2/contract.h"
#include "mediapipe/framework/api2/node.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/tensor.h"
#include "mediapipe/framework/memory_manager.h"
#include "mediapipe/framework/memory_manager_service.h"

namespace mediapipe {
namespace api2 {

namespace {
constexpr char kInputTensorsTag[] = "INPUT_TENSORS";
constexpr char kFeedbackTensorsTag[] = "FEEDBACK_TENSORS";
constexpr char kOutputTensorsTag[] = "TENSORS";

using Tensors = std::vector<Tensor>;
}  // namespace

// FeedbackTensorsCalculator groups the input and the feedback (typically
// recurrent neural network cell state output tensors from the previous run)
// tensor vectors as the input tensor vector for the next recurrent model cell
// inference. On the first step, the feedback tensor is filled with zeros to
// jumpstart the loop.
class FeedbackTensorsCalculator : public Node {
 public:
  static constexpr Input<Tensors> kFeedbackTensorsIn{kFeedbackTensorsTag};
  static constexpr Input<Tensors> kInputTensorsIn{kInputTensorsTag};
  static constexpr Output<Tensors> kTensorsOut{kOutputTensorsTag};

  MEDIAPIPE_NODE_CONTRACT(kFeedbackTensorsIn, kInputTensorsIn, kTensorsOut,
                          TimestampChange::Arbitrary());

  static absl::Status UpdateContract(CalculatorContract* cc) {
    cc->SetProcessTimestampBounds(true);
    cc->UseService(kMemoryManagerService).Optional();
    return absl::OkStatus();
  }

  absl::Status Open(CalculatorContext* cc) override {
    if (cc->Service(kMemoryManagerService).IsAvailable()) {
      memory_manager_ = &cc->Service(kMemoryManagerService).GetObject();
    }
    const auto& options =
        cc->Options<mediapipe::FeedbackTensorsCalculatorOptions>();

    const auto& shape_dims = options.feedback_tensor_shape().dims();
    feedback_tensor_shape_.dims.assign(shape_dims.begin(), shape_dims.end());
    feedback_tensor_size_ = feedback_tensor_shape_.num_elements();

    num_feedback_tensors_ = options.num_feedback_tensors();

    feedback_tensors_location_ = options.location();

    return absl::OkStatus();
  }

  absl::Status Process(CalculatorContext* cc) override {
    if (feedback_tensors_location_ ==
        mediapipe::FeedbackTensorsCalculatorOptions::NONE) {
      kTensorsOut(cc).Send(kInputTensorsIn(cc).packet().As<Tensors>());
      return absl::OkStatus();
    }

    std::vector<Tensor> outputs;
    switch (feedback_tensors_location_) {
      case mediapipe::FeedbackTensorsCalculatorOptions::PREPENDED:
        MP_RETURN_IF_ERROR(AddFeedbackTensors(cc, outputs));
        MP_RETURN_IF_ERROR(AddInputTensors(cc, outputs));
        break;
      case mediapipe::FeedbackTensorsCalculatorOptions::APPENDED:
        MP_RETURN_IF_ERROR(AddInputTensors(cc, outputs));
        MP_RETURN_IF_ERROR(AddFeedbackTensors(cc, outputs));
        break;
      default:
        return absl::InvalidArgumentError(
            "Unsupported feedback tensors location");
    }
    kTensorsOut(cc).Send(std::move(outputs));
    return absl::OkStatus();
  }

 private:
  absl::Status AddInputTensors(CalculatorContext* cc,
                               std::vector<Tensor>& outputs) {
    absl::StatusOr<std::unique_ptr<std::vector<Tensor>>> input_tensors =
        cc->Inputs()
            .Tag(kInputTensorsTag)
            .Value()
            .Consume<std::vector<Tensor>>();
    if (!input_tensors.ok()) {
      return absl::InternalError("The input tensors packet is not consumable");
    }
    RET_CHECK(*input_tensors);
    std::vector<Tensor>& inputs = **input_tensors;
    outputs.insert(outputs.end(), std::make_move_iterator(inputs.begin()),
                   std::make_move_iterator(inputs.end()));
    return absl::OkStatus();
  }

  absl::Status AddFeedbackTensors(CalculatorContext* cc,
                                  std::vector<Tensor>& outputs) {
    if (first_run_) {
      for (int index = 0; index < num_feedback_tensors_; ++index) {
        Tensor initial_feedback_tensor(Tensor::ElementType::kFloat32,
                                       feedback_tensor_shape_, memory_manager_);
        float* data = initial_feedback_tensor.GetCpuWriteView().buffer<float>();
        std::fill_n(data, feedback_tensor_size_, 0.0f);
        outputs.push_back(std::move(initial_feedback_tensor));
      }
      first_run_ = false;
      return absl::OkStatus();
    }

    if (num_feedback_tensors_ != kFeedbackTensorsIn(cc)->size()) {
      return absl::InvalidArgumentError(
          "The number of tensors fed back differs from the configuration");
    }
    absl::StatusOr<std::unique_ptr<std::vector<Tensor>>> feedback_tensors =
        cc->Inputs()
            .Tag(kFeedbackTensorsTag)
            .Value()
            .Consume<std::vector<Tensor>>();
    if (!feedback_tensors.ok()) {
      return absl::InternalError(
          "The feedback tensors packet is not consumable");
    }
    RET_CHECK(*feedback_tensors);
    std::vector<Tensor>& feedbacks = **feedback_tensors;
    for (const auto& feedback : feedbacks) {
      if (feedback.shape().dims != feedback_tensor_shape_.dims) {
        return absl::InvalidArgumentError(
            "The shape of a tensor fed back differs from the configuration");
      }
    }
    outputs.insert(outputs.end(), std::make_move_iterator(feedbacks.begin()),
                   std::make_move_iterator(feedbacks.end()));

    return absl::OkStatus();
  }

  Tensor::Shape feedback_tensor_shape_;
  int num_feedback_tensors_ = 0;
  mediapipe::FeedbackTensorsCalculatorOptions::FeedbackTensorsLocation
      feedback_tensors_location_;

  int feedback_tensor_size_ = 0;
  bool first_run_ = true;

  // Enable pooling of AHWBs in Tensor instances.
  MemoryManager* memory_manager_ = nullptr;
};

MEDIAPIPE_REGISTER_NODE(FeedbackTensorsCalculator);

}  // namespace api2
}  // namespace mediapipe
