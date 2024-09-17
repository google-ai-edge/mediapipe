// Copyright 2024 The MediaPipe Authors.
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
#include <string>
#include <utility>
#include <vector>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/time/time.h"
#include "mediapipe/app/aimatter/cachable_object.h"
#include "mediapipe/app/aimatter/cache_service.h"
#include "mediapipe/calculators/tensor/inference_calculator.h"
#include "mediapipe/calculators/tensor/inference_runner.h"
#include "mediapipe/calculators/tensor/inference_runner_qnn.h"
#include "mediapipe/calculators/tensor/tensor_span.h"
#include "mediapipe/framework/api2/packet.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/tensor.h"
#include "mediapipe/framework/port/ret_check.h"
#include "mediapipe/framework/port/status_macros.h"
#include "mediapipe/util/tflite/tflite_model_loader.h"
#include "tensorflow/lite/core/interpreter.h"

namespace mediapipe {
namespace api2 {

namespace {

using ::tflite::Interpreter;

using ::mediapipe::aimatter::SaveIntoCache;
using ::mediapipe::aimatter::TryGetFromCacheOrCreate;

}  // namespace

// Inference calculator implementation that uses the Qualcomm's QNN Delegate.
// It only supports synchronous inference without support for buffer bindings
// (AHWBs, GPU, etc).
class InferenceCalculatorQnnImpl
    : public InferenceCalculatorNodeImpl<InferenceCalculatorQnn,
                                         InferenceCalculatorQnnImpl> {
 public:
  static absl::Status UpdateContract(CalculatorContract* cc);

  absl::Status Open(CalculatorContext* cc) override;
  absl::Status Close(CalculatorContext* cc) override;

 private:
  absl::Status MaybeGetInferenceRunnerFromCacheAndUpdateIoMapping(
      CalculatorContext* cc) override;
  absl::StatusOr<std::vector<Tensor>> Process(
      CalculatorContext* cc, const TensorSpan& tensor_span) override;
  absl::StatusOr<std::unique_ptr<InferenceRunner>> CreateInferenceRunner(
      CalculatorContext* cc);

  std::unique_ptr<InferenceRunner> inference_runner_;
};

absl::Status InferenceCalculatorQnnImpl::UpdateContract(
    CalculatorContract* cc) {
  MP_RETURN_IF_ERROR(TensorContractCheck(cc));
  const auto& options = cc->Options<InferenceCalculatorOptions>();
  RET_CHECK(!options.model_path().empty() ^ kSideInModel(cc).IsConnected())
      << "Either model as side packet or model path in options is required.";

  RET_CHECK_OK(TfLiteModelLoader::EnableXenoAssetRegistry());
  cc->UseService(mediapipe::aimatter::kCacheService).Optional();
  WarnFeedbackTensorsUnsupported(cc);
  return absl::OkStatus();
}

absl::Status InferenceCalculatorQnnImpl::Open(CalculatorContext* cc) {
  RET_CHECK(kDelegate(cc).IsEmpty()) << "kDelegate isn't supported yet.";
  if (IsCachingAvailable(cc)) {
    MP_ASSIGN_OR_RETURN(
        inference_runner_,
        TryGetFromCacheOrCreate<InferenceRunner>(
            cc, [&]() { return CreateInferenceRunner(cc); }, GetCacheKey(cc),
            absl::ZeroDuration(),
            /*calling_from_open_and_will_retry_in_process=*/true));
    if (inference_runner_) {
      MP_RETURN_IF_ERROR(InferenceCalculatorNodeImpl::UpdateIoMapping(
          cc, inference_runner_->GetInputOutputTensorNames()));
    }
    return absl::OkStatus();
  }

  MP_ASSIGN_OR_RETURN(inference_runner_, CreateInferenceRunner(cc));
  return InferenceCalculatorNodeImpl::UpdateIoMapping(
      cc, inference_runner_->GetInputOutputTensorNames());
}

absl::Status
InferenceCalculatorQnnImpl::MaybeGetInferenceRunnerFromCacheAndUpdateIoMapping(
    CalculatorContext* cc) {
  if (!inference_runner_) {
    // To avoid a deadlock and/or graph error state, this call creates a new
    // InferenceRunner in case it can't be retrieved from the cache within the
    // given duration.
    MP_ASSIGN_OR_RETURN(
        inference_runner_,
        TryGetFromCacheOrCreate<InferenceRunner>(
            cc, [&]() { return CreateInferenceRunner(cc); }, GetCacheKey(cc),
            absl::Seconds(1),
            /*calling_from_open_and_will_retry_in_process=*/false));
    if (inference_runner_) {
      MP_RETURN_IF_ERROR(InferenceCalculatorNodeImpl::UpdateIoMapping(
          cc, inference_runner_->GetInputOutputTensorNames()));
    }
  }
  return absl::OkStatus();
}

absl::StatusOr<std::vector<Tensor>> InferenceCalculatorQnnImpl::Process(
    CalculatorContext* cc, const TensorSpan& tensor_span) {
  std::vector<Tensor> output_tensors;
  MP_ASSIGN_OR_RETURN(output_tensors, inference_runner_->Run(cc, tensor_span));
  return output_tensors;
}

absl::Status InferenceCalculatorQnnImpl::Close(CalculatorContext* cc) {
  if (IsCachingAvailable(cc)) {
    MP_RETURN_IF_ERROR(
        SaveIntoCache(cc, GetCacheKey(cc), std::move(inference_runner_)));
  }
  inference_runner_ = nullptr;
  return absl::OkStatus();
}

absl::StatusOr<std::unique_ptr<InferenceRunner>>
InferenceCalculatorQnnImpl::CreateInferenceRunner(CalculatorContext* cc) {
  const auto& options = cc->Options<mediapipe::InferenceCalculatorOptions>();
  MP_ASSIGN_OR_RETURN(auto model_packet, GetModelAsPacket(cc));
  auto inference_runner = std::make_unique<InferenceRunnerQnn>();
  MP_RETURN_IF_ERROR(inference_runner->Init(options, model_packet));
  return inference_runner;
}

}  // namespace api2
}  // namespace mediapipe
