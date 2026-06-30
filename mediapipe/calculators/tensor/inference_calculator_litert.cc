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
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/time/time.h"
#include "absl/types/optional.h"
#include "mediapipe/app/aimatter/cachable_object.h"
#include "mediapipe/app/aimatter/cache_service.h"
#include "mediapipe/calculators/tensor/inference_calculator.h"
#include "mediapipe/calculators/tensor/inference_runner.h"
#include "mediapipe/calculators/tensor/inference_runner_litert.h"
#include "mediapipe/calculators/tensor/litert/litert_service.h"
#include "mediapipe/calculators/tensor/shared_inference_service.h"
#include "mediapipe/calculators/tensor/tensor_span.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/tensor.h"
#include "mediapipe/framework/memory_manager.h"
#include "mediapipe/framework/memory_manager_service.h"
#include "mediapipe/framework/port/ret_check.h"
#include "mediapipe/framework/port/status_macros.h"
#include "mediapipe/gpu/gl_calculator_helper.h"
#include "mediapipe/util/tflite/tflite_model_loader.h"

#if MEDIAPIPE_METAL_ENABLED
#include "mediapipe/gpu/MPPMetalHelper.h"
#endif  // MEDIAPIPE_METAL_ENABLED

namespace mediapipe {
namespace api2 {

namespace {
using ::mediapipe::aimatter::SaveIntoCache;
using ::mediapipe::aimatter::TryGetFromCacheOrCreate;
}  // namespace

// Inference calculator backend for LiteRt API.
// Note: The model has to be provided as a MP resource.
class InferenceCalculatorLiteRtImpl
    : public InferenceCalculatorNodeImpl<InferenceCalculatorLiteRt,
                                         InferenceCalculatorLiteRtImpl> {
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
  absl::StatusOr<TfLiteDelegatePtr> CreateDelegate(CalculatorContext* cc);

  std::unique_ptr<InferenceRunner> inference_runner_;
  // Enable pooling of AHWBs in Tensor instances.
  MemoryManager* memory_manager_ = nullptr;
  mediapipe::GlCalculatorHelper gpu_helper_;
#if MEDIAPIPE_METAL_ENABLED
  MPPMetalHelper* metal_helper_ = nil;
#endif  // MEDIAPIPE_METAL_ENABLED
};

bool UseGpu(const mediapipe::InferenceCalculatorOptions& options) {
  return options.delegate().litert().has_gpu();
}

absl::Status InferenceCalculatorLiteRtImpl::UpdateContract(
    CalculatorContract* cc) {
  MP_RETURN_IF_ERROR(TensorContractCheck(cc));

  const auto& options = cc->Options<mediapipe::InferenceCalculatorOptions>();
  RET_CHECK(!options.model_path().empty() ^ kSideInModel(cc).IsConnected())
      << "Either model as side packet or model path in options is required.";

  RET_CHECK_OK(TfLiteModelLoader::EnableXenoAssetRegistry());
  cc->UseService(mediapipe::aimatter::kCacheService).Optional();
  cc->UseService(kSharedInferenceService).Optional();
  cc->UseService(kMemoryManagerService).Optional();
  cc->UseService(kLiteRtService).Optional();
  if (UseGpu(options)) {
    MP_RETURN_IF_ERROR(mediapipe::GlCalculatorHelper::UpdateContract(cc));
  }
  return absl::OkStatus();
}

absl::Status InferenceCalculatorLiteRtImpl::Open(CalculatorContext* cc) {
  if (cc->Service(kMemoryManagerService).IsAvailable()) {
    memory_manager_ = &cc->Service(kMemoryManagerService).GetObject();
  }

  if (UseGpu(cc->Options<mediapipe::InferenceCalculatorOptions>())) {
    MP_RETURN_IF_ERROR(gpu_helper_.Open(cc));
  }
  if (IsCachingAvailable(cc)) {
    MP_ASSIGN_OR_RETURN(
        inference_runner_,
        TryGetFromCacheOrCreate<InferenceRunner>(
            cc, [&]() { return CreateInferenceRunner(cc); }, GetCacheKey(cc),
            absl::ZeroDuration(),
            /*calling_from_open_and_will_retry_in_process=*/true));
    if (inference_runner_) {
      return InferenceCalculatorNodeImpl::UpdateIoMapping(
          cc, inference_runner_->GetInputOutputTensorNames());
    }
    return absl::OkStatus();
  }

  // See if sharing is available, and use if so.
  if (IsSharedInferenceAvailable(cc)) {
    MP_ASSIGN_OR_RETURN(
        inference_runner_,
        GetSharedOrCreateInferenceRunner(cc, [&](CalculatorContext* cc) {
          return CreateInferenceRunner(cc);
        }));
    return InferenceCalculatorNodeImpl::UpdateIoMapping(
        cc, inference_runner_->GetInputOutputTensorNames());
  }

  MP_ASSIGN_OR_RETURN(inference_runner_, CreateInferenceRunner(cc));
  return InferenceCalculatorNodeImpl::UpdateIoMapping(
      cc, inference_runner_->GetInputOutputTensorNames());
}

absl::Status InferenceCalculatorLiteRtImpl::
    MaybeGetInferenceRunnerFromCacheAndUpdateIoMapping(CalculatorContext* cc) {
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
    MP_RETURN_IF_ERROR(InferenceCalculatorNodeImpl::UpdateIoMapping(
        cc, inference_runner_->GetInputOutputTensorNames()));
  }
  return absl::OkStatus();
}

absl::StatusOr<std::vector<Tensor>> InferenceCalculatorLiteRtImpl::Process(
    CalculatorContext* cc, const TensorSpan& tensor_span) {
  std::vector<Tensor> output_tensors;
  if (UseGpu(cc->Options<mediapipe::InferenceCalculatorOptions>())) {
    MP_RETURN_IF_ERROR(gpu_helper_.RunInGlContext([&]() -> absl::Status {
      MP_ASSIGN_OR_RETURN(output_tensors,
                          inference_runner_->Run(cc, tensor_span));
      return absl::OkStatus();
    }));
  } else {
    MP_ASSIGN_OR_RETURN(output_tensors,
                        inference_runner_->Run(cc, tensor_span));
  }

  return output_tensors;
}

absl::Status InferenceCalculatorLiteRtImpl::Close(CalculatorContext* cc) {
  if (IsCachingAvailable(cc)) {
    MP_RETURN_IF_ERROR(
        SaveIntoCache(cc, GetCacheKey(cc), std::move(inference_runner_)));
  }
  inference_runner_ = nullptr;
  return absl::OkStatus();
}

absl::StatusOr<std::unique_ptr<InferenceRunner>>
InferenceCalculatorLiteRtImpl::CreateInferenceRunner(CalculatorContext* cc) {
  const auto& options = cc->Options<mediapipe::InferenceCalculatorOptions>();
  MP_ASSIGN_OR_RETURN(auto model_packet, GetModelAsPacket(cc));
  auto litert = options.delegate().litert();

  // If dispatch library path is not specified, try to get it from the service.
  if (litert.has_npu() && litert.npu().dispatch_library_path().empty() &&
      cc->Service(kLiteRtService).IsAvailable()) {
    auto& litert_service = cc->Service(kLiteRtService).GetObject();
    litert.mutable_npu()->set_dispatch_library_path(
        litert_service.GetDispatchLibraryPath());
  }

#if MEDIAPIPE_METAL_ENABLED
  void* metal_helper = nullptr;
  if (UseGpu(options)) {
    metal_helper_ = [[MPPMetalHelper alloc] initWithCalculatorContext:cc];
    metal_helper = (__bridge void*)metal_helper_;
  }
#endif  // MEDIAPIPE_METAL_ENABLED

  return InferenceRunnerLiteRt::Create(
      std::move(model_packet), litert,
      options.has_input_output_config() ? &options.input_output_config()
                                        : nullptr,
      memory_manager_, UseGpu(options) ? &gpu_helper_.GetGlContext() : nullptr,
#if defined(__EMSCRIPTEN__)
      /*.webgpu_service=*/nullptr,
#endif  // __EMSCRIPTEN__
#if MEDIAPIPE_METAL_ENABLED
      metal_helper,
#endif  // MEDIAPIPE_METAL_ENABLED
      /*.litert_options=*/std::nullopt);
}

}  // namespace api2
}  // namespace mediapipe
