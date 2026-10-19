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
#include <optional>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/time/time.h"
#include "mediapipe/calculators/tensor/inference_calculator.h"
#include "mediapipe/calculators/tensor/inference_runner.h"
#include "mediapipe/calculators/tensor/inference_runner_litert.h"
#include "mediapipe/calculators/tensor/litert/litert_service.h"
#include "mediapipe/calculators/tensor/tensor_span.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/tensor.h"
#include "mediapipe/framework/memory_manager.h"
#include "mediapipe/framework/memory_manager_service.h"
#include "mediapipe/framework/port.h"
#include "mediapipe/framework/port/ret_check.h"
#include "mediapipe/framework/port/status_macros.h"
#include "mediapipe/util/tflite/tflite_model_loader.h"

#if !MEDIAPIPE_DISABLE_GPU
#include "mediapipe/gpu/gl_calculator_helper.h"
#endif  // !MEDIAPIPE_DISABLE_GPU

#if MEDIAPIPE_METAL_ENABLED
#include "mediapipe/gpu/MPPMetalHelper.h"
#endif  // MEDIAPIPE_METAL_ENABLED

namespace mediapipe {
namespace api2 {

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
  absl::StatusOr<std::vector<Tensor>> Process(
      CalculatorContext* cc, const TensorSpan& tensor_span) override;
  absl::StatusOr<std::unique_ptr<InferenceRunner>> CreateInferenceRunner(
      CalculatorContext* cc);
  absl::StatusOr<TfLiteDelegatePtr> CreateDelegate(CalculatorContext* cc);

#if !MEDIAPIPE_DISABLE_GPU
  mediapipe::GlCalculatorHelper gpu_helper_;
#endif  // !MEDIAPIPE_DISABLE_GPU
  std::unique_ptr<InferenceRunner> inference_runner_;
  // Enable pooling of AHWBs in Tensor instances.
  MemoryManager* memory_manager_ = nullptr;
};

bool UseGpu(const mediapipe::InferenceCalculatorOptions& options) {
  return options.delegate().litert().has_gpu();
}

absl::Status InferenceCalculatorLiteRtImpl::UpdateContract(
    CalculatorContract* cc) {
  ABSL_RETURN_IF_ERROR(TensorContractCheck(cc));

  const auto& options = cc->Options<mediapipe::InferenceCalculatorOptions>();
  RET_CHECK(!options.model_path().empty() ^ kSideInModel(cc).IsConnected())
      << "Either model as side packet or model path in options is required.";

  cc->UseService(kMemoryManagerService).Optional();
  cc->UseService(kLiteRtService).Optional();
  if (UseGpu(options)) {
#if MEDIAPIPE_DISABLE_GPU
    return absl::UnimplementedError(
        "InferenceCalculatorLiteRt was built without GPU support. Request the "
        "CPU accelerator instead.");
#else
    ABSL_RETURN_IF_ERROR(mediapipe::GlCalculatorHelper::UpdateContract(cc));
#endif  // MEDIAPIPE_DISABLE_GPU
  }
  return absl::OkStatus();
}

absl::Status InferenceCalculatorLiteRtImpl::Open(CalculatorContext* cc) {
  if (cc->Service(kMemoryManagerService).IsAvailable()) {
    memory_manager_ = &cc->Service(kMemoryManagerService).GetObject();
  }

#if !MEDIAPIPE_DISABLE_GPU
  if (UseGpu(cc->Options<mediapipe::InferenceCalculatorOptions>())) {
    ABSL_RETURN_IF_ERROR(gpu_helper_.Open(cc));
  }
#endif  // !MEDIAPIPE_DISABLE_GPU

  ABSL_ASSIGN_OR_RETURN(inference_runner_, CreateInferenceRunner(cc));
  return InferenceCalculatorNodeImpl::UpdateIoMapping(
      cc, inference_runner_->GetInputOutputTensorNames());
}

absl::StatusOr<std::vector<Tensor>> InferenceCalculatorLiteRtImpl::Process(
    CalculatorContext* cc, const TensorSpan& tensor_span) {
  std::vector<Tensor> output_tensors;
#if !MEDIAPIPE_DISABLE_GPU
  if (UseGpu(cc->Options<mediapipe::InferenceCalculatorOptions>())) {
    ABSL_RETURN_IF_ERROR(gpu_helper_.RunInGlContext([&]() -> absl::Status {
      ABSL_ASSIGN_OR_RETURN(output_tensors,
                            inference_runner_->Run(cc, tensor_span));
      return absl::OkStatus();
    }));
    return output_tensors;
  }
#endif  // !MEDIAPIPE_DISABLE_GPU
  ABSL_ASSIGN_OR_RETURN(output_tensors,
                        inference_runner_->Run(cc, tensor_span));
  return output_tensors;
}

absl::Status InferenceCalculatorLiteRtImpl::Close(CalculatorContext* cc) {
  inference_runner_ = nullptr;
  return absl::OkStatus();
}

absl::StatusOr<std::unique_ptr<InferenceRunner>>
InferenceCalculatorLiteRtImpl::CreateInferenceRunner(CalculatorContext* cc) {
  const auto& options = cc->Options<mediapipe::InferenceCalculatorOptions>();
  ABSL_ASSIGN_OR_RETURN(auto model_packet, GetModelAsPacket(cc));
  auto litert = options.delegate().litert();

  LiteRtSystemHandles system_handles;
  if (cc->Service(kLiteRtService).IsAvailable()) {
    const auto& litert_service = cc->Service(kLiteRtService).GetObject();
    system_handles = litert_service.GetSystemHandles();
    // If dispatch library path is not specified, try to get it from the
    // service.
    if (litert.has_npu() && litert.npu().dispatch_library_path().empty()) {
      litert.mutable_npu()->set_dispatch_library_path(
          litert_service.GetDispatchLibraryPath());
    }
  }
#if MEDIAPIPE_METAL_ENABLED
  void* metal_helper = nullptr;
  MPPMetalHelper* helper = nil;
  if (UseGpu(options)) {
    helper = [[MPPMetalHelper alloc] initWithCalculatorContext:cc];
    metal_helper = (__bridge void*)helper;
  }
#endif  // MEDIAPIPE_METAL_ENABLED

  auto get_glcontext_fn = [&]() -> std::shared_ptr<mediapipe::GlContext> {
#if MEDIAPIPE_DISABLE_GPU
    return nullptr;
#else
    return UseGpu(options) ? gpu_helper_.GetSharedGlContext() : nullptr;
#endif  // MEDIAPIPE_DISABLE_GPU
  };

  return InferenceRunnerLiteRt::Create(
      std::move(model_packet), litert,
      options.has_input_output_config() ? &options.input_output_config()
                                        : nullptr,
      memory_manager_, get_glcontext_fn(),
#if defined(__EMSCRIPTEN__)
      /*.webgpu_service=*/nullptr,
#endif  // __EMSCRIPTEN__
#if MEDIAPIPE_METAL_ENABLED
      metal_helper,
#endif  // MEDIAPIPE_METAL_ENABLED
      /*.litert_options=*/std::nullopt, system_handles);
}

}  // namespace api2
}  // namespace mediapipe
