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

#include <algorithm>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/time/time.h"
#include "mediapipe/calculators/tensor/inference_calculator.h"
#include "mediapipe/calculators/tensor/inference_calculator_utils.h"
#include "mediapipe/calculators/tensor/inference_interpreter_delegate_runner.h"
#include "mediapipe/calculators/tensor/inference_runner.h"
#include "mediapipe/calculators/tensor/tensor_span.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/tensor.h"
#include "mediapipe/framework/port/ret_check.h"
#include "mediapipe/framework/port/status_macros.h"
#if defined(MEDIAPIPE_ANDROID)
#include "tensorflow/lite/delegates/nnapi/nnapi_delegate.h"
#endif  // ANDROID
#include "tensorflow/lite/delegates/xnnpack/xnnpack_delegate.h"

namespace mediapipe {
namespace api2 {

class InferenceCalculatorCpuImpl
    : public InferenceCalculatorNodeImpl<InferenceCalculatorCpu,
                                         InferenceCalculatorCpuImpl> {
 public:
  static absl::Status UpdateContract(CalculatorContract* cc);

  absl::Status Open(CalculatorContext* cc) override;
  absl::Status Close(CalculatorContext* cc) override;

 private:
  absl::StatusOr<std::unique_ptr<InferenceRunner>> CreateInferenceRunner(
      CalculatorContext* cc);
  absl::StatusOr<TfLiteDelegatePtr> MaybeCreateDelegate(CalculatorContext* cc);
  absl::StatusOr<std::vector<Tensor>> Process(
      CalculatorContext* cc, const TensorSpan& tensor_span) override;
  std::unique_ptr<InferenceRunner> inference_runner_;
};

absl::Status InferenceCalculatorCpuImpl::UpdateContract(
    CalculatorContract* cc) {
  const auto& options = cc->Options<mediapipe::InferenceCalculatorOptions>();
  RET_CHECK(!options.model_path().empty() ^ kSideInModel(cc).IsConnected())
      << "Either model as side packet or model path in options is required.";

  MP_RETURN_IF_ERROR(TensorContractCheck(cc));

  return absl::OkStatus();
}

absl::Status InferenceCalculatorCpuImpl::Open(CalculatorContext* cc) {
  MP_ASSIGN_OR_RETURN(inference_runner_, CreateInferenceRunner(cc));
  return InferenceCalculatorNodeImpl::UpdateIoMapping(
      cc, inference_runner_->GetInputOutputTensorNames());
}

absl::StatusOr<std::vector<Tensor>> InferenceCalculatorCpuImpl::Process(
    CalculatorContext* cc, const TensorSpan& tensor_span) {
  MP_ASSIGN_OR_RETURN(std::vector<Tensor> output_tensors,
                      inference_runner_->Run(cc, tensor_span));
  return output_tensors;
}

absl::Status InferenceCalculatorCpuImpl::Close(CalculatorContext* cc) {
  inference_runner_ = nullptr;
  return absl::OkStatus();
}

absl::StatusOr<std::unique_ptr<InferenceRunner>>
InferenceCalculatorCpuImpl::CreateInferenceRunner(CalculatorContext* cc) {
  MP_ASSIGN_OR_RETURN(auto model_packet, GetModelAsPacket(cc));
  MP_ASSIGN_OR_RETURN(auto op_resolver_packet, GetOpResolverAsPacket(cc));
  const auto& options = cc->Options<mediapipe::InferenceCalculatorOptions>();
  const int interpreter_num_threads =
      cc->Options<mediapipe::InferenceCalculatorOptions>().cpu_num_thread();
  MP_ASSIGN_OR_RETURN(TfLiteDelegatePtr delegate, MaybeCreateDelegate(cc));
  return CreateInferenceInterpreterDelegateRunner(
      std::move(model_packet), std::move(op_resolver_packet),
      std::move(delegate), interpreter_num_threads,
      &options.input_output_config());
}

absl::StatusOr<TfLiteDelegatePtr>
InferenceCalculatorCpuImpl::MaybeCreateDelegate(CalculatorContext* cc) {
  const auto& calculator_opts =
      cc->Options<mediapipe::InferenceCalculatorOptions>();
  auto opts_delegate = calculator_opts.delegate();
  if (!kDelegate(cc).IsEmpty()) {
    const mediapipe::InferenceCalculatorOptions::Delegate&
        input_side_packet_delegate = kDelegate(cc).Get();
    RET_CHECK(
        input_side_packet_delegate.has_tflite() ||
        input_side_packet_delegate.has_xnnpack() ||
        input_side_packet_delegate.has_nnapi() ||
        input_side_packet_delegate.delegate_case() ==
            mediapipe::InferenceCalculatorOptions::Delegate::DELEGATE_NOT_SET)
        << "inference_calculator_cpu only supports delegate input side packet "
        << "for TFLite, XNNPack and Nnapi";
    opts_delegate.MergeFrom(input_side_packet_delegate);
  }
  const bool opts_has_delegate =
      calculator_opts.has_delegate() || !kDelegate(cc).IsEmpty();
  if (opts_has_delegate && opts_delegate.has_tflite()) {
    // Default tflite inference requested - no need to modify graph.
    return nullptr;
  }

#if defined(MEDIAPIPE_ANDROID)
  const bool nnapi_requested = opts_has_delegate ? opts_delegate.has_nnapi()
                                                 : calculator_opts.use_nnapi();
  if (nnapi_requested) {
    // Attempt to use NNAPI.
    // If not supported, the default CPU delegate will be created and used.
    tflite::StatefulNnApiDelegate::Options options;
    const auto& nnapi = opts_delegate.nnapi();
    options.allow_fp16 = true;
    // Set up cache_dir and model_token for NNAPI compilation cache.
    options.cache_dir =
        nnapi.has_cache_dir() ? nnapi.cache_dir().c_str() : nullptr;
    options.model_token =
        nnapi.has_model_token() ? nnapi.model_token().c_str() : nullptr;
    options.accelerator_name = nnapi.has_accelerator_name()
                                   ? nnapi.accelerator_name().c_str()
                                   : nullptr;
    return TfLiteDelegatePtr(new tflite::StatefulNnApiDelegate(options),
                             [](TfLiteDelegate*) {});
  }
#endif  // MEDIAPIPE_ANDROID

#if defined(__EMSCRIPTEN__) || MEDIAPIPE_FORCE_CPU_INFERENCE
  const bool use_xnnpack = true;
#else
  const bool use_xnnpack = opts_has_delegate && opts_delegate.has_xnnpack();
#endif  // defined(__EMSCRIPTEN__)

  if (use_xnnpack) {
    auto xnnpack_opts = TfLiteXNNPackDelegateOptionsDefault();
    xnnpack_opts.num_threads =
        GetXnnpackNumThreads(opts_has_delegate, opts_delegate);
    return TfLiteDelegatePtr(TfLiteXNNPackDelegateCreate(&xnnpack_opts),
                             &TfLiteXNNPackDelegateDelete);
  }

  return nullptr;
}

}  // namespace api2
}  // namespace mediapipe
