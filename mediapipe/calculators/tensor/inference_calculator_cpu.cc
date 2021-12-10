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

#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include "absl/memory/memory.h"
#include "mediapipe/calculators/tensor/inference_calculator.h"

#if defined(MEDIAPIPE_ANDROID)
#include "tensorflow/lite/delegates/nnapi/nnapi_delegate.h"
#endif  // ANDROID

#if !defined(__EMSCRIPTEN__) || defined(__EMSCRIPTEN_PTHREADS__)
#include "mediapipe/util/cpu_util.h"
#endif  // !__EMSCRIPTEN__ || __EMSCRIPTEN_PTHREADS__

#include "tensorflow/lite/delegates/xnnpack/xnnpack_delegate.h"

namespace mediapipe {
namespace api2 {

namespace {

int GetXnnpackDefaultNumThreads() {
#if defined(MEDIAPIPE_ANDROID) || defined(MEDIAPIPE_IOS) || \
    defined(__EMSCRIPTEN_PTHREADS__)
  constexpr int kMinNumThreadsByDefault = 1;
  constexpr int kMaxNumThreadsByDefault = 4;
  return std::clamp(NumCPUCores() / 2, kMinNumThreadsByDefault,
                    kMaxNumThreadsByDefault);
#else
  return 1;
#endif  // MEDIAPIPE_ANDROID || MEDIAPIPE_IOS || __EMSCRIPTEN_PTHREADS__
}

// Returns number of threads to configure XNNPACK delegate with.
// Returns user provided value if specified. Otherwise, tries to choose optimal
// number of threads depending on the device.
int GetXnnpackNumThreads(
    const bool opts_has_delegate,
    const mediapipe::InferenceCalculatorOptions::Delegate& opts_delegate) {
  static constexpr int kDefaultNumThreads = -1;
  if (opts_has_delegate && opts_delegate.has_xnnpack() &&
      opts_delegate.xnnpack().num_threads() != kDefaultNumThreads) {
    return opts_delegate.xnnpack().num_threads();
  }
  return GetXnnpackDefaultNumThreads();
}

}  // namespace

class InferenceCalculatorCpuImpl
    : public NodeImpl<InferenceCalculatorCpu, InferenceCalculatorCpuImpl> {
 public:
  static absl::Status UpdateContract(CalculatorContract* cc);

  absl::Status Open(CalculatorContext* cc) override;
  absl::Status Process(CalculatorContext* cc) override;
  absl::Status Close(CalculatorContext* cc) override;

 private:
  absl::Status LoadModel(CalculatorContext* cc);
  absl::Status LoadDelegate(CalculatorContext* cc);
  absl::Status LoadDelegateAndAllocateTensors(CalculatorContext* cc);

  // TfLite requires us to keep the model alive as long as the interpreter is.
  Packet<TfLiteModelPtr> model_packet_;
  std::unique_ptr<tflite::Interpreter> interpreter_;
  TfLiteDelegatePtr delegate_;
};

absl::Status InferenceCalculatorCpuImpl::UpdateContract(
    CalculatorContract* cc) {
  const auto& options = cc->Options<::mediapipe::InferenceCalculatorOptions>();
  RET_CHECK(!options.model_path().empty() ^ kSideInModel(cc).IsConnected())
      << "Either model as side packet or model path in options is required.";

  return absl::OkStatus();
}

absl::Status InferenceCalculatorCpuImpl::Open(CalculatorContext* cc) {
  MP_RETURN_IF_ERROR(LoadModel(cc));
  return LoadDelegateAndAllocateTensors(cc);
}

absl::Status InferenceCalculatorCpuImpl::Process(CalculatorContext* cc) {
  if (kInTensors(cc).IsEmpty()) {
    return absl::OkStatus();
  }
  const auto& input_tensors = *kInTensors(cc);
  RET_CHECK(!input_tensors.empty());
  auto output_tensors = absl::make_unique<std::vector<Tensor>>();

  // Read CPU input into tensors.
  for (int i = 0; i < input_tensors.size(); ++i) {
    const Tensor* input_tensor = &input_tensors[i];
    auto input_tensor_view = input_tensor->GetCpuReadView();
    auto input_tensor_buffer = input_tensor_view.buffer<float>();
    float* local_tensor_buffer = interpreter_->typed_input_tensor<float>(i);
    std::memcpy(local_tensor_buffer, input_tensor_buffer,
                input_tensor->bytes());
  }

  // Run inference.
  RET_CHECK_EQ(interpreter_->Invoke(), kTfLiteOk);

  // Output result tensors (CPU).
  const auto& tensor_indexes = interpreter_->outputs();
  output_tensors->reserve(tensor_indexes.size());
  for (int i = 0; i < tensor_indexes.size(); ++i) {
    TfLiteTensor* tensor = interpreter_->tensor(tensor_indexes[i]);
    output_tensors->emplace_back(
        Tensor::ElementType::kFloat32,
        Tensor::Shape{std::vector<int>{
            tensor->dims->data, tensor->dims->data + tensor->dims->size}});
    auto cpu_view = output_tensors->back().GetCpuWriteView();
    std::memcpy(cpu_view.buffer<float>(), tensor->data.f,
                output_tensors->back().bytes());
  }
  kOutTensors(cc).Send(std::move(output_tensors));
  return absl::OkStatus();
}

absl::Status InferenceCalculatorCpuImpl::Close(CalculatorContext* cc) {
  interpreter_ = nullptr;
  delegate_ = nullptr;
  return absl::OkStatus();
}

absl::Status InferenceCalculatorCpuImpl::LoadModel(CalculatorContext* cc) {
  ASSIGN_OR_RETURN(model_packet_, GetModelAsPacket(cc));
  const auto& model = *model_packet_.Get();
  tflite::ops::builtin::BuiltinOpResolver op_resolver =
      kSideInCustomOpResolver(cc).GetOr(
          tflite::ops::builtin::BuiltinOpResolverWithoutDefaultDelegates());

  tflite::InterpreterBuilder(model, op_resolver)(&interpreter_);
  RET_CHECK(interpreter_);

#if defined(__EMSCRIPTEN__)
  interpreter_->SetNumThreads(1);
#else
  interpreter_->SetNumThreads(
      cc->Options<mediapipe::InferenceCalculatorOptions>().cpu_num_thread());
#endif  // __EMSCRIPTEN__

  return absl::OkStatus();
}

absl::Status InferenceCalculatorCpuImpl::LoadDelegateAndAllocateTensors(
    CalculatorContext* cc) {
  MP_RETURN_IF_ERROR(LoadDelegate(cc));

  // AllocateTensors() can be called only after ModifyGraphWithDelegate.
  RET_CHECK_EQ(interpreter_->AllocateTensors(), kTfLiteOk);
  // TODO: Support quantized tensors.
  RET_CHECK_NE(
      interpreter_->tensor(interpreter_->inputs()[0])->quantization.type,
      kTfLiteAffineQuantization);
  return absl::OkStatus();
}

absl::Status InferenceCalculatorCpuImpl::LoadDelegate(CalculatorContext* cc) {
  const auto& calculator_opts =
      cc->Options<mediapipe::InferenceCalculatorOptions>();
  auto opts_delegate = calculator_opts.delegate();
  if (!kDelegate(cc).IsEmpty()) {
    mediapipe::InferenceCalculatorOptions::Delegate input_side_packet_delegate =
        kDelegate(cc).Get();
    CHECK(input_side_packet_delegate.has_tflite() ||
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
    // Default tflite inference requeqsted - no need to modify graph.
    return absl::OkStatus();
  }

#if defined(MEDIAPIPE_ANDROID)
  const bool nnapi_requested = opts_has_delegate ? opts_delegate.has_nnapi()
                                                 : calculator_opts.use_nnapi();
  if (nnapi_requested) {
    // Attempt to use NNAPI.
    // If not supported, the default CPU delegate will be created and used.
    interpreter_->SetAllowFp16PrecisionForFp32(1);
    tflite::StatefulNnApiDelegate::Options options;
    const auto& nnapi = opts_delegate.nnapi();
    // Set up cache_dir and model_token for NNAPI compilation cache.
    options.cache_dir =
        nnapi.has_cache_dir() ? nnapi.cache_dir().c_str() : nullptr;
    options.model_token =
        nnapi.has_model_token() ? nnapi.model_token().c_str() : nullptr;
    delegate_ = TfLiteDelegatePtr(new tflite::StatefulNnApiDelegate(options),
                                  [](TfLiteDelegate*) {});
    RET_CHECK_EQ(interpreter_->ModifyGraphWithDelegate(delegate_.get()),
                 kTfLiteOk);
    return absl::OkStatus();
  }
#endif  // MEDIAPIPE_ANDROID

#if defined(__EMSCRIPTEN__)
  const bool use_xnnpack = true;
#else
  const bool use_xnnpack = opts_has_delegate && opts_delegate.has_xnnpack();
#endif  // defined(__EMSCRIPTEN__)

  if (use_xnnpack) {
    TfLiteXNNPackDelegateOptions xnnpack_opts{};
    xnnpack_opts.num_threads =
        GetXnnpackNumThreads(opts_has_delegate, opts_delegate);
    delegate_ = TfLiteDelegatePtr(TfLiteXNNPackDelegateCreate(&xnnpack_opts),
                                  &TfLiteXNNPackDelegateDelete);
    RET_CHECK_EQ(interpreter_->ModifyGraphWithDelegate(delegate_.get()),
                 kTfLiteOk);
  }

  return absl::OkStatus();
}

}  // namespace api2
}  // namespace mediapipe
