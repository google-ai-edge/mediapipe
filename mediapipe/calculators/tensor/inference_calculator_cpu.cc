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

// Returns number of threads to configure XNNPACK delegate with.
// (Equal to user provided value if specified.  Otherwise, it returns number of
// high cores (hard-coded to 1 for Emscripten without Threads extension))
int GetXnnpackNumThreads(const mediapipe::InferenceCalculatorOptions& opts) {
  static constexpr int kDefaultNumThreads = -1;
  if (opts.has_delegate() && opts.delegate().has_xnnpack() &&
      opts.delegate().xnnpack().num_threads() != kDefaultNumThreads) {
    return opts.delegate().xnnpack().num_threads();
  }
#if !defined(__EMSCRIPTEN__) || defined(__EMSCRIPTEN_PTHREADS__)
  return InferHigherCoreIds().size();
#else
  return 1;
#endif  // !__EMSCRIPTEN__ || __EMSCRIPTEN_PTHREADS__
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
  MP_RETURN_IF_ERROR(LoadDelegate(cc));
  return absl::OkStatus();
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
          tflite::ops::builtin::BuiltinOpResolver());

  tflite::InterpreterBuilder(model, op_resolver)(&interpreter_);
  RET_CHECK(interpreter_);

#if defined(__EMSCRIPTEN__)
  interpreter_->SetNumThreads(1);
#else
  interpreter_->SetNumThreads(
      cc->Options<mediapipe::InferenceCalculatorOptions>().cpu_num_thread());
#endif  // __EMSCRIPTEN__

  RET_CHECK_EQ(interpreter_->AllocateTensors(), kTfLiteOk);
  // TODO: Support quantized tensors.
  CHECK(interpreter_->tensor(interpreter_->inputs()[0])->quantization.type !=
        kTfLiteAffineQuantization);

  return absl::OkStatus();
}

absl::Status InferenceCalculatorCpuImpl::LoadDelegate(CalculatorContext* cc) {
  const auto& calculator_opts =
      cc->Options<mediapipe::InferenceCalculatorOptions>();
  if (calculator_opts.has_delegate() &&
      calculator_opts.delegate().has_tflite()) {
    // Default tflite inference requeqsted - no need to modify graph.
    return absl::OkStatus();
  }

#if defined(MEDIAPIPE_ANDROID)
  const bool nnapi_requested = calculator_opts.has_delegate()
                                   ? calculator_opts.delegate().has_nnapi()
                                   : calculator_opts.use_nnapi();
  if (nnapi_requested) {
    // Attempt to use NNAPI.
    // If not supported, the default CPU delegate will be created and used.
    interpreter_->SetAllowFp16PrecisionForFp32(1);
    delegate_ = TfLiteDelegatePtr(tflite::NnApiDelegate(), [](TfLiteDelegate*) {
      // No need to free according to tflite::NnApiDelegate() documentation.
    });
    RET_CHECK_EQ(interpreter_->ModifyGraphWithDelegate(delegate_.get()),
                 kTfLiteOk);
    return absl::OkStatus();
  }
#endif  // MEDIAPIPE_ANDROID

#if defined(__EMSCRIPTEN__)
  const bool use_xnnpack = true;
#else
  const bool use_xnnpack = calculator_opts.has_delegate() &&
                           calculator_opts.delegate().has_xnnpack();
#endif  // defined(__EMSCRIPTEN__)

  if (use_xnnpack) {
    TfLiteXNNPackDelegateOptions xnnpack_opts{};
    xnnpack_opts.num_threads = GetXnnpackNumThreads(calculator_opts);
    delegate_ = TfLiteDelegatePtr(TfLiteXNNPackDelegateCreate(&xnnpack_opts),
                                  &TfLiteXNNPackDelegateDelete);
    RET_CHECK_EQ(interpreter_->ModifyGraphWithDelegate(delegate_.get()),
                 kTfLiteOk);
  }

  return absl::OkStatus();
}

}  // namespace api2
}  // namespace mediapipe
