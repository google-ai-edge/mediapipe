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
#include "tensorflow/lite/interpreter_builder.h"
#if defined(MEDIAPIPE_ANDROID)
#include "tensorflow/lite/delegates/nnapi/nnapi_delegate.h"
#endif  // ANDROID

#if !defined(__EMSCRIPTEN__) || defined(__EMSCRIPTEN_PTHREADS__)
#include "mediapipe/util/cpu_util.h"
#endif  // !__EMSCRIPTEN__ || __EMSCRIPTEN_PTHREADS__

#include "tensorflow/lite/c/c_api_types.h"
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

template <typename T>
void CopyTensorBuffer(const Tensor& input_tensor,
                      tflite::Interpreter* interpreter,
                      int input_tensor_index) {
  auto input_tensor_view = input_tensor.GetCpuReadView();
  auto input_tensor_buffer = input_tensor_view.buffer<T>();
  T* local_tensor_buffer =
      interpreter->typed_input_tensor<T>(input_tensor_index);
  std::memcpy(local_tensor_buffer, input_tensor_buffer, input_tensor.bytes());
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
  absl::Status InitInterpreter(CalculatorContext* cc);
  absl::Status LoadDelegate(CalculatorContext* cc,
                            tflite::InterpreterBuilder* interpreter_builder);
  absl::Status AllocateTensors();

  // TfLite requires us to keep the model alive as long as the interpreter is.
  Packet<TfLiteModelPtr> model_packet_;
  std::unique_ptr<tflite::Interpreter> interpreter_;
  TfLiteDelegatePtr delegate_;
  TfLiteType input_tensor_type_ = TfLiteType::kTfLiteNoType;
};

absl::Status InferenceCalculatorCpuImpl::UpdateContract(
    CalculatorContract* cc) {
  const auto& options = cc->Options<::mediapipe::InferenceCalculatorOptions>();
  RET_CHECK(!options.model_path().empty() ^ kSideInModel(cc).IsConnected())
      << "Either model as side packet or model path in options is required.";

  return absl::OkStatus();
}

absl::Status InferenceCalculatorCpuImpl::Open(CalculatorContext* cc) {
  return InitInterpreter(cc);
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
    switch (input_tensor_type_) {
      case TfLiteType::kTfLiteFloat16:
      case TfLiteType::kTfLiteFloat32: {
        CopyTensorBuffer<float>(input_tensors[i], interpreter_.get(), i);
        break;
      }
      case TfLiteType::kTfLiteUInt8: {
        CopyTensorBuffer<uint8>(input_tensors[i], interpreter_.get(), i);
        break;
      }
      case TfLiteType::kTfLiteInt8: {
        CopyTensorBuffer<int8>(input_tensors[i], interpreter_.get(), i);
        break;
      }
      default:
        return absl::InvalidArgumentError(
            absl::StrCat("Unsupported input tensor type:", input_tensor_type_));
    }
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

absl::Status InferenceCalculatorCpuImpl::InitInterpreter(
    CalculatorContext* cc) {
  ASSIGN_OR_RETURN(model_packet_, GetModelAsPacket(cc));
  const auto& model = *model_packet_.Get();
  ASSIGN_OR_RETURN(auto op_resolver_packet, GetOpResolverAsPacket(cc));
  const auto& op_resolver = op_resolver_packet.Get();
  tflite::InterpreterBuilder interpreter_builder(model, op_resolver);
  MP_RETURN_IF_ERROR(LoadDelegate(cc, &interpreter_builder));
#if defined(__EMSCRIPTEN__)
  interpreter_builder.SetNumThreads(1);
#else
  interpreter_builder.SetNumThreads(
      cc->Options<mediapipe::InferenceCalculatorOptions>().cpu_num_thread());
#endif  // __EMSCRIPTEN__

  RET_CHECK_EQ(interpreter_builder(&interpreter_), kTfLiteOk);
  RET_CHECK(interpreter_);
  return AllocateTensors();
}

absl::Status InferenceCalculatorCpuImpl::AllocateTensors() {
  RET_CHECK_EQ(interpreter_->AllocateTensors(), kTfLiteOk);
  input_tensor_type_ = interpreter_->tensor(interpreter_->inputs()[0])->type;
  return absl::OkStatus();
}

absl::Status InferenceCalculatorCpuImpl::LoadDelegate(
    CalculatorContext* cc, tflite::InterpreterBuilder* interpreter_builder) {
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
    delegate_ = TfLiteDelegatePtr(new tflite::StatefulNnApiDelegate(options),
                                  [](TfLiteDelegate*) {});
    interpreter_builder->AddDelegate(delegate_.get());
    return absl::OkStatus();
  }
#endif  // MEDIAPIPE_ANDROID

#if defined(__EMSCRIPTEN__)
  const bool use_xnnpack = true;
#else
  const bool use_xnnpack = opts_has_delegate && opts_delegate.has_xnnpack();
#endif  // defined(__EMSCRIPTEN__)

  if (use_xnnpack) {
    auto xnnpack_opts = TfLiteXNNPackDelegateOptionsDefault();
    xnnpack_opts.num_threads =
        GetXnnpackNumThreads(opts_has_delegate, opts_delegate);
    delegate_ = TfLiteDelegatePtr(TfLiteXNNPackDelegateCreate(&xnnpack_opts),
                                  &TfLiteXNNPackDelegateDelete);
    interpreter_builder->AddDelegate(delegate_.get());
  }

  return absl::OkStatus();
}

}  // namespace api2
}  // namespace mediapipe
