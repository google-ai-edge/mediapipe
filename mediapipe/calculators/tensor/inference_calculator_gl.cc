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
#include <utility>
#include <vector>

#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "mediapipe/calculators/tensor/inference_calculator.h"
#include "mediapipe/calculators/tensor/inference_calculator.pb.h"
#include "mediapipe/framework/calculator_context.h"
#include "mediapipe/gpu/gl_calculator_helper.h"
#include "tensorflow/lite/delegates/gpu/gl_delegate.h"

namespace mediapipe {
namespace api2 {

class InferenceCalculatorGlImpl
    : public NodeImpl<InferenceCalculatorGl, InferenceCalculatorGlImpl> {
 public:
  static absl::Status UpdateContract(CalculatorContract* cc);

  absl::Status Open(CalculatorContext* cc) override;
  absl::Status Process(CalculatorContext* cc) override;
  absl::Status Close(CalculatorContext* cc) override;

 private:
  // Helper class that wraps everything related to GPU inference acceleration.
  class GpuInferenceRunner {
   public:
    ~GpuInferenceRunner();

    absl::Status Init(CalculatorContext* cc,
                      const mediapipe::InferenceCalculatorOptions::Delegate&
                          delegate_options);
    absl::Status LoadModel(CalculatorContext* cc);
    absl::Status LoadDelegate(
        CalculatorContext* cc,
        const mediapipe::InferenceCalculatorOptions::Delegate&
            delegate_options);
    absl::Status LoadDelegateAndAllocateTensors(
        CalculatorContext* cc,
        const mediapipe::InferenceCalculatorOptions::Delegate&
            delegate_options);
    absl::Status Process(CalculatorContext* cc,
                         const std::vector<Tensor>& input_tensors,
                         std::vector<Tensor>& output_tensors);

   private:
    // TfLite requires us to keep the model alive as long as the interpreter is.
    Packet<TfLiteModelPtr> model_packet_;
    mediapipe::GlCalculatorHelper gpu_helper_;
    TfLiteDelegatePtr delegate_;
    std::unique_ptr<tflite::Interpreter> interpreter_;
    std::vector<std::unique_ptr<Tensor>> gpu_buffers_in_;
    std::vector<std::unique_ptr<Tensor>> gpu_buffers_out_;
    size_t output_size_ = 0;
  };

  std::unique_ptr<GpuInferenceRunner> gpu_inference_runner_;
};

InferenceCalculatorGlImpl::GpuInferenceRunner::~GpuInferenceRunner() {
  gpu_helper_.RunInGlContext([this]() {
    gpu_buffers_in_.clear();
    gpu_buffers_out_.clear();
    // Delegate must outlive the interpreter, hence the order is important.
    interpreter_ = nullptr;
    delegate_ = nullptr;
  });
}

absl::Status InferenceCalculatorGlImpl::GpuInferenceRunner::Init(
    CalculatorContext* cc,
    const mediapipe::InferenceCalculatorOptions::Delegate& delegate_options) {
  MP_RETURN_IF_ERROR(LoadModel(cc));
  MP_RETURN_IF_ERROR(gpu_helper_.Open(cc));
  return gpu_helper_.RunInGlContext(
      [this, &cc, &delegate_options]() -> absl::Status {
        return LoadDelegateAndAllocateTensors(cc, delegate_options);
      });
}

absl::Status InferenceCalculatorGlImpl::GpuInferenceRunner::LoadModel(
    CalculatorContext* cc) {
  ASSIGN_OR_RETURN(model_packet_, GetModelAsPacket(cc));
  const auto& model = *model_packet_.Get();
  if (kSideInOpResolver(cc).IsConnected()) {
    const tflite::OpResolver& op_resolver = kSideInOpResolver(cc).Get();
    tflite::InterpreterBuilder(model, op_resolver)(&interpreter_);
  } else {
    tflite::ops::builtin::BuiltinOpResolver op_resolver =
        kSideInCustomOpResolver(cc).GetOr(
            tflite::ops::builtin::BuiltinOpResolverWithoutDefaultDelegates());
    tflite::InterpreterBuilder(model, op_resolver)(&interpreter_);
  }
  RET_CHECK(interpreter_);

  interpreter_->SetNumThreads(
      cc->Options<mediapipe::InferenceCalculatorOptions>().cpu_num_thread());

  return absl::OkStatus();
}

absl::Status
InferenceCalculatorGlImpl::GpuInferenceRunner::LoadDelegateAndAllocateTensors(
    CalculatorContext* cc,
    const mediapipe::InferenceCalculatorOptions::Delegate& delegate_options) {
  MP_RETURN_IF_ERROR(LoadDelegate(cc, delegate_options));

  // AllocateTensors() can be called only after ModifyGraphWithDelegate.
  RET_CHECK_EQ(interpreter_->AllocateTensors(), kTfLiteOk);
  // TODO: Support quantized tensors.
  RET_CHECK_NE(
      interpreter_->tensor(interpreter_->inputs()[0])->quantization.type,
      kTfLiteAffineQuantization);
  return absl::OkStatus();
}

absl::Status InferenceCalculatorGlImpl::GpuInferenceRunner::LoadDelegate(
    CalculatorContext* cc,
    const mediapipe::InferenceCalculatorOptions::Delegate& delegate_options) {
  // Configure and create the delegate.
  TfLiteGpuDelegateOptions options = TfLiteGpuDelegateOptionsDefault();
  options.compile_options.precision_loss_allowed =
      (delegate_options.has_gpu() &&
       delegate_options.gpu().allow_precision_loss())
          ? 1
          : 0;
  options.compile_options.preferred_gl_object_type =
      TFLITE_GL_OBJECT_TYPE_FASTEST;
  options.compile_options.dynamic_batch_enabled = 0;
  options.compile_options.inline_parameters = 1;
  delegate_ = TfLiteDelegatePtr(TfLiteGpuDelegateCreate(&options),
                                &TfLiteGpuDelegateDelete);

  // Get input image sizes.
  const auto& input_indices = interpreter_->inputs();
  for (int i = 0; i < input_indices.size(); ++i) {
    const TfLiteTensor* tensor = interpreter_->tensor(input_indices[i]);
    gpu_buffers_in_.emplace_back(absl::make_unique<Tensor>(
        Tensor::ElementType::kFloat32,
        Tensor::Shape{std::vector<int>{
            tensor->dims->data, tensor->dims->data + tensor->dims->size}}));
    RET_CHECK_EQ(TfLiteGpuDelegateBindBufferToTensor(
                     delegate_.get(),
                     gpu_buffers_in_.back()->GetOpenGlBufferWriteView().name(),
                     interpreter_->inputs()[i]),
                 kTfLiteOk);
  }
  interpreter_->SetAllowBufferHandleOutput(true);
  // Get output image sizes.
  const auto& output_indices = interpreter_->outputs();
  output_size_ = output_indices.size();
  // Create and bind output buffers.
  for (int i = 0; i < output_size_; ++i) {
    const TfLiteTensor* tensor = interpreter_->tensor(output_indices[i]);
    gpu_buffers_out_.emplace_back(absl::make_unique<Tensor>(
        Tensor::ElementType::kFloat32,
        Tensor::Shape{std::vector<int>{
            tensor->dims->data, tensor->dims->data + tensor->dims->size}}));
    RET_CHECK_EQ(TfLiteGpuDelegateBindBufferToTensor(
                     delegate_.get(),
                     gpu_buffers_out_.back()->GetOpenGlBufferWriteView().name(),
                     output_indices[i]),
                 kTfLiteOk);
  }

  // Must call this last.
  RET_CHECK_EQ(interpreter_->ModifyGraphWithDelegate(delegate_.get()),
               kTfLiteOk);

  return absl::OkStatus();
}

absl::Status InferenceCalculatorGlImpl::GpuInferenceRunner::Process(
    CalculatorContext* cc, const std::vector<Tensor>& input_tensors,
    std::vector<Tensor>& output_tensors) {
  return gpu_helper_.RunInGlContext(
      [this, &input_tensors, &output_tensors]() -> absl::Status {
        // Explicitly copy input.
        for (int i = 0; i < input_tensors.size(); ++i) {
          glBindBuffer(GL_COPY_READ_BUFFER,
                       input_tensors[i].GetOpenGlBufferReadView().name());
          glBindBuffer(GL_COPY_WRITE_BUFFER,
                       gpu_buffers_in_[i]->GetOpenGlBufferWriteView().name());
          glCopyBufferSubData(GL_COPY_READ_BUFFER, GL_COPY_WRITE_BUFFER, 0, 0,
                              input_tensors[i].bytes());
        }

        // Run inference.
        RET_CHECK_EQ(interpreter_->Invoke(), kTfLiteOk);

        output_tensors.reserve(output_size_);
        for (int i = 0; i < output_size_; ++i) {
          const auto& t = gpu_buffers_out_[i];
          output_tensors.emplace_back(Tensor::ElementType::kFloat32,
                                      gpu_buffers_out_[i]->shape());
          auto read_view = t->GetOpenGlBufferReadView();
          glBindBuffer(GL_COPY_READ_BUFFER, read_view.name());
          auto write_view = output_tensors.back().GetOpenGlBufferWriteView();
          glBindBuffer(GL_COPY_WRITE_BUFFER, write_view.name());
          glCopyBufferSubData(GL_COPY_READ_BUFFER, GL_COPY_WRITE_BUFFER, 0, 0,
                              t->bytes());
        }

        return absl::OkStatus();
      });
}

absl::Status InferenceCalculatorGlImpl::UpdateContract(CalculatorContract* cc) {
  const auto& options = cc->Options<mediapipe::InferenceCalculatorOptions>();
  RET_CHECK(!options.model_path().empty() ^ kSideInModel(cc).IsConnected())
      << "Either model as side packet or model path in options is required.";

  return mediapipe::GlCalculatorHelper::UpdateContract(cc);
}

absl::Status InferenceCalculatorGlImpl::Open(CalculatorContext* cc) {
  const auto& options = cc->Options<mediapipe::InferenceCalculatorOptions>();
  mediapipe::InferenceCalculatorOptions::Delegate delegate = options.delegate();
  if (!kDelegate(cc).IsEmpty()) {
    const mediapipe::InferenceCalculatorOptions::Delegate&
        input_side_packet_delegate = kDelegate(cc).Get();
    RET_CHECK(
        (input_side_packet_delegate.has_gpu() &&
         !input_side_packet_delegate.gpu().use_advanced_gpu_api()) ||
        input_side_packet_delegate.delegate_case() ==
            mediapipe::InferenceCalculatorOptions::Delegate::DELEGATE_NOT_SET)
        << "inference_calculator_gl only supports delegate input side packet "
        << "for Gpu (non advanced)";
    delegate.MergeFrom(input_side_packet_delegate);
  }

  gpu_inference_runner_ = std::make_unique<GpuInferenceRunner>();
  return gpu_inference_runner_->Init(cc, delegate);
}

absl::Status InferenceCalculatorGlImpl::Process(CalculatorContext* cc) {
  if (kInTensors(cc).IsEmpty()) {
    return absl::OkStatus();
  }

  const auto& input_tensors = *kInTensors(cc);
  RET_CHECK(!input_tensors.empty());
  auto output_tensors = absl::make_unique<std::vector<Tensor>>();

  MP_RETURN_IF_ERROR(
      gpu_inference_runner_->Process(cc, input_tensors, *output_tensors));

  kOutTensors(cc).Send(std::move(output_tensors));
  return absl::OkStatus();
}

absl::Status InferenceCalculatorGlImpl::Close(CalculatorContext* cc) {
  gpu_inference_runner_ = nullptr;
  return absl::OkStatus();
}

}  // namespace api2
}  // namespace mediapipe
