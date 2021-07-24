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
#include "absl/status/status.h"
#include "mediapipe/calculators/tensor/inference_calculator.h"
#include "mediapipe/util/tflite/config.h"

#if MEDIAPIPE_TFLITE_GL_INFERENCE
#include "mediapipe/gpu/gl_calculator_helper.h"
#include "mediapipe/gpu/gpu_buffer.h"
#include "mediapipe/util/tflite/tflite_gpu_runner.h"
#include "tensorflow/lite/delegates/gpu/common/shape.h"
#include "tensorflow/lite/delegates/gpu/gl_delegate.h"
#endif  // MEDIAPIPE_TFLITE_GL_INFERENCE

#if defined(MEDIAPIPE_ANDROID)
#include "mediapipe/util/android/file/base/file.h"
#include "mediapipe/util/android/file/base/filesystem.h"
#include "mediapipe/util/android/file/base/helpers.h"
#endif  // ANDROID

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
  absl::Status ReadKernelsFromFile();
  absl::Status WriteKernelsToFile();
  absl::Status LoadModel(CalculatorContext* cc);
  absl::Status LoadDelegate(CalculatorContext* cc);
  absl::Status InitTFLiteGPURunner(CalculatorContext* cc);

  // TfLite requires us to keep the model alive as long as the interpreter is.
  Packet<TfLiteModelPtr> model_packet_;
  std::unique_ptr<tflite::Interpreter> interpreter_;
  TfLiteDelegatePtr delegate_;

#if MEDIAPIPE_TFLITE_GL_INFERENCE
  mediapipe::GlCalculatorHelper gpu_helper_;
  std::unique_ptr<tflite::gpu::TFLiteGPURunner> tflite_gpu_runner_;
  bool allow_precision_loss_ = false;
  mediapipe::InferenceCalculatorOptions::Delegate::Gpu::Api
      tflite_gpu_runner_api_;
  mediapipe::InferenceCalculatorOptions::Delegate::Gpu::InferenceUsage
      tflite_gpu_runner_usage_;
#endif  // MEDIAPIPE_TFLITE_GL_INFERENCE

#if MEDIAPIPE_TFLITE_GPU_SUPPORTED
  std::vector<Tensor::Shape> output_shapes_;
  std::vector<std::unique_ptr<Tensor>> gpu_buffers_in_;
  std::vector<std::unique_ptr<Tensor>> gpu_buffers_out_;
#endif  // MEDIAPIPE_TFLITE_GPU_SUPPORTED

  bool use_advanced_gpu_api_ = false;
  bool use_gpu_delegate_ = false;

  bool use_kernel_caching_ = false;
  std::string cached_kernel_filename_;
};

absl::Status InferenceCalculatorGlImpl::UpdateContract(CalculatorContract* cc) {
  const auto& options = cc->Options<::mediapipe::InferenceCalculatorOptions>();
  RET_CHECK(!options.model_path().empty() ^ kSideInModel(cc).IsConnected())
      << "Either model as side packet or model path in options is required.";

  MP_RETURN_IF_ERROR(mediapipe::GlCalculatorHelper::UpdateContract(cc));
  return absl::OkStatus();
}

absl::Status InferenceCalculatorGlImpl::Open(CalculatorContext* cc) {
  const auto& options = cc->Options<::mediapipe::InferenceCalculatorOptions>();
  use_advanced_gpu_api_ = options.has_delegate() &&
                          options.delegate().has_gpu() &&
                          options.delegate().gpu().use_advanced_gpu_api();
  allow_precision_loss_ = options.delegate().gpu().allow_precision_loss();
  tflite_gpu_runner_api_ = options.delegate().gpu().api();
  tflite_gpu_runner_usage_ = options.delegate().gpu().usage();
  use_kernel_caching_ = use_advanced_gpu_api_ &&
                        options.delegate().gpu().has_cached_kernel_path();
  use_gpu_delegate_ = !use_advanced_gpu_api_;

  if (use_kernel_caching_) {
#ifdef MEDIAPIPE_ANDROID
    cached_kernel_filename_ = options.delegate().gpu().cached_kernel_path() +
                              mediapipe::File::Basename(options.model_path()) +
                              ".ker";
#endif  // MEDIAPIPE_ANDROID
  }

  // When use_advanced_gpu_api_, model loading is handled in InitTFLiteGPURunner
  // for everything.
  if (!use_advanced_gpu_api_) {
    MP_RETURN_IF_ERROR(LoadModel(cc));
  }

  MP_RETURN_IF_ERROR(gpu_helper_.Open(cc));
  MP_RETURN_IF_ERROR(gpu_helper_.RunInGlContext([this,
                                                 &cc]() -> ::mediapipe::Status {
    return use_advanced_gpu_api_ ? InitTFLiteGPURunner(cc) : LoadDelegate(cc);
  }));
  return absl::OkStatus();
}

absl::Status InferenceCalculatorGlImpl::Process(CalculatorContext* cc) {
  if (kInTensors(cc).IsEmpty()) {
    return absl::OkStatus();
  }
  const auto& input_tensors = *kInTensors(cc);
  RET_CHECK(!input_tensors.empty());
  auto output_tensors = absl::make_unique<std::vector<Tensor>>();

  if (use_advanced_gpu_api_) {
    MP_RETURN_IF_ERROR(gpu_helper_.RunInGlContext(
        [this, &input_tensors, &output_tensors]() -> ::mediapipe::Status {
          for (int i = 0; i < input_tensors.size(); ++i) {
            MP_RETURN_IF_ERROR(tflite_gpu_runner_->BindSSBOToInputTensor(
                input_tensors[i].GetOpenGlBufferReadView().name(), i));
          }
          output_tensors->reserve(output_shapes_.size());
          for (int i = 0; i < output_shapes_.size(); ++i) {
            output_tensors->emplace_back(Tensor::ElementType::kFloat32,
                                         output_shapes_[i]);
            MP_RETURN_IF_ERROR(tflite_gpu_runner_->BindSSBOToOutputTensor(
                output_tensors->back().GetOpenGlBufferWriteView().name(), i));
          }
          return absl::OkStatus();
        }));
  } else {
    MP_RETURN_IF_ERROR(gpu_helper_.RunInGlContext(
        [this, &input_tensors]() -> ::mediapipe::Status {
          // Explicitly copy input.
          for (int i = 0; i < input_tensors.size(); ++i) {
            glBindBuffer(GL_COPY_READ_BUFFER,
                         input_tensors[i].GetOpenGlBufferReadView().name());
            glBindBuffer(GL_COPY_WRITE_BUFFER,
                         gpu_buffers_in_[i]->GetOpenGlBufferWriteView().name());
            glCopyBufferSubData(GL_COPY_READ_BUFFER, GL_COPY_WRITE_BUFFER, 0, 0,
                                input_tensors[i].bytes());
          }
          return absl::OkStatus();
        }));
  }

  // Run inference.
  if (use_advanced_gpu_api_) {
    RET_CHECK(tflite_gpu_runner_->Invoke().ok());
  } else {
    RET_CHECK_EQ(interpreter_->Invoke(), kTfLiteOk);
  }

  if (use_gpu_delegate_) {
    MP_RETURN_IF_ERROR(gpu_helper_.RunInGlContext(
        [this, &output_tensors]() -> ::mediapipe::Status {
          output_tensors->reserve(output_shapes_.size());
          for (int i = 0; i < output_shapes_.size(); ++i) {
            const auto& t = gpu_buffers_out_[i];
            output_tensors->emplace_back(Tensor::ElementType::kFloat32,
                                         gpu_buffers_out_[i]->shape());
            auto read_view = t->GetOpenGlBufferReadView();
            glBindBuffer(GL_COPY_READ_BUFFER, read_view.name());
            auto write_view = output_tensors->back().GetOpenGlBufferWriteView();
            glBindBuffer(GL_COPY_WRITE_BUFFER, write_view.name());
            glCopyBufferSubData(GL_COPY_READ_BUFFER, GL_COPY_WRITE_BUFFER, 0, 0,
                                t->bytes());
          }
          return absl::OkStatus();
        }));
  }
  // Output tensors are already bound if use_advanced_gpu_api_ is true.

  kOutTensors(cc).Send(std::move(output_tensors));
  return absl::OkStatus();
}

absl::Status InferenceCalculatorGlImpl::WriteKernelsToFile() {
#ifdef MEDIAPIPE_ANDROID
  if (use_kernel_caching_) {
    // Save kernel file.
    auto kernel_cache = absl::make_unique<std::vector<uint8_t>>(
        tflite_gpu_runner_->GetSerializedBinaryCache());
    std::string cache_str(kernel_cache->begin(), kernel_cache->end());
    MP_RETURN_IF_ERROR(
        mediapipe::file::SetContents(cached_kernel_filename_, cache_str));
  }
#endif  // MEDIAPIPE_ANDROID
  return absl::OkStatus();
}

absl::Status InferenceCalculatorGlImpl::Close(CalculatorContext* cc) {
  MP_RETURN_IF_ERROR(WriteKernelsToFile());
  if (use_gpu_delegate_) {
    MP_RETURN_IF_ERROR(gpu_helper_.RunInGlContext([this]() -> Status {
      gpu_buffers_in_.clear();
      gpu_buffers_out_.clear();
      return absl::OkStatus();
    }));
  }

  interpreter_ = nullptr;
  delegate_ = nullptr;
  return absl::OkStatus();
}

absl::Status InferenceCalculatorGlImpl::ReadKernelsFromFile() {
#ifdef MEDIAPIPE_ANDROID
  if (use_kernel_caching_) {
    // Load pre-compiled kernel file.
    if (mediapipe::File::Exists(cached_kernel_filename_)) {
      std::string cache_str;
      MP_RETURN_IF_ERROR(
          mediapipe::file::GetContents(cached_kernel_filename_, &cache_str));
      std::vector<uint8_t> cache_vec(cache_str.begin(), cache_str.end());
      tflite_gpu_runner_->SetSerializedBinaryCache(std::move(cache_vec));
    }
  }
#endif  // MEDIAPIPE_ANDROID
  return absl::OkStatus();
}

absl::Status InferenceCalculatorGlImpl::InitTFLiteGPURunner(
    CalculatorContext* cc) {
  ASSIGN_OR_RETURN(model_packet_, GetModelAsPacket(cc));
  const auto& model = *model_packet_.Get();
  tflite::ops::builtin::BuiltinOpResolver op_resolver =
      kSideInCustomOpResolver(cc).GetOr(
          tflite::ops::builtin::BuiltinOpResolverWithoutDefaultDelegates());

  // Create runner
  tflite::gpu::InferenceOptions options;
  options.priority1 = allow_precision_loss_
                          ? tflite::gpu::InferencePriority::MIN_LATENCY
                          : tflite::gpu::InferencePriority::MAX_PRECISION;
  options.priority2 = tflite::gpu::InferencePriority::AUTO;
  options.priority3 = tflite::gpu::InferencePriority::AUTO;
  switch (tflite_gpu_runner_usage_) {
    case mediapipe::InferenceCalculatorOptions::Delegate::Gpu::
        FAST_SINGLE_ANSWER: {
      options.usage = tflite::gpu::InferenceUsage::FAST_SINGLE_ANSWER;
      break;
    }
    case mediapipe::InferenceCalculatorOptions::Delegate::Gpu::
        SUSTAINED_SPEED: {
      options.usage = tflite::gpu::InferenceUsage::SUSTAINED_SPEED;
      break;
    }
    case mediapipe::InferenceCalculatorOptions::Delegate::Gpu::UNSPECIFIED: {
      return absl::InternalError("inference usage need to be specified.");
    }
  }
  tflite_gpu_runner_ = std::make_unique<tflite::gpu::TFLiteGPURunner>(options);
  switch (tflite_gpu_runner_api_) {
    case mediapipe::InferenceCalculatorOptions::Delegate::Gpu::ANY: {
      // Do not need to force any specific API.
      break;
    }
    case mediapipe::InferenceCalculatorOptions::Delegate::Gpu::OPENGL: {
      tflite_gpu_runner_->ForceOpenGL();
      break;
    }
    case mediapipe::InferenceCalculatorOptions::Delegate::Gpu::OPENCL: {
      tflite_gpu_runner_->ForceOpenCL();
      break;
    }
  }
  MP_RETURN_IF_ERROR(tflite_gpu_runner_->InitializeWithModel(
      model, op_resolver, /*allow_quant_ops=*/true));

  // Create and bind OpenGL buffers for outputs.
  // The buffers are created once and their ids are passed to calculator outputs
  output_shapes_.resize(tflite_gpu_runner_->outputs_size());
  for (int i = 0; i < tflite_gpu_runner_->outputs_size(); ++i) {
    output_shapes_[i] = {tflite_gpu_runner_->GetOutputShapes()[i].b,
                         tflite_gpu_runner_->GetOutputShapes()[i].h,
                         tflite_gpu_runner_->GetOutputShapes()[i].w,
                         tflite_gpu_runner_->GetOutputShapes()[i].c};
  }

  MP_RETURN_IF_ERROR(ReadKernelsFromFile());

  MP_RETURN_IF_ERROR(tflite_gpu_runner_->Build());

  return absl::OkStatus();
}

absl::Status InferenceCalculatorGlImpl::LoadModel(CalculatorContext* cc) {
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

  RET_CHECK_EQ(interpreter_->AllocateTensors(), kTfLiteOk);
  // TODO: Support quantized tensors.
  CHECK(interpreter_->tensor(interpreter_->inputs()[0])->quantization.type !=
        kTfLiteAffineQuantization);

  return absl::OkStatus();
}

absl::Status InferenceCalculatorGlImpl::LoadDelegate(CalculatorContext* cc) {
  // Configure and create the delegate.
  TfLiteGpuDelegateOptions options = TfLiteGpuDelegateOptionsDefault();
  options.compile_options.precision_loss_allowed =
      allow_precision_loss_ ? 1 : 0;
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
  output_shapes_.resize(output_indices.size());
  // Create and bind output buffers.
  for (int i = 0; i < output_shapes_.size(); ++i) {
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

}  // namespace api2
}  // namespace mediapipe
