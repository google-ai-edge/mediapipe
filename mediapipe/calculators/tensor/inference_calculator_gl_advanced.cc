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

#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "mediapipe/calculators/tensor/inference_calculator.h"
#include "mediapipe/gpu/gl_calculator_helper.h"
#include "mediapipe/util/tflite/tflite_gpu_runner.h"

#if defined(MEDIAPIPE_ANDROID)
#include "mediapipe/framework/deps/file_path.h"
#include "mediapipe/util/android/file/base/file.h"
#include "mediapipe/util/android/file/base/filesystem.h"
#include "mediapipe/util/android/file/base/helpers.h"
#endif  // ANDROID

namespace mediapipe {
namespace api2 {

// Runs TFLite GPU delegate API2 directly, bypassing interpreter usage, and
// allows choosing specific API.
//
// To trigger this code path:
//   [mediapipe.InferenceCalculatorOptions.ext] {
//     delegate {
//       gpu {
//         use_advanced_gpu_api: true
//         api: OPENCL  # or OPENGL or ANY
//       }
//     }
//   }
class InferenceCalculatorGlAdvancedImpl
    : public NodeImpl<InferenceCalculatorGlAdvanced,
                      InferenceCalculatorGlAdvancedImpl> {
 public:
  static absl::Status UpdateContract(CalculatorContract* cc);

  absl::Status Open(CalculatorContext* cc) override;
  absl::Status Process(CalculatorContext* cc) override;
  absl::Status Close(CalculatorContext* cc) override;

 private:
  absl::Status ReadGpuCaches();
  absl::Status SaveGpuCaches();
  absl::Status InitTFLiteGPURunner(CalculatorContext* cc);

  // TfLite requires us to keep the model alive as long as the interpreter is.
  Packet<TfLiteModelPtr> model_packet_;

  mediapipe::GlCalculatorHelper gpu_helper_;
  std::unique_ptr<tflite::gpu::TFLiteGPURunner> tflite_gpu_runner_;
  bool allow_precision_loss_ = false;
  mediapipe::InferenceCalculatorOptions::Delegate::Gpu::Api
      tflite_gpu_runner_api_;
  mediapipe::InferenceCalculatorOptions::Delegate::Gpu::InferenceUsage
      tflite_gpu_runner_usage_;

  std::vector<Tensor::Shape> output_shapes_;

  bool use_kernel_caching_ = false;
  std::string cached_kernel_filename_;
  bool use_serialized_model_ = false;
  std::string serialized_model_path_;
};

absl::Status InferenceCalculatorGlAdvancedImpl::UpdateContract(
    CalculatorContract* cc) {
  const auto& options = cc->Options<::mediapipe::InferenceCalculatorOptions>();
  RET_CHECK(!options.model_path().empty() ^ kSideInModel(cc).IsConnected())
      << "Either model as side packet or model path in options is required.";

  MP_RETURN_IF_ERROR(mediapipe::GlCalculatorHelper::UpdateContract(cc));
  return absl::OkStatus();
}

absl::Status InferenceCalculatorGlAdvancedImpl::Open(CalculatorContext* cc) {
  const auto& options = cc->Options<::mediapipe::InferenceCalculatorOptions>();
  mediapipe::InferenceCalculatorOptions::Delegate delegate = options.delegate();
  if (!kDelegate(cc).IsEmpty()) {
    mediapipe::InferenceCalculatorOptions::Delegate input_side_packet_delegate =
        kDelegate(cc).Get();
    CHECK(input_side_packet_delegate.has_gpu() ||
          input_side_packet_delegate.delegate_case() ==
              mediapipe::InferenceCalculatorOptions::Delegate::DELEGATE_NOT_SET)
        << "inference_calculator_gl_advanced only supports delegate input side "
           "packet for Gpu";
    delegate.MergeFrom(input_side_packet_delegate);
  }
  allow_precision_loss_ = delegate.gpu().allow_precision_loss();
  tflite_gpu_runner_api_ = delegate.gpu().api();
  tflite_gpu_runner_usage_ = delegate.gpu().usage();
  use_kernel_caching_ = delegate.gpu().has_cached_kernel_path();
  use_serialized_model_ = delegate.gpu().has_serialized_model_dir() &&
                          delegate.gpu().has_model_token();

  if (use_kernel_caching_) {
#ifdef MEDIAPIPE_ANDROID
    cached_kernel_filename_ = delegate.gpu().cached_kernel_path() +
                              mediapipe::File::Basename(options.model_path()) +
                              ".ker";
#endif  // MEDIAPIPE_ANDROID
  }
  if (use_serialized_model_) {
#ifdef MEDIAPIPE_ANDROID
    serialized_model_path_ = mediapipe::file::JoinPath(
        delegate.gpu().serialized_model_dir(), delegate.gpu().model_token());
#endif  // MEDIAPIPE_ANDROID
  }

  MP_RETURN_IF_ERROR(gpu_helper_.Open(cc));
  return gpu_helper_.RunInGlContext(
      [this, &cc]() -> absl::Status { return InitTFLiteGPURunner(cc); });
}

absl::Status InferenceCalculatorGlAdvancedImpl::Process(CalculatorContext* cc) {
  if (kInTensors(cc).IsEmpty()) {
    return absl::OkStatus();
  }
  const auto& input_tensors = *kInTensors(cc);
  RET_CHECK(!input_tensors.empty());
  auto output_tensors = absl::make_unique<std::vector<Tensor>>();

  MP_RETURN_IF_ERROR(gpu_helper_.RunInGlContext(
      [this, &input_tensors, &output_tensors]() -> absl::Status {
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

  // Run inference.
  MP_RETURN_IF_ERROR(tflite_gpu_runner_->Invoke());
  kOutTensors(cc).Send(std::move(output_tensors));
  return absl::OkStatus();
}

absl::Status InferenceCalculatorGlAdvancedImpl::SaveGpuCaches() {
#ifdef MEDIAPIPE_ANDROID
  if (use_kernel_caching_) {
    // Save kernel file.
    auto kernel_cache = absl::make_unique<std::vector<uint8_t>>(
        tflite_gpu_runner_->GetSerializedBinaryCache());
    std::string cache_str(kernel_cache->begin(), kernel_cache->end());
    MP_RETURN_IF_ERROR(
        mediapipe::file::SetContents(cached_kernel_filename_, cache_str));
  }
  if (use_serialized_model_) {
    // Save serialized model file.
    ASSIGN_OR_RETURN(std::vector<uint8_t> serialized_model_vec,
                     tflite_gpu_runner_->GetSerializedModel());
    absl::string_view serialized_model(
        reinterpret_cast<char*>(serialized_model_vec.data()),
        serialized_model_vec.size());
    MP_RETURN_IF_ERROR(
        mediapipe::file::SetContents(serialized_model_path_, serialized_model));
  }
#endif  // MEDIAPIPE_ANDROID
  return absl::OkStatus();
}

absl::Status InferenceCalculatorGlAdvancedImpl::Close(CalculatorContext* cc) {
  MP_RETURN_IF_ERROR(SaveGpuCaches());
  return gpu_helper_.RunInGlContext([this]() -> absl::Status {
    tflite_gpu_runner_.reset();
    return absl::OkStatus();
  });
}

absl::Status InferenceCalculatorGlAdvancedImpl::ReadGpuCaches() {
#ifdef MEDIAPIPE_ANDROID
  if (use_kernel_caching_ && File::Exists(cached_kernel_filename_)) {
    // Load pre-compiled kernel file.
    std::string cache_str;
    MP_RETURN_IF_ERROR(
        mediapipe::file::GetContents(cached_kernel_filename_, &cache_str));
    std::vector<uint8_t> cache_vec(cache_str.begin(), cache_str.end());
    tflite_gpu_runner_->SetSerializedBinaryCache(std::move(cache_vec));
  }
  if (use_serialized_model_ && File::Exists(serialized_model_path_)) {
    // Load serialized model file.
    std::string serialized_model_str;
    MP_RETURN_IF_ERROR(
        file::GetContents(serialized_model_path_, &serialized_model_str));
    std::vector<uint8_t> serialized_model_vec(serialized_model_str.begin(),
                                              serialized_model_str.end());
    tflite_gpu_runner_->SetSerializedModel(std::move(serialized_model_vec));
  }
#endif  // MEDIAPIPE_ANDROID
  return absl::OkStatus();
}

absl::Status InferenceCalculatorGlAdvancedImpl::InitTFLiteGPURunner(
    CalculatorContext* cc) {
  ASSIGN_OR_RETURN(model_packet_, GetModelAsPacket(cc));
  const auto& model = *model_packet_.Get();

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
  if (kSideInOpResolver(cc).IsConnected()) {
    const tflite::OpResolver& op_resolver = kSideInOpResolver(cc).Get();
    MP_RETURN_IF_ERROR(tflite_gpu_runner_->InitializeWithModel(
        model, op_resolver, /*allow_quant_ops=*/true));
  } else {
    tflite::ops::builtin::BuiltinOpResolver op_resolver =
        kSideInCustomOpResolver(cc).GetOr(
            tflite::ops::builtin::BuiltinOpResolverWithoutDefaultDelegates());
    MP_RETURN_IF_ERROR(tflite_gpu_runner_->InitializeWithModel(
        model, op_resolver, /*allow_quant_ops=*/true));
  }

  // Create and bind OpenGL buffers for outputs.
  // The buffers are created once and their ids are passed to calculator outputs
  output_shapes_.resize(tflite_gpu_runner_->outputs_size());
  for (int i = 0; i < tflite_gpu_runner_->outputs_size(); ++i) {
    output_shapes_[i] = {tflite_gpu_runner_->GetOutputShapes()[i].b,
                         tflite_gpu_runner_->GetOutputShapes()[i].h,
                         tflite_gpu_runner_->GetOutputShapes()[i].w,
                         tflite_gpu_runner_->GetOutputShapes()[i].c};
  }

  MP_RETURN_IF_ERROR(ReadGpuCaches());
  return tflite_gpu_runner_->Build();
}

}  // namespace api2
}  // namespace mediapipe
