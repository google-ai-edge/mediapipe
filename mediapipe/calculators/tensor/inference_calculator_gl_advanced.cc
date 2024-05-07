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

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/log/absl_log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/time/time.h"
#include "mediapipe/calculators/tensor/inference_calculator.h"
#include "mediapipe/calculators/tensor/inference_io_mapper.h"
#include "mediapipe/calculators/tensor/tensor_span.h"
#include "mediapipe/framework/api2/packet.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/deps/file_path.h"
#include "mediapipe/framework/formats/tensor.h"
#include "mediapipe/framework/port/file_helpers.h"
#include "mediapipe/framework/port/ret_check.h"
#include "mediapipe/framework/port/status_macros.h"
#include "mediapipe/gpu/gl_calculator_helper.h"
#include "mediapipe/gpu/gl_context.h"
#include "mediapipe/util/tflite/tflite_gpu_runner.h"
#include "mediapipe/util/tflite/tflite_model_loader.h"

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
    : public InferenceCalculatorNodeImpl<InferenceCalculatorGlAdvanced,
                                         InferenceCalculatorGlAdvancedImpl> {
 public:
  static absl::Status UpdateContract(CalculatorContract* cc);

  absl::Status Open(CalculatorContext* cc) override;
  absl::Status Close(CalculatorContext* cc) override;

 private:
  // Helper class that saves binary data to disk, or read from disk.
  class OnDiskCacheHelper {
   public:
    absl::Status Init(
        const mediapipe::InferenceCalculatorOptions& options,
        const mediapipe::InferenceCalculatorOptions::Delegate::Gpu&
            gpu_delegate_options);
    absl::Status ReadGpuCaches(tflite::gpu::TFLiteGPURunner& gpu_runner) const;
    // Writes caches to disk based on |cache_writing_behavior_|.
    absl::Status SaveGpuCachesBasedOnBehavior(
        tflite::gpu::TFLiteGPURunner& gpu_runner) const;
    bool UseSerializedModel() const { return use_serialized_model_; }

   private:
    // Writes caches to disk, returns error on failure.
    absl::Status SaveGpuCaches(tflite::gpu::TFLiteGPURunner& gpu_runner) const;

    bool use_kernel_caching_ = false;
    std::string cached_kernel_filename_;
    bool use_serialized_model_ = false;
    std::string serialized_model_path_;
    mediapipe::InferenceCalculatorOptions::Delegate::Gpu::CacheWritingBehavior
        cache_writing_behavior_;
  };

  // Helper class that wraps everything related to GPU inference acceleration.
  class GpuInferenceRunner {
   public:
    ~GpuInferenceRunner();

    absl::Status Init(CalculatorContext* cc,
                      std::shared_ptr<GlContext> gl_context);

    absl::StatusOr<std::vector<Tensor>> Process(
        CalculatorContext* cc, const TensorSpan& input_tensors);

    const InputOutputTensorNames& GetInputOutputTensorNames() const;

   private:
    absl::Status InitTFLiteGPURunner(
        CalculatorContext* cc,
        const mediapipe::InferenceCalculatorOptions::Delegate& delegate);

    // TfLite requires us to keep the model alive as long as the interpreter
    // is.
    Packet<TfLiteModelPtr> model_packet_;

    std::shared_ptr<GlContext> gl_context_;
    std::unique_ptr<tflite::gpu::TFLiteGPURunner> tflite_gpu_runner_;

    std::vector<Tensor::Shape> output_shapes_;

    OnDiskCacheHelper on_disk_cache_helper_;

    InputOutputTensorNames input_output_tensor_names_;
  };

  absl::StatusOr<std::vector<Tensor>> Process(
      CalculatorContext* cc, const TensorSpan& tensor_span) override;
  absl::StatusOr<std::unique_ptr<GpuInferenceRunner>> CreateInferenceRunner(
      CalculatorContext* cc);

  std::unique_ptr<GpuInferenceRunner> gpu_inference_runner_;
  mediapipe::GlCalculatorHelper gpu_helper_;
};

InferenceCalculatorGlAdvancedImpl::GpuInferenceRunner::~GpuInferenceRunner() {
  const auto success = gl_context_->Run([this]() -> absl::Status {
    tflite_gpu_runner_.reset();
    return absl::OkStatus();
  });
  if (!success.ok()) {
    ABSL_LOG(DFATAL) << "Failed to close gpu inference runner: " << success;
  }
}

absl::Status InferenceCalculatorGlAdvancedImpl::GpuInferenceRunner::Init(
    CalculatorContext* cc, std::shared_ptr<GlContext> gl_context) {
  gl_context_ = gl_context;
  const auto& options = cc->Options<mediapipe::InferenceCalculatorOptions>();

  mediapipe::InferenceCalculatorOptions::Delegate delegate = options.delegate();
  if (!kDelegate(cc).IsEmpty()) {
    const mediapipe::InferenceCalculatorOptions::Delegate&
        input_side_packet_delegate = kDelegate(cc).Get();
    RET_CHECK(
        input_side_packet_delegate.has_gpu() ||
        input_side_packet_delegate.delegate_case() ==
            mediapipe::InferenceCalculatorOptions::Delegate::DELEGATE_NOT_SET)
        << "inference_calculator_gl_advanced only supports gpu delegate "
           "configuration through side packet.";
    delegate.MergeFrom(input_side_packet_delegate);
  }

  MP_RETURN_IF_ERROR(on_disk_cache_helper_.Init(options, delegate.gpu()));

  return gl_context_->Run([this, &cc, &delegate]() -> absl::Status {
    return InitTFLiteGPURunner(cc, delegate);
  });
}

absl::StatusOr<std::vector<Tensor>>
InferenceCalculatorGlAdvancedImpl::GpuInferenceRunner::Process(
    CalculatorContext* cc, const TensorSpan& input_tensors) {
  std::vector<Tensor> output_tensors;

  MP_RETURN_IF_ERROR(gl_context_->Run(
      [this, cc, &input_tensors, &output_tensors]() -> absl::Status {
        for (int i = 0; i < input_tensors.size(); ++i) {
          MP_RETURN_IF_ERROR(tflite_gpu_runner_->BindSSBOToInputTensor(
              input_tensors[i].GetOpenGlBufferReadView().name(), i));
        }
        output_tensors.reserve(output_shapes_.size());
        for (int i = 0; i < output_shapes_.size(); ++i) {
          output_tensors.emplace_back(Tensor::ElementType::kFloat32,
                                      output_shapes_[i]);
          MP_RETURN_IF_ERROR(tflite_gpu_runner_->BindSSBOToOutputTensor(
              output_tensors.back().GetOpenGlBufferWriteView().name(), i));
        }
        // Run inference.
        {
          MEDIAPIPE_PROFILING(GPU_TASK_INVOKE_ADVANCED, cc);
          return tflite_gpu_runner_->Invoke();
        }
      }));

  return output_tensors;
}

const InputOutputTensorNames& InferenceCalculatorGlAdvancedImpl::
    GpuInferenceRunner::GetInputOutputTensorNames() const {
  return input_output_tensor_names_;
}

absl::Status
InferenceCalculatorGlAdvancedImpl::GpuInferenceRunner::InitTFLiteGPURunner(
    CalculatorContext* cc,
    const mediapipe::InferenceCalculatorOptions::Delegate& delegate) {
  MP_ASSIGN_OR_RETURN(model_packet_, GetModelAsPacket(cc));
  const auto& model = *model_packet_.Get();

  bool allow_precision_loss = delegate.gpu().allow_precision_loss();

  // Create runner
  tflite::gpu::InferenceOptions options;
  options.priority1 = allow_precision_loss
                          ? tflite::gpu::InferencePriority::MIN_LATENCY
                          : tflite::gpu::InferencePriority::MAX_PRECISION;
  options.priority2 = tflite::gpu::InferencePriority::AUTO;
  options.priority3 = tflite::gpu::InferencePriority::AUTO;
  switch (delegate.gpu().usage()) {
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
  switch (delegate.gpu().api()) {
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
    MP_ASSIGN_OR_RETURN(input_output_tensor_names_,
                        InferenceIoMapper::GetInputOutputTensorNamesFromModel(
                            model, op_resolver));
  } else {
    tflite::ops::builtin::BuiltinOpResolver op_resolver =
        kSideInCustomOpResolver(cc).GetOr(
            tflite::ops::builtin::BuiltinOpResolverWithoutDefaultDelegates());
    MP_RETURN_IF_ERROR(tflite_gpu_runner_->InitializeWithModel(
        model, op_resolver, /*allow_quant_ops=*/true));
    MP_ASSIGN_OR_RETURN(input_output_tensor_names_,
                        InferenceIoMapper::GetInputOutputTensorNamesFromModel(
                            model, op_resolver));
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

  if (on_disk_cache_helper_.UseSerializedModel()) {
    tflite_gpu_runner_->ForceOpenCLInitFromSerializedModel();
  }

  MP_RETURN_IF_ERROR(on_disk_cache_helper_.ReadGpuCaches(*tflite_gpu_runner_));
  MP_RETURN_IF_ERROR(tflite_gpu_runner_->Build());
  return on_disk_cache_helper_.SaveGpuCachesBasedOnBehavior(
      *tflite_gpu_runner_);
}

absl::Status InferenceCalculatorGlAdvancedImpl::OnDiskCacheHelper::Init(
    const mediapipe::InferenceCalculatorOptions& options,
    const mediapipe::InferenceCalculatorOptions::Delegate::Gpu&
        gpu_delegate_options) {
  // The kernel cache needs a unique filename based on either model_path or the
  // model token, to prevent the cache from being overwritten if the graph has
  // more than one model.
  use_kernel_caching_ =
      gpu_delegate_options.has_cached_kernel_path() &&
      (options.has_model_path() || gpu_delegate_options.has_model_token());
  use_serialized_model_ = gpu_delegate_options.has_serialized_model_dir() &&
                          gpu_delegate_options.has_model_token();

  if (use_kernel_caching_) {
    absl::string_view basename =
        options.has_model_path()
            ? mediapipe::file::Basename(options.model_path())
            : gpu_delegate_options.model_token();
    cached_kernel_filename_ =
        mediapipe::file::JoinPath(gpu_delegate_options.cached_kernel_path(),
                                  absl::StrCat(basename, ".ker"));
  }
  if (use_serialized_model_) {
    serialized_model_path_ =
        mediapipe::file::JoinPath(gpu_delegate_options.serialized_model_dir(),
                                  gpu_delegate_options.model_token());
  }
  cache_writing_behavior_ = gpu_delegate_options.has_cache_writing_behavior()
                                ? gpu_delegate_options.cache_writing_behavior()
                                : mediapipe::InferenceCalculatorOptions::
                                      Delegate::Gpu::WRITE_OR_ERROR;
  return absl::OkStatus();
}

absl::Status InferenceCalculatorGlAdvancedImpl::OnDiskCacheHelper::
    SaveGpuCachesBasedOnBehavior(
        tflite::gpu::TFLiteGPURunner& gpu_runner) const {
  switch (cache_writing_behavior_) {
    case mediapipe::InferenceCalculatorOptions::Delegate::Gpu::NO_WRITE:
      return absl::OkStatus();
    case mediapipe::InferenceCalculatorOptions::Delegate::Gpu::TRY_WRITE: {
      auto status = SaveGpuCaches(gpu_runner);
      if (!status.ok()) {
        ABSL_LOG_FIRST_N(WARNING, 1) << "Failed to save gpu caches: " << status;
      }
      return absl::OkStatus();
    }
    case mediapipe::InferenceCalculatorOptions::Delegate::Gpu::WRITE_OR_ERROR:
      return SaveGpuCaches(gpu_runner);
    default:
      ABSL_LOG_FIRST_N(ERROR, 1)
          << "Unknown cache writing behavior: "
          << static_cast<uint32_t>(cache_writing_behavior_);
      return absl::InvalidArgumentError("Unknown cache writing behavior.");
  }
}

absl::Status
InferenceCalculatorGlAdvancedImpl::OnDiskCacheHelper::SaveGpuCaches(
    tflite::gpu::TFLiteGPURunner& gpu_runner) const {
  if (use_kernel_caching_ && gpu_runner.CanGenerateSerializedBinaryCache()) {
    // Save kernel file.
    MP_ASSIGN_OR_RETURN(std::vector<uint8_t> kernel_cache,
                        gpu_runner.GetSerializedBinaryCache());
    std::string cache_str(kernel_cache.begin(), kernel_cache.end());
    MP_RETURN_IF_ERROR(
        mediapipe::file::SetContents(cached_kernel_filename_, cache_str));
  }
  if (use_serialized_model_ && gpu_runner.CanGenerateSerializedModel()) {
    // Save serialized model file.
    MP_ASSIGN_OR_RETURN(std::vector<uint8_t> serialized_model_vec,
                        gpu_runner.GetSerializedModel());
    absl::string_view serialized_model(
        reinterpret_cast<char*>(serialized_model_vec.data()),
        serialized_model_vec.size());
    MP_RETURN_IF_ERROR(
        mediapipe::file::SetContents(serialized_model_path_, serialized_model));
  }
  return absl::OkStatus();
}

absl::Status
InferenceCalculatorGlAdvancedImpl::OnDiskCacheHelper::ReadGpuCaches(
    tflite::gpu::TFLiteGPURunner& gpu_runner) const {
  if (use_kernel_caching_ &&
      mediapipe::file::Exists(cached_kernel_filename_).ok()) {
    // Load pre-compiled kernel file.
    std::string cache_str;
    MP_RETURN_IF_ERROR(
        mediapipe::file::GetContents(cached_kernel_filename_, &cache_str));
    std::vector<uint8_t> cache_vec(cache_str.begin(), cache_str.end());
    gpu_runner.SetSerializedBinaryCache(std::move(cache_vec));
  }
  if (use_serialized_model_ &&
      mediapipe::file::Exists(serialized_model_path_).ok()) {
    // Load serialized model file.
    std::string serialized_model_str;
    MP_RETURN_IF_ERROR(
        file::GetContents(serialized_model_path_, &serialized_model_str));
    std::vector<uint8_t> serialized_model_vec(serialized_model_str.begin(),
                                              serialized_model_str.end());
    gpu_runner.SetSerializedModel(std::move(serialized_model_vec));
  }
  return absl::OkStatus();
}

absl::Status InferenceCalculatorGlAdvancedImpl::UpdateContract(
    CalculatorContract* cc) {
  MP_RETURN_IF_ERROR(TensorContractCheck(cc));

  const auto& options = cc->Options<mediapipe::InferenceCalculatorOptions>();
  RET_CHECK(!options.model_path().empty() ^ kSideInModel(cc).IsConnected())
      << "Either model as side packet or model path in options is required.";

  WarnFeedbackTensorsUnsupported(cc);
  MP_RETURN_IF_ERROR(mediapipe::GlCalculatorHelper::UpdateContract(cc));
  return absl::OkStatus();
}

absl::Status InferenceCalculatorGlAdvancedImpl::Open(CalculatorContext* cc) {
  MP_RETURN_IF_ERROR(gpu_helper_.Open(cc));
  gpu_inference_runner_ = std::make_unique<GpuInferenceRunner>();
  MP_RETURN_IF_ERROR(
      gpu_inference_runner_->Init(cc, gpu_helper_.GetSharedGlContext()));
  return InferenceCalculatorNodeImpl::UpdateIoMapping(
      cc, gpu_inference_runner_->GetInputOutputTensorNames());
}

absl::StatusOr<std::vector<Tensor>> InferenceCalculatorGlAdvancedImpl::Process(
    CalculatorContext* cc, const TensorSpan& tensor_span) {
  MP_ASSIGN_OR_RETURN(std::vector<Tensor> output_tensors,
                      gpu_inference_runner_->Process(cc, tensor_span));
  return output_tensors;
}

absl::Status InferenceCalculatorGlAdvancedImpl::Close(CalculatorContext* cc) {
  gpu_inference_runner_.reset();

  return absl::OkStatus();
}

absl::StatusOr<
    std::unique_ptr<InferenceCalculatorGlAdvancedImpl::GpuInferenceRunner>>
InferenceCalculatorGlAdvancedImpl::CreateInferenceRunner(
    CalculatorContext* cc) {
  auto gpu_inference_runner = std::make_unique<GpuInferenceRunner>();
  MP_RETURN_IF_ERROR(
      gpu_inference_runner->Init(cc, gpu_helper_.GetSharedGlContext()));
  return gpu_inference_runner;
}

}  // namespace api2
}  // namespace mediapipe
