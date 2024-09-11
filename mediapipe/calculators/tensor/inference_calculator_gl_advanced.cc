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

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/log/absl_log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/time/time.h"
#include "mediapipe/calculators/tensor/inference_calculator.h"
#include "mediapipe/calculators/tensor/inference_io_mapper.h"
#include "mediapipe/calculators/tensor/inference_on_disk_cache_helper.h"
#include "mediapipe/calculators/tensor/tensor_span.h"
#include "mediapipe/framework/api2/packet.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/tensor.h"
#include "mediapipe/framework/mediapipe_profiling.h"
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

    std::shared_ptr<GlContext> initialization_gl_context_;
    std::unique_ptr<tflite::gpu::TFLiteGPURunner> tflite_gpu_runner_;

    std::vector<Tensor::Shape> output_shapes_;

    InferenceOnDiskCacheHelper on_disk_cache_helper_;

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
  const auto success =
      initialization_gl_context_->Run([this]() -> absl::Status {
        tflite_gpu_runner_.reset();
        return absl::OkStatus();
      });
  if (!success.ok()) {
    ABSL_LOG(DFATAL) << "Failed to close gpu inference runner: " << success;
  }
}

absl::Status InferenceCalculatorGlAdvancedImpl::GpuInferenceRunner::Init(
    CalculatorContext* cc, std::shared_ptr<GlContext> gl_context) {
  initialization_gl_context_ = gl_context;
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

  return initialization_gl_context_->Run(
      [this, &cc, &delegate]() -> absl::Status {
        return InitTFLiteGPURunner(cc, delegate);
      });
}

absl::StatusOr<std::vector<Tensor>>
InferenceCalculatorGlAdvancedImpl::GpuInferenceRunner::Process(
    CalculatorContext* cc, const TensorSpan& input_tensors) {
  std::vector<Tensor> output_tensors;
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
    MP_RETURN_IF_ERROR(tflite_gpu_runner_->Invoke());
  }
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
  std::vector<Tensor> output_tensors;
  MP_RETURN_IF_ERROR(gpu_helper_.RunInGlContext([&]() -> absl::Status {
    MP_ASSIGN_OR_RETURN(output_tensors,
                        gpu_inference_runner_->Process(cc, tensor_span));
    return absl::OkStatus();
  }));
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
