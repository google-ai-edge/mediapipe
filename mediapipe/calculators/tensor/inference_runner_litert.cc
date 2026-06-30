#include "mediapipe/calculators/tensor/inference_runner_litert.h"

#include <optional>

#if MEDIAPIPE_METAL_ENABLED
#import <Metal/Metal.h>

#include "mediapipe/gpu/MPPMetalHelper.h"
#endif  // MEDIAPIPE_METAL_ENABLED

#include <algorithm>
#include <cstdint>
#include <cstring>  // IWYU pragma: keep
#include <iterator>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/base/nullability.h"
#include "absl/log/absl_log.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"  // IWYU pragma: keep
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "litert/cc/internal/litert_extended_model.h"
#include "litert/cc/litert_buffer_ref.h"
#include "litert/cc/litert_common.h"
#include "litert/cc/litert_compiled_model.h"
#include "litert/cc/litert_environment.h"
#include "litert/cc/litert_layout.h"
#include "litert/cc/litert_macros.h"
#include "litert/cc/litert_opaque_options.h"
#include "litert/cc/litert_options.h"
#include "litert/cc/litert_ranked_tensor_type.h"
#include "litert/cc/litert_tensor_buffer.h"
#include "litert/cc/litert_tensor_buffer_requirements.h"
#include "litert/cc/litert_tensor_buffer_types.h"
#include "litert/cc/options/litert_cpu_options.h"
#include "litert/cc/options/litert_gpu_options.h"
#include "litert/cc/options/litert_qualcomm_options.h"
#include "mediapipe/calculators/tensor/inference_calculator.pb.h"
#include "mediapipe/calculators/tensor/inference_calculator_utils.h"
#include "mediapipe/calculators/tensor/inference_feedback_manager_litert.h"
#include "mediapipe/calculators/tensor/inference_io_mapper.h"
#include "mediapipe/calculators/tensor/litert/litert_utils.h"
#include "mediapipe/calculators/tensor/tensor_span.h"
#include "mediapipe/framework/api2/packet.h"
#include "mediapipe/framework/calculator_context.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/tensor.h"
#include "mediapipe/framework/memory_manager.h"
#include "mediapipe/framework/port.h"  // IWYU pragma: keep
#include "mediapipe/framework/port/ret_check.h"
#include "mediapipe/framework/port/status_macros.h"
#include "mediapipe/util/tflite/tflite_model_loader.h"
#include "mediapipe/util/tflite/tflite_signature_reader.h"

#if MEDIAPIPE_METAL_ENABLED
#include "mediapipe/framework/formats/tensor_mtl_buffer_view.h"
#include "tensorflow/lite/delegates/gpu/metal/metal_device.h"
#endif  // MEDIAPIPE_METAL_ENABLED
#if MEDIAPIPE_OPENGL_ES_VERSION >= MEDIAPIPE_OPENGL_ES_31
#include "mediapipe/gpu/gl_context.h"
#endif

#if MEDIAPIPE_TENSOR_USE_AHWB
#include "mediapipe/framework/formats/shared_fd.h"
#include "mediapipe/framework/formats/tensor_fd_finished_func.h"  // IWYU pragma: keep
#include "mediapipe/util/sync_wait.h"
#endif

#if defined(__EMSCRIPTEN__)
#include "mediapipe/gpu/webgpu/webgpu_service.h"
#endif  // __EMSCRIPTEN__

namespace mediapipe {

namespace {

// Preferred buffer types for input and output tensors.
// Note: Order indicates the priority of the buffer type, i.e. the order in
// which LiteRT's buffer requirements will be checked.
constexpr litert::TensorBufferType kPreferredBufferTypes[] = {
    litert::TensorBufferType::kWebGpuTexture,
#if MEDIAPIPE_TENSOR_USE_AHWB
    litert::TensorBufferType::kAhwb,
#endif  // MEDIAPIPE_TENSOR_USE_AHWB
    litert::TensorBufferType::kHostMemory,
    litert::TensorBufferType::kFastRpc,
    litert::TensorBufferType::kDmaBuf,
#if MEDIAPIPE_METAL_ENABLED
    litert::TensorBufferType::kMetalBufferPacked,
#endif  // MEDIAPIPE_METAL_ENABLED
    litert::TensorBufferType::kGlBuffer,
    litert::TensorBufferType::kOpenClTextureFp16,
    litert::TensorBufferType::kOpenClBuffer,
    litert::TensorBufferType::kOpenClBufferPacked,
    litert::TensorBufferType::kWebGpuBuffer,
    litert::TensorBufferType::kWebGpuBufferPacked,
};

// Buffer types that require copying to/from the MP tensor and are managed by
// LiteRT. Should be a subset of kPreferredBufferTypes.
constexpr litert::TensorBufferType kLiteRtManagedBufferTypes[] = {
    litert::TensorBufferType::kFastRpc,
    litert::TensorBufferType::kDmaBuf,
    litert::TensorBufferType::kOpenClBuffer,
    litert::TensorBufferType::kOpenClTextureFp16,
    litert::TensorBufferType::kOpenClBufferPacked,
    litert::TensorBufferType::kWebGpuBuffer,
    litert::TensorBufferType::kWebGpuBufferPacked,
};

// Returns the best supported LiteRT tensor buffer type.
absl::StatusOr<litert::TensorBufferType> ChooseBestBufferType(
    const litert::TensorBufferRequirements& buffer_requirements) {
  LITERT_MP_ASSIGN_OR_RETURN(const auto supported_types,
                             buffer_requirements.SupportedTypes());
  for (const auto& buffer_type : kPreferredBufferTypes) {
    if (absl::c_contains(supported_types, buffer_type)) {
      return buffer_type;
    }
  }
  std::vector<std::string> supported_types_str;
  absl::c_transform(supported_types, std::back_inserter(supported_types_str),
                    litert::BufferTypeToStringCC);
  std::vector<std::string> preferred_types_str;
  absl::c_transform(kPreferredBufferTypes,
                    std::back_inserter(preferred_types_str),
                    litert::BufferTypeToStringCC);
  return absl::InternalError(absl::StrCat(
      "LiteRT buffer requirements not supported. Supported types: ",
      absl::StrJoin(supported_types_str, ", "),
      ". Preferred types: ", absl::StrJoin(preferred_types_str, ", ")));
}

absl::StatusOr<litert::HwAcceleratorSet> GetLiteRtHwAccelerators(
    const InferenceCalculatorOptions::Delegate::LiteRt& options) {
  litert::HwAcceleratorSet accelerators(litert::HwAccelerators::kNone);
  if (options.has_cpu()) {
    accelerators |= litert::HwAccelerators::kCpu;
  }
  if (options.has_gpu()) {
    accelerators |= litert::HwAccelerators::kGpu;
  }
  if (options.has_npu()) {
    accelerators |= litert::HwAccelerators::kNpu;
  }
#if defined(__EMSCRIPTEN__)
  if (options.has_webnn()) {
    accelerators = accelerators | litert::HwAccelerators::kWebNn;
  }
#endif  // __EMSCRIPTEN__
  if (accelerators.value == static_cast<int>(litert::HwAccelerators::kNone)) {
    return absl::InvalidArgumentError("No accelerator is specified");
  }
  return accelerators;
}

absl::StatusOr<litert::Environment> InitializeLiteRtEnvironment(
    const InferenceCalculatorOptions::Delegate::LiteRt& options,
    const mediapipe::GlContext* gl_context
#if MEDIAPIPE_METAL_ENABLED
    ,
    void* metal_helper
#endif  // MEDIAPIPE_METAL_ENABLED
#if defined(__EMSCRIPTEN__)
    ,
    WebGpuService* webgpu_service
#endif  // __EMSCRIPTEN__
) {
  std::vector<litert::Environment::Option> environment_options;
#if MEDIAPIPE_METAL_ENABLED
  if (metal_helper) {
    MPPMetalHelper* helper = (__bridge MPPMetalHelper*)metal_helper;
    environment_options.push_back(litert::Environment::Option{
        /*.tag=*/litert::Environment::OptionTag::MetalCommandQueue,
        /*.value=*/(__bridge void*)helper.mtlCommandQueue,
    });
    environment_options.push_back(litert::Environment::Option{
        /*.tag=*/litert::Environment::OptionTag::MetalDevice,
        /*.value=*/(__bridge void*)helper.mtlDevice,
    });
  }
#endif  // MEDIAPIPE_METAL_ENABLED
  if (options.has_npu()) {
    if (options.npu().has_compiler_plugin_library_path()) {
      environment_options.push_back(litert::Environment::Option{
          /*.tag=*/litert::Environment::OptionTag::CompilerPluginLibraryDir,
          /*.value=*/options.npu().compiler_plugin_library_path().c_str(),
      });
    }
    if (options.npu().has_dispatch_library_path()) {
      environment_options.push_back(litert::Environment::Option{
          /*.tag=*/litert::Environment::OptionTag::DispatchLibraryDir,
          /*.value=*/options.npu().dispatch_library_path().c_str(),
      });
    }
  }

#if defined(__EMSCRIPTEN__)
  if (webgpu_service != nullptr) {
    const wgpu::Device& device = webgpu_service->device();
    environment_options.push_back(litert::Environment::Option{
        /*.tag=*/litert::Environment::OptionTag::WebGpuDevice,
        /*.value=*/reinterpret_cast<int64_t>(device.Get()),
    });
    environment_options.push_back(litert::Environment::Option{
        /*.tag=*/litert::Environment::OptionTag::WebGpuQueue,
        /*.value=*/reinterpret_cast<int64_t>(device.GetQueue().Get()),
    });
  }

#endif  // defined(__EMSCRIPTEN__)

#if MEDIAPIPE_OPENGL_ES_VERSION >= MEDIAPIPE_OPENGL_ES_31
  if (gl_context != nullptr) {
    // We expect EGL context and display to outlive the LiteRT environment.
    environment_options.push_back(litert::Environment::Option{
        /*.tag=*/litert::Environment::OptionTag::EglContext,
        /*.value=*/reinterpret_cast<int64_t>(gl_context->egl_context()),
    });
    environment_options.push_back(litert::Environment::Option{
        /*.tag=*/litert::Environment::OptionTag::EglDisplay,
        /*.value=*/reinterpret_cast<int64_t>(gl_context->egl_display()),
    });
  }
#endif  // MEDIAPIPE_OPENGL_ES_VERSION >= MEDIAPIPE_OPENGL_ES_31

  auto environment = litert::Environment::Create(environment_options);
  if (!environment) {
    return absl::InternalError(
        absl::StrCat("Failed to create LiteRt environment: ",
                     environment.Error().Message()));
  }
  return std::move(*environment);
}

// Returns true if the given MP tensor buffer is aligned to the given alignment.
bool IsTensorAligned(const Tensor& tensor, size_t alignment) {
  const auto cpu_read_view = tensor.GetCpuReadView();
  const void* ptr = cpu_read_view.buffer<void>();
  return reinterpret_cast<uintptr_t>(ptr) % alignment == 0;
}

// Copies the source MP tensor to the destination MP tensor.
absl::Status CopyMpTensorToMpTensor(const Tensor& src_tensor,
                                    Tensor& dst_tensor) {
  RET_CHECK_EQ(src_tensor.bytes(), dst_tensor.bytes())
      << "Source and destination tensors must have the same size";
  const auto src_tensor_view = src_tensor.GetCpuReadView();
  auto dst_tensor_view = dst_tensor.GetCpuWriteView();
  std::memcpy(dst_tensor_view.buffer<void>(), src_tensor_view.buffer<void>(),
              src_tensor.bytes());
  return absl::OkStatus();
}

}  // namespace

absl::StatusOr<std::unique_ptr<InferenceRunnerLiteRt>>
InferenceRunnerLiteRt::Create(
    api2::Packet<TfLiteModelPtr> model_packet,
    const InferenceCalculatorOptions::Delegate::LiteRt& options,
    const InferenceCalculatorOptions::
        InputOutputConfig* /*absl_nullable - not yet supported*/
            input_output_config,
    MemoryManager* memory_manager, const mediapipe::GlContext* gl_context,
#if defined(__EMSCRIPTEN__)
    WebGpuService* webgpu_service,
#endif  // __EMSCRIPTEN__
#if MEDIAPIPE_METAL_ENABLED
    void* metal_helper,
#endif  // MEDIAPIPE_METAL_ENABLED
    std::optional<litert::Options> litert_options) {
  MP_ASSIGN_OR_RETURN(auto environment,
                      InitializeLiteRtEnvironment(options, gl_context
#if MEDIAPIPE_METAL_ENABLED
                                                  ,
                                                  metal_helper
#endif  // MEDIAPIPE_METAL_ENABLED
#if defined(__EMSCRIPTEN__)
                                                  ,
                                                  webgpu_service
#endif  // __EMSCRIPTEN__
                                                  ));

  // Repackage the model into a litert::Model.
  const auto* model_allocation = (*model_packet)->allocation();
  LITERT_MP_ASSIGN_OR_RETURN(
      auto litert_model,
      litert::ExtendedModel::CreateFromBuffer(litert::BufferRef<uint8_t>(
          model_allocation->base(), model_allocation->bytes())));
  MP_ASSIGN_OR_RETURN(litert::HwAcceleratorSet accelerator,
                      GetLiteRtHwAccelerators(options));

  litert::Options jit_compilation_options;
  if (litert_options) {
    jit_compilation_options = *std::move(litert_options);
  } else {
    LITERT_MP_ASSIGN_OR_RETURN(jit_compilation_options,
                               litert::Options::Create());
  }

  LITERT_MP_RETURN_IF_ERROR(
      jit_compilation_options.SetHardwareAccelerators(accelerator).HasValue());

  if (accelerator & litert::HwAccelerators::kCpu) {
    LITERT_MP_ASSIGN_OR_RETURN(auto& cpu_options,
                               jit_compilation_options.GetCpuOptions());
    cpu_options.SetNumThreads(options.cpu().has_num_threads()
                                  ? options.cpu().num_threads()
                                  : GetCpuDefaultNumThreads());
    if (options.cpu().has_xnnpack() && options.cpu().xnnpack().has_flags()) {
      LITERT_MP_RETURN_IF_ERROR(
          cpu_options.SetXNNPackFlags(options.cpu().xnnpack().flags()));
    }
  }
  if (accelerator & litert::HwAccelerators::kGpu) {
    LITERT_MP_ASSIGN_OR_RETURN(auto& gpu_options,
                               jit_compilation_options.GetGpuOptions());
    if (options.gpu().has_precision()) {
      switch (options.gpu().precision()) {
        case InferenceCalculatorOptions::Delegate::LiteRt::Gpu::FP16:
          gpu_options.SetPrecision(litert::GpuOptions::Precision::kFp16);
          break;
        case InferenceCalculatorOptions::Delegate::LiteRt::Gpu::FP32:
          gpu_options.SetPrecision(litert::GpuOptions::Precision::kFp32);
          break;
        case InferenceCalculatorOptions::Delegate::LiteRt::Gpu::DEFAULT:
          gpu_options.SetPrecision(litert::GpuOptions::Precision::kDefault);
          break;
        default:
          return absl::InvalidArgumentError("Unsupported GPU precision");
      }
    }
    if (options.gpu().enable_constant_tensors_sharing()) {
      gpu_options.EnableConstantTensorSharing(true);
    }

    if (options.gpu().hint_fully_delegated_to_single_delegate()) {
      gpu_options.SetHintFullyDelegatedToSingleDelegate(true);
    }

    if (options.gpu().has_backend()) {
      switch (options.gpu().backend()) {
        case InferenceCalculatorOptions::Delegate::LiteRt::Gpu::OPENGL:
          gpu_options.SetBackend(litert::GpuOptions::Backend::kOpenGl);
          break;
        case InferenceCalculatorOptions::Delegate::LiteRt::Gpu::OPENCL:
          gpu_options.SetBackend(litert::GpuOptions::Backend::kOpenCl);
          break;
        case InferenceCalculatorOptions::Delegate::LiteRt::Gpu::AUTOMATIC:
          gpu_options.SetBackend(litert::GpuOptions::Backend::kAutomatic);
          break;
        case InferenceCalculatorOptions::Delegate::LiteRt::Gpu::WEBGPU:
          gpu_options.SetBackend(litert::GpuOptions::Backend::kWebGpu);
          break;
        default:
          return absl::InvalidArgumentError("Unsupported GPU backend");
      }
    }
    if (options.gpu().has_wait_type()) {
      switch (options.gpu().wait_type()) {
        case InferenceCalculatorOptions::Delegate::LiteRt::Gpu::
            WAIT_TYPE_PASSIVE:
          gpu_options.SetSyncExecutionModeWaitType(
              litert::GpuOptions::SyncExecutionModeWaitType::kPassive);
          break;
        case InferenceCalculatorOptions::Delegate::LiteRt::Gpu::
            WAIT_TYPE_ACTIVE:
          gpu_options.SetSyncExecutionModeWaitType(
              litert::GpuOptions::SyncExecutionModeWaitType::kActive);
          break;
        case InferenceCalculatorOptions::Delegate::LiteRt::Gpu::
            WAIT_TYPE_DO_NOT_WAIT:
          gpu_options.SetSyncExecutionModeWaitType(
              litert::GpuOptions::SyncExecutionModeWaitType::kDoNotWait);
          break;
        case InferenceCalculatorOptions::Delegate::LiteRt::Gpu::
            WAIT_TYPE_UNSPECIFIED:
          gpu_options.SetSyncExecutionModeWaitType(
              litert::GpuOptions::SyncExecutionModeWaitType::kDefault);
          break;
      }
    }
    if (options.gpu().has_serialize_program_cache()) {
      LITERT_MP_RETURN_IF_ERROR(gpu_options.SetSerializeProgramCache(
          options.gpu().serialize_program_cache()));
    }
    if (options.gpu().has_serialization_dir()) {
      LITERT_MP_RETURN_IF_ERROR(gpu_options.SetSerializationDir(
          options.gpu().serialization_dir().c_str()));
    }
    if (options.gpu().has_model_cache_key()) {
      LITERT_MP_RETURN_IF_ERROR(gpu_options.SetModelCacheKey(
          options.gpu().model_cache_key().c_str()));
    }
  }
  if (accelerator & litert::HwAccelerators::kNpu) {
    if (options.npu().has_darwinn()) {
    } else if (options.npu().has_qualcomm()) {
      LITERT_MP_ASSIGN_OR_RETURN(auto& qualcomm_options,
                                 jit_compilation_options.GetQualcommOptions());
      if (options.npu().qualcomm().has_htp_performance_mode()) {
        using QualcommDelegateOptions =
            InferenceCalculatorOptions::Delegate::LiteRt::Npu::Qualcomm;
        litert::qualcomm::QualcommOptions::HtpPerformanceMode mode;
        switch (options.npu().qualcomm().htp_performance_mode()) {
          case QualcommDelegateOptions::DEFAULT:
            mode =
                litert::qualcomm::QualcommOptions::HtpPerformanceMode::kDefault;
            break;
          case QualcommDelegateOptions::SUSTAINED_HIGH_PERFORMANCE:
            mode = litert::qualcomm::QualcommOptions::HtpPerformanceMode::
                kSustainedHighPerformance;
            break;
          case QualcommDelegateOptions::BURST:
            mode =
                litert::qualcomm::QualcommOptions::HtpPerformanceMode::kBurst;
            break;
          case QualcommDelegateOptions::HIGH_PERFORMANCE:
            mode = litert::qualcomm::QualcommOptions::HtpPerformanceMode::
                kHighPerformance;
            break;
          case QualcommDelegateOptions::POWER_SAVER:
            mode = litert::qualcomm::QualcommOptions::HtpPerformanceMode::
                kPowerSaver;
            break;
          case QualcommDelegateOptions::LOW_POWER_SAVER:
            mode = litert::qualcomm::QualcommOptions::HtpPerformanceMode::
                kLowPowerSaver;
            break;
          case QualcommDelegateOptions::HIGH_POWER_SAVER:
            mode = litert::qualcomm::QualcommOptions::HtpPerformanceMode::
                kHighPowerSaver;
            break;
          case QualcommDelegateOptions::LOW_BALANCED:
            mode = litert::qualcomm::QualcommOptions::HtpPerformanceMode::
                kLowBalanced;
            break;
          case QualcommDelegateOptions::BALANCED:
            mode = litert::qualcomm::QualcommOptions::HtpPerformanceMode::
                kBalanced;
            break;
          case QualcommDelegateOptions::EXTREME_POWER_SAVER:
            mode = litert::qualcomm::QualcommOptions::HtpPerformanceMode::
                kExtremePowerSaver;
            break;
          default:
            return absl::InvalidArgumentError(
                "Unsupported Qualcomm HTP performance mode");
        }
        qualcomm_options.SetHtpPerformanceMode(mode);
      }
    }
  }

  LITERT_MP_ASSIGN_OR_RETURN(
      auto compiled_model,
      litert::CompiledModel::Create(environment, litert_model.Get(),
                                    jit_compilation_options));
  LITERT_MP_ASSIGN_OR_RETURN(auto signatures, litert_model.GetSignatures());
  const int num_signatures = signatures.size();
  RET_CHECK_GT(num_signatures, 0) << "Model must have at least one signature";

  LITERT_MP_ASSIGN_OR_RETURN(bool is_fully_accelerated,
                             compiled_model.IsFullyAccelerated());
  bool release_model_packet = false;
  // Release the model packet if the model is fully accelerated and the
  // accelerator is GPU. This is aiming to reduce the peak memory usage.
  if (is_fully_accelerated && (accelerator & litert::HwAccelerators::kGpu)) {
    release_model_packet = true;
  }
  int signature_index = 0;
  if (num_signatures > 1) {
    // Find default signature for models with multiple signatures.
    auto default_signature_itr =
        absl::c_find_if(signatures, [](const auto& signature) {
          return signature.Key() == litert::Model::DefaultSignatureKey();
        });
    RET_CHECK(default_signature_itr != signatures.end())
        << "Model must have default signature key";
    signature_index = std::distance(signatures.begin(), default_signature_itr);
  }
  LITERT_MP_ASSIGN_OR_RETURN(
      auto subgraph,
      litert_model.Subgraph(signatures.at(signature_index).Key()));

  bool run_async = options.run_async();
  bool enable_dynamic_resize = options.enable_dynamic_resize();

  InputOutputTensorNames input_output_tensor_names =
      CreateInputOutputTensorNames(signatures, signature_index);
  std::unique_ptr<InferenceFeedbackManagerLiteRt> feedback_manager;
  if (input_output_config) {
    // The feedback manager is only using the LiteRT objects (subgraph, model,
    // compiled_model) during initialization, so it's safe to move them after
    // the feedback manager is created.
    feedback_manager = std::make_unique<InferenceFeedbackManagerLiteRt>();
    MP_RETURN_IF_ERROR(feedback_manager->Init(
        *input_output_config, input_output_tensor_names, &subgraph,
        &litert_model, &compiled_model, signature_index));
  }

  auto runner =
      std::unique_ptr<InferenceRunnerLiteRt>(new InferenceRunnerLiteRt(
          memory_manager,
#if defined(__EMSCRIPTEN__)
          webgpu_service,
#endif  // __EMSCRIPTEN__
          std::move(model_packet),
          std::make_unique<litert::Environment>(std::move(environment)),
          std::make_unique<litert::Model>(std::move(litert_model)),
          std::make_unique<litert::CompiledModel>(std::move(compiled_model)),
          run_async, enable_dynamic_resize, release_model_packet,
          std::make_unique<litert::Subgraph>(std::move(subgraph)),
          std::make_unique<std::vector<litert::Signature>>(
              std::move(signatures)),
          signature_index, std::move(feedback_manager)
#if MEDIAPIPE_METAL_ENABLED
                               ,
          metal_helper
#endif  // MEDIAPIPE_METAL_ENABLED
          ));

  std::vector<litert::TensorBufferRequirements> reqs;
  size_t num_inputs = runner->subgraph_->Inputs().size();
  reqs.reserve(num_inputs);
  for (int i = 0; i < num_inputs; ++i) {
    LITERT_MP_ASSIGN_OR_RETURN(
        auto req, runner->compiled_model_->GetInputBufferRequirements(
                      runner->signature_index_, i));
    reqs.push_back(std::move(req));
  }
  runner->cached_input_buffer_requirements_ = std::move(reqs);

  if (release_model_packet) {
    runner->ClearModelPacket();
  }

  return runner;
}

InferenceRunnerLiteRt::InferenceRunnerLiteRt(
    MemoryManager* memory_manager,
#if defined(__EMSCRIPTEN__)
    WebGpuService* webgpu_service,
#endif  // __EMSCRIPTEN__
    api2::Packet<TfLiteModelPtr> model_packet,
    std::unique_ptr<litert::Environment> environment,
    std::unique_ptr<litert::Model> model,
    std::unique_ptr<litert::CompiledModel> compiled_model, bool run_async,
    bool enable_dynamic_resize, bool release_model_packet,
    std::unique_ptr<litert::Subgraph> subgraph,
    std::unique_ptr<std::vector<litert::Signature>> signatures,
    int signature_index,
    std::unique_ptr<InferenceFeedbackManagerLiteRt> feedback_manager
#if MEDIAPIPE_METAL_ENABLED
    ,
    void* metal_helper
#endif  // MEDIAPIPE_METAL_ENABLED
    )
    : memory_manager_(memory_manager),
#if defined(__EMSCRIPTEN__)
      webgpu_service_(webgpu_service),
#endif  // __EMSCRIPTEN__
      model_packet_(std::move(model_packet)),
      environment_(std::move(environment)),
      model_(std::move(model)),
      compiled_model_(std::move(compiled_model)),
      run_async_(run_async),
      subgraph_(std::move(subgraph)),
      signatures_(std::move(signatures)),
      signature_index_(signature_index),
      enable_dynamic_resize_(enable_dynamic_resize),
      input_output_tensor_names_(
          CreateInputOutputTensorNames(*signatures_, signature_index)),
      feedback_manager_(std::move(feedback_manager)),
      managed_input_buffers_(subgraph_->Inputs().size()),
      managed_output_buffers_(subgraph_->Outputs().size()) {
#if MEDIAPIPE_METAL_ENABLED
  metal_helper_ = metal_helper;
#endif  // MEDIAPIPE_METAL_ENABLED
}

InferenceRunnerLiteRt::~InferenceRunnerLiteRt() {
#if MEDIAPIPE_METAL_ENABLED
  if (metal_command_buffer_) {
    id<MTLCommandBuffer> command_buffer =
        (__bridge_transfer id<MTLCommandBuffer>)metal_command_buffer_;
    (void)command_buffer;
    metal_command_buffer_ = nullptr;
  }
#endif  // MEDIAPIPE_METAL_ENABLED
  absl::Status status = Close();
  if (!status.ok()) {
    ABSL_LOG(DFATAL) << "Failed to shut down InferenceRunnerLiteRt: " << status;
  }
}

absl::Status InferenceRunnerLiteRt::Close() {
#if MEDIAPIPE_TENSOR_USE_AHWB
  MP_RETURN_IF_ERROR(CleanAsyncRunStates(/*wait_for_all=*/true));
#endif  // MEDIAPIPE_TENSOR_USE_AHWB
  return absl::OkStatus();
}

// Creates a ranked tensor type for the given input tensor.
absl::StatusOr<litert::RankedTensorType>
InferenceRunnerLiteRt::CreateInputRankedTensorType(
    const litert::Tensor& model_tensor, const Tensor& mp_input_tensor,
    int tensor_index) {
  LITERT_MP_ASSIGN_OR_RETURN(litert::RankedTensorType tensor_type,
                             model_tensor.RankedTensorType());

  absl::Span<const int> litert_shape = tensor_type.Layout().Dimensions();

  // Check if dynamic resizing is enabled and the tensor has a dynamic
  // dimension.
  if (enable_dynamic_resize_ && absl::c_linear_search(litert_shape, -1)) {
    has_dynamic_dimension_ = true;
    // Check if shapes are compatible first (allowing for dynamic dimensions)
    MP_RETURN_IF_ERROR(AreTensorSpecsCompatible(tensor_type, mp_input_tensor));

    // Shapes are compatible, try resize
    const auto& mp_dims = mp_input_tensor.shape().dims;
    VLOG(1) << "Resizing input tensor " << tensor_index << " from "
            << absl::StrJoin(litert_shape, "x") << " to "
            << absl::StrJoin(mp_dims, "x");

    LITERT_MP_RETURN_IF_ERROR(compiled_model_->ResizeInputTensor(
        signature_index_, tensor_index, mp_dims));

    return litert::RankedTensorType(
        tensor_type.ElementType(),
        litert::Layout(litert::BuildLayout(mp_dims)));
  } else {
    MP_RETURN_IF_ERROR(AreTensorSpecsEqual(tensor_type, mp_input_tensor));
  }
  return tensor_type;
}

// Creates a ranked tensor type for the given output tensor.
absl::StatusOr<litert::RankedTensorType>
InferenceRunnerLiteRt::CreateOutputRankedTensorType(
    const litert::Tensor& tensor, litert::Layout layout) {
  LITERT_MP_ASSIGN_OR_RETURN(const litert::RankedTensorType ranked_tensor_type,
                             tensor.RankedTensorType());
  if (!has_dynamic_dimension_) {
    return ranked_tensor_type;
  }

  return litert::RankedTensorType(ranked_tensor_type.ElementType(),
                                  std::move(layout));
}

InputOutputTensorNames InferenceRunnerLiteRt::CreateInputOutputTensorNames(
    const std::vector<litert::Signature>& signatures, int signature_index) {
  InputOutputTensorNames input_output_tensor_names;
  for (const auto& signature : signatures) {
    SignatureInputOutputTensorNames signature_input_output_tensor_names;
    for (const auto& tensor_name : signature.InputNames()) {
      signature_input_output_tensor_names.input_tensor_names.emplace_back(
          tensor_name);
    }
    for (const auto& tensor_name : signature.OutputNames()) {
      signature_input_output_tensor_names.output_tensor_names.emplace_back(
          tensor_name);
    }
    input_output_tensor_names[signature.Key()] =
        std::move(signature_input_output_tensor_names);
  }
  return input_output_tensor_names;
}

#if MEDIAPIPE_OPENGL_ES_VERSION >= MEDIAPIPE_OPENGL_ES_31
absl::StatusOr<litert::TensorBuffer>
InferenceRunnerLiteRt::CreateGlInputTensorBufferFromMpTensor(
    const Tensor& mp_input_tensor, const litert::RankedTensorType& tensor_type,
    std::vector<InferenceRunnerLiteRt::MpTensorReadView>&
        mp_input_tensor_views) {
  auto input_tensor_view = mp_input_tensor.GetOpenGlBufferReadView();
  LITERT_MP_ASSIGN_OR_RETURN(
      auto litert_buffer,
      litert::TensorBuffer::CreateFromGlBuffer(
          *environment_, tensor_type, GL_SHADER_STORAGE_BUFFER,
          input_tensor_view.name(), mp_input_tensor.bytes(), /*offset=*/0));
  mp_input_tensor_views.push_back(std::move(input_tensor_view));
  return litert_buffer;
}

absl::StatusOr<litert::TensorBuffer>
InferenceRunnerLiteRt::CreateGlOutputTensorBufferFromMpTensor(
    const Tensor& mp_output_tensor, const litert::RankedTensorType& tensor_type,
    std::vector<InferenceRunnerLiteRt::MpTensorWriteView>&
        mp_output_tensor_views) {
  auto output_tensor_view = mp_output_tensor.GetOpenGlBufferWriteView();
  LITERT_MP_ASSIGN_OR_RETURN(
      auto litert_buffer,
      litert::TensorBuffer::CreateFromGlBuffer(
          *environment_, tensor_type, GL_SHADER_STORAGE_BUFFER,
          output_tensor_view.name(), mp_output_tensor.bytes(), /*offset=*/0));
  mp_output_tensor_views.push_back(std::move(output_tensor_view));
  return litert_buffer;
}
#endif

#if defined(__EMSCRIPTEN__)
absl::StatusOr<litert::TensorBuffer>
InferenceRunnerLiteRt::CreateWebGpuInputTensorBufferFromMpTensor(
    const Tensor& mp_input_tensor, const litert::RankedTensorType& tensor_type,
    std::vector<InferenceRunnerLiteRt::MpTensorReadView>&
        mp_input_tensor_views) {
  auto input_tensor_view =
      mp_input_tensor.GetWebGpuTexture2dReadView(*webgpu_service_);
  LITERT_MP_ASSIGN_OR_RETURN(
      auto litert_buffer,
      litert::TensorBuffer::CreateFromWebGpuTexture(
          *environment_, tensor_type, input_tensor_view.name().Get(),
          mp_input_tensor.bytes()));
  mp_input_tensor_views.push_back(std::move(input_tensor_view));
  return litert_buffer;
}

absl::StatusOr<litert::TensorBuffer>
InferenceRunnerLiteRt::CreateWebGpuOutputTensorBufferFromMpTensor(
    const Tensor& mp_output_tensor, const litert::RankedTensorType& tensor_type,
    std::vector<InferenceRunnerLiteRt::MpTensorWriteView>&
        mp_output_tensor_views) {
  auto output_tensor_view =
      mp_output_tensor.GetWebGpuTexture2dWriteView(*webgpu_service_);
  LITERT_MP_ASSIGN_OR_RETURN(
      auto litert_buffer,
      litert::TensorBuffer::CreateFromWebGpuTexture(
          *environment_, tensor_type, output_tensor_view.name().Get(),
          mp_output_tensor.bytes()));
  mp_output_tensor_views.push_back(std::move(output_tensor_view));
  return litert_buffer;
}
#endif  // defined(__EMSCRIPTEN__)

#if MEDIAPIPE_METAL_ENABLED

id<MTLCommandBuffer> InferenceRunnerLiteRt::GetMetalCommandBuffer() {
  id<MTLCommandBuffer> command_buffer =
      (__bridge id<MTLCommandBuffer>)metal_command_buffer_;
  if (command_buffer) {
    return command_buffer;
  }

  // Get command buffer from the helper.
  MPPMetalHelper* helper = (__bridge MPPMetalHelper*)metal_helper_;
  if (helper) {
    command_buffer = [helper commandBuffer];
    metal_command_buffer_ = (__bridge_retained void*)command_buffer;
  }
  return command_buffer;
}

absl::StatusOr<const Tensor*>
InferenceRunnerLiteRt::ConvertTensorChannelsIfNeeded(
    const Tensor& mp_input_tensor, const litert::RankedTensorType& tensor_type,
    InferenceRunContext& ctx) {
  auto model_dims = tensor_type.Layout().Dimensions();
  const auto& mp_dims = mp_input_tensor.shape().dims;

  int model_h = 1, model_w = 1, model_c = 1, model_n = 1;
  if (model_dims.size() == 4) {
    model_n = model_dims[0];
    model_h = model_dims[1];
    model_w = model_dims[2];
    model_c = model_dims[3];
  } else if (model_dims.size() == 3) {
    model_h = model_dims[0];
    model_w = model_dims[1];
    model_c = model_dims[2];
  }

  int mp_h = 1, mp_w = 1, mp_c = 1, mp_n = 1;
  if (mp_dims.size() == 4) {
    mp_n = mp_dims[0];
    mp_h = mp_dims[1];
    mp_w = mp_dims[2];
    mp_c = mp_dims[3];
  } else if (mp_dims.size() == 3) {
    mp_h = mp_dims[0];
    mp_w = mp_dims[1];
    mp_c = mp_dims[2];
  }

  if (model_c != mp_c && model_h == mp_h && model_w == mp_w) {
    auto converted_tensor = std::make_unique<Tensor>(
        mp_input_tensor.element_type(),
        Tensor::Shape{mp_n, mp_h, mp_w, model_c}, memory_manager_);

    auto src_view = mp_input_tensor.GetCpuReadView();
    auto dst_view = converted_tensor->GetCpuWriteView();

    int num_pixels = mp_n * mp_h * mp_w;
    size_t element_size = mp_input_tensor.element_size();
    size_t pixel_size_src = mp_c * element_size;
    size_t pixel_size_dst = model_c * element_size;
    const uint8_t* src_ptr =
        reinterpret_cast<const uint8_t*>(src_view.buffer<void>());
    uint8_t* dst_ptr = reinterpret_cast<uint8_t*>(dst_view.buffer<void>());

    size_t bytes_to_copy = std::min(mp_c, model_c) * element_size;

    for (int p = 0; p < num_pixels; ++p) {
      // Copy matching channels
      std::memcpy(dst_ptr + p * pixel_size_dst, src_ptr + p * pixel_size_src,
                  bytes_to_copy);

      // Pad with zeros if model expects more channels
      if (model_c > mp_c) {
        std::memset(dst_ptr + p * pixel_size_dst + bytes_to_copy, 0,
                    (model_c - mp_c) * element_size);
      }
    }

    const Tensor* result = converted_tensor.get();
    ctx.temporary_tensors.push_back(std::move(converted_tensor));
    return result;
  }

  return &mp_input_tensor;
}

absl::StatusOr<litert::TensorBuffer>
InferenceRunnerLiteRt::CreateMetalInputTensorBufferFromMpTensor(
    const Tensor& mp_input_tensor, const litert::RankedTensorType& tensor_type,
    InferenceRunContext& ctx) {
  id<MTLCommandBuffer> command_buffer = GetMetalCommandBuffer();
  if (!command_buffer) {
    return absl::InternalError("Failed to create metal command buffer");
  }

  const Tensor* tensor_to_use = &mp_input_tensor;

  auto mtl_input_view =
      MtlBufferView::GetReadView(*tensor_to_use, command_buffer);

  LITERT_MP_ASSIGN_OR_RETURN(
      auto litert_buffer,
      litert::TensorBuffer::CreateFromMetalBuffer(
          *environment_, tensor_type,
          litert::TensorBufferType::kMetalBufferPacked,
          (__bridge void*)mtl_input_view.buffer(), tensor_to_use->bytes()));

  ctx.active_input_views.push_back(std::move(mtl_input_view));
  return litert_buffer;
}

absl::StatusOr<litert::TensorBuffer>
InferenceRunnerLiteRt::CreateMetalOutputTensorBufferFromMpTensor(
    const Tensor& mp_output_tensor, const litert::RankedTensorType& tensor_type,
    std::vector<InferenceRunnerLiteRt::MpTensorWriteView>&
        mp_output_tensor_views) {
  id<MTLCommandBuffer> command_buffer = GetMetalCommandBuffer();
  if (!command_buffer) {
    return absl::InternalError("Failed to create metal command buffer");
  }
  auto mtl_output_view =
      MtlBufferView::GetWriteView(mp_output_tensor, command_buffer);

  LITERT_MP_ASSIGN_OR_RETURN(
      auto litert_buffer,
      litert::TensorBuffer::CreateFromMetalBuffer(
          *environment_, tensor_type,
          litert::TensorBufferType::kMetalBufferPacked,
          (__bridge void*)mtl_output_view.buffer(), mp_output_tensor.bytes()));
  mp_output_tensor_views.push_back(std::move(mtl_output_view));
  return litert_buffer;
}
#endif  // MEDIAPIPE_METAL_ENABLED

#if MEDIAPIPE_TENSOR_USE_AHWB

absl::StatusOr<int> DupFd(int fd) {
  RET_CHECK_GE(fd, 0) << "Invalid fd: " << fd;
  int dup_fd = -1;
  if (dup_fd = dup(fd); dup_fd < 0) {
    return absl::InternalError(
        absl::StrCat("Failed to duplicate fd: ", fd, ": ", strerror(errno)));
  }
  return dup_fd;
}

absl::StatusOr<litert::TensorBuffer>
InferenceRunnerLiteRt::CreateAhwbInputTensorBufferFromMpTensor(
    const Tensor& mp_input_tensor, const litert::RankedTensorType& tensor_type,
    std::vector<InferenceRunnerLiteRt::MpTensorReadView>&
        mp_input_tensor_views) {
  auto input_tensor_view = mp_input_tensor.GetAHardwareBufferReadView();

  AHardwareBuffer* ahwb = input_tensor_view.handle();
  LITERT_MP_ASSIGN_OR_RETURN(
      auto litert_buffer,
      litert::TensorBuffer::CreateFromAhwb(*environment_, tensor_type, ahwb,
                                           /*ahwb_offset=*/0));

  const int write_complete_fence_fd =
      input_tensor_view.GetWriteCompleteFenceFd();
  if (write_complete_fence_fd != -1) {
    // If the write complete fence fd is valid, then we need to set it on the
    // LiteRt buffer.
    MP_ASSIGN_OR_RETURN(int dup_fd, DupFd(write_complete_fence_fd));
    LITERT_MP_ASSIGN_OR_RETURN(
        auto event, litert::Event::CreateFromSyncFenceFd(*environment_, dup_fd,
                                                         /*owns_fd=*/true));
    LITERT_MP_RETURN_IF_ERROR(litert_buffer.SetEvent(std::move(event)));
  }
  mp_input_tensor_views.push_back(std::move(input_tensor_view));
  return litert_buffer;
}

absl::StatusOr<litert::TensorBuffer>
InferenceRunnerLiteRt::CreateAhwbOutputTensorBufferFromMpTensor(
    const Tensor& mp_output_tensor, const litert::RankedTensorType& tensor_type,
    std::vector<InferenceRunnerLiteRt::MpTensorWriteView>&
        mp_output_tensor_views) {
  auto output_tensor_view = mp_output_tensor.GetAHardwareBufferWriteView();
  AHardwareBuffer* ahwb = output_tensor_view.handle();
  LITERT_MP_ASSIGN_OR_RETURN(
      auto output_buffer,
      litert::TensorBuffer::CreateFromAhwb(*environment_, tensor_type, ahwb,
                                           /*ahwb_offset=*/0));
  mp_output_tensor_views.push_back(std::move(output_tensor_view));
  return output_buffer;
}

#endif  // MEDIAPIPE_TENSOR_USE_AHWB

absl::StatusOr<litert::TensorBuffer>
InferenceRunnerLiteRt::CreateCpuInputTensorBufferFromMpTensor(
    const Tensor& tensor, const litert::RankedTensorType& tensor_type,
    std::vector<InferenceRunnerLiteRt::MpTensorReadView>& mp_input_tensor_views,
    std::vector<std::unique_ptr<Tensor>>& mp_aligned_input_tensors) {
  if (IsTensorAligned(tensor, LITERT_HOST_MEMORY_BUFFER_ALIGNMENT)) {
    // Alignment requirement met, create with zero-copy.
    auto input_tensor_view = tensor.GetCpuReadView();
    const void* input_tensor_data = input_tensor_view.buffer<void>();
    LITERT_MP_ASSIGN_OR_RETURN(
        auto litert_buffer,
        litert::TensorBuffer::CreateFromHostMemory(
            *environment_, tensor_type, const_cast<void*>(input_tensor_data),
            tensor.bytes()));
    mp_input_tensor_views.push_back(std::move(input_tensor_view));
    return litert_buffer;
  }
  // Alignment requirement not met, create a new aligned MP tensor.
  ABSL_LOG_FIRST_N(WARNING, 1)
      << "Input tensor memory is not aligned according to LiteRt's "
         "memory "
         "alignment requirements. Reallocating memory and copying data.";
  MP_ASSIGN_OR_RETURN(
      Tensor aligned_mp_input_tensor,
      CreateTensorFromLiteRtRankedTensorType(
          tensor_type, memory_manager_, LITERT_HOST_MEMORY_BUFFER_ALIGNMENT));
  auto aligned_mp_input_tensor_ptr =
      std::make_unique<Tensor>(std::move(aligned_mp_input_tensor));
  MP_RETURN_IF_ERROR(
      CopyMpTensorToMpTensor(tensor, *aligned_mp_input_tensor_ptr));
  auto input_tensor_view = aligned_mp_input_tensor_ptr->GetCpuReadView();
  const void* input_tensor_data = input_tensor_view.buffer<void>();
  LITERT_MP_ASSIGN_OR_RETURN(
      auto litert_buffer,
      litert::TensorBuffer::CreateFromHostMemory(
          *environment_, tensor_type, const_cast<void*>(input_tensor_data),
          aligned_mp_input_tensor_ptr->bytes()));
  mp_aligned_input_tensors.push_back(std::move(aligned_mp_input_tensor_ptr));
  mp_input_tensor_views.push_back(std::move(input_tensor_view));
  return litert_buffer;
}

absl::StatusOr<litert::TensorBuffer>
InferenceRunnerLiteRt::CreateCpuOutputTensorBufferFromMpTensor(
    const Tensor& mp_output_tensor, const litert::RankedTensorType& tensor_type,
    std::vector<InferenceRunnerLiteRt::MpTensorWriteView>&
        mp_output_tensor_views) {
  const int output_tensor_size_bytes = mp_output_tensor.bytes();
  auto output_tensor_view = mp_output_tensor.GetCpuWriteView();
  void* output_tensor_data = output_tensor_view.buffer<void>();
  LITERT_MP_ASSIGN_OR_RETURN(auto output_buffer,
                             litert::TensorBuffer::CreateFromHostMemory(
                                 *environment_, tensor_type, output_tensor_data,
                                 output_tensor_size_bytes));
  mp_output_tensor_views.push_back(std::move(output_tensor_view));
  return output_buffer;
}

absl::StatusOr<litert::TensorBuffer>
InferenceRunnerLiteRt::CreateInputTensorBufferFromMpTensor(
    litert::TensorBufferType buffer_type, const Tensor& tensor,
    const litert::RankedTensorType& tensor_type, InferenceRunContext& ctx) {
  switch (buffer_type) {
    case litert::TensorBufferType::kHostMemory:
      return CreateCpuInputTensorBufferFromMpTensor(
          tensor, tensor_type, ctx.active_input_views, ctx.temporary_tensors);
#if MEDIAPIPE_TENSOR_USE_AHWB
    case litert::TensorBufferType::kAhwb:
      return CreateAhwbInputTensorBufferFromMpTensor(tensor, tensor_type,
                                                     ctx.active_input_views);
#endif  // MEDIAPIPE_TENSOR_USE_AHWB
#if MEDIAPIPE_METAL_ENABLED
    case litert::TensorBufferType::kMetalBufferPacked:
      return CreateMetalInputTensorBufferFromMpTensor(tensor, tensor_type, ctx);
#endif  // MEDIAPIPE_METAL_ENABLED
#if MEDIAPIPE_OPENGL_ES_VERSION >= MEDIAPIPE_OPENGL_ES_31
    case litert::TensorBufferType::kGlBuffer:
      return CreateGlInputTensorBufferFromMpTensor(tensor, tensor_type,
                                                   ctx.active_input_views);
#endif  // MEDIAPIPE_OPENGL_ES_VERSION >= MEDIAPIPE_OPENGL_ES_31
#if defined(__EMSCRIPTEN__)
    case litert::TensorBufferType::kWebGpuTexture:
      if (webgpu_service_ != nullptr) {
        return CreateWebGpuInputTensorBufferFromMpTensor(
            tensor, tensor_type, ctx.active_input_views);
      } else {
        return absl::InvalidArgumentError(absl::StrCat(
            "Unsupported buffer type to create input tensor: ", buffer_type));
      }
#endif  // defined(__EMSCRIPTEN__)
    default:
      return absl::InvalidArgumentError(absl::StrCat(
          "Unsupported buffer type to create input tensor: ", buffer_type));
  }
}

absl::StatusOr<bool> InferenceRunnerLiteRt::NeedsReallocation(
    const std::optional<litert::TensorBuffer>& managed_buffer,
    size_t buffer_size, const litert::RankedTensorType& tensor_type) const {
  if (!managed_buffer.has_value()) return true;

  LITERT_MP_ASSIGN_OR_RETURN(size_t cache_size, managed_buffer->Size());
  if (cache_size != buffer_size) return true;

  LITERT_MP_ASSIGN_OR_RETURN(auto cached_tensor_type,
                             managed_buffer->TensorType());
  return !absl::c_equal(cached_tensor_type.Layout().Dimensions(),
                        tensor_type.Layout().Dimensions());
}

absl::StatusOr<bool> InferenceRunnerLiteRt::TryAppendFeedbackBuffer(
    bool is_input, int tensor_index, absl::string_view tensor_name,
    std::vector<litert::TensorBuffer>& litert_buffers) const {
  if (!feedback_manager_) {
    return false;
  }
  bool is_feedback_tensor =
      is_input ? feedback_manager_->IsFeedbackInputTensorAtIndex(tensor_index)
               : feedback_manager_->IsFeedbackOutputTensorAtIndex(tensor_index);
  if (!is_feedback_tensor) {
    return false;
  }
  if (!feedback_manager_->Contains(tensor_name)) {
    return absl::InternalError(absl::StrCat(
        "Feedback tensor buffer not found for tensor: ", tensor_name));
  }
  // TODO: Perform lookups based on tensor index, not name.
  LITERT_MP_ASSIGN_OR_RETURN(
      const litert::TensorBuffer& feedback_buffer,
      feedback_manager_->GetFeedbackTensorBuffer(tensor_name));
  LITERT_MP_ASSIGN_OR_RETURN(litert::TensorBuffer buffer_dup,
                             feedback_buffer.Duplicate());
  litert_buffers.push_back(std::move(buffer_dup));
  return true;
}

absl::Status InferenceRunnerLiteRt::PrepareInputBuffers(
    const TensorSpan& tensor_span,
    const litert::SubgraphInputs& model_input_tensors,
    InferenceRunContext& ctx) {
  ctx.litert_inputs.reserve(model_input_tensors.size());
  LITERT_MP_ASSIGN_OR_RETURN(
      std::vector<absl::string_view> model_input_tensor_names,
      model_->GetSignatureInputNames(signature_index_));

  // When feedback tensors are present then the model has more inputs than MP
  // has tensors, so we use a separate counter to track which MP tensor maps to
  // which model input.
  int mp_tensor_index = 0;
  for (int i = 0; i < model_input_tensors.size(); ++i) {
    MP_ASSIGN_OR_RETURN(
        bool is_feedback_tensor,
        TryAppendFeedbackBuffer(true, i, model_input_tensor_names[i],
                                ctx.litert_inputs));
    if (is_feedback_tensor) continue;
    // We have a MP tensor for this model input tensor.
    const Tensor& mp_input_tensor = tensor_span[mp_tensor_index++];
    const Tensor* tensor_to_use = &mp_input_tensor;

    const auto& input_buffer_requirements =
        cached_input_buffer_requirements_[i];
    MP_ASSIGN_OR_RETURN(litert::TensorBufferType buffer_type,
                        ChooseBestBufferType(input_buffer_requirements));

#if MEDIAPIPE_METAL_ENABLED
    if (buffer_type == litert::TensorBufferType::kMetalBufferPacked) {
      LITERT_MP_ASSIGN_OR_RETURN(litert::RankedTensorType model_tensor_type,
                                 model_input_tensors[i].RankedTensorType());
      MP_ASSIGN_OR_RETURN(
          tensor_to_use, ConvertTensorChannelsIfNeeded(*tensor_to_use,
                                                       model_tensor_type, ctx));
    }
#endif  // MEDIAPIPE_METAL_ENABLED

    MP_ASSIGN_OR_RETURN(
        const litert::RankedTensorType tensor_type,
        CreateInputRankedTensorType(model_input_tensors[i], *tensor_to_use, i));

    LITERT_MP_ASSIGN_OR_RETURN(size_t buffer_size,
                               input_buffer_requirements.BufferSize());
    RET_CHECK_EQ(buffer_size, tensor_to_use->bytes())
        << "LiteRt input buffer size differs from MP input tensor bytes.";

    if (absl::c_linear_search(kLiteRtManagedBufferTypes, buffer_type)) {
      MP_ASSIGN_OR_RETURN(bool needs_realloc,
                          NeedsReallocation(managed_input_buffers_[i],
                                            buffer_size, tensor_type));

      if (needs_realloc) {
        LITERT_MP_ASSIGN_OR_RETURN(
            auto litert_buffer,
            litert::TensorBuffer::CreateManaged(*environment_, buffer_type,
                                                tensor_type, buffer_size));
        managed_input_buffers_[i] = std::move(litert_buffer);
      }
      MP_RETURN_IF_ERROR(CopyMpTensorToLiteRtBuffer(
          *tensor_to_use, *managed_input_buffers_[i]));
      LITERT_MP_ASSIGN_OR_RETURN(litert::TensorBuffer duplicated_buffer,
                                 managed_input_buffers_[i]->Duplicate());
      ctx.litert_inputs.push_back(std::move(duplicated_buffer));
    } else {
      MP_ASSIGN_OR_RETURN(litert::TensorBuffer litert_input_tensor,
                          CreateInputTensorBufferFromMpTensor(
                              buffer_type, *tensor_to_use, tensor_type, ctx));
      ctx.litert_inputs.push_back(std::move(litert_input_tensor));
    }
  }

  RET_CHECK_EQ(tensor_span.size() + GetNumberOfFeedbackTensors(),
               ctx.litert_inputs.size());
  return absl::OkStatus();
}

absl::StatusOr<std::vector<Tensor>>
InferenceRunnerLiteRt::CreateMpOutputTensors(
    const litert::SubgraphOutputs& model_output_tensors,
    const std::vector<litert::Layout>& output_tensor_layouts) {
  std::vector<Tensor> mp_output_tensors;
  mp_output_tensors.reserve(model_output_tensors.size());

  RET_CHECK_EQ(model_output_tensors.size(), output_tensor_layouts.size());

  LITERT_MP_ASSIGN_OR_RETURN(
      std::vector<absl::string_view> model_output_tensor_names,
      model_->GetSignatureOutputNames(signature_index_));

  for (int i = 0; i < model_output_tensors.size(); ++i) {
    if (feedback_manager_ &&
        feedback_manager_->IsFeedbackOutputTensorAtIndex(i)) {
      continue;
    }
    MP_ASSIGN_OR_RETURN(const litert::RankedTensorType tensor_type,
                        CreateOutputRankedTensorType(model_output_tensors[i],
                                                     output_tensor_layouts[i]));

    MP_ASSIGN_OR_RETURN(Tensor mp_output_tensor,
                        CreateTensorFromLiteRtRankedTensorType(
                            tensor_type, /*memory_manager=*/nullptr,
                            LITERT_HOST_MEMORY_BUFFER_ALIGNMENT));
    mp_output_tensors.push_back(std::move(mp_output_tensor));
  }
  return mp_output_tensors;
}

absl::StatusOr<litert::TensorBuffer>
InferenceRunnerLiteRt::CreateOutputTensorBufferFromMpTensor(
    litert::TensorBufferType buffer_type, const Tensor& tensor,
    const litert::RankedTensorType& tensor_type, InferenceRunContext& ctx) {
  switch (buffer_type) {
    case litert::TensorBufferType::kHostMemory:
      return CreateCpuOutputTensorBufferFromMpTensor(tensor, tensor_type,
                                                     ctx.active_output_views);
#if MEDIAPIPE_METAL_ENABLED
    case litert::TensorBufferType::kMetalBufferPacked:
      return CreateMetalOutputTensorBufferFromMpTensor(tensor, tensor_type,
                                                       ctx.active_output_views);
#endif  // MEDIAPIPE_METAL_ENABLED
#if MEDIAPIPE_TENSOR_USE_AHWB
    case litert::TensorBufferType::kAhwb:
      return CreateAhwbOutputTensorBufferFromMpTensor(tensor, tensor_type,
                                                      ctx.active_output_views);
#endif  // MEDIAPIPE_TENSOR_USE_AHWB
#if MEDIAPIPE_OPENGL_ES_VERSION >= MEDIAPIPE_OPENGL_ES_31
    case litert::TensorBufferType::kGlBuffer:
      return CreateGlOutputTensorBufferFromMpTensor(tensor, tensor_type,
                                                    ctx.active_output_views);
#endif  // MEDIAPIPE_OPENGL_ES_VERSION >= MEDIAPIPE_OPENGL_ES_31
#if defined(__EMSCRIPTEN__)
    case litert::TensorBufferType::kWebGpuTexture:
      if (webgpu_service_ != nullptr) {
        return CreateWebGpuOutputTensorBufferFromMpTensor(
            tensor, tensor_type, ctx.active_output_views);
      } else {
        return absl::InvalidArgumentError(absl::StrCat(
            "Unsupported buffer type to create output tensor: ", buffer_type));
      }
#endif  // defined(__EMSCRIPTEN__)
    default:
      return absl::InvalidArgumentError(absl::StrCat(
          "Unsupported buffer type to create output tensor: ", buffer_type));
  }
}

absl::Status InferenceRunnerLiteRt::PrepareOutputBuffers(
    const litert::SubgraphOutputs& model_output_tensors,
    const std::vector<litert::Layout>& output_tensor_layouts,
    const std::vector<Tensor>& mp_output_tensors, InferenceRunContext& ctx) {
  ctx.litert_outputs.reserve(model_output_tensors.size());

  LITERT_MP_ASSIGN_OR_RETURN(
      std::vector<absl::string_view> model_output_tensor_names,
      model_->GetSignatureOutputNames(signature_index_));

  int mp_output_tensor_index = 0;
  for (int i = 0; i < model_output_tensors.size(); ++i) {
    MP_ASSIGN_OR_RETURN(
        bool is_feedback_tensor,
        TryAppendFeedbackBuffer(false, i, model_output_tensor_names[i],
                                ctx.litert_outputs));
    if (is_feedback_tensor) continue;
    MP_ASSIGN_OR_RETURN(const litert::RankedTensorType tensor_type,
                        CreateOutputRankedTensorType(model_output_tensors[i],
                                                     output_tensor_layouts[i]));

    LITERT_MP_ASSIGN_OR_RETURN(
        const auto output_buffer_requirements,
        compiled_model_->GetOutputBufferRequirements(signature_index_, i),
        _ << "Failed to get output buffer requirements for output " << i);

    LITERT_MP_ASSIGN_OR_RETURN(size_t buffer_size,
                               output_buffer_requirements.BufferSize());
    RET_CHECK_EQ(buffer_size, mp_output_tensors[mp_output_tensor_index].bytes())
        << "LiteRt output buffer size differs from MP output tensor bytes.";

    MP_ASSIGN_OR_RETURN(litert::TensorBufferType buffer_type,
                        ChooseBestBufferType(output_buffer_requirements));

    if (absl::c_linear_search(kLiteRtManagedBufferTypes, buffer_type)) {
      MP_ASSIGN_OR_RETURN(bool needs_realloc,
                          NeedsReallocation(managed_output_buffers_[i],
                                            buffer_size, tensor_type));

      if (needs_realloc) {
        LITERT_MP_ASSIGN_OR_RETURN(
            auto litert_buffer,
            litert::TensorBuffer::CreateManaged(*environment_, buffer_type,
                                                tensor_type, buffer_size));
        managed_output_buffers_[i] = std::move(litert_buffer);
      }
      LITERT_MP_ASSIGN_OR_RETURN(litert::TensorBuffer duplicated_buffer,
                                 managed_output_buffers_[i]->Duplicate());
      ctx.litert_outputs.push_back(std::move(duplicated_buffer));
      mp_output_tensor_index++;
    } else {
      MP_ASSIGN_OR_RETURN(
          litert::TensorBuffer output_tensor,
          CreateOutputTensorBufferFromMpTensor(
              buffer_type, mp_output_tensors[mp_output_tensor_index++],
              tensor_type, ctx));
      ctx.litert_outputs.push_back(std::move(output_tensor));
    }
  }
  return absl::OkStatus();
}

#if MEDIAPIPE_TENSOR_USE_AHWB
absl::Status InferenceRunnerLiteRt::CleanAsyncRunStates(bool wait_for_all) {
  absl::MutexLock lock(&async_runs_mutex_);
  int count = 0;
  auto it = active_async_runs_.begin();
  while (it != active_async_runs_.end()) {
    bool is_signaled = true;
    for (const auto& fence : it->async_run_fences) {
      if (fence.Get() >= 0) {
        if (wait_for_all) {
          MP_RETURN_IF_ERROR(SyncWait(fence, absl::InfiniteDuration()));
        } else {
          MP_ASSIGN_OR_RETURN(bool fence_signaled, IsSignaled(fence));
          if (!fence_signaled) {
            is_signaled = false;
            break;
          }
        }
      }
    }

    if (is_signaled) {
      it = active_async_runs_.erase(it);
    } else {
      ++it;
    }
    count++;
  }
  return absl::OkStatus();
}
#endif  // MEDIAPIPE_TENSOR_USE_AHWB

absl::StatusOr<std::vector<Tensor>> InferenceRunnerLiteRt::Run(
    CalculatorContext* cc, const TensorSpan& tensor_span) {
  InferenceRunContext ctx;

  auto model_input_tensors = subgraph_->Inputs();

#if MEDIAPIPE_METAL_ENABLED
  if (metal_command_buffer_) {
    id<MTLCommandBuffer> command_buffer =
        (__bridge_transfer id<MTLCommandBuffer>)metal_command_buffer_;
    (void)command_buffer;
    metal_command_buffer_ = nullptr;
  }
#endif  // MEDIAPIPE_METAL_ENABLED

  MP_RETURN_IF_ERROR(
      PrepareInputBuffers(tensor_span, model_input_tensors, ctx));

#if MEDIAPIPE_METAL_ENABLED
  id<MTLCommandBuffer> input_command_buffer = nil;
  if (metal_command_buffer_) {
    input_command_buffer = (__bridge id<MTLCommandBuffer>)metal_command_buffer_;
    metal_command_buffer_ = nullptr;
  }
#endif  // MEDIAPIPE_METAL_ENABLED

  // Get final output tensor layouts from the compiled model.
  LITERT_MP_ASSIGN_OR_RETURN(
      std::vector<litert::Layout> output_tensor_layouts,
      compiled_model_->GetOutputTensorLayouts(
          signature_index_, /*update_allocation=*/has_dynamic_dimension_));

  auto model_output_tensors = subgraph_->Outputs();
  MP_ASSIGN_OR_RETURN(
      std::vector<Tensor> mp_output_tensors,
      CreateMpOutputTensors(model_output_tensors, output_tensor_layouts));

  MP_RETURN_IF_ERROR(PrepareOutputBuffers(
      model_output_tensors, output_tensor_layouts, mp_output_tensors, ctx));

#if MEDIAPIPE_METAL_ENABLED
  id<MTLCommandBuffer> output_command_buffer = nil;
  if (metal_command_buffer_) {
    output_command_buffer =
        (__bridge_transfer id<MTLCommandBuffer>)metal_command_buffer_;
    metal_command_buffer_ = nullptr;
  }

  if (input_command_buffer) {
    [input_command_buffer commit];
    [input_command_buffer waitUntilScheduled];
  }
#endif  // MEDIAPIPE_METAL_ENABLED

  // Execute model.
  if (run_async_) {
#if MEDIAPIPE_TENSOR_USE_AHWB
    // First, free any previously registered buffer handles from finished runs.
    MP_RETURN_IF_ERROR(CleanAsyncRunStates(/*wait_for_all=*/false));

    if (GetNumberOfFeedbackTensors() > 0) {
      return absl::InternalError(
          "Async execution with feedback tensors is not supported yet.");
    }
    bool async = true;
    LITERT_MP_RETURN_IF_ERROR(compiled_model_->RunAsync(
        signature_index_, ctx.litert_inputs, ctx.litert_outputs, async));
    RET_CHECK(async) << "LiteRT async execution requested but failed";

    // Create vector of SharedFds from LiteRT output buffer events.
    std::vector<SharedFd> litert_output_buffer_fds;
    litert_output_buffer_fds.reserve(ctx.litert_outputs.size());
    for (auto& output_buffer : ctx.litert_outputs) {
      LITERT_MP_ASSIGN_OR_RETURN(auto output_event, output_buffer.GetEvent());
      LITERT_MP_ASSIGN_OR_RETURN(int fd, output_event.DupFd());
      litert_output_buffer_fds.push_back(SharedFd(UniqueFd(fd)));
    }

    // Synchronize when reading is finished for each MP input buffer.
    // This is done by checking that all LiteRT output buffer events (vector
    // of SharedFds) are signaled.
    for (auto& view : ctx.active_input_views) {
      auto& ahwb_view = std::get<Tensor::AHardwareBufferView>(view);
      ahwb_view.SetReadingFinishedFunc(
          MultipleFdsFinishedFunc(litert_output_buffer_fds));
    }
    // Synchronize when writing is finished for each MP output buffer.
    // This is done by checking each respective LiteRT output buffer's event.
    RET_CHECK_EQ(ctx.litert_outputs.size(), ctx.active_output_views.size());
    for (int i = 0; i < ctx.litert_outputs.size(); ++i) {
      LITERT_MP_ASSIGN_OR_RETURN(auto output_event,
                                 ctx.litert_outputs[i].GetEvent());
      auto& ahwb_view =
          std::get<Tensor::AHardwareBufferView>(ctx.active_output_views[i]);
      LITERT_MP_ASSIGN_OR_RETURN(int write_finished_fd, output_event.DupFd());
      LITERT_MP_ASSIGN_OR_RETURN(int write_finished_fd_for_func,
                                 output_event.DupFd());
      ahwb_view.SetWritingFinishedFD(
          write_finished_fd,
          FdFinishedFunc(SharedFd(UniqueFd(write_finished_fd_for_func))));
    }
    ctx.active_output_views.clear();
    ctx.active_input_views.clear();

    // Move the context and fences to the async run state and add to the queue.
    AsyncRunState async_state;
    async_state.context = std::move(ctx);
    async_state.async_run_fences = std::move(litert_output_buffer_fds);
    {
      absl::MutexLock lock(&async_runs_mutex_);
      active_async_runs_.push_back(std::move(async_state));
    }
#else
    return absl::InternalError("LiteRT requires AHWB support to run async");
#endif  // MEDIAPIPE_TENSOR_USE_AHWB
  } else {
    LITERT_MP_RETURN_IF_ERROR(compiled_model_->Run(
        signature_index_, ctx.litert_inputs, ctx.litert_outputs));

#if MEDIAPIPE_METAL_ENABLED
    if (output_command_buffer) {
      [output_command_buffer commit];
      // The below call is found (manual testing) to resolve flickering issues
      // for some use cases where multiple Metal calculators are involved. See
      // b/302372363.
      [output_command_buffer waitUntilScheduled];
    }
#endif  // MEDIAPIPE_METAL_ENABLED

    ctx.active_output_views.clear();
    ctx.active_input_views.clear();

    MP_RETURN_IF_ERROR(CopyLiteRtManagedOutputBuffersToMpTensors(
        ctx.litert_outputs, mp_output_tensors));
  }
  return std::move(mp_output_tensors);
}

absl::Status InferenceRunnerLiteRt::CopyLiteRtManagedOutputBuffersToMpTensors(
    std::vector<litert::TensorBuffer>& output_buffers,
    std::vector<Tensor>& mp_output_tensors) {
  RET_CHECK_EQ(output_buffers.size(),
               mp_output_tensors.size() + GetNumberOfFeedbackTensors());
  int mp_output_tensor_index = 0;
  for (int i = 0; i < output_buffers.size(); ++i) {
    if (feedback_manager_ &&
        feedback_manager_->IsFeedbackOutputTensorAtIndex(i)) {
      continue;
    }
    LITERT_MP_ASSIGN_OR_RETURN(
        auto buffer_type, output_buffers[i].BufferType(),
        _ << "Failed to get output buffer type for output " << i);
    if (absl::c_linear_search(kLiteRtManagedBufferTypes, buffer_type)) {
      MP_RETURN_IF_ERROR(CopyLiteRtBufferToMpTensor(
          output_buffers[i], mp_output_tensors[mp_output_tensor_index]));
    }
    mp_output_tensor_index++;
  }
  return absl::OkStatus();
}

}  // namespace mediapipe
