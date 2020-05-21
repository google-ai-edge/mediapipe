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
#include "mediapipe/calculators/tflite/tflite_inference_calculator.pb.h"
#include "mediapipe/calculators/tflite/util.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/port/ret_check.h"

#if !defined(__EMSCRIPTEN__) || defined(__EMSCRIPTEN_PTHREADS__)
#include "mediapipe/util/cpu_util.h"
#endif  // !__EMSCRIPTEN__ || __EMSCRIPTEN_PTHREADS__

#include "mediapipe/util/resource_util.h"
#include "tensorflow/lite/error_reporter.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"

#if !defined(MEDIAPIPE_DISABLE_GL_COMPUTE)
#include "mediapipe/gpu/gl_calculator_helper.h"
#include "mediapipe/gpu/gpu_buffer.h"
#include "mediapipe/util/tflite/tflite_gpu_runner.h"
#include "tensorflow/lite/delegates/gpu/common/shape.h"
#include "tensorflow/lite/delegates/gpu/gl/gl_buffer.h"
#include "tensorflow/lite/delegates/gpu/gl/gl_program.h"
#include "tensorflow/lite/delegates/gpu/gl/gl_shader.h"
#include "tensorflow/lite/delegates/gpu/gl_delegate.h"
#endif  //  !MEDIAPIPE_DISABLE_GL_COMPUTE

#if defined(MEDIAPIPE_IOS)
#import <CoreVideo/CoreVideo.h>
#import <Metal/Metal.h>
#import <MetalKit/MetalKit.h>

#import "mediapipe/gpu/MPPMetalHelper.h"
#include "mediapipe/gpu/MPPMetalUtil.h"
#include "mediapipe/gpu/gpu_buffer.h"
#include "tensorflow/lite/delegates/gpu/common/shape.h"
#include "tensorflow/lite/delegates/gpu/metal/buffer_convert.h"
#include "tensorflow/lite/delegates/gpu/metal_delegate.h"
#include "tensorflow/lite/delegates/gpu/metal_delegate_internal.h"
#endif  // iOS

#if !defined(MEDIAPIPE_EDGE_TPU)
#include "tensorflow/lite/delegates/xnnpack/xnnpack_delegate.h"
#endif  // !EDGETPU
#if defined(MEDIAPIPE_ANDROID)
#include "tensorflow/lite/delegates/nnapi/nnapi_delegate.h"
#endif  // ANDROID

namespace {
// Commonly used to compute the number of blocks to launch in a kernel.
int NumGroups(const int size, const int group_size) {  // NOLINT
  return (size + group_size - 1) / group_size;
}

#if !defined(MEDIAPIPE_DISABLE_GL_COMPUTE)
typedef ::tflite::gpu::gl::GlBuffer GpuTensor;
#elif defined(MEDIAPIPE_IOS)
typedef id<MTLBuffer> GpuTensor;
#endif

// Round up n to next multiple of m.
size_t RoundUp(size_t n, size_t m) { return ((n + m - 1) / m) * m; }  // NOLINT

constexpr char kTensorsTag[] = "TENSORS";
constexpr char kTensorsGpuTag[] = "TENSORS_GPU";
}  // namespace

#if defined(MEDIAPIPE_EDGE_TPU)
#include "edgetpu.h"

// Creates and returns an Edge TPU interpreter to run the given edgetpu model.
std::unique_ptr<tflite::Interpreter> BuildEdgeTpuInterpreter(
    const tflite::FlatBufferModel& model,
    tflite::ops::builtin::BuiltinOpResolver* resolver,
    edgetpu::EdgeTpuContext* edgetpu_context) {
  resolver->AddCustom(edgetpu::kCustomOp, edgetpu::RegisterCustomOp());
  std::unique_ptr<tflite::Interpreter> interpreter;
  if (tflite::InterpreterBuilder(model, *resolver)(&interpreter) != kTfLiteOk) {
    std::cerr << "Failed to build edge TPU interpreter." << std::endl;
  }
  interpreter->SetExternalContext(kTfLiteEdgeTpuContext, edgetpu_context);
  interpreter->SetNumThreads(1);
  if (interpreter->AllocateTensors() != kTfLiteOk) {
    std::cerr << "Failed to allocate edge TPU tensors." << std::endl;
  }
  return interpreter;
}
#endif  // MEDIAPIPE_EDGE_TPU

// TfLiteInferenceCalculator File Layout:
//  * Header
//  * Core
//  * Aux
namespace mediapipe {

#if !defined(MEDIAPIPE_DISABLE_GL_COMPUTE)
using ::tflite::gpu::gl::CopyBuffer;
using ::tflite::gpu::gl::CreateReadWriteShaderStorageBuffer;
using ::tflite::gpu::gl::GlBuffer;
#endif

#if !defined(MEDIAPIPE_DISABLE_GPU) && !defined(__EMSCRIPTEN__)
namespace {
struct GPUData {
  int elements = 1;
  GpuTensor buffer;
  ::tflite::gpu::BHWC shape;
};
}  // namespace
#endif

// Returns number of threads to configure XNNPACK delegate with.
// (Equal to user provided value if specified.  Otherwise, it returns number of
// high cores (hard-coded to 1 for Emscripten without Threads extension))
int GetXnnpackNumThreads(
    const mediapipe::TfLiteInferenceCalculatorOptions& opts) {
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

// Calculator Header Section

// Runs inference on the provided input TFLite tensors and TFLite model.
//
// Creates an interpreter with given model and calls invoke().
// Optionally run inference on CPU/GPU.
//
// This calculator is designed to be used with the TfLiteConverterCalcualtor,
// to get the appropriate inputs.
//
// When the input tensors are on CPU, gpu inference is optional and can be
// specified in the calculator options.
// When the input tensors are on GPU, inference is GPU and output can be CPU or
// GPU.
//
// Input:
//  TENSORS - Vector of TfLiteTensor of type kTfLiteFloat32 or kTfLiteUInt8
//  TENSORS_GPU - Vector of GlBuffer or MTLBuffer
//
// Output:
//  TENSORS - Vector of TfLiteTensor of type kTfLiteFloat32 or kTfLiteUInt8
//  TENSORS_GPU - Vector of GlBuffer or MTLBuffer
//
// Input side packet:
//  CUSTOM_OP_RESOLVER (optional) - Use a custom op resolver,
//                                  instead of the builtin one.
//  MODEL (optional) - Use to specify TfLite model
//                     (std::unique_ptr<tflite::FlatBufferModel,
//                       std::function<void(tflite::FlatBufferModel*)>>)
//
// Example use:
// node {
//   calculator: "TfLiteInferenceCalculator"
//   input_stream: "TENSORS:tensor_image"
//   output_stream: "TENSORS:tensors"
//   options: {
//     [mediapipe.TfLiteInferenceCalculatorOptions.ext] {
//       model_path: "modelname.tflite"
//       delegate { gpu {} }
//     }
//   }
// }
//
// or
//
// node {
//   calculator: "TfLiteInferenceCalculator"
//   input_stream: "TENSORS:tensor_image"
//   input_side_packet: "MODEL:model"
//   output_stream: "TENSORS:tensors"
//   options: {
//     [mediapipe.TfLiteInferenceCalculatorOptions.ext] {
//       delegate { gpu {} }
//     }
//   }
// }
//
// IMPORTANT Notes:
//  Tensors are assumed to be ordered correctly (sequentially added to model).
//  Input tensors are assumed to be of the correct size and already normalized.
//  All output TfLiteTensors will be destroyed when the graph closes,
//  (i.e. after calling graph.WaitUntilDone()).
//  GPU tensors are currently only supported on Android and iOS.
//  This calculator uses FixedSizeInputStreamHandler by default.
//
class TfLiteInferenceCalculator : public CalculatorBase {
 public:
  using TfLiteDelegatePtr =
      std::unique_ptr<TfLiteDelegate, std::function<void(TfLiteDelegate*)>>;
  using TfLiteModelPtr =
      std::unique_ptr<tflite::FlatBufferModel,
                      std::function<void(tflite::FlatBufferModel*)>>;

  static ::mediapipe::Status GetContract(CalculatorContract* cc);

  ::mediapipe::Status Open(CalculatorContext* cc) override;
  ::mediapipe::Status Process(CalculatorContext* cc) override;
  ::mediapipe::Status Close(CalculatorContext* cc) override;

 private:
  ::mediapipe::Status LoadModel(CalculatorContext* cc);
  ::mediapipe::StatusOr<Packet> GetModelAsPacket(const CalculatorContext& cc);
  ::mediapipe::Status LoadDelegate(CalculatorContext* cc);
  ::mediapipe::Status InitTFLiteGPURunner();

  Packet model_packet_;
  std::unique_ptr<tflite::Interpreter> interpreter_;
  TfLiteDelegatePtr delegate_;

#if !defined(MEDIAPIPE_DISABLE_GL_COMPUTE)
  mediapipe::GlCalculatorHelper gpu_helper_;
  std::vector<std::unique_ptr<GPUData>> gpu_data_in_;
  std::vector<std::unique_ptr<GPUData>> gpu_data_out_;
  std::unique_ptr<tflite::gpu::TFLiteGPURunner> tflite_gpu_runner_;
#elif defined(MEDIAPIPE_IOS)
  MPPMetalHelper* gpu_helper_ = nullptr;
  std::vector<std::unique_ptr<GPUData>> gpu_data_in_;
  std::vector<std::unique_ptr<GPUData>> gpu_data_out_;
  id<MTLComputePipelineState> fp32_to_fp16_program_;
  TFLBufferConvert* converter_from_BPHWC4_ = nil;
#endif

#if defined(MEDIAPIPE_EDGE_TPU)
  std::shared_ptr<edgetpu::EdgeTpuContext> edgetpu_context_ =
      edgetpu::EdgeTpuManager::GetSingleton()->OpenDevice();
#endif

  bool gpu_inference_ = false;
  bool gpu_input_ = false;
  bool gpu_output_ = false;
  bool use_quantized_tensors_ = false;

  bool use_advanced_gpu_api_ = false;
};
REGISTER_CALCULATOR(TfLiteInferenceCalculator);

// Calculator Core Section

::mediapipe::Status TfLiteInferenceCalculator::GetContract(
    CalculatorContract* cc) {
  RET_CHECK(cc->Inputs().HasTag(kTensorsTag) ^
            cc->Inputs().HasTag(kTensorsGpuTag));
  RET_CHECK(cc->Outputs().HasTag(kTensorsTag) ^
            cc->Outputs().HasTag(kTensorsGpuTag));

  const auto& options =
      cc->Options<::mediapipe::TfLiteInferenceCalculatorOptions>();
  RET_CHECK(!options.model_path().empty() ^
            cc->InputSidePackets().HasTag("MODEL"))
      << "Either model as side packet or model path in options is required.";

  bool use_gpu =
      options.has_delegate() ? options.delegate().has_gpu() : options.use_gpu();

  if (cc->Inputs().HasTag(kTensorsTag))
    cc->Inputs().Tag(kTensorsTag).Set<std::vector<TfLiteTensor>>();
#if !defined(MEDIAPIPE_DISABLE_GPU) && !defined(__EMSCRIPTEN__)
  if (cc->Inputs().HasTag(kTensorsGpuTag)) {
    RET_CHECK(!options.has_delegate() || options.delegate().has_gpu())
        << "GPU input is compatible with GPU delegate only.";

    cc->Inputs().Tag(kTensorsGpuTag).Set<std::vector<GpuTensor>>();
    use_gpu |= true;
  }
#endif  //  !MEDIAPIPE_DISABLE_GPU

  if (cc->Outputs().HasTag(kTensorsTag))
    cc->Outputs().Tag(kTensorsTag).Set<std::vector<TfLiteTensor>>();
#if !defined(MEDIAPIPE_DISABLE_GPU) && !defined(__EMSCRIPTEN__)
  if (cc->Outputs().HasTag(kTensorsGpuTag)) {
    RET_CHECK(!options.has_delegate() || options.delegate().has_gpu())
        << "GPU output is compatible with GPU delegate only.";

    cc->Outputs().Tag(kTensorsGpuTag).Set<std::vector<GpuTensor>>();
    use_gpu |= true;
  }
#endif  //  !MEDIAPIPE_DISABLE_GPU

  if (cc->InputSidePackets().HasTag("CUSTOM_OP_RESOLVER")) {
    cc->InputSidePackets()
        .Tag("CUSTOM_OP_RESOLVER")
        .Set<tflite::ops::builtin::BuiltinOpResolver>();
  }
  if (cc->InputSidePackets().HasTag("MODEL")) {
    cc->InputSidePackets().Tag("MODEL").Set<TfLiteModelPtr>();
  }

  if (use_gpu) {
#if !defined(MEDIAPIPE_DISABLE_GL_COMPUTE)
    MP_RETURN_IF_ERROR(mediapipe::GlCalculatorHelper::UpdateContract(cc));
#elif defined(MEDIAPIPE_IOS)
    MP_RETURN_IF_ERROR([MPPMetalHelper updateContract:cc]);
#endif
  }

  // Assign this calculator's default InputStreamHandler.
  cc->SetInputStreamHandler("FixedSizeInputStreamHandler");

  return ::mediapipe::OkStatus();
}

::mediapipe::Status TfLiteInferenceCalculator::Open(CalculatorContext* cc) {
  cc->SetOffset(TimestampDiff(0));

  const auto& options =
      cc->Options<::mediapipe::TfLiteInferenceCalculatorOptions>();
  gpu_inference_ = options.use_gpu();

  if (cc->Inputs().HasTag(kTensorsGpuTag)) {
#if !defined(MEDIAPIPE_DISABLE_GPU) && !defined(__EMSCRIPTEN__)
    gpu_input_ = true;
    gpu_inference_ = true;  // Inference must be on GPU also.
#else
    RET_CHECK(!cc->Inputs().HasTag(kTensorsGpuTag))
        << "GPU processing not enabled.";
#endif  //  !MEDIAPIPE_DISABLE_GPU
  }

  if (cc->Outputs().HasTag(kTensorsGpuTag)) {
#if !defined(MEDIAPIPE_DISABLE_GPU) && !defined(__EMSCRIPTEN__)
    gpu_output_ = true;
    RET_CHECK(cc->Inputs().HasTag(kTensorsGpuTag))
        << "GPU output must also have GPU Input.";
#else
    RET_CHECK(!cc->Inputs().HasTag(kTensorsGpuTag))
        << "GPU processing not enabled.";
#endif  //  !MEDIAPIPE_DISABLE_GPU
  }

  const auto& calculator_opts =
      cc->Options<mediapipe::TfLiteInferenceCalculatorOptions>();
  use_advanced_gpu_api_ = false;

  MP_RETURN_IF_ERROR(LoadModel(cc));

  if (gpu_inference_) {
#if !defined(MEDIAPIPE_DISABLE_GL_COMPUTE)
    MP_RETURN_IF_ERROR(gpu_helper_.Open(cc));
#elif defined(MEDIAPIPE_IOS)
    gpu_helper_ = [[MPPMetalHelper alloc] initWithCalculatorContext:cc];
    RET_CHECK(gpu_helper_);
#endif

#if !defined(MEDIAPIPE_DISABLE_GL_COMPUTE)
    MP_RETURN_IF_ERROR(
        gpu_helper_.RunInGlContext([this, &cc]() -> ::mediapipe::Status {
          return use_advanced_gpu_api_ ? InitTFLiteGPURunner()
                                       : LoadDelegate(cc);
        }));
    if (use_advanced_gpu_api_) return ::mediapipe::OkStatus();
#else
    MP_RETURN_IF_ERROR(LoadDelegate(cc));
#endif
  } else {
#if defined(__EMSCRIPTEN__) || defined(MEDIAPIPE_ANDROID)
    MP_RETURN_IF_ERROR(LoadDelegate(cc));
#endif  // __EMSCRIPTEN__ || ANDROID
  }
  return ::mediapipe::OkStatus();
}

::mediapipe::Status TfLiteInferenceCalculator::InitTFLiteGPURunner() {
#if !defined(MEDIAPIPE_DISABLE_GL_COMPUTE)
  // Create and bind OpenGL buffers for outputs.
  // These buffers are created onve and later their ids are jut passed to the
  // calculator outputs.

  gpu_data_out_.resize(tflite_gpu_runner_->outputs_size());
  for (int i = 0; i < tflite_gpu_runner_->outputs_size(); ++i) {
    gpu_data_out_[i] = absl::make_unique<GPUData>();
    ASSIGN_OR_RETURN(gpu_data_out_[i]->elements,
                     tflite_gpu_runner_->GetOutputElements(i));
    // Create and bind input buffer.
    RET_CHECK_CALL(::tflite::gpu::gl::CreateReadWriteShaderStorageBuffer<float>(
        gpu_data_out_[i]->elements, &gpu_data_out_[i]->buffer));
  }
  RET_CHECK_CALL(tflite_gpu_runner_->Build());
#endif
  return ::mediapipe::OkStatus();
}

::mediapipe::Status TfLiteInferenceCalculator::Process(CalculatorContext* cc) {
  // 1. Receive pre-processed tensor inputs.
  if (use_advanced_gpu_api_) {
#if !defined(MEDIAPIPE_DISABLE_GL_COMPUTE)
    if (cc->Inputs().Tag(kTensorsGpuTag).IsEmpty()) {
      return ::mediapipe::OkStatus();
    }
    const auto& input_tensors =
        cc->Inputs().Tag(kTensorsGpuTag).Get<std::vector<GpuTensor>>();
    RET_CHECK(!input_tensors.empty());
    MP_RETURN_IF_ERROR(gpu_helper_.RunInGlContext(
        [this, &input_tensors]() -> ::mediapipe::Status {
          for (int i = 0; i < input_tensors.size(); ++i) {
            MP_RETURN_IF_ERROR(tflite_gpu_runner_->BindSSBOToInputTensor(
                input_tensors[i].id(), i));
          }
          for (int i = 0; i < gpu_data_out_.size(); ++i) {
            MP_RETURN_IF_ERROR(tflite_gpu_runner_->BindSSBOToOutputTensor(
                gpu_data_out_[i]->buffer.id(), i));
          }
          return ::mediapipe::OkStatus();
        }));
#endif
  } else if (gpu_input_) {
    // Read GPU input into SSBO.
#if !defined(MEDIAPIPE_DISABLE_GL_COMPUTE)
    if (cc->Inputs().Tag(kTensorsGpuTag).IsEmpty()) {
      return ::mediapipe::OkStatus();
    }
    const auto& input_tensors =
        cc->Inputs().Tag(kTensorsGpuTag).Get<std::vector<GpuTensor>>();
    RET_CHECK_GT(input_tensors.size(), 0);
    MP_RETURN_IF_ERROR(gpu_helper_.RunInGlContext(
        [this, &input_tensors]() -> ::mediapipe::Status {
          // Explicit copy input.
          gpu_data_in_.resize(input_tensors.size());
          for (int i = 0; i < input_tensors.size(); ++i) {
            RET_CHECK_CALL(
                CopyBuffer(input_tensors[i], gpu_data_in_[i]->buffer));
          }

          return ::mediapipe::OkStatus();
        }));
#elif defined(MEDIAPIPE_IOS)
    if (cc->Inputs().Tag(kTensorsGpuTag).IsEmpty()) {
      return ::mediapipe::OkStatus();
    }
    const auto& input_tensors =
        cc->Inputs().Tag(kTensorsGpuTag).Get<std::vector<GpuTensor>>();
    RET_CHECK_GT(input_tensors.size(), 0);
    // Explicit copy input with conversion float 32 bits to 16 bits.
    gpu_data_in_.resize(input_tensors.size());
    id<MTLCommandBuffer> command_buffer = [gpu_helper_ commandBuffer];
    command_buffer.label = @"TfLiteInferenceCalculatorConvert";
    id<MTLComputeCommandEncoder> compute_encoder =
        [command_buffer computeCommandEncoder];
    [compute_encoder setComputePipelineState:fp32_to_fp16_program_];
    for (int i = 0; i < input_tensors.size(); ++i) {
      [compute_encoder setBuffer:input_tensors[i] offset:0 atIndex:0];
      [compute_encoder setBuffer:gpu_data_in_[i]->buffer offset:0 atIndex:1];
      constexpr int kWorkgroupSize = 64;  // Block size for GPU shader.
      MTLSize threads_per_group = MTLSizeMake(kWorkgroupSize, 1, 1);
      const int threadgroups =
          NumGroups(gpu_data_in_[i]->elements, kWorkgroupSize);
      [compute_encoder dispatchThreadgroups:MTLSizeMake(threadgroups, 1, 1)
                      threadsPerThreadgroup:threads_per_group];
    }
    [compute_encoder endEncoding];
    [command_buffer commit];
#else
    RET_CHECK_FAIL() << "GPU processing not enabled.";
#endif
  } else {
    if (cc->Inputs().Tag(kTensorsTag).IsEmpty()) {
      return ::mediapipe::OkStatus();
    }
    // Read CPU input into tensors.
    const auto& input_tensors =
        cc->Inputs().Tag(kTensorsTag).Get<std::vector<TfLiteTensor>>();
    RET_CHECK_GT(input_tensors.size(), 0);
    for (int i = 0; i < input_tensors.size(); ++i) {
      const TfLiteTensor* input_tensor = &input_tensors[i];
      RET_CHECK(input_tensor->data.raw);
      if (use_quantized_tensors_) {
        const uint8* input_tensor_buffer = input_tensor->data.uint8;
        uint8* local_tensor_buffer = interpreter_->typed_input_tensor<uint8>(i);
        std::memcpy(local_tensor_buffer, input_tensor_buffer,
                    input_tensor->bytes);
      } else {
        const float* input_tensor_buffer = input_tensor->data.f;
        float* local_tensor_buffer = interpreter_->typed_input_tensor<float>(i);
        std::memcpy(local_tensor_buffer, input_tensor_buffer,
                    input_tensor->bytes);
      }
    }
  }

  // 2. Run inference.
  if (gpu_inference_) {
#if !defined(MEDIAPIPE_DISABLE_GL_COMPUTE)
    MP_RETURN_IF_ERROR(
        gpu_helper_.RunInGlContext([this]() -> ::mediapipe::Status {
          if (use_advanced_gpu_api_) {
            RET_CHECK(tflite_gpu_runner_->Invoke().ok());
          } else {
            RET_CHECK_EQ(interpreter_->Invoke(), kTfLiteOk);
          }
          return ::mediapipe::OkStatus();
        }));
#elif defined(MEDIAPIPE_IOS)
    RET_CHECK_EQ(interpreter_->Invoke(), kTfLiteOk);
#endif
  } else {
    RET_CHECK_EQ(interpreter_->Invoke(), kTfLiteOk);
  }

  // 3. Output processed tensors.
  if (use_advanced_gpu_api_) {
#if !defined(MEDIAPIPE_DISABLE_GL_COMPUTE)
    auto output_tensors = absl::make_unique<std::vector<GpuTensor>>();
    output_tensors->resize(gpu_data_out_.size());
    for (int i = 0; i < gpu_data_out_.size(); ++i) {
      output_tensors->at(i) = gpu_data_out_[i]->buffer.MakeRef();
    }
    cc->Outputs()
        .Tag(kTensorsGpuTag)
        .Add(output_tensors.release(), cc->InputTimestamp());
#endif
  } else if (gpu_output_) {
#if !defined(MEDIAPIPE_DISABLE_GL_COMPUTE)
    // Output result tensors (GPU).
    auto output_tensors = absl::make_unique<std::vector<GpuTensor>>();
    MP_RETURN_IF_ERROR(gpu_helper_.RunInGlContext(
        [this, &output_tensors]() -> ::mediapipe::Status {
          output_tensors->resize(gpu_data_out_.size());
          for (int i = 0; i < gpu_data_out_.size(); ++i) {
            GpuTensor& tensor = output_tensors->at(i);
            RET_CHECK_CALL(CreateReadWriteShaderStorageBuffer<float>(
                gpu_data_out_[i]->elements, &tensor));
            RET_CHECK_CALL(CopyBuffer(gpu_data_out_[i]->buffer, tensor));
          }
          return ::mediapipe::OkStatus();
        }));
    cc->Outputs()
        .Tag(kTensorsGpuTag)
        .Add(output_tensors.release(), cc->InputTimestamp());
#elif defined(MEDIAPIPE_IOS)
    // Output result tensors (GPU).
    auto output_tensors = absl::make_unique<std::vector<GpuTensor>>();
    output_tensors->resize(gpu_data_out_.size());
    id<MTLDevice> device = gpu_helper_.mtlDevice;
    id<MTLCommandBuffer> command_buffer = [gpu_helper_ commandBuffer];
    command_buffer.label = @"TfLiteInferenceBPHWC4Convert";
    id<MTLComputeCommandEncoder> convert_command =
        [command_buffer computeCommandEncoder];
    for (int i = 0; i < gpu_data_out_.size(); ++i) {
      output_tensors->at(i) =
          [device newBufferWithLength:gpu_data_out_[i]->elements * sizeof(float)
                              options:MTLResourceStorageModeShared];
      // Reshape tensor.
      [converter_from_BPHWC4_ convertWithEncoder:convert_command
                                           shape:gpu_data_out_[i]->shape
                                    sourceBuffer:gpu_data_out_[i]->buffer
                                 convertedBuffer:output_tensors->at(i)];
    }
    [convert_command endEncoding];
    [command_buffer commit];
    cc->Outputs()
        .Tag(kTensorsGpuTag)
        .Add(output_tensors.release(), cc->InputTimestamp());
#else
    RET_CHECK_FAIL() << "GPU processing not enabled.";
#endif  //  !MEDIAPIPE_DISABLE_GPU
  } else {
    // Output result tensors (CPU).
    const auto& tensor_indexes = interpreter_->outputs();
    auto output_tensors = absl::make_unique<std::vector<TfLiteTensor>>();
    for (int i = 0; i < tensor_indexes.size(); ++i) {
      TfLiteTensor* tensor = interpreter_->tensor(tensor_indexes[i]);
      output_tensors->emplace_back(*tensor);
    }
    cc->Outputs()
        .Tag(kTensorsTag)
        .Add(output_tensors.release(), cc->InputTimestamp());
  }

  return ::mediapipe::OkStatus();
}

::mediapipe::Status TfLiteInferenceCalculator::Close(CalculatorContext* cc) {
  if (delegate_) {
    if (gpu_inference_) {
#if !defined(MEDIAPIPE_DISABLE_GL_COMPUTE)
      MP_RETURN_IF_ERROR(gpu_helper_.RunInGlContext([this]() -> Status {
        interpreter_ = nullptr;
        delegate_ = nullptr;
        for (int i = 0; i < gpu_data_in_.size(); ++i) {
          gpu_data_in_[i].reset();
        }
        for (int i = 0; i < gpu_data_out_.size(); ++i) {
          gpu_data_out_[i].reset();
        }
        return ::mediapipe::OkStatus();
      }));
#elif defined(MEDIAPIPE_IOS)
      interpreter_ = nullptr;
      delegate_ = nullptr;
      for (int i = 0; i < gpu_data_in_.size(); ++i) {
        gpu_data_in_[i].reset();
      }
      for (int i = 0; i < gpu_data_out_.size(); ++i) {
        gpu_data_out_[i].reset();
      }
#endif
    } else {
      interpreter_ = nullptr;
      delegate_ = nullptr;
    }
  }
#if defined(MEDIAPIPE_EDGE_TPU)
  edgetpu_context_.reset();
#endif
  return ::mediapipe::OkStatus();
}

// Calculator Auxiliary Section

::mediapipe::Status TfLiteInferenceCalculator::LoadModel(
    CalculatorContext* cc) {
  ASSIGN_OR_RETURN(model_packet_, GetModelAsPacket(*cc));
  const auto& model = *model_packet_.Get<TfLiteModelPtr>();

  tflite::ops::builtin::BuiltinOpResolver op_resolver;
  if (cc->InputSidePackets().HasTag("CUSTOM_OP_RESOLVER")) {
    op_resolver = cc->InputSidePackets()
                      .Tag("CUSTOM_OP_RESOLVER")
                      .Get<tflite::ops::builtin::BuiltinOpResolver>();
  }

#if !defined(MEDIAPIPE_DISABLE_GL_COMPUTE)
  if (use_advanced_gpu_api_) {
    tflite::gpu::InferenceOptions options;
    options.priority1 = tflite::gpu::InferencePriority::MIN_LATENCY;
    options.priority2 = tflite::gpu::InferencePriority::AUTO;
    options.priority3 = tflite::gpu::InferencePriority::AUTO;
    options.usage = tflite::gpu::InferenceUsage::SUSTAINED_SPEED;
    tflite_gpu_runner_ =
        std::make_unique<tflite::gpu::TFLiteGPURunner>(options);
    return tflite_gpu_runner_->InitializeWithModel(model, op_resolver);
  }
#endif

#if defined(MEDIAPIPE_EDGE_TPU)
  interpreter_ =
      BuildEdgeTpuInterpreter(model, &op_resolver, edgetpu_context_.get());
#else
  tflite::InterpreterBuilder(model, op_resolver)(&interpreter_);
#endif  // MEDIAPIPE_EDGE_TPU

  RET_CHECK(interpreter_);

#if defined(__EMSCRIPTEN__) || defined(MEDIAPIPE_EDGE_TPU)
  interpreter_->SetNumThreads(1);
#else
  interpreter_->SetNumThreads(
      cc->Options<mediapipe::TfLiteInferenceCalculatorOptions>()
          .cpu_num_thread());
#endif  // __EMSCRIPTEN__

  if (gpu_output_) {
    use_quantized_tensors_ = false;
  } else {
    RET_CHECK_EQ(interpreter_->AllocateTensors(), kTfLiteOk);
    use_quantized_tensors_ =
        (interpreter_->tensor(interpreter_->inputs()[0])->quantization.type ==
         kTfLiteAffineQuantization);
    if (use_quantized_tensors_) gpu_inference_ = false;
  }

  return ::mediapipe::OkStatus();
}

::mediapipe::StatusOr<Packet> TfLiteInferenceCalculator::GetModelAsPacket(
    const CalculatorContext& cc) {
  const auto& options =
      cc.Options<mediapipe::TfLiteInferenceCalculatorOptions>();
  if (!options.model_path().empty()) {
    std::string model_path = options.model_path();

    ASSIGN_OR_RETURN(model_path, mediapipe::PathToResourceAsFile(model_path));

    auto model = tflite::FlatBufferModel::BuildFromFile(model_path.c_str());
    RET_CHECK(model) << "Failed to load model from path.";
    return MakePacket<TfLiteModelPtr>(TfLiteModelPtr(
        model.release(), [](tflite::FlatBufferModel* model) { delete model; }));
  }
  if (cc.InputSidePackets().HasTag("MODEL")) {
    return cc.InputSidePackets().Tag("MODEL");
  }
  return ::mediapipe::Status(
      ::mediapipe::StatusCode::kNotFound,
      "Must specify TFLite model as path or loaded model.");
}

::mediapipe::Status TfLiteInferenceCalculator::LoadDelegate(
    CalculatorContext* cc) {
  const auto& calculator_opts =
      cc->Options<mediapipe::TfLiteInferenceCalculatorOptions>();
  if (calculator_opts.has_delegate() &&
      calculator_opts.delegate().has_tflite()) {
    // Default tflite inference requeqsted - no need to modify graph.
    return ::mediapipe::OkStatus();
  }

  if (!gpu_inference_) {
#if defined(MEDIAPIPE_ANDROID)
    const bool nnapi_requested = calculator_opts.has_delegate()
                                     ? calculator_opts.delegate().has_nnapi()
                                     : calculator_opts.use_nnapi();
    if (nnapi_requested) {
      // Attempt to use NNAPI.
      // If not supported, the default CPU delegate will be created and used.
      interpreter_->SetAllowFp16PrecisionForFp32(1);
      delegate_ =
          TfLiteDelegatePtr(tflite::NnApiDelegate(), [](TfLiteDelegate*) {
            // No need to free according to tflite::NnApiDelegate()
            // documentation.
          });
      RET_CHECK_EQ(interpreter_->ModifyGraphWithDelegate(delegate_.get()),
                   kTfLiteOk);
      return ::mediapipe::OkStatus();
    }
#endif  // MEDIAPIPE_ANDROID

#if defined(__EMSCRIPTEN__)
    const bool xnnpack_requested = true;
#else
    const bool xnnpack_requested = calculator_opts.has_delegate() &&
                                   calculator_opts.delegate().has_xnnpack();
#endif  // __EMSCRIPTEN__

#if !defined(MEDIAPIPE_EDGE_TPU)
    if (xnnpack_requested) {
      TfLiteXNNPackDelegateOptions xnnpack_opts{};
      xnnpack_opts.num_threads = GetXnnpackNumThreads(calculator_opts);
      delegate_ = TfLiteDelegatePtr(TfLiteXNNPackDelegateCreate(&xnnpack_opts),
                                    &TfLiteXNNPackDelegateDelete);
      RET_CHECK_EQ(interpreter_->ModifyGraphWithDelegate(delegate_.get()),
                   kTfLiteOk);
    }
#endif  // !EDGETPU

    // Return, no need for GPU delegate below.
    return ::mediapipe::OkStatus();
  }

#if !defined(MEDIAPIPE_DISABLE_GL_COMPUTE)
  // Configure and create the delegate.
  TfLiteGpuDelegateOptions options = TfLiteGpuDelegateOptionsDefault();
  options.compile_options.precision_loss_allowed = 1;
  options.compile_options.preferred_gl_object_type =
      TFLITE_GL_OBJECT_TYPE_FASTEST;
  options.compile_options.dynamic_batch_enabled = 0;
  options.compile_options.inline_parameters = 1;
  if (!delegate_)
    delegate_ = TfLiteDelegatePtr(TfLiteGpuDelegateCreate(&options),
                                  &TfLiteGpuDelegateDelete);

  if (gpu_input_) {
    // Get input image sizes.
    const auto& input_indices = interpreter_->inputs();
    gpu_data_in_.resize(input_indices.size());
    for (int i = 0; i < input_indices.size(); ++i) {
      const TfLiteTensor* tensor = interpreter_->tensor(input_indices[i]);
      gpu_data_in_[i] = absl::make_unique<GPUData>();
      gpu_data_in_[i]->elements = 1;
      for (int d = 0; d < tensor->dims->size; ++d) {
        gpu_data_in_[i]->elements *= tensor->dims->data[d];
      }
      // Create and bind input buffer.
      RET_CHECK_CALL(
          ::tflite::gpu::gl::CreateReadWriteShaderStorageBuffer<float>(
              gpu_data_in_[i]->elements, &gpu_data_in_[i]->buffer));
      RET_CHECK_EQ(TfLiteGpuDelegateBindBufferToTensor(
                       delegate_.get(), gpu_data_in_[i]->buffer.id(),
                       interpreter_->inputs()[i]),
                   kTfLiteOk);
    }
  }
  if (gpu_output_) {
    // Get output image sizes.
    const auto& output_indices = interpreter_->outputs();
    gpu_data_out_.resize(output_indices.size());
    for (int i = 0; i < gpu_data_out_.size(); ++i) {
      const TfLiteTensor* tensor = interpreter_->tensor(output_indices[i]);
      gpu_data_out_[i] = absl::make_unique<GPUData>();
      gpu_data_out_[i]->elements = 1;
      // TODO handle *2 properly on some dialated models
      for (int d = 0; d < tensor->dims->size; ++d) {
        gpu_data_out_[i]->elements *= tensor->dims->data[d];
      }
    }
    // Create and bind output buffers.
    interpreter_->SetAllowBufferHandleOutput(true);
    for (int i = 0; i < gpu_data_out_.size(); ++i) {
      RET_CHECK_CALL(CreateReadWriteShaderStorageBuffer<float>(
          gpu_data_out_[i]->elements, &gpu_data_out_[i]->buffer));
      RET_CHECK_EQ(TfLiteGpuDelegateBindBufferToTensor(
                       delegate_.get(), gpu_data_out_[i]->buffer.id(),
                       output_indices[i]),
                   kTfLiteOk);
    }
  }

  // Must call this last.
  RET_CHECK_EQ(interpreter_->ModifyGraphWithDelegate(delegate_.get()),
               kTfLiteOk);
#endif  // OpenGL

#if defined(MEDIAPIPE_IOS)
  const int kHalfSize = 2;  // sizeof(half)
  // Configure and create the delegate.
  TFLGpuDelegateOptions options;
  options.allow_precision_loss = true;
  options.wait_type = TFLGpuDelegateWaitType::TFLGpuDelegateWaitTypePassive;
  if (!delegate_)
    delegate_ = TfLiteDelegatePtr(TFLGpuDelegateCreate(&options),
                                  &TFLGpuDelegateDelete);
  id<MTLDevice> device = gpu_helper_.mtlDevice;

  if (gpu_input_) {
    // Get input image sizes.
    const auto& input_indices = interpreter_->inputs();
    gpu_data_in_.resize(input_indices.size());
    for (int i = 0; i < input_indices.size(); ++i) {
      const TfLiteTensor* tensor = interpreter_->tensor(input_indices[i]);
      gpu_data_in_[i] = absl::make_unique<GPUData>();
      gpu_data_in_[i]->shape.b = tensor->dims->data[0];
      gpu_data_in_[i]->shape.h = tensor->dims->data[1];
      gpu_data_in_[i]->shape.w = tensor->dims->data[2];
      // On iOS GPU, input must be 4 channels, regardless of what model expects.
      gpu_data_in_[i]->shape.c = 4;
      gpu_data_in_[i]->elements =
          gpu_data_in_[i]->shape.b * gpu_data_in_[i]->shape.h *
          gpu_data_in_[i]->shape.w * gpu_data_in_[i]->shape.c;
      // Input to model can be RGBA only.
      if (tensor->dims->data[3] != 4) {
        LOG(WARNING) << "Please ensure input GPU tensor is 4 channels.";
      }
      const std::string shader_source =
          absl::Substitute(R"(#include <metal_stdlib>
        using namespace metal;
        kernel void convertKernel(device float4* const input_buffer [[buffer(0)]],
                                  device half4* output_buffer [[buffer(1)]],
                                  uint gid [[thread_position_in_grid]]) {
          if (gid >= $0) return;
          output_buffer[gid] = half4(input_buffer[gid]);
        })",
                           gpu_data_in_[i]->elements / 4);
      NSString* library_source =
          [NSString stringWithUTF8String:shader_source.c_str()];
      NSError* error = nil;
      id<MTLLibrary> library =
          [device newLibraryWithSource:library_source options:nil error:&error];
      RET_CHECK(library != nil) << "Couldn't create shader library "
                                << [[error localizedDescription] UTF8String];
      id<MTLFunction> kernel_func = nil;
      kernel_func = [library newFunctionWithName:@"convertKernel"];
      RET_CHECK(kernel_func != nil) << "Couldn't create kernel function.";
      fp32_to_fp16_program_ =
          [device newComputePipelineStateWithFunction:kernel_func error:&error];
      RET_CHECK(fp32_to_fp16_program_ != nil)
          << "Couldn't create pipeline state "
          << [[error localizedDescription] UTF8String];

      // Create and bind input buffer.
      gpu_data_in_[i]->buffer =
          [device newBufferWithLength:gpu_data_in_[i]->elements * kHalfSize
                              options:MTLResourceStorageModeShared];
      RET_CHECK_EQ(interpreter_->ModifyGraphWithDelegate(delegate_.get()),
                   kTfLiteOk);
      RET_CHECK_EQ(
          TFLGpuDelegateBindMetalBufferToTensor(
              delegate_.get(), input_indices[i], gpu_data_in_[i]->buffer),
          true);
    }
  }
  if (gpu_output_) {
    // Get output image sizes.
    const auto& output_indices = interpreter_->outputs();
    gpu_data_out_.resize(output_indices.size());
    for (int i = 0; i < gpu_data_out_.size(); ++i) {
      const TfLiteTensor* tensor = interpreter_->tensor(output_indices[i]);
      gpu_data_out_[i] = absl::make_unique<GPUData>();
      gpu_data_out_[i]->elements = 1;
      // TODO handle *2 properly on some dialated models
      for (int d = 0; d < tensor->dims->size; ++d) {
        // Pad each dim for BHWC4 conversion inside delegate.
        gpu_data_out_[i]->elements *= RoundUp(tensor->dims->data[d], 4);
      }
      // Save dimensions for reshaping back later.
      gpu_data_out_[i]->shape.b = tensor->dims->data[0];
      switch (tensor->dims->size) {
        case 2:
          gpu_data_out_[i]->shape.h = 1;
          gpu_data_out_[i]->shape.w = 1;
          gpu_data_out_[i]->shape.c = tensor->dims->data[1];
          break;
        case 3:
          gpu_data_out_[i]->shape.h = 1;
          gpu_data_out_[i]->shape.w = tensor->dims->data[1];
          gpu_data_out_[i]->shape.c = tensor->dims->data[2];
          break;
        case 4:
          gpu_data_out_[i]->shape.h = tensor->dims->data[1];
          gpu_data_out_[i]->shape.w = tensor->dims->data[2];
          gpu_data_out_[i]->shape.c = tensor->dims->data[3];
          break;
        default:
          return mediapipe::InternalError("Unsupported tensor shape.");
      }
    }
    // Create and bind output buffers.
    interpreter_->SetAllowBufferHandleOutput(true);
    for (int i = 0; i < gpu_data_out_.size(); ++i) {
      gpu_data_out_[i]->buffer =
          [device newBufferWithLength:gpu_data_out_[i]->elements * kHalfSize
                              options:MTLResourceStorageModeShared];
      RET_CHECK_EQ(
          TFLGpuDelegateBindMetalBufferToTensor(
              delegate_.get(), output_indices[i], gpu_data_out_[i]->buffer),
          true);
    }

    // Create converter for GPU output.
    converter_from_BPHWC4_ = [[TFLBufferConvert alloc] initWithDevice:device
                                                            isFloat16:true
                                                      convertToPBHWC4:false];
    if (converter_from_BPHWC4_ == nil) {
      return mediapipe::InternalError(
          "Error initializating output buffer converter");
    }
  }
#endif  // iOS

  return ::mediapipe::OkStatus();
}

}  // namespace mediapipe
