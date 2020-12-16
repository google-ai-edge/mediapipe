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
#include "mediapipe/calculators/tensor/inference_calculator.pb.h"
#include "mediapipe/framework/api2/node.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/tensor.h"
#include "mediapipe/framework/port/ret_check.h"
#include "mediapipe/util/tflite/config.h"
#include "mediapipe/util/tflite/tflite_model_loader.h"

#if !defined(__EMSCRIPTEN__) || defined(__EMSCRIPTEN_PTHREADS__)
#include "mediapipe/util/cpu_util.h"
#endif  // !__EMSCRIPTEN__ || __EMSCRIPTEN_PTHREADS__

#include "tensorflow/lite/error_reporter.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"

#if defined(MEDIAPIPE_ANDROID)
#include "mediapipe/util/android/file/base/file.h"
#include "mediapipe/util/android/file/base/filesystem.h"
#include "mediapipe/util/android/file/base/helpers.h"
#endif  // ANDROID

#if MEDIAPIPE_TFLITE_GL_INFERENCE
#include "mediapipe/gpu/gl_calculator_helper.h"
#include "mediapipe/gpu/gpu_buffer.h"
#include "mediapipe/util/tflite/tflite_gpu_runner.h"
#include "tensorflow/lite/delegates/gpu/common/shape.h"
#include "tensorflow/lite/delegates/gpu/gl_delegate.h"
#endif  // MEDIAPIPE_TFLITE_GL_INFERENCE

#if MEDIAPIPE_TFLITE_METAL_INFERENCE
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
#endif  // MEDIAPIPE_TFLITE_METAL_INFERENCE

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

// Round up n to next multiple of m.
template <typename T>
T RoundUp(T n, T m) {
  return ((n + m - T{1}) / m) * m;
}

bool ShouldUseGpu(const mediapipe::InferenceCalculatorOptions& options) {
  return (
      !options.has_delegate() ||  // Use GPU delegate if delegate not specified
      (options.has_delegate() && options.delegate().has_gpu()));
}

constexpr char kTensorsTag[] = "TENSORS";

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

}  // namespace

namespace mediapipe {
namespace api2 {

#if MEDIAPIPE_TFLITE_METAL_INFERENCE
namespace {
tflite::gpu::BHWC BhwcFromTensorShape(const Tensor::Shape& shape) {
  tflite::gpu::BHWC result;
  result.b = shape.dims[0];
  switch (shape.dims.size()) {
    case 1:
      // result.b is already filled.
      break;
    case 2:
      result.h = 1;
      result.w = 1;
      result.c = shape.dims[1];
      break;
    case 3:
      result.h = 1;
      result.w = shape.dims[1];
      result.c = shape.dims[2];
      break;
    case 4:
      result.h = shape.dims[1];
      result.w = shape.dims[2];
      result.c = shape.dims[3];
      break;
    default:
      // Handles 0 and >4.
      LOG(FATAL)
          << "Dimensions size must be in range [1,4] for GPU inference, but "
          << shape.dims.size() << " is provided";
  }
  return result;
}
}  // namespace
#endif  // MEDIAPIPE_TFLITE_METAL_INFERENCE

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

// Calculator Header Section

// Runs inference on the provided input Tensors and TFLite model.
//
// Creates an interpreter with given model and calls invoke().
// Optionally run inference on CPU/GPU.
//
// This calculator can be used with TensorConverterCalculator to get the
// appropriate inputs.
//
// When the input tensors are on CPU, gpu inference is optional and can be
// specified in the calculator options.
// When the input tensors are on GPU, inference is GPU and output can be CPU or
// GPU.
//
// Input:
//  TENSORS - Vector of Tensors
//
// Output:
//  TENSORS - Vector of Tensors
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
//   calculator: "InferenceCalculator"
//   input_stream: "TENSORS:tensor_image"
//   output_stream: "TENSORS:tensors"
//   options: {
//     [mediapipe.InferenceCalculatorOptions.ext] {
//       model_path: "modelname.tflite"
//     }
//   }
// }
//
// or
//
// node {
//   calculator: "InferenceCalculator"
//   input_stream: "TENSORS:tensor_image"
//   input_side_packet: "MODEL:model"
//   output_stream: "TENSORS:tensors"
//   options: {
//     [mediapipe.InferenceCalculatorOptions.ext] {
//       model_path: "modelname.tflite"
//       delegate { gpu {} }
//     }
//   }
// }
//
// IMPORTANT Notes:
//  Tensors are assumed to be ordered correctly (sequentially added to model).
//  Input tensors are assumed to be of the correct size and already normalized.

class InferenceCalculator : public Node {
 public:
  using TfLiteDelegatePtr =
      std::unique_ptr<TfLiteDelegate, std::function<void(TfLiteDelegate*)>>;

  static constexpr Input<std::vector<Tensor>> kInTensors{"TENSORS"};
  static constexpr SideInput<tflite::ops::builtin::BuiltinOpResolver>::Optional
      kSideInCustomOpResolver{"CUSTOM_OP_RESOLVER"};
  static constexpr SideInput<TfLiteModelPtr>::Optional kSideInModel{"MODEL"};
  static constexpr Output<std::vector<Tensor>> kOutTensors{"TENSORS"};
  MEDIAPIPE_NODE_CONTRACT(kInTensors, kSideInCustomOpResolver, kSideInModel,
                          kOutTensors);
  static mediapipe::Status UpdateContract(CalculatorContract* cc);

  mediapipe::Status Open(CalculatorContext* cc) override;
  mediapipe::Status Process(CalculatorContext* cc) override;
  mediapipe::Status Close(CalculatorContext* cc) override;

 private:
  mediapipe::Status ReadKernelsFromFile();
  mediapipe::Status WriteKernelsToFile();
  mediapipe::Status LoadModel(CalculatorContext* cc);
  mediapipe::StatusOr<mediapipe::Packet> GetModelAsPacket(
      const CalculatorContext& cc);
  mediapipe::Status LoadDelegate(CalculatorContext* cc);
  mediapipe::Status InitTFLiteGPURunner(CalculatorContext* cc);

  mediapipe::Packet model_packet_;
  std::unique_ptr<tflite::Interpreter> interpreter_;
  TfLiteDelegatePtr delegate_;

#if MEDIAPIPE_TFLITE_GL_INFERENCE
  mediapipe::GlCalculatorHelper gpu_helper_;
  std::unique_ptr<tflite::gpu::TFLiteGPURunner> tflite_gpu_runner_;
  bool allow_precision_loss_ = false;
  mediapipe::InferenceCalculatorOptions::Delegate::Gpu::API
      tflite_gpu_runner_api_;
#elif MEDIAPIPE_TFLITE_METAL_INFERENCE
  MPPMetalHelper* gpu_helper_ = nullptr;
  TFLBufferConvert* converter_to_BPHWC4_ = nil;
  TFLBufferConvert* converter_from_BPHWC4_ = nil;
#endif  // MEDIAPIPE_TFLITE_GL_INFERENCE

#if MEDIAPIPE_TFLITE_GPU_SUPPORTED
  std::vector<Tensor::Shape> output_shapes_;
  std::vector<std::unique_ptr<Tensor>> gpu_buffers_in_;
  std::vector<std::unique_ptr<Tensor>> gpu_buffers_out_;
#endif  // MEDIAPIPE_TFLITE_GPU_SUPPORTED

#if defined(MEDIAPIPE_EDGE_TPU)
  std::shared_ptr<edgetpu::EdgeTpuContext> edgetpu_context_ =
      edgetpu::EdgeTpuManager::GetSingleton()->OpenDevice();
#endif

  bool use_advanced_gpu_api_ = false;
  bool use_gpu_delegate_ = false;

  bool use_kernel_caching_ = false;
  std::string cached_kernel_filename_;
};

MEDIAPIPE_REGISTER_NODE(InferenceCalculator);

mediapipe::Status InferenceCalculator::UpdateContract(CalculatorContract* cc) {
  const auto& options = cc->Options<::mediapipe::InferenceCalculatorOptions>();
  RET_CHECK(!options.model_path().empty() ^ kSideInModel(cc).IsConnected())
      << "Either model as side packet or model path in options is required.";

  if (ShouldUseGpu(options)) {
#if MEDIAPIPE_TFLITE_GL_INFERENCE
    MP_RETURN_IF_ERROR(mediapipe::GlCalculatorHelper::UpdateContract(cc));
#elif MEDIAPIPE_TFLITE_METAL_INFERENCE
    MP_RETURN_IF_ERROR([MPPMetalHelper updateContract:cc]);
#endif
  }
  return mediapipe::OkStatus();
}

mediapipe::Status InferenceCalculator::Open(CalculatorContext* cc) {
#if MEDIAPIPE_TFLITE_GL_INFERENCE || MEDIAPIPE_TFLITE_METAL_INFERENCE
  const auto& options = cc->Options<::mediapipe::InferenceCalculatorOptions>();
  if (ShouldUseGpu(options)) {
#if MEDIAPIPE_TFLITE_GL_INFERENCE
    use_advanced_gpu_api_ = options.has_delegate() &&
                            options.delegate().has_gpu() &&
                            options.delegate().gpu().use_advanced_gpu_api();
    allow_precision_loss_ = options.delegate().gpu().allow_precision_loss();
    tflite_gpu_runner_api_ = options.delegate().gpu().api();
    use_kernel_caching_ =
        use_advanced_gpu_api_ && options.delegate().gpu().use_kernel_caching();
#endif  // MEDIAPIPE_TFLITE_GL_INFERENCE
    use_gpu_delegate_ = !use_advanced_gpu_api_;
  }
#endif  // MEDIAPIPE_TFLITE_GL_INFERENCE || MEDIAPIPE_TFLITE_METAL_INFERENCE

  if (use_kernel_caching_) {
#if MEDIAPIPE_TFLITE_GL_INFERENCE && defined(MEDIAPIPE_ANDROID)
    cached_kernel_filename_ =
        "/sdcard/" + mediapipe::File::Basename(options.model_path()) + ".ker";
#endif  // MEDIAPIPE_TFLITE_GL_INFERENCE && MEDIAPIPE_ANDROID
  }

  // When use_advanced_gpu_api_, model loading is handled in InitTFLiteGPURunner
  // for everything.
  if (!use_advanced_gpu_api_) {
    MP_RETURN_IF_ERROR(LoadModel(cc));
  }

  if (use_gpu_delegate_ || use_advanced_gpu_api_) {
#if MEDIAPIPE_TFLITE_GL_INFERENCE
    MP_RETURN_IF_ERROR(gpu_helper_.Open(cc));
    MP_RETURN_IF_ERROR(
        gpu_helper_.RunInGlContext([this, &cc]() -> ::mediapipe::Status {
          return use_advanced_gpu_api_ ? InitTFLiteGPURunner(cc)
                                       : LoadDelegate(cc);
        }));
#elif MEDIAPIPE_TFLITE_METAL_INFERENCE
    gpu_helper_ = [[MPPMetalHelper alloc] initWithCalculatorContext:cc];
    RET_CHECK(gpu_helper_);
    MP_RETURN_IF_ERROR(LoadDelegate(cc));
#endif
  } else {
    MP_RETURN_IF_ERROR(LoadDelegate(cc));
  }
  return mediapipe::OkStatus();
}

mediapipe::Status InferenceCalculator::Process(CalculatorContext* cc) {
  if (kInTensors(cc).IsEmpty()) {
    return mediapipe::OkStatus();
  }
  const auto& input_tensors = *kInTensors(cc);
  RET_CHECK(!input_tensors.empty());
  auto output_tensors = absl::make_unique<std::vector<Tensor>>();
#if MEDIAPIPE_TFLITE_METAL_INFERENCE
  id<MTLCommandBuffer> command_buffer;
  id<MTLComputeCommandEncoder> compute_encoder;
#endif  // MEDIAPIPE_TFLITE_METAL_INFERENCE

  if (use_gpu_delegate_ || use_advanced_gpu_api_) {
#if MEDIAPIPE_TFLITE_GL_INFERENCE
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
            return mediapipe::OkStatus();
          }));
    } else {
      MP_RETURN_IF_ERROR(gpu_helper_.RunInGlContext(
          [this, &input_tensors]() -> ::mediapipe::Status {
            // Explicitly copy input.
            for (int i = 0; i < input_tensors.size(); ++i) {
              glBindBuffer(GL_COPY_READ_BUFFER,
                           input_tensors[i].GetOpenGlBufferReadView().name());
              glBindBuffer(
                  GL_COPY_WRITE_BUFFER,
                  gpu_buffers_in_[i]->GetOpenGlBufferWriteView().name());
              glCopyBufferSubData(GL_COPY_READ_BUFFER, GL_COPY_WRITE_BUFFER, 0,
                                  0, input_tensors[i].bytes());
            }
            return mediapipe::OkStatus();
          }));
    }
#elif MEDIAPIPE_TFLITE_METAL_INFERENCE
    command_buffer = [gpu_helper_ commandBuffer];
    command_buffer.label = @"InferenceCalculator";
    compute_encoder = [command_buffer computeCommandEncoder];
    // Explicit copy input with conversion float 32 bits to 16 bits.
    for (int i = 0; i < input_tensors.size(); ++i) {
      auto input_view = input_tensors[i].GetMtlBufferReadView(command_buffer);
      // Reshape tensor.
      tflite::gpu::BHWC shape = BhwcFromTensorShape(input_tensors[i].shape());
      auto gpu_buffer_view =
          gpu_buffers_in_[i]->GetMtlBufferWriteView(command_buffer);
      [converter_to_BPHWC4_ convertWithEncoder:compute_encoder
                                         shape:shape
                                  sourceBuffer:input_view.buffer()
                               convertedBuffer:gpu_buffer_view.buffer()];
    }
#endif  // MEDIAPIPE_TFLITE_GL_INFERENCE
  } else {
    // Read CPU input into tensors.
    for (int i = 0; i < input_tensors.size(); ++i) {
      const Tensor* input_tensor = &input_tensors[i];
      auto input_tensor_view = input_tensor->GetCpuReadView();
      auto input_tensor_buffer = input_tensor_view.buffer<float>();
      float* local_tensor_buffer = interpreter_->typed_input_tensor<float>(i);
      std::memcpy(local_tensor_buffer, input_tensor_buffer,
                  input_tensor->bytes());
    }
  }

  // Run inference.
#if MEDIAPIPE_TFLITE_GL_INFERENCE
  if (use_advanced_gpu_api_) {
    RET_CHECK(tflite_gpu_runner_->Invoke().ok());
  } else {
    RET_CHECK_EQ(interpreter_->Invoke(), kTfLiteOk);
  }
#else
#if MEDIAPIPE_TFLITE_METAL_INFERENCE
  if (use_gpu_delegate_) {
    RET_CHECK(
        TFLGpuDelegateSetCommandEncoder(delegate_.get(), compute_encoder));
  }
#endif  // MEDIAPIPE_TFLITE_METAL_INFERENCE
  RET_CHECK_EQ(interpreter_->Invoke(), kTfLiteOk);
#endif  // MEDIAPIPE_TFLITE_GL_INFERENCE

  if (use_gpu_delegate_ || use_advanced_gpu_api_) {
#if MEDIAPIPE_TFLITE_GL_INFERENCE
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
              auto write_view =
                  output_tensors->back().GetOpenGlBufferWriteView();
              glBindBuffer(GL_COPY_WRITE_BUFFER, write_view.name());
              glCopyBufferSubData(GL_COPY_READ_BUFFER, GL_COPY_WRITE_BUFFER, 0,
                                  0, t->bytes());
            }
            return mediapipe::OkStatus();
          }));
    }
    // Output tensors are already bound if use_advanced_gpu_api_ is true.
#elif MEDIAPIPE_TFLITE_METAL_INFERENCE
    output_tensors->reserve(output_shapes_.size());
    for (int i = 0; i < output_shapes_.size(); ++i) {
      output_tensors->emplace_back(Tensor::ElementType::kFloat32,
                                   output_shapes_[i]);
      // Reshape tensor.
      tflite::gpu::BHWC shape = BhwcFromTensorShape(output_shapes_[i]);
      auto read_view =
          gpu_buffers_out_[i]->GetMtlBufferReadView(command_buffer);
      auto write_view =
          output_tensors->at(i).GetMtlBufferWriteView(command_buffer);
      [converter_from_BPHWC4_ convertWithEncoder:compute_encoder
                                           shape:shape
                                    sourceBuffer:read_view.buffer()
                                 convertedBuffer:write_view.buffer()];
    }
    [compute_encoder endEncoding];
    [command_buffer commit];
#endif  // MEDIAPIPE_TFLITE_GL_INFERENCE
  } else {
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
  }
  kOutTensors(cc).Send(std::move(output_tensors));
  return mediapipe::OkStatus();
}

mediapipe::Status InferenceCalculator::WriteKernelsToFile() {
#if MEDIAPIPE_TFLITE_GL_INFERENCE && defined(MEDIAPIPE_ANDROID)
  if (use_kernel_caching_) {
    // Save kernel file.
    auto kernel_cache = absl::make_unique<std::vector<uint8_t>>(
        tflite_gpu_runner_->GetSerializedBinaryCache());
    std::string cache_str(kernel_cache->begin(), kernel_cache->end());
    MP_RETURN_IF_ERROR(
        mediapipe::file::SetContents(cached_kernel_filename_, cache_str));
  }
#endif  // MEDIAPIPE_TFLITE_GL_INFERENCE && MEDIAPIPE_ANDROID
  return mediapipe::OkStatus();
}

mediapipe::Status InferenceCalculator::Close(CalculatorContext* cc) {
  MP_RETURN_IF_ERROR(WriteKernelsToFile());
#if MEDIAPIPE_TFLITE_GL_INFERENCE
  if (use_gpu_delegate_) {
    MP_RETURN_IF_ERROR(gpu_helper_.RunInGlContext([this]() -> Status {
      gpu_buffers_in_.clear();
      gpu_buffers_out_.clear();
      return mediapipe::OkStatus();
    }));
  }
#elif MEDIAPIPE_TFLITE_METAL_INFERENCE
  converter_to_BPHWC4_ = nil;
  converter_from_BPHWC4_ = nil;
  gpu_buffers_in_.clear();
  gpu_buffers_out_.clear();
#endif  // MEDIAPIPE_TFLITE_GL_INFERENCE

#if defined(MEDIAPIPE_EDGE_TPU)
  edgetpu_context_.reset();
#endif
  interpreter_ = nullptr;
  delegate_ = nullptr;
  return mediapipe::OkStatus();
}

mediapipe::Status InferenceCalculator::ReadKernelsFromFile() {
#if MEDIAPIPE_TFLITE_GL_INFERENCE && defined(MEDIAPIPE_ANDROID)
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
#endif  // MEDIAPIPE_TFLITE_GL_INFERENCE && MEDIAPIPE_ANDROID
  return mediapipe::OkStatus();
}

mediapipe::Status InferenceCalculator::InitTFLiteGPURunner(
    CalculatorContext* cc) {
#if MEDIAPIPE_TFLITE_GL_INFERENCE
  ASSIGN_OR_RETURN(model_packet_, GetModelAsPacket(*cc));
  const auto& model = *model_packet_.Get<TfLiteModelPtr>();
  tflite::ops::builtin::BuiltinOpResolver op_resolver =
      kSideInCustomOpResolver(cc).GetOr(
          tflite::ops::builtin::BuiltinOpResolver());

  // Create runner
  tflite::gpu::InferenceOptions options;
  options.priority1 = allow_precision_loss_
                          ? tflite::gpu::InferencePriority::MIN_LATENCY
                          : tflite::gpu::InferencePriority::MAX_PRECISION;
  options.priority2 = tflite::gpu::InferencePriority::AUTO;
  options.priority3 = tflite::gpu::InferencePriority::AUTO;
  options.usage = tflite::gpu::InferenceUsage::SUSTAINED_SPEED;
  tflite_gpu_runner_ = std::make_unique<tflite::gpu::TFLiteGPURunner>(options);
  switch (tflite_gpu_runner_api_) {
    case mediapipe::InferenceCalculatorOptions::Delegate::Gpu::OPENGL: {
      tflite_gpu_runner_->ForceOpenGL();
      break;
    }
    case mediapipe::InferenceCalculatorOptions::Delegate::Gpu::OPENCL: {
      tflite_gpu_runner_->ForceOpenCL();
      break;
    }
    case mediapipe::InferenceCalculatorOptions::Delegate::Gpu::ANY: {
      // Do not need to force any specific API.
      break;
    }
  }
  MP_RETURN_IF_ERROR(
      tflite_gpu_runner_->InitializeWithModel(model, op_resolver));

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
#endif  // MEDIAPIPE_TFLITE_GL_INFERENCE

  return mediapipe::OkStatus();
}

mediapipe::Status InferenceCalculator::LoadModel(CalculatorContext* cc) {
  ASSIGN_OR_RETURN(model_packet_, GetModelAsPacket(*cc));
  const auto& model = *model_packet_.Get<TfLiteModelPtr>();
  tflite::ops::builtin::BuiltinOpResolver op_resolver =
      kSideInCustomOpResolver(cc).GetOr(
          tflite::ops::builtin::BuiltinOpResolver());

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
      cc->Options<mediapipe::InferenceCalculatorOptions>().cpu_num_thread());
#endif  // __EMSCRIPTEN__

  RET_CHECK_EQ(interpreter_->AllocateTensors(), kTfLiteOk);
  // TODO: Support quantized tensors.
  CHECK(interpreter_->tensor(interpreter_->inputs()[0])->quantization.type !=
        kTfLiteAffineQuantization);

  return mediapipe::OkStatus();
}

mediapipe::StatusOr<mediapipe::Packet> InferenceCalculator::GetModelAsPacket(
    const CalculatorContext& cc) {
  const auto& options = cc.Options<mediapipe::InferenceCalculatorOptions>();
  if (!options.model_path().empty()) {
    return TfLiteModelLoader::LoadFromPath(options.model_path());
  }
  if (cc.InputSidePackets().HasTag("MODEL")) {
    return cc.InputSidePackets().Tag("MODEL");
  }
  return mediapipe::Status(
      mediapipe::StatusCode::kNotFound,
      "Must specify TFLite model as path or loaded model.");
}

mediapipe::Status InferenceCalculator::LoadDelegate(CalculatorContext* cc) {
  const auto& calculator_opts =
      cc->Options<mediapipe::InferenceCalculatorOptions>();
  if (calculator_opts.has_delegate() &&
      calculator_opts.delegate().has_tflite()) {
    // Default tflite inference requeqsted - no need to modify graph.
    return mediapipe::OkStatus();
  }

  if (!use_gpu_delegate_) {
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
      return mediapipe::OkStatus();
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
      return mediapipe::OkStatus();
    }
#endif  // !EDGETPU

    // Return, no need for GPU delegate below.
    return mediapipe::OkStatus();
  } else {
#if MEDIAPIPE_TFLITE_GL_INFERENCE
    // Configure and create the delegate.
    TfLiteGpuDelegateOptions options = TfLiteGpuDelegateOptionsDefault();
    options.compile_options.precision_loss_allowed = 1;
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
      RET_CHECK_EQ(
          TfLiteGpuDelegateBindBufferToTensor(
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
      RET_CHECK_EQ(
          TfLiteGpuDelegateBindBufferToTensor(
              delegate_.get(),
              gpu_buffers_out_.back()->GetOpenGlBufferWriteView().name(),
              output_indices[i]),
          kTfLiteOk);
    }

    // Must call this last.
    RET_CHECK_EQ(interpreter_->ModifyGraphWithDelegate(delegate_.get()),
                 kTfLiteOk);
#elif MEDIAPIPE_TFLITE_METAL_INFERENCE
    // Configure and create the delegate.
    TFLGpuDelegateOptions options;
    options.allow_precision_loss = true;
    options.wait_type = TFLGpuDelegateWaitType::TFLGpuDelegateWaitTypeDoNotWait;
    delegate_ = TfLiteDelegatePtr(TFLGpuDelegateCreate(&options),
                                  &TFLGpuDelegateDelete);
    RET_CHECK_EQ(interpreter_->ModifyGraphWithDelegate(delegate_.get()),
                 kTfLiteOk);
    id<MTLDevice> device = gpu_helper_.mtlDevice;

    // Get input image sizes.
    const auto& input_indices = interpreter_->inputs();
    for (int i = 0; i < input_indices.size(); ++i) {
      const TfLiteTensor* tensor = interpreter_->tensor(input_indices[i]);
      // Create and bind input buffer.
      std::vector<int> dims{tensor->dims->data,
                            tensor->dims->data + tensor->dims->size};
      dims.back() = RoundUp(dims.back(), 4);
      gpu_buffers_in_.emplace_back(absl::make_unique<Tensor>(
          Tensor::ElementType::kFloat16, Tensor::Shape{dims}));
      auto buffer_view =
          gpu_buffers_in_[i]->GetMtlBufferWriteView(gpu_helper_.mtlDevice);
      RET_CHECK_EQ(TFLGpuDelegateBindMetalBufferToTensor(
                       delegate_.get(), input_indices[i], buffer_view.buffer()),
                   true);
    }

    interpreter_->SetAllowBufferHandleOutput(true);
    // Get output image sizes.
    const auto& output_indices = interpreter_->outputs();
    output_shapes_.resize(output_indices.size());
    for (int i = 0; i < output_shapes_.size(); ++i) {
      const TfLiteTensor* tensor = interpreter_->tensor(output_indices[i]);
      RET_CHECK(tensor->dims->size <= 4);
      // Create and bind output buffers.
      // Channels are always padded to multiple of 4.
      std::vector<int> dims{tensor->dims->data,
                            tensor->dims->data + tensor->dims->size};
      output_shapes_[i] = {dims};
      dims.back() = RoundUp(dims.back(), 4);
      gpu_buffers_out_.emplace_back(absl::make_unique<Tensor>(
          Tensor::ElementType::kFloat16, Tensor::Shape{dims}));
      RET_CHECK_EQ(TFLGpuDelegateBindMetalBufferToTensor(
                       delegate_.get(), output_indices[i],
                       gpu_buffers_out_[i]
                           ->GetMtlBufferWriteView(gpu_helper_.mtlDevice)
                           .buffer()),
                   true);
    }

    // Create converter for GPU input.
    converter_to_BPHWC4_ = [[TFLBufferConvert alloc] initWithDevice:device
                                                          isFloat16:true
                                                    convertToPBHWC4:true];
    if (converter_to_BPHWC4_ == nil) {
      return mediapipe::InternalError(
          "Error initializating input buffer converter");
    }
    // Create converter for GPU output.
    converter_from_BPHWC4_ = [[TFLBufferConvert alloc] initWithDevice:device
                                                            isFloat16:true
                                                      convertToPBHWC4:false];
    if (converter_from_BPHWC4_ == nil) {
      return mediapipe::InternalError(
          "Error initializating output buffer converter");
    }
#endif  // MEDIAPIPE_TFLITE_GL_INFERENCE
  }

  return mediapipe::OkStatus();
}

}  // namespace api2
}  // namespace mediapipe
