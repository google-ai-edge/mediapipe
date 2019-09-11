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

#include <string>
#include <vector>

#include "mediapipe/calculators/tflite/tflite_inference_calculator.pb.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/port/ret_check.h"
#include "mediapipe/util/resource_util.h"
#include "tensorflow/lite/error_reporter.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"

#if defined(__ANDROID__)
#include "mediapipe/gpu/gl_calculator_helper.h"
#include "mediapipe/gpu/gpu_buffer.h"
#include "tensorflow/lite/delegates/gpu/gl/gl_buffer.h"
#include "tensorflow/lite/delegates/gpu/gl/gl_program.h"
#include "tensorflow/lite/delegates/gpu/gl/gl_shader.h"
#include "tensorflow/lite/delegates/gpu/gl_delegate.h"
#endif  // __ANDROID__

#if defined(__APPLE__) && !TARGET_OS_OSX  // iOS
#import <CoreVideo/CoreVideo.h>
#import <Metal/Metal.h>
#import <MetalKit/MetalKit.h>

#import "mediapipe/gpu/MPPMetalHelper.h"
#include "tensorflow/lite/delegates/gpu/metal_delegate.h"
#endif  // iOS

#if defined(__ANDROID__)
typedef ::tflite::gpu::gl::GlBuffer GpuTensor;
#elif defined(__APPLE__) && !TARGET_OS_OSX  // iOS
typedef id<MTLBuffer> GpuTensor;
#endif

// TfLiteInferenceCalculator File Layout:
//  * Header
//  * Core
//  * Aux
namespace mediapipe {

#if defined(__ANDROID__)
using ::tflite::gpu::gl::GlBuffer;
using ::tflite::gpu::gl::GlProgram;
using ::tflite::gpu::gl::GlShader;
struct GPUData {
  int elements = 1;
  GlBuffer buffer;
};
#elif defined(__APPLE__) && !TARGET_OS_OSX  // iOS
struct GPUData {
  int elements = 1;
  id<MTLBuffer> buffer;
};
#endif

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
//
// Example use:
// node {
//   calculator: "TfLiteInferenceCalculator"
//   input_stream: "TENSORS:tensor_image"
//   output_stream: "TENSORS:tensors"
//   options: {
//     [mediapipe.TfLiteInferenceCalculatorOptions.ext] {
//       model_path: "modelname.tflite"
//       use_gpu: true
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
  static ::mediapipe::Status GetContract(CalculatorContract* cc);

  ::mediapipe::Status Open(CalculatorContext* cc) override;
  ::mediapipe::Status Process(CalculatorContext* cc) override;
  ::mediapipe::Status Close(CalculatorContext* cc) override;

 private:
  ::mediapipe::Status LoadOptions(CalculatorContext* cc);
  ::mediapipe::Status LoadModel(CalculatorContext* cc);
  ::mediapipe::Status LoadDelegate(CalculatorContext* cc);

  std::unique_ptr<tflite::Interpreter> interpreter_;
  std::unique_ptr<tflite::FlatBufferModel> model_;
  TfLiteDelegate* delegate_ = nullptr;

#if defined(__ANDROID__)
  mediapipe::GlCalculatorHelper gpu_helper_;
  std::unique_ptr<GPUData> gpu_data_in_;
  std::vector<std::unique_ptr<GPUData>> gpu_data_out_;
#elif defined(__APPLE__) && !TARGET_OS_OSX  // iOS
  MPPMetalHelper* gpu_helper_ = nullptr;
  std::unique_ptr<GPUData> gpu_data_in_;
  std::vector<std::unique_ptr<GPUData>> gpu_data_out_;
#endif

  std::string model_path_ = "";
  bool gpu_inference_ = false;
  bool gpu_input_ = false;
  bool gpu_output_ = false;
  bool use_quantized_tensors_ = false;
};
REGISTER_CALCULATOR(TfLiteInferenceCalculator);

// Calculator Core Section

::mediapipe::Status TfLiteInferenceCalculator::GetContract(
    CalculatorContract* cc) {
  RET_CHECK(cc->Inputs().HasTag("TENSORS") ^
            cc->Inputs().HasTag("TENSORS_GPU"));
  RET_CHECK(cc->Outputs().HasTag("TENSORS") ^
            cc->Outputs().HasTag("TENSORS_GPU"));

  if (cc->Inputs().HasTag("TENSORS"))
    cc->Inputs().Tag("TENSORS").Set<std::vector<TfLiteTensor>>();
#if defined(__ANDROID__) || (defined(__APPLE__) && !TARGET_OS_OSX)
  if (cc->Inputs().HasTag("TENSORS_GPU"))
    cc->Inputs().Tag("TENSORS_GPU").Set<std::vector<GpuTensor>>();
#endif

  if (cc->Outputs().HasTag("TENSORS"))
    cc->Outputs().Tag("TENSORS").Set<std::vector<TfLiteTensor>>();
#if defined(__ANDROID__) || (defined(__APPLE__) && !TARGET_OS_OSX)
  if (cc->Outputs().HasTag("TENSORS_GPU"))
    cc->Outputs().Tag("TENSORS_GPU").Set<std::vector<GpuTensor>>();
#endif

  if (cc->InputSidePackets().HasTag("CUSTOM_OP_RESOLVER")) {
    cc->InputSidePackets()
        .Tag("CUSTOM_OP_RESOLVER")
        .Set<tflite::ops::builtin::BuiltinOpResolver>();
  }

#if defined(__ANDROID__)
  MP_RETURN_IF_ERROR(mediapipe::GlCalculatorHelper::UpdateContract(cc));
#elif defined(__APPLE__) && !TARGET_OS_OSX  // iOS
  MP_RETURN_IF_ERROR([MPPMetalHelper updateContract:cc]);
#endif

  // Assign this calculator's default InputStreamHandler.
  cc->SetInputStreamHandler("FixedSizeInputStreamHandler");

  return ::mediapipe::OkStatus();
}

::mediapipe::Status TfLiteInferenceCalculator::Open(CalculatorContext* cc) {
  cc->SetOffset(TimestampDiff(0));

  MP_RETURN_IF_ERROR(LoadOptions(cc));

  if (cc->Inputs().HasTag("TENSORS_GPU")) {
#if defined(__ANDROID__) || (defined(__APPLE__) && !TARGET_OS_OSX)
    gpu_input_ = true;
    gpu_inference_ = true;  // Inference must be on GPU also.
#else
    RET_CHECK_FAIL() << "GPU processing is for Android and iOS only.";
#endif
  }

  if (cc->Outputs().HasTag("TENSORS_GPU")) {
#if defined(__ANDROID__) || (defined(__APPLE__) && !TARGET_OS_OSX)
    gpu_output_ = true;
    RET_CHECK(cc->Inputs().HasTag("TENSORS_GPU"))
        << "GPU output must also have GPU Input.";
#else
    RET_CHECK_FAIL() << "GPU processing is for Android and iOS only.";
#endif
  }

  MP_RETURN_IF_ERROR(LoadModel(cc));

  if (gpu_inference_) {
#if defined(__ANDROID__)
    MP_RETURN_IF_ERROR(gpu_helper_.Open(cc));
#elif defined(__APPLE__) && !TARGET_OS_OSX  // iOS
    gpu_helper_ = [[MPPMetalHelper alloc] initWithCalculatorContext:cc];
    RET_CHECK(gpu_helper_);
#endif

    MP_RETURN_IF_ERROR(LoadDelegate(cc));
  }

  return ::mediapipe::OkStatus();
}

::mediapipe::Status TfLiteInferenceCalculator::Process(CalculatorContext* cc) {
  // 1. Receive pre-processed tensor inputs.
  if (gpu_input_) {
    // Read GPU input into SSBO.
#if defined(__ANDROID__)
    const auto& input_tensors =
        cc->Inputs().Tag("TENSORS_GPU").Get<std::vector<GpuTensor>>();
    RET_CHECK_EQ(input_tensors.size(), 1);
    MP_RETURN_IF_ERROR(gpu_helper_.RunInGlContext(
        [this, &input_tensors]() -> ::mediapipe::Status {
          // Explicit copy input.
          tflite::gpu::gl::CopyBuffer(input_tensors[0], gpu_data_in_->buffer);
          return ::mediapipe::OkStatus();
        }));
#elif defined(__APPLE__) && !TARGET_OS_OSX  // iOS
    const auto& input_tensors =
        cc->Inputs().Tag("TENSORS_GPU").Get<std::vector<GpuTensor>>();
    RET_CHECK_EQ(input_tensors.size(), 1);
    id<MTLCommandBuffer> command_buffer = [gpu_helper_ commandBuffer];
    command_buffer.label = @"TfLiteInferenceCalculatorInput";
    id<MTLBlitCommandEncoder> blit_command =
        [command_buffer blitCommandEncoder];
    // Explicit copy input.
    [blit_command copyFromBuffer:input_tensors[0]
                    sourceOffset:0
                        toBuffer:gpu_data_in_->buffer
               destinationOffset:0
                            size:gpu_data_in_->elements * sizeof(float)];
    [blit_command endEncoding];
    [command_buffer commit];
    [command_buffer waitUntilCompleted];
#else
    RET_CHECK_FAIL() << "GPU processing is for Android and iOS only.";
#endif
  } else {
    // Read CPU input into tensors.
    const auto& input_tensors =
        cc->Inputs().Tag("TENSORS").Get<std::vector<TfLiteTensor>>();
    RET_CHECK_GT(input_tensors.size(), 0);
    for (int i = 0; i < input_tensors.size(); ++i) {
      const TfLiteTensor* input_tensor = &input_tensors[i];
      RET_CHECK(input_tensor->data.raw);
      if (use_quantized_tensors_) {
        const uint8* input_tensor_buffer = input_tensor->data.uint8;
        uint8* local_tensor_buffer = interpreter_->typed_input_tensor<uint8>(i);
        memcpy(local_tensor_buffer, input_tensor_buffer, input_tensor->bytes);
      } else {
        const float* input_tensor_buffer = input_tensor->data.f;
        float* local_tensor_buffer = interpreter_->typed_input_tensor<float>(i);
        memcpy(local_tensor_buffer, input_tensor_buffer, input_tensor->bytes);
      }
    }
  }

  // 2. Run inference.
  if (gpu_inference_) {
#if defined(__ANDROID__)
    MP_RETURN_IF_ERROR(
        gpu_helper_.RunInGlContext([this]() -> ::mediapipe::Status {
          RET_CHECK_EQ(interpreter_->Invoke(), kTfLiteOk);
          return ::mediapipe::OkStatus();
        }));
#elif defined(__APPLE__) && !TARGET_OS_OSX  // iOS
    RET_CHECK_EQ(interpreter_->Invoke(), kTfLiteOk);
#endif
  } else {
    RET_CHECK_EQ(interpreter_->Invoke(), kTfLiteOk);
  }

  // 3. Output processed tensors.
  if (gpu_output_) {
#if defined(__ANDROID__)
    // Output result tensors (GPU).
    auto output_tensors = absl::make_unique<std::vector<GpuTensor>>();
    output_tensors->resize(gpu_data_out_.size());
    for (int i = 0; i < gpu_data_out_.size(); ++i) {
      GlBuffer& tensor = output_tensors->at(i);
      using ::tflite::gpu::gl::CreateReadWriteShaderStorageBuffer;
      auto status = CreateReadWriteShaderStorageBuffer<float>(
          gpu_data_out_[i]->elements, &tensor);
      if (!status.ok()) {
        return ::mediapipe::InternalError(status.error_message());
      }
      tflite::gpu::gl::CopyBuffer(gpu_data_out_[i]->buffer, tensor);
    }
    cc->Outputs()
        .Tag("TENSORS_GPU")
        .Add(output_tensors.release(), cc->InputTimestamp());
#elif defined(__APPLE__) && !TARGET_OS_OSX  // iOS
    // Output result tensors (GPU).
    auto output_tensors = absl::make_unique<std::vector<GpuTensor>>();
    id<MTLDevice> device = gpu_helper_.mtlDevice;
    id<MTLCommandBuffer> command_buffer = [gpu_helper_ commandBuffer];
    command_buffer.label = @"TfLiteInferenceCalculatorOutput";
    for (int i = 0; i < gpu_data_out_.size(); ++i) {
      id<MTLBuffer> tensor =
          [device newBufferWithLength:gpu_data_out_[i]->elements * sizeof(float)
                              options:MTLResourceStorageModeShared];
      id<MTLBlitCommandEncoder> blit_command =
          [command_buffer blitCommandEncoder];
      // Explicit copy input.
      [blit_command copyFromBuffer:gpu_data_out_[i]->buffer
                      sourceOffset:0
                          toBuffer:tensor
                 destinationOffset:0
                              size:gpu_data_out_[i]->elements * sizeof(float)];
      [blit_command endEncoding];
      [command_buffer commit];
      [command_buffer waitUntilCompleted];
      output_tensors->push_back(tensor);
    }
    cc->Outputs()
        .Tag("TENSORS_GPU")
        .Add(output_tensors.release(), cc->InputTimestamp());
#else
    RET_CHECK_FAIL() << "GPU processing is for Android and iOS only.";
#endif
  } else {
    // Output result tensors (CPU).
    const auto& tensor_indexes = interpreter_->outputs();
    auto output_tensors = absl::make_unique<std::vector<TfLiteTensor>>();
    for (int i = 0; i < tensor_indexes.size(); ++i) {
      TfLiteTensor* tensor = interpreter_->tensor(tensor_indexes[i]);
      output_tensors->emplace_back(*tensor);
    }
    cc->Outputs().Tag("TENSORS").Add(output_tensors.release(),
                                     cc->InputTimestamp());
  }

  return ::mediapipe::OkStatus();
}

::mediapipe::Status TfLiteInferenceCalculator::Close(CalculatorContext* cc) {
  if (delegate_) {
#if defined(__ANDROID__)
    MP_RETURN_IF_ERROR(gpu_helper_.RunInGlContext([this]() -> Status {
      TfLiteGpuDelegateDelete(delegate_);
      gpu_data_in_.reset();
      for (int i = 0; i < gpu_data_out_.size(); ++i) {
        gpu_data_out_[i].reset();
      }
      return ::mediapipe::OkStatus();
    }));
#elif defined(__APPLE__) && !TARGET_OS_OSX  // iOS
    TFLGpuDelegateDelete(delegate_);
    gpu_data_in_.reset();
    for (int i = 0; i < gpu_data_out_.size(); ++i) {
      gpu_data_out_[i].reset();
    }
#endif
    delegate_ = nullptr;
  }
  return ::mediapipe::OkStatus();
}

// Calculator Auxiliary Section

::mediapipe::Status TfLiteInferenceCalculator::LoadOptions(
    CalculatorContext* cc) {
  // Get calculator options specified in the graph.
  const auto& options =
      cc->Options<::mediapipe::TfLiteInferenceCalculatorOptions>();

  // Get model name.
  if (!options.model_path().empty()) {
    auto model_path = options.model_path();

    ASSIGN_OR_RETURN(model_path_, mediapipe::PathToResourceAsFile(model_path));
  } else {
    LOG(ERROR) << "Must specify path to TFLite model.";
    return ::mediapipe::Status(::mediapipe::StatusCode::kNotFound,
                               "Must specify path to TFLite model.");
  }

  // Get execution modes.
  gpu_inference_ = options.use_gpu();

  return ::mediapipe::OkStatus();
}

::mediapipe::Status TfLiteInferenceCalculator::LoadModel(
    CalculatorContext* cc) {
  model_ = tflite::FlatBufferModel::BuildFromFile(model_path_.c_str());
  RET_CHECK(model_);

  if (cc->InputSidePackets().HasTag("CUSTOM_OP_RESOLVER")) {
    const auto& op_resolver =
        cc->InputSidePackets()
            .Tag("CUSTOM_OP_RESOLVER")
            .Get<tflite::ops::builtin::BuiltinOpResolver>();
    tflite::InterpreterBuilder(*model_, op_resolver)(&interpreter_);
  } else {
    const tflite::ops::builtin::BuiltinOpResolver op_resolver;
    tflite::InterpreterBuilder(*model_, op_resolver)(&interpreter_);
  }

  RET_CHECK(interpreter_);

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

::mediapipe::Status TfLiteInferenceCalculator::LoadDelegate(
    CalculatorContext* cc) {
#if defined(__ANDROID__)
  // Configure and create the delegate.
  TfLiteGpuDelegateOptions options = TfLiteGpuDelegateOptionsDefault();
  options.compile_options.precision_loss_allowed = 1;
  options.compile_options.preferred_gl_object_type =
      TFLITE_GL_OBJECT_TYPE_FASTEST;
  options.compile_options.dynamic_batch_enabled = 0;
  options.compile_options.inline_parameters = 1;
  if (!delegate_) delegate_ = TfLiteGpuDelegateCreate(&options);

  if (gpu_input_) {
    // Get input image sizes.
    gpu_data_in_ = absl::make_unique<GPUData>();
    const auto& input_indices = interpreter_->inputs();
    RET_CHECK_EQ(input_indices.size(), 1);  // TODO accept > 1.
    const TfLiteTensor* tensor = interpreter_->tensor(input_indices[0]);
    gpu_data_in_->elements = 1;
    for (int d = 0; d < tensor->dims->size; ++d) {
      gpu_data_in_->elements *= tensor->dims->data[d];
    }
    // Input to model can be either RGB/RGBA only.
    RET_CHECK_GE(tensor->dims->data[3], 3);
    RET_CHECK_LE(tensor->dims->data[3], 4);
    // Create and bind input buffer.
    auto status = ::tflite::gpu::gl::CreateReadWriteShaderStorageBuffer<float>(
        gpu_data_in_->elements, &gpu_data_in_->buffer);
    if (!status.ok()) {
      return ::mediapipe::InternalError(status.error_message());
    }
    RET_CHECK_EQ(TfLiteGpuDelegateBindBufferToTensor(
                     delegate_, gpu_data_in_->buffer.id(),
                     interpreter_->inputs()[0]),  // First tensor only
                 kTfLiteOk);
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
      using ::tflite::gpu::gl::CreateReadWriteShaderStorageBuffer;
      auto status = CreateReadWriteShaderStorageBuffer<float>(
          gpu_data_out_[i]->elements, &gpu_data_out_[i]->buffer);
      if (!status.ok()) {
        return ::mediapipe::InternalError(status.error_message());
      }
      RET_CHECK_EQ(
          TfLiteGpuDelegateBindBufferToTensor(
              delegate_, gpu_data_out_[i]->buffer.id(), output_indices[i]),
          kTfLiteOk);
    }
  }

  // Must call this last.
  RET_CHECK_EQ(interpreter_->ModifyGraphWithDelegate(delegate_), kTfLiteOk);
#endif  // __ANDROID__

#if defined(__APPLE__) && !TARGET_OS_OSX  // iOS
  // Configure and create the delegate.
  GpuDelegateOptions options;
  options.allow_precision_loss = false;  // Must match converter, F=float/T=half
  options.wait_type = GpuDelegateOptions::WaitType::kActive;
  if (!delegate_) delegate_ = TFLGpuDelegateCreate(&options);

  if (gpu_input_) {
    // Get input image sizes.
    gpu_data_in_ = absl::make_unique<GPUData>();
    const auto& input_indices = interpreter_->inputs();
    RET_CHECK_EQ(input_indices.size(), 1);
    const TfLiteTensor* tensor = interpreter_->tensor(input_indices[0]);
    gpu_data_in_->elements = 1;
    // On iOS GPU, input must be 4 channels, regardless of what model expects.
    {
      gpu_data_in_->elements *= tensor->dims->data[0];  // batch
      gpu_data_in_->elements *= tensor->dims->data[1];  // height
      gpu_data_in_->elements *= tensor->dims->data[2];  // width
      gpu_data_in_->elements *= 4;                      // channels
    }
    // Input to model can be RGBA only.
    if (tensor->dims->data[3] != 4) {
      LOG(WARNING) << "Please ensure input GPU tensor is 4 channels.";
    }
    // Create and bind input buffer.
    id<MTLDevice> device = gpu_helper_.mtlDevice;
    gpu_data_in_->buffer =
        [device newBufferWithLength:gpu_data_in_->elements * sizeof(float)
                            options:MTLResourceStorageModeShared];
    // Must call this before TFLGpuDelegateBindMetalBufferToTensor.
    RET_CHECK_EQ(interpreter_->ModifyGraphWithDelegate(delegate_), kTfLiteOk);
    RET_CHECK_EQ(TFLGpuDelegateBindMetalBufferToTensor(
                     delegate_,
                     input_indices[0],  // First tensor only
                     gpu_data_in_->buffer),
                 true);
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
    id<MTLDevice> device = gpu_helper_.mtlDevice;
    for (int i = 0; i < gpu_data_out_.size(); ++i) {
      gpu_data_out_[i]->buffer =
          [device newBufferWithLength:gpu_data_out_[i]->elements * sizeof(float)
                              options:MTLResourceStorageModeShared];
      RET_CHECK_EQ(TFLGpuDelegateBindMetalBufferToTensor(
                       delegate_, output_indices[i], gpu_data_out_[i]->buffer),
                   true);
    }
  }
#endif  // iOS

  return ::mediapipe::OkStatus();
}

}  // namespace mediapipe
