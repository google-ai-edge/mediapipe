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
//

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
#endif  // ANDROID

#if defined(__APPLE__) && !TARGET_OS_OSX  // iOS
#if defined(__OBJC__)
#import <CoreVideo/CoreVideo.h>
#import <MetalKit/MetalKit.h>
#endif  // OBJC
#import "mediapipe/framework/ios/NSError+util_status.h"
#import "mediapipe/gpu/MediaPipeMetalHelper.h"
#include "tensorflow/lite/delegates/gpu/metal_delegate.h"
#endif  // APPLE && !TARGET_OS_OSX

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
  GlBuffer ssbo;
  GlShader shader;
  GlProgram program;
};
#endif  // ANDROID

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
//  TENSORS - Vector of TfLiteTensor of type kTfLiteFloat32
//  TENSORS_GPU - Vector of GlBuffer (assumed to be RGB image)
//
// Output:
//  TENSORS - Vector of TfLiteTensor of type kTfLiteFloat32
//  TENSORS_GPU - Vector of GlBuffer
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
//  GPU tensors are currently only supported on Android.
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
#endif
#if defined(__APPLE__) && !TARGET_OS_OSX  // iOS
  MediaPipeMetalHelper* gpu_helper_ = nullptr;
#endif

  std::string model_path_ = "";
  bool gpu_inference_ = false;
  bool gpu_input_ = false;
  bool gpu_output_ = false;
};  // TfLiteInferenceCalculator

REGISTER_CALCULATOR(TfLiteInferenceCalculator);

// Calculator Core Section

::mediapipe::Status TfLiteInferenceCalculator::GetContract(
    CalculatorContract* cc) {
  RET_CHECK(cc->Inputs().HasTag("TENSORS") ||
            cc->Inputs().HasTag("TENSORS_GPU"));
  RET_CHECK(cc->Outputs().HasTag("TENSORS") ||
            cc->Outputs().HasTag("TENSORS_GPU"));

  if (cc->Inputs().HasTag("TENSORS"))
    cc->Inputs().Tag("TENSORS").Set<std::vector<TfLiteTensor>>();
#if defined(__ANDROID__)
  if (cc->Inputs().HasTag("TENSORS_GPU"))
    cc->Inputs().Tag("TENSORS_GPU").Set<std::vector<GlBuffer>>();
#endif

  if (cc->Outputs().HasTag("TENSORS"))
    cc->Outputs().Tag("TENSORS").Set<std::vector<TfLiteTensor>>();
#if defined(__ANDROID__)
  if (cc->Outputs().HasTag("TENSORS_GPU"))
    cc->Outputs().Tag("TENSORS_GPU").Set<std::vector<GlBuffer>>();
#endif

  if (cc->InputSidePackets().HasTag("CUSTOM_OP_RESOLVER")) {
    cc->InputSidePackets()
        .Tag("CUSTOM_OP_RESOLVER")
        .Set<tflite::ops::builtin::BuiltinOpResolver>();
  }

#if defined(__ANDROID__)
  RETURN_IF_ERROR(mediapipe::GlCalculatorHelper::UpdateContract(cc));
#elif defined(__APPLE__) && !TARGET_OS_OSX  // iOS
  RETURN_IF_ERROR([MediaPipeMetalHelper updateContract:cc]);
#endif

  // Assign this calculator's default InputStreamHandler.
  cc->SetInputStreamHandler("FixedSizeInputStreamHandler");

  return ::mediapipe::OkStatus();
}

::mediapipe::Status TfLiteInferenceCalculator::Open(CalculatorContext* cc) {
  RETURN_IF_ERROR(LoadOptions(cc));

  if (cc->Inputs().HasTag("TENSORS_GPU")) {
#if defined(__ANDROID__)
    gpu_input_ = true;
    gpu_inference_ = true;  // Inference must be on GPU also.
#else
    RET_CHECK(!cc->Inputs().HasTag("TENSORS_GPU"))
        << "GPU input for non-Android not supported yet.";
#endif
  }

  if (cc->Outputs().HasTag("TENSORS_GPU")) {
#if defined(__ANDROID__)
    gpu_output_ = true;
    RET_CHECK(cc->Inputs().HasTag("TENSORS_GPU"))
        << "GPU output must also have GPU Input.";
#else
    RET_CHECK(!cc->Inputs().HasTag("TENSORS_GPU"))
        << "GPU output for non-Android not supported yet.";
#endif
  }

  RETURN_IF_ERROR(LoadModel(cc));

  if (gpu_inference_) {
#if defined(__ANDROID__)
    RETURN_IF_ERROR(gpu_helper_.Open(cc));
#elif defined(__APPLE__) && !TARGET_OS_OSX  // iOS
    gpu_helper_ = [[MediaPipeMetalHelper alloc] initWithCalculatorContext:cc];
    RET_CHECK(gpu_helper_);
#endif

    RETURN_IF_ERROR(LoadDelegate(cc));
  }

  return ::mediapipe::OkStatus();
}

::mediapipe::Status TfLiteInferenceCalculator::Process(CalculatorContext* cc) {
  // Receive pre-processed tensor inputs.
  if (gpu_input_) {
    // Read GPU input into SSBO.
#if defined(__ANDROID__)
    const auto& input_tensors =
        cc->Inputs().Tag("TENSORS_GPU").Get<std::vector<GlBuffer>>();
    RET_CHECK_EQ(input_tensors.size(), 1);
    RETURN_IF_ERROR(gpu_helper_.RunInGlContext(
        [this, &input_tensors]() -> ::mediapipe::Status {
          // Explicit copy input.
          tflite::gpu::gl::CopyBuffer(input_tensors[0], gpu_data_in_->ssbo);
          // Run inference.
          RET_CHECK_EQ(interpreter_->Invoke(), kTfLiteOk);
          return ::mediapipe::OkStatus();
        }));
#else
    RET_CHECK_FAIL()
        << "GPU input on non-Android devices is not supported yet.";
#endif
  } else {
    // Read CPU input into tensors.
    const auto& input_tensors =
        cc->Inputs().Tag("TENSORS").Get<std::vector<TfLiteTensor>>();
    RET_CHECK_GT(input_tensors.size(), 0);
    for (int i = 0; i < input_tensors.size(); ++i) {
      const TfLiteTensor* input_tensor = &input_tensors[i];
      const float* input_tensor_buffer = input_tensor->data.f;
      RET_CHECK(input_tensor_buffer);

      float* local_tensor_buffer = interpreter_->typed_input_tensor<float>(i);
      RET_CHECK(local_tensor_buffer);

      memcpy(local_tensor_buffer, input_tensor_buffer, input_tensor->bytes);
    }

    // Run inference.
    if (gpu_inference_) {
#if defined(__ANDROID__)
      RETURN_IF_ERROR(
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
  }

  if (gpu_output_) {
#if defined(__ANDROID__)
    // Output result tensors (GPU).
    auto output_tensors = absl::make_unique<std::vector<GlBuffer>>();
    output_tensors->resize(gpu_data_out_.size());
    for (int i = 0; i < gpu_data_out_.size(); ++i) {
      GlBuffer& tensor = output_tensors->at(i);
      using ::tflite::gpu::gl::CreateReadWriteShaderStorageBuffer;
      auto status = CreateReadWriteShaderStorageBuffer<float>(
          gpu_data_out_[i]->elements, &tensor);
      if (!status.ok()) {
        return ::mediapipe::InternalError(status.error_message());
      }
      tflite::gpu::gl::CopyBuffer(gpu_data_out_[i]->ssbo, tensor);
    }
    cc->Outputs()
        .Tag("TENSORS_GPU")
        .Add(output_tensors.release(), cc->InputTimestamp());
#else
    LOG(ERROR) << "GPU output on non-Android not supported yet.";
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
    RETURN_IF_ERROR(gpu_helper_.RunInGlContext([this]() -> Status {
      TfLiteGpuDelegateDelete(delegate_);
      gpu_data_in_.reset();
      for (int i = 0; i < gpu_data_out_.size(); ++i) {
        gpu_data_out_[i].reset();
      }
      return ::mediapipe::OkStatus();
    }));
#elif defined(__APPLE__) && !TARGET_OS_OSX  // iOS
    DeleteGpuDelegate(delegate_);
#endif
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
    ASSIGN_OR_RETURN(model_path_,
                     mediapipe::PathToResourceAsFile(options.model_path()));
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

#if !defined(__ANDROID__) && !(defined(__APPLE__) && !TARGET_OS_OSX)
  LOG(WARNING) << "GPU only supported on mobile platforms. Using CPU fallback.";
  gpu_inference_ = false;
#endif

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

  if (!gpu_output_) {
    RET_CHECK_EQ(interpreter_->AllocateTensors(), kTfLiteOk);
  }

  return ::mediapipe::OkStatus();
}

::mediapipe::Status TfLiteInferenceCalculator::LoadDelegate(
    CalculatorContext* cc) {
#if defined(__ANDROID__)
  // Get input image sizes.
  if (gpu_input_) {
    gpu_data_in_ = absl::make_unique<GPUData>();
    const auto& input_indices = interpreter_->inputs();
    // TODO accept > 1.
    RET_CHECK_EQ(input_indices.size(), 1);
    const TfLiteTensor* tensor = interpreter_->tensor(input_indices[0]);
    gpu_data_in_->elements = 1;
    for (int d = 0; d < tensor->dims->size; ++d) {
      gpu_data_in_->elements *= tensor->dims->data[d];
    }
    // Input to model can be either RGB/RGBA only.
    RET_CHECK_GE(tensor->dims->data[3], 3);
    RET_CHECK_LE(tensor->dims->data[3], 4);
  }
  // Get output image sizes.
  if (gpu_output_) {
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
  }
  // Configure and create the delegate.
  TfLiteGpuDelegateOptions options;
  options.metadata = nullptr;
  options.compile_options.precision_loss_allowed = 1;
  options.compile_options.preferred_gl_object_type =
      TFLITE_GL_OBJECT_TYPE_FASTEST;
  options.compile_options.dynamic_batch_enabled = 0;
  if (!delegate_) delegate_ = TfLiteGpuDelegateCreate(&options);
  // Shader to convert GL texture to SSBO.
  if (gpu_input_) {
    auto status = ::tflite::gpu::gl::CreateReadWriteShaderStorageBuffer<float>(
        gpu_data_in_->elements, &gpu_data_in_->ssbo);
    if (!status.ok()) {
      return ::mediapipe::InternalError(status.error_message());
    }
    RET_CHECK_EQ(TfLiteGpuDelegateBindBufferToTensor(
                     delegate_, gpu_data_in_->ssbo.id(),
                     interpreter_->inputs()[0]),  // First tensor only
                 kTfLiteOk);
  }
  // Create output SSBO buffers.
  if (gpu_output_) {
    interpreter_->SetAllowBufferHandleOutput(true);
    const auto& output_indices = interpreter_->outputs();
    for (int i = 0; i < gpu_data_out_.size(); ++i) {
      using ::tflite::gpu::gl::CreateReadWriteShaderStorageBuffer;
      auto status = CreateReadWriteShaderStorageBuffer<float>(
          gpu_data_out_[i]->elements, &gpu_data_out_[i]->ssbo);
      if (!status.ok()) {
        return ::mediapipe::InternalError(status.error_message());
      }
      RET_CHECK_EQ(
          TfLiteGpuDelegateBindBufferToTensor(
              delegate_, gpu_data_out_[i]->ssbo.id(), output_indices[i]),
          kTfLiteOk);
    }
  }
  // Must call this last.
  RET_CHECK_EQ(interpreter_->ModifyGraphWithDelegate(delegate_), kTfLiteOk);
  return ::mediapipe::OkStatus();
#elif defined(__APPLE__) && !TARGET_OS_OSX  // iOS
  GpuDelegateOptions options;
  options.allow_precision_loss = 1;
  options.wait_type = GpuDelegateOptions::WaitType::kPassive;
  if (!delegate_) delegate_ = NewGpuDelegate(&options);
  RET_CHECK_EQ(interpreter_->ModifyGraphWithDelegate(delegate_), kTfLiteOk);
  return ::mediapipe::OkStatus();
#endif                                      // ANDROID or iOS

  return ::mediapipe::OkStatus();
}

}  // namespace mediapipe
