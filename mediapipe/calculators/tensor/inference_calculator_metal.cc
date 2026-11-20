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

#import <CoreVideo/CoreVideo.h>
#import <Metal/Metal.h>
#import <MetalKit/MetalKit.h>

#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include "absl/log/absl_log.h"          // from @com_google_absl
#include "absl/memory/memory.h"         // from @com_google_absl
#include "absl/status/status.h"         // from @com_google_absl
#include "absl/status/status_macros.h"  // from @com_google_absl
#include "absl/strings/str_format.h"    // from @com_google_absl
#include "absl/strings/string_view.h"   // from @com_google_absl
#include "mediapipe/calculators/tensor/inference_calculator.h"
#include "mediapipe/calculators/tensor/inference_io_mapper.h"
#include "mediapipe/calculators/tensor/tensor_span.h"
#include "mediapipe/framework/formats/tensor.h"
#include "mediapipe/framework/formats/tensor_mtl_buffer_view.h"
#import "mediapipe/gpu/MPPMetalHelper.h"
#include "mediapipe/gpu/MPPMetalUtil.h"
#include "mediapipe/gpu/gpu_buffer.h"
#include "mediapipe/util/tflite/config.h"
#include "tflite/delegates/gpu/common/shape.h"             // from @litert
#include "tflite/delegates/gpu/metal/buffer_convert.h"     // from @litert
#include "tflite/delegates/gpu/metal_delegate.h"           // from @litert
#include "tflite/delegates/gpu/metal_delegate_internal.h"  // from @litert

namespace {

// Round up n to next multiple of m.
template <typename T>
T RoundUp(T n, T m) {
  return ((n + m - T{1}) / m) * m;
}

}  // namespace

namespace mediapipe {
namespace api2 {

#if MEDIAPIPE_TFLITE_METAL_INFERENCE
namespace {
tflite::gpu::BHWC BhwcFromTensorShape(const Tensor::Shape& shape) {
  tflite::gpu::BHWC result;
  result.b = shape.dims[0];
  switch (shape.dims.size()) {
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
      // Handles <2 and >4.
      ABSL_LOG(FATAL)
          << "Dimensions size must be in range [2,4] for GPU inference, but "
          << shape.dims.size() << " is provided";
  }
  return result;
}
}  // namespace
#endif  // MEDIAPIPE_TFLITE_METAL_INFERENCE

class InferenceCalculatorMetalImpl
    : public InferenceCalculatorNodeImpl<InferenceCalculatorMetal,
                                         InferenceCalculatorMetalImpl> {
 public:
  static absl::Status UpdateContract(CalculatorContract* cc);

  absl::Status Open(CalculatorContext* cc) override;
  absl::Status Close(CalculatorContext* cc) override;

 private:
  absl::StatusOr<std::vector<Tensor>> Process(
      CalculatorContext* cc, const TensorSpan& tensor_span) override;
  absl::Status InitInterpreter(CalculatorContext* cc);
  void CreateDelegate();
  absl::Status CreateConverters(CalculatorContext* cc);

  // TfLite requires us to keep the model and delegate alive as long as the
  // interpreter is; declare `interpreter_` after both so reverse member
  // destruction on `Open()` failure tears down the interpreter first.
  Packet<TfLiteModelPtr> model_packet_;
  TfLiteDelegatePtr delegate_;
  std::unique_ptr<tflite::Interpreter> interpreter_;
  bool allow_precision_loss_ = false;

#if MEDIAPIPE_TFLITE_METAL_INFERENCE
  MPPMetalHelper* gpu_helper_ = nullptr;
  TFLBufferConvert* converter_to_BPHWC4_ = nil;
  TFLBufferConvert* converter_from_BPHWC4_ = nil;
#endif  // MEDIAPIPE_TFLITE_GL_INFERENCE

#if MEDIAPIPE_TFLITE_GPU_SUPPORTED
  std::vector<Tensor::Shape> output_shapes_;
  std::vector<std::unique_ptr<Tensor>> gpu_buffers_in_;
  std::vector<std::unique_ptr<Tensor>> gpu_buffers_out_;
#endif  // MEDIAPIPE_TFLITE_GPU_SUPPORTED
};

absl::Status InferenceCalculatorMetalImpl::UpdateContract(
    CalculatorContract* cc) {
  ABSL_RETURN_IF_ERROR(TensorContractCheck(cc));

  RET_CHECK(!kDelegate(cc).IsConnected())
      << "Delegate configuration through side packet is not supported.";
  const auto& options = cc->Options<mediapipe::InferenceCalculatorOptions>();
  RET_CHECK(!options.model_path().empty() ^ kSideInModel(cc).IsConnected())
      << "Either model as side packet or model path in options is required.";

  WarnFeedbackTensorsUnsupported(cc);
  ABSL_RETURN_IF_ERROR([MPPMetalHelper updateContract:cc]);
  return absl::OkStatus();
}

absl::Status InferenceCalculatorMetalImpl::Open(CalculatorContext* cc) {
  const auto& options = cc->Options<::mediapipe::InferenceCalculatorOptions>();
  allow_precision_loss_ = options.delegate().gpu().allow_precision_loss();

  gpu_helper_ = [[MPPMetalHelper alloc] initWithCalculatorContext:cc];
  RET_CHECK(gpu_helper_);
  return InitInterpreter(cc);
}

absl::StatusOr<std::vector<Tensor>> InferenceCalculatorMetalImpl::Process(
    CalculatorContext* cc, const TensorSpan& tensor_span) {
  const auto& input_indices = interpreter_->inputs();
  if (tensor_span.size() != input_indices.size()) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "Input tensor count (%d) does not match model input count (%d).",
        tensor_span.size(), input_indices.size()));
  }
  for (int i = 0; i < tensor_span.size(); ++i) {
    if (tensor_span[i].element_type() != Tensor::ElementType::kFloat32) {
      return absl::InvalidArgumentError(
          "Only float32 input tensors are supported by the macOS/iOS GPU "
          "delegate.");
    }
    const TfLiteTensor* expected = interpreter_->tensor(input_indices[i]);
    const auto& actual_dims = tensor_span[i].shape().dims;
    bool matches =
        expected != nullptr && expected->dims != nullptr &&
        expected->dims->size >= 2 &&
        actual_dims.size() == static_cast<size_t>(expected->dims->size);
    if (matches) {
      const int rank = expected->dims->size;
      for (int d = 0; matches && d < rank - 1; ++d) {
        matches = (actual_dims[d] == expected->dims->data[d]);
      }
      // On Metal, ImageToTensorCalculator and ImageToBatchTensorCalculator pad
      // 1- or 3-channel image tensors to 4-channel RGBA ([B, H, W, 4]),
      // matching CreateConverters() rounding the bound Metal input buffer's
      // channel dimension up to 4.
      const int expected_last = expected->dims->data[rank - 1];
      const int actual_last = actual_dims[rank - 1];
      const bool allow_rgba_padding = (rank == 4 && expected_last > 0 &&
                                       expected_last < 4 && actual_last == 4);
      matches = matches && (actual_last == expected_last || allow_rgba_padding);
    }
    if (!matches) {
      return absl::InvalidArgumentError(
          "Dynamic-sized tensors or input shapes differing from model "
          "dimensions are not supported by the macOS/iOS GPU delegate.");
    }
  }

  std::vector<Tensor> output_tensors;

  id<MTLCommandBuffer> command_buffer;

  command_buffer = [gpu_helper_ commandBuffer];
  command_buffer.label = @"InferenceCalculator";
  // Explicit copy input with conversion float 32 bits to 16 bits.
  for (int i = 0; i < tensor_span.size(); ++i) {
    auto input_view =
        MtlBufferView::GetReadView(tensor_span[i], command_buffer);
    // Reshape tensor.
    tflite::gpu::BHWC shape = BhwcFromTensorShape(tensor_span[i].shape());
    auto gpu_buffer_view =
        MtlBufferView::GetWriteView(*gpu_buffers_in_[i], command_buffer);
    id<MTLComputeCommandEncoder> input_encoder =
        [command_buffer computeCommandEncoder];
    [converter_to_BPHWC4_ convertWithEncoder:input_encoder
                                       shape:shape
                                sourceBuffer:input_view.buffer()
                             convertedBuffer:gpu_buffer_view.buffer()];
    [input_encoder endEncoding];
  }

  // Run inference.
  RET_CHECK(TFLGpuDelegateSetCommandBuffer(delegate_.get(), command_buffer));
  RET_CHECK_EQ(interpreter_->Invoke(), kTfLiteOk);

  output_tensors.reserve(output_shapes_.size());
  for (int i = 0; i < output_shapes_.size(); ++i) {
    output_tensors.emplace_back(Tensor::ElementType::kFloat32,
                                output_shapes_[i]);
    // Reshape tensor.
    tflite::gpu::BHWC shape = BhwcFromTensorShape(output_shapes_[i]);
    auto read_view =
        MtlBufferView::GetReadView(*gpu_buffers_out_[i], command_buffer);
    auto write_view =
        MtlBufferView::GetWriteView(output_tensors[i], command_buffer);
    id<MTLComputeCommandEncoder> output_encoder =
        [command_buffer computeCommandEncoder];
    [converter_from_BPHWC4_ convertWithEncoder:output_encoder
                                         shape:shape
                                  sourceBuffer:read_view.buffer()
                               convertedBuffer:write_view.buffer()];
    [output_encoder endEncoding];
  }
  [command_buffer commit];
  // The below call is found (manual testing) to resolve flickering issues for
  // some use cases where multiple Metal calculators are involved.
  // TODO: investigate and ensure proper synchronization
  // (e.g. fences/barriers/events).
  [command_buffer waitUntilScheduled];

  return output_tensors;
}

absl::Status InferenceCalculatorMetalImpl::Close(CalculatorContext* cc) {
  converter_to_BPHWC4_ = nil;
  converter_from_BPHWC4_ = nil;
  gpu_buffers_in_.clear();
  gpu_buffers_out_.clear();
  interpreter_ = nullptr;
  delegate_ = nullptr;
  return absl::OkStatus();
}

namespace {

absl::Status ValidateMetalIoTensor(const TfLiteTensor* tensor, int tensor_index,
                                   absl::string_view io_kind) {
  if (tensor == nullptr) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "%s tensor at index [%d] is null.", io_kind, tensor_index));
  }
  // TODO: Support quantized tensors.
  if (tensor->quantization.type == kTfLiteAffineQuantization) {
    return absl::InvalidArgumentError(
        "Quantized tensors are not supported by the macOS/iOS GPU delegate.");
  }
  if (tensor->type != kTfLiteFloat32) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "%s tensor at index [%d] has type %s, which is not supported by the "
        "macOS/iOS GPU delegate (only float32 input/output tensors are "
        "supported).",
        io_kind, tensor_index, TfLiteTypeGetName(tensor->type)));
  }
  if (tensor->dims == nullptr || tensor->dims->size < 2 ||
      tensor->dims->size > 4) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "%s tensor at index [%d] has rank %d, which is not supported by the "
        "macOS/iOS GPU delegate (expected rank in [2, 4]).",
        io_kind, tensor_index, tensor->dims ? tensor->dims->size : 0));
  }
  if (tensor->allocation_type == kTfLiteDynamic) {
    return absl::InvalidArgumentError(
        "Dynamic-sized tensors are not supported by the macOS/iOS GPU "
        "delegate.");
  }
  for (int d = 0; d < tensor->dims->size; ++d) {
    if (tensor->dims->data[d] <= 0) {
      return absl::InvalidArgumentError(
          "Dynamic-sized tensors are not supported by the macOS/iOS GPU "
          "delegate.");
    }
  }
  // Keras/TFLite Converter often populates dims_signature[0] = -1 (symbolic
  // batch) on single-batch models whose dims[0] == 1 (e.g.
  // mobilenet_v3_small, face_landmarks_detector). Skip index 0 only when it is
  // a unit batch axis on a multi-dimensional tensor (Process() verifies exact
  // runtime input shapes against tensor->dims) while rejecting all other
  // dynamic dimensions up front.
  if (tensor->dims_signature != nullptr) {
    const int first_non_batch_dim =
        (tensor->dims->size > 1 && tensor->dims->data[0] == 1) ? 1 : 0;
    for (int d = first_non_batch_dim; d < tensor->dims_signature->size; ++d) {
      if (tensor->dims_signature->data[d] <= 0) {
        return absl::InvalidArgumentError(
            "Dynamic-sized tensors are not supported by the macOS/iOS GPU "
            "delegate.");
      }
    }
  }
  return absl::OkStatus();
}

}  // namespace

absl::Status InferenceCalculatorMetalImpl::InitInterpreter(
    CalculatorContext* cc) {
  ABSL_ASSIGN_OR_RETURN(model_packet_, GetModelAsPacket(cc));
  const auto& model = *model_packet_.Get();
  ABSL_ASSIGN_OR_RETURN(auto op_resolver_packet, GetOpResolverAsPacket(cc));
  const auto& op_resolver = op_resolver_packet.Get();
  tflite::InterpreterBuilder interpreter_builder(model, op_resolver);
  interpreter_builder.SetNumThreads(
      cc->Options<mediapipe::InferenceCalculatorOptions>().cpu_num_thread());
  if (interpreter_builder(&interpreter_) != kTfLiteOk || !interpreter_) {
    return absl::InvalidArgumentError(
        "Failed to build TFLite interpreter for the macOS/iOS GPU delegate.");
  }

  for (int input_index : interpreter_->inputs()) {
    ABSL_RETURN_IF_ERROR(ValidateMetalIoTensor(
        interpreter_->tensor(input_index), input_index, "Input"));
  }
  for (int output_index : interpreter_->outputs()) {
    ABSL_RETURN_IF_ERROR(ValidateMetalIoTensor(
        interpreter_->tensor(output_index), output_index, "Output"));
  }

  CreateDelegate();
  const TfLiteStatus delegate_status =
      interpreter_->ModifyGraphWithDelegate(delegate_.get());
  if (delegate_status == kTfLiteApplicationError) {
    return absl::InvalidArgumentError(
        "Dynamic-sized tensors are not supported by the macOS/iOS GPU "
        "delegate.");
  }
  if (delegate_status != kTfLiteOk) {
    return absl::InvalidArgumentError(
        "Model contains operations or tensor shapes that are not supported by "
        "the macOS/iOS GPU delegate.");
  }

  ABSL_ASSIGN_OR_RETURN(
      const auto& io_mapping,
      InferenceIoMapper::GetInputOutputTensorNamesFromInterpreter(
          *interpreter_));
  ABSL_RETURN_IF_ERROR(
      InferenceCalculatorNodeImpl::UpdateIoMapping(cc, io_mapping));

  ABSL_RETURN_IF_ERROR(CreateConverters(cc));
  RET_CHECK_EQ(interpreter_->AllocateTensors(), kTfLiteOk);
  return absl::OkStatus();
}

void InferenceCalculatorMetalImpl::CreateDelegate() {
  // Configure and create the delegate.
  TFLGpuDelegateOptions options;
  // `enable_quantization` enables the run of sparse models i.e. the models with
  // DENSIFY op preceding DEQUINTIZE op. Both ops get removed from the execution
  // graph after the tensor of the weights is read.
  options.enable_quantization = true;
  options.allow_precision_loss = allow_precision_loss_;
  options.wait_type = TFLGpuDelegateWaitType::TFLGpuDelegateWaitTypeDoNotWait;
  delegate_ =
      TfLiteDelegatePtr(TFLGpuDelegateCreate(&options), &TFLGpuDelegateDelete);
}

absl::Status InferenceCalculatorMetalImpl::CreateConverters(
    CalculatorContext* cc) {
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
        allow_precision_loss_ ? Tensor::ElementType::kFloat16
                              : Tensor::ElementType::kFloat32,
        Tensor::Shape{dims}));
    auto buffer_view =
        MtlBufferView::GetWriteView(*gpu_buffers_in_[i], gpu_helper_.mtlDevice);
    if (!TFLGpuDelegateBindMetalBufferToTensor(
            delegate_.get(), input_indices[i], buffer_view.buffer())) {
      return absl::InvalidArgumentError(absl::StrFormat(
          "Failed to bind Metal buffer to input tensor [%d]; the model "
          "contains operations not supported by the macOS/iOS GPU delegate.",
          input_indices[i]));
    }
  }

  interpreter_->SetAllowBufferHandleOutput(true);
  // Get output image sizes.
  const auto& output_indices = interpreter_->outputs();
  output_shapes_.resize(output_indices.size());
  for (int i = 0; i < output_shapes_.size(); ++i) {
    const TfLiteTensor* tensor = interpreter_->tensor(output_indices[i]);
    // Create and bind output buffers.
    // Channels are always padded to multiple of 4.
    std::vector<int> dims{tensor->dims->data,
                          tensor->dims->data + tensor->dims->size};
    output_shapes_[i] = {dims};
    dims.back() = RoundUp(dims.back(), 4);
    gpu_buffers_out_.emplace_back(absl::make_unique<Tensor>(
        allow_precision_loss_ ? Tensor::ElementType::kFloat16
                              : Tensor::ElementType::kFloat32,
        Tensor::Shape{dims}));
    if (!TFLGpuDelegateBindMetalBufferToTensor(
            delegate_.get(), output_indices[i],
            MtlBufferView::GetWriteView(*gpu_buffers_out_[i],
                                        gpu_helper_.mtlDevice)
                .buffer())) {
      return absl::InvalidArgumentError(absl::StrFormat(
          "Failed to bind Metal buffer to output tensor [%d]; the model "
          "contains operations not supported by the macOS/iOS GPU delegate.",
          output_indices[i]));
    }
  }

  // Create converter for GPU input.
  converter_to_BPHWC4_ =
      [[TFLBufferConvert alloc] initWithDevice:device
                                     isFloat16:allow_precision_loss_
                               convertToPBHWC4:true];
  if (converter_to_BPHWC4_ == nil) {
    return absl::InternalError("Error initializating input buffer converter");
  }
  // Create converter for GPU output.
  converter_from_BPHWC4_ =
      [[TFLBufferConvert alloc] initWithDevice:device
                                     isFloat16:allow_precision_loss_
                               convertToPBHWC4:false];
  if (converter_from_BPHWC4_ == nil) {
    return absl::InternalError("Error initializating output buffer converter");
  }

  return absl::OkStatus();
}

}  // namespace api2
}  // namespace mediapipe
