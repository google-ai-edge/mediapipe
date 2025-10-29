// Copyright 2021 The MediaPipe Authors.
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
#include "mediapipe/calculators/tensor/tensors_to_segmentation_calculator.h"

#include <memory>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "mediapipe/calculators/tensor/tensors_to_segmentation_calculator.pb.h"
#include "mediapipe/calculators/tensor/tensors_to_segmentation_converter.h"
#include "mediapipe/calculators/tensor/tensors_to_segmentation_utils.h"
#include "mediapipe/framework/api3/calculator.h"
#include "mediapipe/framework/api3/calculator_context.h"
#include "mediapipe/framework/api3/calculator_contract.h"
#include "mediapipe/framework/calculator_context.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/image.h"
#include "mediapipe/framework/formats/tensor.h"
#include "mediapipe/framework/port.h"
#include "mediapipe/framework/port/ret_check.h"
#include "mediapipe/framework/port/status_macros.h"

#if !MEDIAPIPE_DISABLE_GPU
#include "mediapipe/gpu/gl_calculator_helper.h"

#if MEDIAPIPE_OPENGL_ES_VERSION >= MEDIAPIPE_OPENGL_ES_31
#include "mediapipe/calculators/tensor/tensors_to_segmentation_converter_gl_buffer.h"
#elif MEDIAPIPE_METAL_ENABLED
#include "mediapipe/calculators/tensor/tensors_to_segmentation_converter_metal.h"
#import "mediapipe/gpu/MPPMetalHelper.h"
#else
#include "mediapipe/calculators/tensor/tensors_to_segmentation_converter_gl_texture.h"
#endif  // MEDIAPIPE_OPENGL_ES_VERSION >= MEDIAPIPE_OPENGL_ES_31
#endif  // !MEDIAPIPE_DISABLE_GPU

#if !MEDIAPIPE_DISABLE_OPENCV
#include "mediapipe/calculators/tensor/tensors_to_segmentation_converter_opencv.h"
#endif  // !MEDIAPIPE_DISABLE_OPENCV

namespace mediapipe::api3 {

namespace {

using ::mediapipe::tensors_to_segmentation_utils::CanUseGpu;
using ::mediapipe::tensors_to_segmentation_utils::GetHwcFromDims;

}  // namespace

class TensorsToSegmentationCalculator
    : public Calculator<TensorsToSegmentationNode,
                        TensorsToSegmentationCalculator> {
 public:
  static absl::Status UpdateContract(
      CalculatorContract<TensorsToSegmentationNode>& cc);

  absl::Status Open(CalculatorContext<TensorsToSegmentationNode>& cc) override;
  absl::Status Process(
      CalculatorContext<TensorsToSegmentationNode>& cc) override;

 private:
  absl::Status InitConverterIfNecessary(bool use_gpu,
                                        mediapipe::CalculatorContext* cc) {
    if (use_gpu) {
#if !MEDIAPIPE_DISABLE_GPU
      if (!gpu_converter_) {
#if MEDIAPIPE_OPENGL_ES_VERSION >= MEDIAPIPE_OPENGL_ES_31
        MP_ASSIGN_OR_RETURN(gpu_converter_,
                            CreateGlBufferConverter(cc, options_));
#elif MEDIAPIPE_METAL_ENABLED
        MP_ASSIGN_OR_RETURN(gpu_converter_, CreateMetalConverter(cc, options_));
#else
        MP_ASSIGN_OR_RETURN(gpu_converter_,
                            CreateGlTextureConverter(cc, options_));
#endif  // MEDIAPIPE_OPENGL_ES_VERSION >= MEDIAPIPE_OPENGL_ES_31
      }
#else
      RET_CHECK_FAIL() << "Cannot initialize GPU converter because GPU "
                          "processing is disabled.";
#endif  // !MEDIAPIPE_DISABLE_GPU
    } else {
#if !MEDIAPIPE_DISABLE_OPENCV
      if (!cpu_converter_) {
        MP_ASSIGN_OR_RETURN(cpu_converter_, CreateOpenCvConverter(options_));
      }
#else
      RET_CHECK_FAIL() << "Cannot initialize OpenCV converter because OpenCV "
                          "processing is disabled.";
#endif  // !MEDIAPIPE_DISABLE_OPENCV
    }
    return absl::OkStatus();
  }

  mediapipe::TensorsToSegmentationCalculatorOptions options_;
  std::unique_ptr<TensorsToSegmentationConverter> cpu_converter_;
  std::unique_ptr<TensorsToSegmentationConverter> gpu_converter_;
};

// static
absl::Status TensorsToSegmentationCalculator::UpdateContract(
    CalculatorContract<TensorsToSegmentationNode>& cc) {
  RET_CHECK(cc.tensors_in.IsConnected() ^ cc.tensor_in.IsConnected())
      << "Either TENSOR or TENSORS must be connected";
  if (CanUseGpu()) {
#if !MEDIAPIPE_DISABLE_GPU
    MP_RETURN_IF_ERROR(mediapipe::GlCalculatorHelper::UpdateContract(
        &cc.GetGenericContract(), /*request_gpu_as_optional=*/true));
#if MEDIAPIPE_METAL_ENABLED
    MP_RETURN_IF_ERROR(
        [MPPMetalHelper updateContract:&cc.GetGenericContract()]);
#endif  // MEDIAPIPE_METAL_ENABLED
#endif  // !MEDIAPIPE_DISABLE_GPU
  }

  return absl::OkStatus();
}

absl::Status TensorsToSegmentationCalculator::Open(
    CalculatorContext<TensorsToSegmentationNode>& cc) {
  options_ = cc.options.Get();
  return absl::OkStatus();
}

absl::Status TensorsToSegmentationCalculator::Process(
    CalculatorContext<TensorsToSegmentationNode>& cc) {
  const Tensor* input_tensor = nullptr;
  if (cc.tensors_in.IsConnected()) {
    if (!cc.tensors_in) return absl::OkStatus();
    RET_CHECK(!cc.tensors_in.GetOrDie().empty());
    input_tensor = &cc.tensors_in.GetOrDie()[0];
  } else {
    RET_CHECK(cc.tensor_in.IsConnected());
    if (!cc.tensor_in) return absl::OkStatus();
    input_tensor = &cc.tensor_in.GetOrDie();
  }
  RET_CHECK_NE(input_tensor, nullptr);

  bool use_gpu = false;
  if (CanUseGpu()) {
    // Use GPU processing only if at least one input tensor is already on GPU.
    use_gpu = input_tensor->ready_on_gpu();
  }

  // Validate tensor channels and activation type.
  {
    RET_CHECK(input_tensor->element_type() == Tensor::ElementType::kFloat32);
    MP_ASSIGN_OR_RETURN(auto hwc, GetHwcFromDims(input_tensor->shape().dims));
    int tensor_channels = std::get<2>(hwc);
    using Options = ::mediapipe::TensorsToSegmentationCalculatorOptions;
    switch (options_.activation()) {
      case Options::NONE:
        RET_CHECK_EQ(tensor_channels, 1);
        break;
      case Options::SIGMOID:
        RET_CHECK_EQ(tensor_channels, 1);
        break;
      case Options::SOFTMAX:
        RET_CHECK_EQ(tensor_channels, 2);
        break;
    }
  }

  // Get dimensions.
  MP_ASSIGN_OR_RETURN(auto hwc, GetHwcFromDims(input_tensor->shape().dims));
  auto [tensor_height, tensor_width, tensor_channels] = hwc;
  int output_width = tensor_width, output_height = tensor_height;
  if (cc.output_size_in.IsConnected()) {
    RET_CHECK(cc.output_size_in) << "Output size is empty.";
    const auto& size = cc.output_size_in.GetOrDie();
    output_width = size.first;
    output_height = size.second;
  }

  if (use_gpu) {
#if !MEDIAPIPE_DISABLE_GPU
    // Lazily initialize converter
    MP_RETURN_IF_ERROR(
        InitConverterIfNecessary(use_gpu, &cc.GetGenericContext()));
    MP_ASSIGN_OR_RETURN(
        std::unique_ptr<Image> output_mask,
        gpu_converter_->Convert(*input_tensor, output_width, output_height));
    cc.mask_out.Send(std::move(output_mask));
#else
    RET_CHECK_FAIL() << "GPU processing disabled.";
#endif  // !MEDIAPIPE_DISABLE_GPU
  } else {
#if !MEDIAPIPE_DISABLE_OPENCV
    // Lazily initialize converter.
    MP_RETURN_IF_ERROR(
        InitConverterIfNecessary(use_gpu, &cc.GetGenericContext()));
    MP_ASSIGN_OR_RETURN(
        std::unique_ptr<Image> output_mask,
        cpu_converter_->Convert(*input_tensor, output_width, output_height));
    cc.mask_out.Send(std::move(output_mask));
#else
    RET_CHECK_FAIL() << "OpenCV processing disabled.";
#endif  // !MEDIAPIPE_DISABLE_OPENCV
  }

  return absl::OkStatus();
}

}  // namespace mediapipe::api3
