// Copyright 2020 The MediaPipe Authors.
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
#include "mediapipe/calculators/tensor/image_to_tensor_calculator.h"

#include <array>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include "absl/log/absl_log.h"
#include "absl/status/status.h"
#include "mediapipe/calculators/tensor/image_to_tensor_calculator.pb.h"
#include "mediapipe/calculators/tensor/image_to_tensor_converter.h"
#include "mediapipe/calculators/tensor/image_to_tensor_utils.h"
#include "mediapipe/framework/api3/calculator.h"
#include "mediapipe/framework/api3/calculator_context.h"
#include "mediapipe/framework/api3/calculator_contract.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/image.h"
#include "mediapipe/framework/formats/image_frame.h"
#include "mediapipe/framework/formats/rect.pb.h"
#include "mediapipe/framework/formats/tensor.h"
#include "mediapipe/framework/memory_manager.h"
#include "mediapipe/framework/memory_manager_service.h"
#include "mediapipe/framework/port.h"
#include "mediapipe/framework/port/canonical_errors.h"
#include "mediapipe/framework/port/ret_check.h"
#include "mediapipe/framework/port/status.h"
#include "mediapipe/framework/port/statusor.h"
#include "mediapipe/gpu/gpu_origin.pb.h"
#include "mediapipe/gpu/gpu_origin_utils.h"
#include "mediapipe/gpu/webgpu/webgpu_check.h"

#if !MEDIAPIPE_DISABLE_OPENCV
#include "mediapipe/calculators/tensor/image_to_tensor_converter_opencv.h"
#elif MEDIAPIPE_ENABLE_HALIDE
#include "mediapipe/calculators/tensor/image_to_tensor_converter_frame_buffer.h"
#endif

#if !MEDIAPIPE_DISABLE_GPU
#include "mediapipe/gpu/gpu_buffer.h"

#if MEDIAPIPE_METAL_ENABLED
#include "mediapipe/calculators/tensor/image_to_tensor_converter_metal.h"
#include "mediapipe/gpu/MPPMetalHelper.h"
#elif MEDIAPIPE_OPENGL_ES_VERSION >= MEDIAPIPE_OPENGL_ES_31
#include "mediapipe/calculators/tensor/image_to_tensor_converter_gl_buffer.h"
#include "mediapipe/gpu/gl_calculator_helper.h"
#include "mediapipe/gpu/gpu_service.h"
#else
#include "mediapipe/calculators/tensor/image_to_tensor_converter_gl_texture.h"
#include "mediapipe/gpu/gl_calculator_helper.h"
#include "mediapipe/gpu/gpu_service.h"
#if MEDIAPIPE_USE_WEBGPU
#include "mediapipe/gpu/webgpu/image_to_tensor_converter_webgpu_texture.h"
#include "mediapipe/gpu/webgpu/webgpu_service.h"
#include "mediapipe/gpu/webgpu/webgpu_texture_buffer.h"
#endif  // MEDIAPIPE_USE_WEBGPU
#endif  // MEDIAPIPE_METAL_ENABLED
#endif  // !MEDIAPIPE_DISABLE_GPU

namespace mediapipe {
namespace api3 {

class ImageToTensorNodeImpl
    : public Calculator<ImageToTensorNode, ImageToTensorNodeImpl> {
 public:
  static absl::Status UpdateContract(
      CalculatorContract<ImageToTensorNode>& cc) {
    const auto& options = cc.options.Get();

    RET_CHECK_OK(ValidateOptionOutputDims(options));
    RET_CHECK(cc.in.IsConnected() ^ cc.in_gpu.IsConnected())
        << "One and only one of IMAGE and IMAGE_GPU input is expected.";
    RET_CHECK(cc.out_tensors.IsConnected() ^ cc.out_tensor.IsConnected())
        << "One and only one of TENSORS and TENSOR output is supported.";

#if MEDIAPIPE_DISABLE_GPU
    if (cc.in_gpu.IsConnected()) {
      return absl::UnimplementedError(
          "GPU processing is disabled in build flags");
    }
#else  // !MEDIAPIPE_DISABLE_GPU
#if MEDIAPIPE_METAL_ENABLED
    MP_RETURN_IF_ERROR(
        [MPPMetalHelper updateContract:&cc.GetGenericContract()]);
#else

    cc.UseService(kGpuService).Optional();
#if MEDIAPIPE_USE_WEBGPU
    cc.UseService(kWebGpuService).Optional();
#endif  // MEDIAPIPE_USE_WEBGPU
#endif  // MEDIAPIPE_METAL_ENABLED
#endif  // MEDIAPIPE_DISABLE_GPU

    cc.UseService(kMemoryManagerService).Optional();
    return absl::OkStatus();
  }

  absl::Status Open(CalculatorContext<ImageToTensorNode>& cc) final {
    if (cc.Service(kMemoryManagerService).IsAvailable()) {
      memory_manager_ = &cc.Service(kMemoryManagerService).GetObject();
    }
    options_ = cc.options.Get();
    params_ = GetOutputTensorParams(options_);
    return absl::OkStatus();
  }

  absl::Status Process(CalculatorContext<ImageToTensorNode>& cc) final {
    if (!cc.in && !cc.in_gpu) {
      // Timestamp bound update happens automatically.
      return absl::OkStatus();
    }

    std::optional<mediapipe::NormalizedRect> norm_rect;
    if (cc.in_norm_rect.IsConnected()) {
      if (!cc.in_norm_rect) {
        // Timestamp bound update happens automatically. (See Open().)
        return absl::OkStatus();
      }
      norm_rect = cc.in_norm_rect.GetOrDie();
      if (norm_rect->width() == 0 && norm_rect->height() == 0) {
        // WORKAROUND: some existing graphs may use sentinel rects {width=0,
        // height=0, ...} quite often and calculator has to handle them
        // gracefully by updating timestamp bound instead of returning failure.
        // Timestamp bound update happens automatically. (See Open().)
        // NOTE: usage of sentinel rects should be avoided.
        ABSL_DLOG(WARNING)
            << "Updating timestamp bound in response to a sentinel rect";
        return absl::OkStatus();
      }
    }

    std::shared_ptr<const Image> image;
    if (cc.in.IsConnected()) {
      image = cc.in.VisitOrDie(
          [](const Image& image) {
            return std::shared_ptr<const Image>(
                &image, [](const Image* image) { /*do nothing*/ });
          },
          [](const ImageFrame& image_frame) {
            return std::make_shared<const Image>(std::shared_ptr<ImageFrame>(
                const_cast<ImageFrame*>(&image_frame),
                [](ImageFrame* image) { /*do nothing*/ }));
          });
    }
#if !MEDIAPIPE_DISABLE_GPU
    if (cc.in_gpu.IsConnected()) {
      image = std::make_shared<const Image>(cc.in_gpu.GetOrDie());
    }
#endif  // !MEDIAPIPE_DISABLE_GPU
    RET_CHECK(image) << "Input image is missing.";

    RotatedRect roi = GetRoi(image->width(), image->height(), norm_rect);
    const int tensor_width = params_.output_width.value_or(image->width());
    const int tensor_height = params_.output_height.value_or(image->height());
    MP_ASSIGN_OR_RETURN(auto padding,
                        PadRoi(tensor_width, tensor_height,
                               options_.keep_aspect_ratio(), &roi));
    if (cc.out_letterbox_padding.IsConnected()) {
      cc.out_letterbox_padding.Send(padding);
    }
    if (cc.out_matrix.IsConnected()) {
      std::array<float, 16> matrix;
      GetRotatedSubRectToRectTransformMatrix(
          roi, image->width(), image->height(),
          /*flip_horizontally=*/false, &matrix);
      cc.out_matrix.Send(std::move(matrix));
    }

    // Lazy initialization of the GPU or CPU converter.
    MP_RETURN_IF_ERROR(
        InitConverterIfNecessary(&cc.GetGenericContext(), *image));

    Tensor::ElementType output_tensor_type =
        GetOutputTensorType(image->UsesGpu(), params_);
    Tensor tensor(
        output_tensor_type,
        {1, tensor_height, tensor_width, GetNumOutputChannels(*image)},
        memory_manager_);
    MP_RETURN_IF_ERROR((image->UsesGpu() ? gpu_converter_ : cpu_converter_)
                           ->Convert(*image, roi, params_.range_min,
                                     params_.range_max,
                                     /*tensor_buffer_offset=*/0, tensor));

    if (cc.out_tensors.IsConnected()) {
      auto result = std::make_unique<std::vector<Tensor>>();
      result->push_back(std::move(tensor));
      cc.out_tensors.Send(std::move(result));
    } else {
      cc.out_tensor.Send(std::move(tensor));
    }
    return absl::OkStatus();
  }

 private:
  absl::Status InitConverterIfNecessary(mediapipe::CalculatorContext* cc,
                                        const Image& image) {
    // Lazy initialization of the GPU or CPU converter.
    if (image.UsesGpu()) {
      if (!params_.is_float_output) {
        return absl::UnimplementedError(
            "ImageToTensorConverter for the input GPU image currently doesn't "
            "support quantization.");
      }
      if (!gpu_converter_) {
#if !MEDIAPIPE_DISABLE_GPU
#if MEDIAPIPE_METAL_ENABLED
        MP_ASSIGN_OR_RETURN(
            gpu_converter_,
            CreateMetalConverter(cc, GetBorderMode(options_.border_mode())));
#elif MEDIAPIPE_OPENGL_ES_VERSION >= MEDIAPIPE_OPENGL_ES_31
        MP_ASSIGN_OR_RETURN(bool input_starts_at_bottom,
                            IsGpuOriginAtBottom(options_.gpu_origin()));
        MP_ASSIGN_OR_RETURN(gpu_converter_,
                            CreateImageToGlBufferTensorConverter(
                                cc, input_starts_at_bottom,
                                GetBorderMode(options_.border_mode())));
#else
        if (IsWebGpuAvailable()) {
#if MEDIAPIPE_USE_WEBGPU
          MP_ASSIGN_OR_RETURN(gpu_converter_,
                              CreateImageToWebGpuTextureTensorConverter(cc));
#endif  // MEDIAPIPE_USE_WEBGPU
        }
#if MEDIAPIPE_OPENGL_ES_VERSION >= MEDIAPIPE_OPENGL_ES_30
        if (!gpu_converter_) {
          MP_ASSIGN_OR_RETURN(bool input_starts_at_bottom,
                              IsGpuOriginAtBottom(options_.gpu_origin()));
          MP_ASSIGN_OR_RETURN(gpu_converter_,
                              CreateImageToGlTextureTensorConverter(
                                  cc, input_starts_at_bottom,
                                  GetBorderMode(options_.border_mode())));
        }
#endif  // MEDIAPIPE_OPENGL_ES_VERSION >= MEDIAPIPE_OPENGL_ES_30
        if (!gpu_converter_) {
          return absl::UnimplementedError(
              "ImageToTensorConverter for the input GPU image is unavailable.");
        }
#endif  // MEDIAPIPE_METAL_ENABLED
#endif  // !MEDIAPIPE_DISABLE_GPU
      }
    } else {
      if (!cpu_converter_) {
#if !MEDIAPIPE_DISABLE_OPENCV
        MP_ASSIGN_OR_RETURN(
            cpu_converter_,
            CreateOpenCvConverter(
                cc, GetBorderMode(options_.border_mode()),
                GetOutputTensorType(/*uses_gpu=*/false, params_)));
// TODO: FrameBuffer-based converter needs to call GetGpuBuffer()
// to get access to a FrameBuffer view. Investigate if GetGpuBuffer() can be
// made available even with MEDIAPIPE_DISABLE_GPU set.
#elif MEDIAPIPE_ENABLE_HALIDE
        MP_ASSIGN_OR_RETURN(
            cpu_converter_,
            CreateFrameBufferConverter(
                cc, GetBorderMode(options_.border_mode()),
                GetOutputTensorType(/*uses_gpu=*/false, params_)));
#else
        ABSL_LOG(FATAL) << "Cannot create image to tensor CPU converter since "
                           "MEDIAPIPE_DISABLE_OPENCV is defined and "
                           "MEDIAPIPE_ENABLE_HALIDE is not defined.";
#endif  // !MEDIAPIPE_DISABLE_HALIDE
      }
    }
    return absl::OkStatus();
  }

  std::unique_ptr<ImageToTensorConverter> gpu_converter_;
  std::unique_ptr<ImageToTensorConverter> cpu_converter_;
  mediapipe::ImageToTensorCalculatorOptions options_;
  OutputTensorParams params_;
  MemoryManager* memory_manager_ = nullptr;
};

}  // namespace api3
}  // namespace mediapipe
