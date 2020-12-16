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

#include <array>
#include <memory>
#include <vector>

#include "mediapipe/calculators/tensor/image_to_tensor_calculator.pb.h"
#include "mediapipe/calculators/tensor/image_to_tensor_converter.h"
#include "mediapipe/calculators/tensor/image_to_tensor_converter_opencv.h"
#include "mediapipe/calculators/tensor/image_to_tensor_utils.h"
#include "mediapipe/framework/api2/node.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/image_frame.h"
#include "mediapipe/framework/formats/rect.pb.h"
#include "mediapipe/framework/formats/tensor.h"
#include "mediapipe/framework/port.h"
#include "mediapipe/framework/port/canonical_errors.h"
#include "mediapipe/framework/port/ret_check.h"
#include "mediapipe/framework/port/status.h"

#if !MEDIAPIPE_DISABLE_GPU
#include "mediapipe/gpu/gpu_buffer.h"

#if MEDIAPIPE_METAL_ENABLED
#include "mediapipe/calculators/tensor/image_to_tensor_converter_metal.h"
#include "mediapipe/gpu/MPPMetalHelper.h"
#elif MEDIAPIPE_OPENGL_ES_VERSION >= MEDIAPIPE_OPENGL_ES_31
#include "mediapipe/calculators/tensor/image_to_tensor_converter_gl_buffer.h"
#include "mediapipe/gpu/gl_calculator_helper.h"
#else
#include "mediapipe/calculators/tensor/image_to_tensor_converter_gl_texture.h"
#include "mediapipe/gpu/gl_calculator_helper.h"
#endif  // MEDIAPIPE_METAL_ENABLED

#endif  // !MEDIAPIPE_DISABLE_GPU

namespace mediapipe {
namespace api2 {

#if MEDIAPIPE_DISABLE_GPU
// Just a placeholder to not have to depend on mediapipe::GpuBuffer.
using GpuBuffer = AnyType;
#else
using GpuBuffer = mediapipe::GpuBuffer;
#endif  // MEDIAPIPE_DISABLE_GPU

// Converts image into Tensor, possibly with cropping, resizing and
// normalization, according to specified inputs and options.
//
// Inputs:
//   IMAGE - ImageFrame [ImageFormat::SRGB/SRGBA]
//     Image to extract from.
//   IMAGE_GPU - GpuBuffer [GpuBufferFormat::kBGRA32]
//     Image to extract from.
//   (Either IMAGE or IMAGE_GPU has to be specified.)
//
//   NORM_RECT - NormalizedRect @Optional
//     Describes region of image to extract.
//     @Optional: rect covering the whole image is used if not specified.
//
// Outputs:
//   TENSORS - std::vector<Tensor>
//     Vector containing a single Tensor populated with an extrated RGB image.
//   MATRIX - std::array<float, 16> @Optional
//     An std::array<float, 16> representing a 4x4 row-major-order matrix which
//     can be used to map a point on the output tensor to a point on the input
//     image.
//   LETTERBOX_PADDING - std::array<float, 4> @Optional
//     An std::array<float, 4> representing the letterbox padding from the 4
//     sides ([left, top, right, bottom]) of the output image, normalized to
//     [0.f, 1.f] by the output dimensions. The padding values are non-zero only
//     when the "keep_aspect_ratio" is true.
//
//     For instance, when the input image is 10x10 (width x height) and the
//     output dimensions specified in the calculator option are 20x40 and
//     "keep_aspect_ratio" is true, the calculator scales the input image to
//     20x20 and places it in the middle of the output image with an equal
//     padding of 10 pixels at the top and the bottom. The resulting array is
//     therefore [0.f, 0.25f, 0.f, 0.25f] (10/40 = 0.25f).
//
// Example:
// node {
//   calculator: "ImageToTensorCalculator"
//   input_stream: "IMAGE:image"  # or "IMAGE_GPU:image"
//   input_stream: "NORM_RECT:roi"
//   output_stream: "TENSORS:tensors"
//   output_stream: "MATRIX:matrix"
//   options {
//     [mediapipe.ImageToTensorCalculatorOptions.ext] {
//       output_tensor_width: 256
//       output_tensor_height: 256
//       keep_aspect_ratio: false
//       output_tensor_float_range {
//         min: 0.0
//         max: 1.0
//       }
//       # gpu_origin: CONVENTIONAL # or TOP_LEFT
//     }
//   }
// }
class ImageToTensorCalculator : public Node {
 public:
  static constexpr Input<mediapipe::ImageFrame>::Optional kInCpu{"IMAGE"};
  static constexpr Input<GpuBuffer>::Optional kInGpu{"IMAGE_GPU"};
  static constexpr Input<mediapipe::NormalizedRect>::Optional kInNormRect{
      "NORM_RECT"};
  static constexpr Output<std::vector<Tensor>> kOutTensors{"TENSORS"};
  static constexpr Output<std::array<float, 4>>::Optional kOutLetterboxPadding{
      "LETTERBOX_PADDING"};
  static constexpr Output<std::array<float, 16>>::Optional kOutMatrix{"MATRIX"};

  MEDIAPIPE_NODE_CONTRACT(kInCpu, kInGpu, kInNormRect, kOutTensors,
                          kOutLetterboxPadding, kOutMatrix);

  static ::mediapipe::Status UpdateContract(CalculatorContract* cc) {
    const auto& options =
        cc->Options<mediapipe::ImageToTensorCalculatorOptions>();

    RET_CHECK(options.has_output_tensor_float_range())
        << "Output tensor range is required.";
    RET_CHECK_LT(options.output_tensor_float_range().min(),
                 options.output_tensor_float_range().max())
        << "Valid output tensor range is required.";
    RET_CHECK_GT(options.output_tensor_width(), 0)
        << "Valid output tensor width is required.";
    RET_CHECK_GT(options.output_tensor_height(), 0)
        << "Valid output tensor height is required.";

    RET_CHECK(kInCpu(cc).IsConnected() ^ kInGpu(cc).IsConnected())
        << "One and only one of CPU or GPU input is expected.";

    if (kInGpu(cc).IsConnected()) {
#if MEDIAPIPE_DISABLE_GPU
      return mediapipe::UnimplementedError("GPU processing is disabled");
#else

#if MEDIAPIPE_METAL_ENABLED
      MP_RETURN_IF_ERROR([MPPMetalHelper updateContract:cc]);
#else
      MP_RETURN_IF_ERROR(mediapipe::GlCalculatorHelper::UpdateContract(cc));
#endif  // MEDIAPIPE_METAL_ENABLED

#endif  // MEDIAPIPE_DISABLE_GPU
    }
    return mediapipe::OkStatus();
  }

  mediapipe::Status Open(CalculatorContext* cc) {
    options_ = cc->Options<mediapipe::ImageToTensorCalculatorOptions>();
    output_width_ = options_.output_tensor_width();
    output_height_ = options_.output_tensor_height();
    range_min_ = options_.output_tensor_float_range().min();
    range_max_ = options_.output_tensor_float_range().max();

    if (kInCpu(cc).IsConnected()) {
      ASSIGN_OR_RETURN(converter_, CreateOpenCvConverter(cc, GetBorderMode()));
    } else {
#if MEDIAPIPE_DISABLE_GPU
      return mediapipe::UnimplementedError("GPU processing is disabled");
#else

#if MEDIAPIPE_METAL_ENABLED
      ASSIGN_OR_RETURN(converter_, CreateMetalConverter(cc, GetBorderMode()));
#elif MEDIAPIPE_OPENGL_ES_VERSION >= MEDIAPIPE_OPENGL_ES_31
      ASSIGN_OR_RETURN(converter_,
                       CreateImageToGlBufferTensorConverter(
                           cc, DoesInputStartAtBottom(), GetBorderMode()));
#else
      ASSIGN_OR_RETURN(converter_,
                       CreateImageToGlTextureTensorConverter(
                           cc, DoesInputStartAtBottom(), GetBorderMode()));
#endif  // MEDIAPIPE_METAL_ENABLED

#endif  // MEDIAPIPE_DISABLE_GPU
    }
    return mediapipe::OkStatus();
  }

  mediapipe::Status Process(CalculatorContext* cc) {
    const PacketBase& image_packet =
        kInCpu(cc).IsConnected() ? kInCpu(cc).packet() : kInGpu(cc).packet();
    if (image_packet.IsEmpty()) {
      // Timestamp bound update happens automatically. (See Open().)
      return mediapipe::OkStatus();
    }

    absl::optional<mediapipe::NormalizedRect> norm_rect;
    if (kInNormRect(cc).IsConnected()) {
      if (kInNormRect(cc).IsEmpty()) {
        // Timestamp bound update happens automatically. (See Open().)
        return mediapipe::OkStatus();
      }
      norm_rect = *kInNormRect(cc);
      if (norm_rect->width() == 0 && norm_rect->height() == 0) {
        // WORKAROUND: some existing graphs may use sentinel rects {width=0,
        // height=0, ...} quite often and calculator has to handle them
        // gracefully by updating timestamp bound instead of returning failure.
        // Timestamp bound update happens automatically. (See Open().)
        // NOTE: usage of sentinel rects should be avoided.
        DLOG(WARNING)
            << "Updating timestamp bound in response to a sentinel rect";
        return mediapipe::OkStatus();
      }
    }

    const Size& size = converter_->GetImageSize(image_packet);
    RotatedRect roi = GetRoi(size.width, size.height, norm_rect);
    ASSIGN_OR_RETURN(auto padding, PadRoi(options_.output_tensor_width(),
                                          options_.output_tensor_height(),
                                          options_.keep_aspect_ratio(), &roi));
    if (kOutLetterboxPadding(cc).IsConnected()) {
      kOutLetterboxPadding(cc).Send(padding);
    }
    if (kOutMatrix(cc).IsConnected()) {
      std::array<float, 16> matrix;
      GetRotatedSubRectToRectTransformMatrix(roi, size.width, size.height,
                                             /*flip_horizontaly=*/false,
                                             &matrix);
      kOutMatrix(cc).Send(std::move(matrix));
    }

    ASSIGN_OR_RETURN(
        Tensor tensor,
        converter_->Convert(image_packet, roi, {output_width_, output_height_},
                            range_min_, range_max_));

    auto result = std::make_unique<std::vector<Tensor>>();
    result->push_back(std::move(tensor));
    kOutTensors(cc).Send(std::move(result));

    return mediapipe::OkStatus();
  }

 private:
  bool DoesInputStartAtBottom() {
    return options_.gpu_origin() != mediapipe::GpuOrigin_Mode_TOP_LEFT;
  }

  BorderMode GetBorderMode() {
    switch (options_.border_mode()) {
      case mediapipe::
          ImageToTensorCalculatorOptions_BorderMode_BORDER_UNSPECIFIED:
        return BorderMode::kReplicate;
      case mediapipe::ImageToTensorCalculatorOptions_BorderMode_BORDER_ZERO:
        return BorderMode::kZero;
      case mediapipe::
          ImageToTensorCalculatorOptions_BorderMode_BORDER_REPLICATE:
        return BorderMode::kReplicate;
    }
  }

  std::unique_ptr<ImageToTensorConverter> converter_;
  mediapipe::ImageToTensorCalculatorOptions options_;
  int output_width_ = 0;
  int output_height_ = 0;
  float range_min_ = 0.0f;
  float range_max_ = 1.0f;
};

MEDIAPIPE_REGISTER_NODE(ImageToTensorCalculator);

}  // namespace api2
}  // namespace mediapipe
