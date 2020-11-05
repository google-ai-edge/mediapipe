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

#include "mediapipe/calculators/tensor/image_to_tensor_calculator.pb.h"
#include "mediapipe/calculators/tensor/image_to_tensor_converter.h"
#include "mediapipe/calculators/tensor/image_to_tensor_converter_opencv.h"
#include "mediapipe/calculators/tensor/image_to_tensor_utils.h"
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

namespace {
constexpr char kInputCpu[] = "IMAGE";
constexpr char kInputGpu[] = "IMAGE_GPU";
constexpr char kOutputMatrix[] = "MATRIX";
constexpr char kOutput[] = "TENSORS";
constexpr char kInputNormRect[] = "NORM_RECT";
constexpr char kOutputLetterboxPadding[] = "LETTERBOX_PADDING";
}  // namespace

namespace mediapipe {

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
class ImageToTensorCalculator : public CalculatorBase {
 public:
  static ::mediapipe::Status GetContract(CalculatorContract* cc) {
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

    if (cc->Inputs().HasTag(kInputNormRect)) {
      cc->Inputs().Tag(kInputNormRect).Set<mediapipe::NormalizedRect>();
    }
    if (cc->Outputs().HasTag(kOutputLetterboxPadding)) {
      cc->Outputs().Tag(kOutputLetterboxPadding).Set<std::array<float, 4>>();
    }
    if (cc->Outputs().HasTag(kOutputMatrix)) {
      cc->Outputs().Tag(kOutputMatrix).Set<std::array<float, 16>>();
    }

    const bool has_cpu_input = cc->Inputs().HasTag(kInputCpu);
    const bool has_gpu_input = cc->Inputs().HasTag(kInputGpu);
    RET_CHECK_EQ((has_cpu_input ? 1 : 0) + (has_gpu_input ? 1 : 0), 1)
        << "Either CPU or GPU input is expected, not both.";

    if (has_cpu_input) {
      cc->Inputs().Tag(kInputCpu).Set<mediapipe::ImageFrame>();
    } else if (has_gpu_input) {
#if MEDIAPIPE_DISABLE_GPU
      return mediapipe::UnimplementedError("GPU processing is disabled");
#else

#if MEDIAPIPE_METAL_ENABLED
      MP_RETURN_IF_ERROR([MPPMetalHelper updateContract:cc]);
#else
      MP_RETURN_IF_ERROR(mediapipe::GlCalculatorHelper::UpdateContract(cc));
#endif  // MEDIAPIPE_METAL_ENABLED
      cc->Inputs().Tag(kInputGpu).Set<mediapipe::GpuBuffer>();

#endif  // MEDIAPIPE_DISABLE_GPU
    }
    cc->Outputs().Tag(kOutput).Set<std::vector<Tensor>>();
    return ::mediapipe::OkStatus();
  }

  ::mediapipe::Status Open(CalculatorContext* cc) {
    // Makes sure outputs' next timestamp bound update is handled automatically
    // by the framework.
    cc->SetOffset(TimestampDiff(0));
    options_ = cc->Options<mediapipe::ImageToTensorCalculatorOptions>();
    output_width_ = options_.output_tensor_width();
    output_height_ = options_.output_tensor_height();
    range_min_ = options_.output_tensor_float_range().min();
    range_max_ = options_.output_tensor_float_range().max();

    if (cc->Inputs().HasTag(kInputCpu)) {
      ASSIGN_OR_RETURN(converter_, CreateOpenCvConverter(cc));
    } else {
#if MEDIAPIPE_DISABLE_GPU
      return mediapipe::UnimplementedError("GPU processing is disabled");
#else

#if MEDIAPIPE_METAL_ENABLED
      ASSIGN_OR_RETURN(converter_, CreateMetalConverter(cc));
#elif MEDIAPIPE_OPENGL_ES_VERSION >= MEDIAPIPE_OPENGL_ES_31
      ASSIGN_OR_RETURN(converter_, CreateImageToGlBufferTensorConverter(
                                       cc, DoesInputStartAtBottom()));
#else
      ASSIGN_OR_RETURN(converter_, CreateImageToGlTextureTensorConverter(
                                       cc, DoesInputStartAtBottom()));
#endif  // MEDIAPIPE_METAL_ENABLED

#endif  // MEDIAPIPE_DISABLE_GPU
    }
    return ::mediapipe::OkStatus();
  }

  ::mediapipe::Status Process(CalculatorContext* cc) {
    const InputStreamShard& input = cc->Inputs().Tag(
        cc->Inputs().HasTag(kInputCpu) ? kInputCpu : kInputGpu);
    if (input.IsEmpty()) {
      // Timestamp bound update happens automatically. (See Open().)
      return ::mediapipe::OkStatus();
    }

    absl::optional<mediapipe::NormalizedRect> norm_rect;
    if (cc->Inputs().HasTag(kInputNormRect)) {
      if (cc->Inputs().Tag(kInputNormRect).IsEmpty()) {
        // Timestamp bound update happens automatically. (See Open().)
        return ::mediapipe::OkStatus();
      }
      norm_rect =
          cc->Inputs().Tag(kInputNormRect).Get<mediapipe::NormalizedRect>();
      if (norm_rect->width() == 0 && norm_rect->height() == 0) {
        // WORKAROUND: some existing graphs may use sentinel rects {width=0,
        // height=0, ...} quite often and calculator has to handle them
        // gracefully by updating timestamp bound instead of returning failure.
        // Timestamp bound update happens automatically. (See Open().)
        // NOTE: usage of sentinel rects should be avoided.
        DLOG(WARNING)
            << "Updating timestamp bound in response to a sentinel rect";
        return ::mediapipe::OkStatus();
      }
    }

    const Packet& image_packet = input.Value();
    const Size& size = converter_->GetImageSize(image_packet);
    RotatedRect roi = GetRoi(size.width, size.height, norm_rect);
    ASSIGN_OR_RETURN(auto padding, PadRoi(options_.output_tensor_width(),
                                          options_.output_tensor_height(),
                                          options_.keep_aspect_ratio(), &roi));
    if (cc->Outputs().HasTag(kOutputLetterboxPadding)) {
      cc->Outputs()
          .Tag(kOutputLetterboxPadding)
          .AddPacket(MakePacket<std::array<float, 4>>(padding).At(
              cc->InputTimestamp()));
    }
    if (cc->Outputs().HasTag(kOutputMatrix)) {
      std::array<float, 16> matrix;
      GetRotatedSubRectToRectTransformMatrix(roi, size.width, size.height,
                                             /*flip_horizontaly=*/false,
                                             &matrix);
      cc->Outputs()
          .Tag(kOutputMatrix)
          .AddPacket(MakePacket<std::array<float, 16>>(std::move(matrix))
                         .At(cc->InputTimestamp()));
    }

    ASSIGN_OR_RETURN(
        Tensor tensor,
        converter_->Convert(image_packet, roi, {output_width_, output_height_},
                            range_min_, range_max_));

    std::vector<Tensor> result;
    result.push_back(std::move(tensor));
    cc->Outputs().Tag(kOutput).AddPacket(
        MakePacket<std::vector<Tensor>>(std::move(result))
            .At(cc->InputTimestamp()));

    return ::mediapipe::OkStatus();
  }

 private:
  bool DoesInputStartAtBottom() {
    return options_.gpu_origin() != mediapipe::GpuOrigin_Mode_TOP_LEFT;
  }

  std::unique_ptr<ImageToTensorConverter> converter_;
  mediapipe::ImageToTensorCalculatorOptions options_;
  int output_width_ = 0;
  int output_height_ = 0;
  float range_min_ = 0.0f;
  float range_max_ = 1.0f;
};

REGISTER_CALCULATOR(ImageToTensorCalculator);

}  // namespace mediapipe
