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

#include "mediapipe/calculators/tensor/image_to_tensor_utils.h"

#include <array>
#include <cmath>
#include <memory>
#include <utility>

#include "absl/status/statusor.h"
#include "absl/types/optional.h"
#include "mediapipe/framework/api2/packet.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/image.h"
#include "mediapipe/framework/formats/image_frame.h"
#include "mediapipe/framework/formats/tensor.h"
#include "mediapipe/framework/port/ret_check.h"
#include "mediapipe/framework/port/status_macros.h"
#if !MEDIAPIPE_DISABLE_GPU
#include "mediapipe/gpu/gpu_buffer.h"
#endif  // !MEDIAPIPE_DISABLE_GPU

namespace mediapipe {

RotatedRect GetRoi(int input_width, int input_height,
                   absl::optional<mediapipe::NormalizedRect> norm_rect) {
  if (norm_rect) {
    return {/*center_x=*/norm_rect->x_center() * input_width,
            /*center_y =*/norm_rect->y_center() * input_height,
            /*width =*/norm_rect->width() * input_width,
            /*height =*/norm_rect->height() * input_height,
            /*rotation =*/norm_rect->rotation()};
  }
  return {/*center_x=*/0.5f * input_width,
          /*center_y =*/0.5f * input_height,
          /*width =*/static_cast<float>(input_width),
          /*height =*/static_cast<float>(input_height),
          /*rotation =*/0};
}

absl::StatusOr<std::array<float, 4>> PadRoi(int input_tensor_width,
                                            int input_tensor_height,
                                            bool keep_aspect_ratio,
                                            RotatedRect* roi) {
  if (!keep_aspect_ratio) {
    return std::array<float, 4>{0.0f, 0.0f, 0.0f, 0.0f};
  }

  RET_CHECK(input_tensor_width > 0 && input_tensor_height > 0)
      << "Input tensor width and height must be > 0.";
  const float tensor_aspect_ratio =
      static_cast<float>(input_tensor_height) / input_tensor_width;

  RET_CHECK(roi->width > 0 && roi->height > 0)
      << "ROI width and height must be > 0.";
  const float roi_aspect_ratio = roi->height / roi->width;

  float vertical_padding = 0.0f;
  float horizontal_padding = 0.0f;
  float new_width;
  float new_height;
  if (tensor_aspect_ratio > roi_aspect_ratio) {
    new_width = roi->width;
    new_height = roi->width * tensor_aspect_ratio;
    vertical_padding = (1.0f - roi_aspect_ratio / tensor_aspect_ratio) / 2.0f;
  } else {
    new_width = roi->height / tensor_aspect_ratio;
    new_height = roi->height;
    horizontal_padding = (1.0f - tensor_aspect_ratio / roi_aspect_ratio) / 2.0f;
  }

  roi->width = new_width;
  roi->height = new_height;

  return std::array<float, 4>{horizontal_padding, vertical_padding,
                              horizontal_padding, vertical_padding};
}

absl::StatusOr<ValueTransformation> GetValueRangeTransformation(
    float from_range_min, float from_range_max, float to_range_min,
    float to_range_max) {
  RET_CHECK_LT(from_range_min, from_range_max)
      << "Invalid FROM range: min >= max.";
  RET_CHECK_LT(to_range_min, to_range_max) << "Invalid TO range: min >= max.";
  const float scale =
      (to_range_max - to_range_min) / (from_range_max - from_range_min);
  const float offset = to_range_min - from_range_min * scale;
  return ValueTransformation{scale, offset};
}

void GetRotatedSubRectToRectTransformMatrix(const RotatedRect& sub_rect,
                                            int rect_width, int rect_height,
                                            bool flip_horizontally,
                                            std::array<float, 16>* matrix_ptr) {
  std::array<float, 16>& matrix = *matrix_ptr;
  // The resulting matrix is multiplication of below commented out matrices:
  //   post_scale_matrix
  //     * translate_matrix
  //     * rotate_matrix
  //     * flip_matrix
  //     * scale_matrix
  //     * initial_translate_matrix

  // Matrix to convert X,Y to [-0.5, 0.5] range "initial_translate_matrix"
  // { 1.0f,  0.0f, 0.0f, -0.5f}
  // { 0.0f,  1.0f, 0.0f, -0.5f}
  // { 0.0f,  0.0f, 1.0f,  0.0f}
  // { 0.0f,  0.0f, 0.0f,  1.0f}

  const float a = sub_rect.width;
  const float b = sub_rect.height;
  // Matrix to scale X,Y,Z to sub rect "scale_matrix"
  // Z has the same scale as X.
  // {   a, 0.0f, 0.0f, 0.0f}
  // {0.0f,    b, 0.0f, 0.0f}
  // {0.0f, 0.0f,    a, 0.0f}
  // {0.0f, 0.0f, 0.0f, 1.0f}

  const float flip = flip_horizontally ? -1 : 1;
  // Matrix for optional horizontal flip around middle of output image.
  // { fl  , 0.0f, 0.0f, 0.0f}
  // { 0.0f, 1.0f, 0.0f, 0.0f}
  // { 0.0f, 0.0f, 1.0f, 0.0f}
  // { 0.0f, 0.0f, 0.0f, 1.0f}

  const float c = std::cos(sub_rect.rotation);
  const float d = std::sin(sub_rect.rotation);
  // Matrix to do rotation around Z axis "rotate_matrix"
  // {    c,   -d, 0.0f, 0.0f}
  // {    d,    c, 0.0f, 0.0f}
  // { 0.0f, 0.0f, 1.0f, 0.0f}
  // { 0.0f, 0.0f, 0.0f, 1.0f}

  const float e = sub_rect.center_x;
  const float f = sub_rect.center_y;
  // Matrix to do X,Y translation of sub rect within parent rect
  // "translate_matrix"
  // {1.0f, 0.0f, 0.0f, e   }
  // {0.0f, 1.0f, 0.0f, f   }
  // {0.0f, 0.0f, 1.0f, 0.0f}
  // {0.0f, 0.0f, 0.0f, 1.0f}

  const float g = 1.0f / rect_width;
  const float h = 1.0f / rect_height;
  // Matrix to scale X,Y,Z to [0.0, 1.0] range "post_scale_matrix"
  // {g,    0.0f, 0.0f, 0.0f}
  // {0.0f, h,    0.0f, 0.0f}
  // {0.0f, 0.0f,    g, 0.0f}
  // {0.0f, 0.0f, 0.0f, 1.0f}

  // row 1
  matrix[0] = a * c * flip * g;
  matrix[1] = -b * d * g;
  matrix[2] = 0.0f;
  matrix[3] = (-0.5f * a * c * flip + 0.5f * b * d + e) * g;

  // row 2
  matrix[4] = a * d * flip * h;
  matrix[5] = b * c * h;
  matrix[6] = 0.0f;
  matrix[7] = (-0.5f * b * c - 0.5f * a * d * flip + f) * h;

  // row 3
  matrix[8] = 0.0f;
  matrix[9] = 0.0f;
  matrix[10] = a * g;
  matrix[11] = 0.0f;

  // row 4
  matrix[12] = 0.0f;
  matrix[13] = 0.0f;
  matrix[14] = 0.0f;
  matrix[15] = 1.0f;
}

void GetTransposedRotatedSubRectToRectTransformMatrix(
    const RotatedRect& sub_rect, int rect_width, int rect_height,
    bool flip_horizontally, std::array<float, 16>* matrix_ptr) {
  std::array<float, 16>& matrix = *matrix_ptr;
  // See comments in GetRotatedSubRectToRectTransformMatrix for detailed
  // calculations.
  const float a = sub_rect.width;
  const float b = sub_rect.height;
  const float flip = flip_horizontally ? -1 : 1;
  const float c = std::cos(sub_rect.rotation);
  const float d = std::sin(sub_rect.rotation);
  const float e = sub_rect.center_x;
  const float f = sub_rect.center_y;
  const float g = 1.0f / rect_width;
  const float h = 1.0f / rect_height;

  // row 1 (indices 0,4,8,12 from non-transposed fcn)
  matrix[0] = a * c * flip * g;
  matrix[1] = a * d * flip * h;
  matrix[2] = 0.0f;
  matrix[3] = 0.0f;

  // row 2 (indices 1,5,9,13 from non-transposed fcn)
  matrix[4] = -b * d * g;
  matrix[5] = b * c * h;
  matrix[6] = 0.0f;
  matrix[7] = 0.0f;

  // row 3 (indices 2,6,10,14 from non-transposed fcn)
  matrix[8] = 0.0f;
  matrix[9] = 0.0f;
  matrix[10] = a * g;
  matrix[11] = 0.0f;

  // row 4 (indices 3,7,11,15 from non-transposed fcn)
  matrix[12] = (-0.5f * a * c * flip + 0.5f * b * d + e) * g;
  matrix[13] = (-0.5f * b * c - 0.5f * a * d * flip + f) * h;
  matrix[14] = 0.0f;
  matrix[15] = 1.0f;
}

BorderMode GetBorderMode(
    const mediapipe::ImageToTensorCalculatorOptions::BorderMode& mode) {
  switch (mode) {
    case mediapipe::
        ImageToTensorCalculatorOptions_BorderMode_BORDER_UNSPECIFIED:
      return BorderMode::kReplicate;
    case mediapipe::ImageToTensorCalculatorOptions_BorderMode_BORDER_ZERO:
      return BorderMode::kZero;
    case mediapipe::ImageToTensorCalculatorOptions_BorderMode_BORDER_REPLICATE:
      return BorderMode::kReplicate;
  }
}

Tensor::ElementType GetOutputTensorType(bool uses_gpu,
                                        const OutputTensorParams& params) {
  if (!uses_gpu) {
    if (params.is_float_output) {
      return Tensor::ElementType::kFloat32;
    }
    if (params.range_min < 0) {
      return Tensor::ElementType::kInt8;
    } else {
      return Tensor::ElementType::kUInt8;
    }
  }
  // Always use float32 when GPU is enabled.
  return Tensor::ElementType::kFloat32;
}

int GetNumOutputChannels(const mediapipe::Image& image) {
#if !MEDIAPIPE_DISABLE_GPU
#if MEDIAPIPE_METAL_ENABLED
  if (image.UsesGpu()) {
    return 4;
  }
#endif  // MEDIAPIPE_METAL_ENABLED
#endif  // !MEDIAPIPE_DISABLE_GPU
  // TODO: Add a unittest here to test the behavior on GPU, i.e.
  // failure.
  // Only output channel == 1 when running on CPU and the input image channel
  // is 1. Ideally, we want to also support GPU for output channel == 1. But
  // setting this on the safer side to prevent unintentional failure.
  if (!image.UsesGpu() && image.channels() == 1) {
    return 1;
  }
  return 3;
}

absl::StatusOr<std::shared_ptr<const mediapipe::Image>> GetInputImage(
    const api2::Packet<api2::OneOf<Image, mediapipe::ImageFrame>>&
        image_packet) {
  return image_packet.Visit(
      [&image_packet](const mediapipe::Image&) {
        return image_packet.Share<mediapipe::Image>();
      },
      [&image_packet](const mediapipe::ImageFrame&)
          -> absl::StatusOr<std::shared_ptr<const mediapipe::Image>> {
        MP_ASSIGN_OR_RETURN(
            std::shared_ptr<const mediapipe::ImageFrame> image_frame,
            image_packet.Share<mediapipe::ImageFrame>());
        return std::make_shared<const mediapipe::Image>(
            std::const_pointer_cast<mediapipe::ImageFrame>(
                std::move(image_frame)));
      });
}

#if !MEDIAPIPE_DISABLE_GPU
absl::StatusOr<std::shared_ptr<const mediapipe::Image>> GetInputImage(
    const api2::Packet<mediapipe::GpuBuffer>& image_gpu_packet) {
  // A shallow copy is okay since the resulting 'image' object is local in
  // Process(), and thus never outlives 'input'.
  return std::make_shared<const mediapipe::Image>(image_gpu_packet.Get());
}
#endif  // !MEDIAPIPE_DISABLE_GPU

}  // namespace mediapipe
