// Copyright 2023 The MediaPipe Authors.
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

#include "mediapipe/calculators/tensor/image_to_tensor_converter_frame_buffer.h"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "mediapipe/calculators/tensor/image_to_tensor_converter.h"
#include "mediapipe/calculators/tensor/image_to_tensor_utils.h"
#include "mediapipe/framework/calculator_context.h"
#include "mediapipe/framework/formats/frame_buffer.h"
#include "mediapipe/framework/formats/image.h"
#include "mediapipe/framework/formats/tensor.h"
#include "mediapipe/framework/port/status_macros.h"
#include "mediapipe/gpu/frame_buffer_view.h"
#include "mediapipe/util/frame_buffer/frame_buffer_util.h"

namespace mediapipe {

namespace {

// Converts from radians (clockwise) to degrees (counter-clockwise) in [0,360).
int RadiansToDegrees(float radians) {
  int degrees = static_cast<int>(std::round(-radians * 180 / M_PI)) % 360;
  if (degrees < 0) {
    degrees += 360;
  }
  return degrees;
}

// FrameBuffer-based implementation of ImageToTensorConverter.
class ImageToTensorFrameBufferConverter : public ImageToTensorConverter {
 public:
  explicit ImageToTensorFrameBufferConverter(Tensor::ElementType tensor_type)
      : tensor_type_(tensor_type) {}

  absl::Status Convert(const mediapipe::Image& input, const RotatedRect& roi,
                       float range_min, float range_max,
                       int tensor_buffer_offset,
                       Tensor& output_tensor) override;

 private:
  absl::Status ValidateTensorShape(const Tensor::Shape& output_shape);
  // Crops, rotates and resizes the input based on the provided
  // region-of-interest.
  absl::Status CropRotateResize90Degrees(
      std::shared_ptr<const FrameBuffer> input, const RotatedRect& roi,
      std::shared_ptr<FrameBuffer> output);
  // Converts the input FrameBuffer to a float Tensor. Output tensor must have
  // type kFloat32.
  absl::Status ConvertToFloatTensor(
      std::shared_ptr<const FrameBuffer> input_frame, float range_min,
      float range_max, Tensor& output_tensor);

  Tensor::ElementType tensor_type_;

  // Temporary buffers and their respective sizes.
  std::unique_ptr<uint8_t[]> cropped_buffer_;
  size_t cropped_buffer_size_ = 0;
  std::unique_ptr<uint8_t[]> rotated_buffer_;
  size_t rotated_buffer_size_ = 0;
  std::unique_ptr<uint8_t[]> output_buffer_;
  size_t output_buffer_size_ = 0;
};

absl::Status ImageToTensorFrameBufferConverter::Convert(
    const mediapipe::Image& input, const RotatedRect& roi, float range_min,
    float range_max, int tensor_buffer_offset, Tensor& output_tensor) {
  // TODO: add support for non-zero tensor buffer offset.
  RET_CHECK_EQ(tensor_buffer_offset, 0)
      << "Non-zero tensor_buffer_offset input is not supported yet.";

  // Range other than [0,255] is not supported for uint8 tensor outputs.
  if (tensor_type_ == Tensor::ElementType::kUInt8) {
    RET_CHECK(static_cast<int>(range_min) == 0 &&
              static_cast<int>(range_max) == 255);
  }

  auto input_frame =
      input.GetGpuBuffer(/*upload_to_gpu=*/false).GetReadView<FrameBuffer>();
  const auto& output_shape = output_tensor.shape();
  MP_RETURN_IF_ERROR(ValidateTensorShape(output_shape));
  FrameBuffer::Dimension output_dimension{/*width=*/output_shape.dims[2],
                                          /*height=*/output_shape.dims[1]};

  // Optimized path for multiples of 90°.
  if (RadiansToDegrees(roi.rotation) % 90 == 0) {
    if (tensor_type_ == Tensor::ElementType::kUInt8) {
      auto view = output_tensor.GetCpuWriteView();
      uint8_t* data = view.buffer<uint8_t>();
      auto output_frame =
          frame_buffer::CreateFromRgbRawBuffer(data, output_dimension);
      return CropRotateResize90Degrees(input_frame, roi, output_frame);
    } else {
      size_t output_buffer_size = frame_buffer::GetFrameBufferByteSize(
          output_dimension, FrameBuffer::Format::kRGB);
      if (output_buffer_size > output_buffer_size_) {
        output_buffer_ = std::make_unique<uint8_t[]>(output_buffer_size);
        output_buffer_size_ = output_buffer_size;
      }
      auto output_frame = frame_buffer::CreateFromRgbRawBuffer(
          output_buffer_.get(), output_dimension);
      MP_RETURN_IF_ERROR(
          CropRotateResize90Degrees(input_frame, roi, output_frame));
      return ConvertToFloatTensor(output_frame, range_min, range_max,
                                  output_tensor);
    }
  } else {
    // TODO: add support for arbitrary rotations
    return absl::UnimplementedError(
        "FrameBufferConverter doesn't yet support rotations that are not "
        "multiples of 90°.");
  }
  return absl::OkStatus();
}

absl::Status ImageToTensorFrameBufferConverter::ValidateTensorShape(
    const Tensor::Shape& shape) {
  RET_CHECK_EQ(shape.dims.size(), 4)
      << "Wrong output dims size: " << shape.dims.size();
  RET_CHECK_EQ(shape.dims[0], 1)
      << "Handling batch dimension not equal to 1 is not implemented in this "
         "converter.";
  RET_CHECK_EQ(shape.dims[3], 3) << "Wrong output channel: " << shape.dims[3];
  return absl::OkStatus();
}

absl::Status ImageToTensorFrameBufferConverter::CropRotateResize90Degrees(
    std::shared_ptr<const FrameBuffer> input, const RotatedRect& roi,
    std::shared_ptr<FrameBuffer> output) {
  int rotation_degrees = RadiansToDegrees(roi.rotation);
  bool rotation_required = rotation_degrees != 0;
  bool conversion_required = input->format() != output->format();

  // First, crop and resize.
  std::shared_ptr<FrameBuffer> cropped = output;
  FrameBuffer::Dimension cropped_dims = output->dimension();
  int left, right, top, bottom;
  if (rotation_degrees % 180 != 0) {
    cropped_dims.Swap();
    left = roi.center_x - roi.height / 2;
    right = left + roi.height - 1;
    top = roi.center_y - roi.width / 2;
    bottom = top + roi.width - 1;
  } else {
    left = roi.center_x - roi.width / 2;
    right = left + roi.width - 1;
    top = roi.center_y - roi.height / 2;
    bottom = top + roi.height - 1;
  }
  if (rotation_required || conversion_required) {
    // Create temporary FrameBuffer from recycled buffer.
    size_t cropped_buffer_size =
        frame_buffer::GetFrameBufferByteSize(cropped_dims, input->format());
    if (cropped_buffer_size > cropped_buffer_size_) {
      cropped_buffer_ = std::make_unique<uint8_t[]>(cropped_buffer_size);
      cropped_buffer_size_ = cropped_buffer_size;
    }
    MP_ASSIGN_OR_RETURN(
        cropped, frame_buffer::CreateFromRawBuffer(
                     cropped_buffer_.get(), cropped_dims, input->format()));
  }
  MP_RETURN_IF_ERROR(
      frame_buffer::Crop(*input, left, top, right, bottom, cropped.get()));

  // Then rotate if needed.
  std::shared_ptr<FrameBuffer> rotated = output;
  if (rotation_required) {
    if (conversion_required) {
      // Create temporary FrameBuffer from recycled buffer.
      FrameBuffer::Dimension rotated_dims = output->dimension();
      size_t rotated_buffer_size =
          frame_buffer::GetFrameBufferByteSize(rotated_dims, cropped->format());
      if (rotated_buffer_size > rotated_buffer_size_) {
        rotated_buffer_ = std::make_unique<uint8_t[]>(rotated_buffer_size);
        rotated_buffer_size_ = rotated_buffer_size;
      }
      MP_ASSIGN_OR_RETURN(auto rotated, frame_buffer::CreateFromRawBuffer(
                                            rotated_buffer_.get(), rotated_dims,
                                            cropped->format()));
    }
    MP_RETURN_IF_ERROR(
        frame_buffer::Rotate(*cropped, rotation_degrees, rotated.get()));
  } else {
    rotated = cropped;
  }

  // Then convert if needed.
  if (conversion_required) {
    return frame_buffer::Convert(*rotated, output.get());
  }
  return absl::OkStatus();
}

absl::Status ImageToTensorFrameBufferConverter::ConvertToFloatTensor(
    std::shared_ptr<const FrameBuffer> input_frame, float range_min,
    float range_max, Tensor& output_tensor) {
  RET_CHECK(output_tensor.element_type() == Tensor::ElementType::kFloat32);
  constexpr float kInputImageRangeMin = 0.0f;
  constexpr float kInputImageRangeMax = 255.0f;
  MP_ASSIGN_OR_RETURN(
      auto transform,
      GetValueRangeTransformation(kInputImageRangeMin, kInputImageRangeMax,
                                  range_min, range_max));
  return frame_buffer::ToFloatTensor(*input_frame, transform.scale,
                                     transform.offset, output_tensor);
}

}  // namespace

absl::StatusOr<std::unique_ptr<ImageToTensorConverter>>
CreateFrameBufferConverter(CalculatorContext* cc, BorderMode border_mode,
                           Tensor::ElementType tensor_type) {
  if (tensor_type != Tensor::ElementType::kUInt8 &&
      tensor_type != Tensor::ElementType::kFloat32) {
    return absl::InvalidArgumentError(
        absl::StrFormat("Tensor type is currently not supported by "
                        "ImageToTensorFrameBufferConverter, type: %d.",
                        tensor_type));
  }
  // TODO: add support for BorderMode:kZero.
  if (border_mode == BorderMode::kZero) {
    return absl::UnimplementedError(
        "BorderMode::kZero is not yet supported by "
        "ImageToTensorFrameBufferConverter");
  }
  return std::make_unique<ImageToTensorFrameBufferConverter>(tensor_type);
}

}  // namespace mediapipe
