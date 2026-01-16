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

#include "mediapipe/util/frame_buffer/frame_buffer_util.h"

#include <cstdint>
#include <memory>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "mediapipe/framework/formats/frame_buffer.h"
#include "mediapipe/framework/formats/tensor.h"
#include "mediapipe/framework/port/status_macros.h"
#include "mediapipe/util/frame_buffer/float_buffer.h"
#include "mediapipe/util/frame_buffer/gray_buffer.h"
#include "mediapipe/util/frame_buffer/rgb_buffer.h"
#include "mediapipe/util/frame_buffer/yuv_buffer.h"

namespace mediapipe {
namespace frame_buffer {

namespace {

constexpr int kRgbaChannels = 4;
constexpr int kRgbaPixelBytes = 4;
constexpr int kRgbChannels = 3;
constexpr int kRgbPixelBytes = 3;
constexpr int kGrayChannel = 1;
constexpr int kGrayPixelBytes = 1;

// YUV helpers.
//------------------------------------------------------------------------------

// Returns whether the buffer is part of the supported Yuv format.
bool IsSupportedYuvBuffer(const FrameBuffer& buffer) {
  return buffer.format() == FrameBuffer::Format::kNV21 ||
         buffer.format() == FrameBuffer::Format::kNV12 ||
         buffer.format() == FrameBuffer::Format::kYV12 ||
         buffer.format() == FrameBuffer::Format::kYV21;
}

// Returns the number of channels for the provided buffer. Returns an error if
// the buffer is not using an interleaved single-planar format.
absl::StatusOr<int> NumberOfChannels(const FrameBuffer& buffer) {
  switch (buffer.format()) {
    case FrameBuffer::Format::kGRAY:
      return kGrayChannel;
    case FrameBuffer::Format::kRGB:
      return kRgbChannels;
    case FrameBuffer::Format::kRGBA:
      return kRgbaChannels;
    default:
      return absl::InvalidArgumentError(
          absl::StrFormat("Unsupported buffer format: %i.", buffer.format()));
  }
}

// Shared validation functions.
//------------------------------------------------------------------------------

// Indicates whether the given buffers have the same dimensions.
bool AreBufferDimsEqual(const FrameBuffer& buffer1,
                        const FrameBuffer& buffer2) {
  return buffer1.dimension() == buffer2.dimension();
}

// Indicates whether the given buffers formats are compatible. Same formats are
// compatible and all YUV family formats (e.g. NV21, NV12, YV12, YV21, etc) are
// compatible.
bool AreBufferFormatsCompatible(const FrameBuffer& buffer1,
                                const FrameBuffer& buffer2) {
  switch (buffer1.format()) {
    case FrameBuffer::Format::kRGBA:
    case FrameBuffer::Format::kRGB:
      return (buffer2.format() == FrameBuffer::Format::kRGBA ||
              buffer2.format() == FrameBuffer::Format::kRGB);
    case FrameBuffer::Format::kNV12:
    case FrameBuffer::Format::kNV21:
    case FrameBuffer::Format::kYV12:
    case FrameBuffer::Format::kYV21:
      return (buffer2.format() == FrameBuffer::Format::kNV12 ||
              buffer2.format() == FrameBuffer::Format::kNV21 ||
              buffer2.format() == FrameBuffer::Format::kYV12 ||
              buffer2.format() == FrameBuffer::Format::kYV21);
    case FrameBuffer::Format::kGRAY:
    default:
      return buffer1.format() == buffer2.format();
  }
}

absl::Status ValidateBufferFormat(const FrameBuffer& buffer) {
  switch (buffer.format()) {
    case FrameBuffer::Format::kGRAY:
    case FrameBuffer::Format::kRGB:
    case FrameBuffer::Format::kRGBA:
      if (buffer.plane_count() == 1) return absl::OkStatus();
      return absl::InvalidArgumentError(
          "Plane count must be 1 for grayscale and RGB[a] buffers.");
    case FrameBuffer::Format::kNV21:
    case FrameBuffer::Format::kNV12:
    case FrameBuffer::Format::kYV21:
    case FrameBuffer::Format::kYV12:
      return absl::OkStatus();
    default:
      return absl::InternalError(
          absl::StrFormat("Unsupported buffer format: %i.", buffer.format()));
  }
}

absl::Status ValidateBufferFormats(const FrameBuffer& buffer1,
                                   const FrameBuffer& buffer2) {
  MP_RETURN_IF_ERROR(ValidateBufferFormat(buffer1));
  MP_RETURN_IF_ERROR(ValidateBufferFormat(buffer2));
  return absl::OkStatus();
}

absl::Status ValidateResizeBufferInputs(const FrameBuffer& buffer,
                                        const FrameBuffer& output_buffer) {
  bool valid_format = false;
  switch (buffer.format()) {
    case FrameBuffer::Format::kGRAY:
    case FrameBuffer::Format::kRGB:
    case FrameBuffer::Format::kNV12:
    case FrameBuffer::Format::kNV21:
    case FrameBuffer::Format::kYV12:
    case FrameBuffer::Format::kYV21:
      valid_format = (buffer.format() == output_buffer.format());
      break;
    case FrameBuffer::Format::kRGBA:
      valid_format = (output_buffer.format() == FrameBuffer::Format::kRGBA ||
                      output_buffer.format() == FrameBuffer::Format::kRGB);
      break;
    default:
      return absl::InternalError(
          absl::StrFormat("Unsupported buffer format: %i.", buffer.format()));
  }
  if (!valid_format) {
    return absl::InvalidArgumentError(
        "Input and output buffer formats must match.");
  }
  return ValidateBufferFormats(buffer, output_buffer);
}

absl::Status ValidateRotateBufferInputs(const FrameBuffer& buffer,
                                        const FrameBuffer& output_buffer,
                                        int angle_deg) {
  if (!AreBufferFormatsCompatible(buffer, output_buffer)) {
    return absl::InvalidArgumentError(
        "Input and output buffer formats must match.");
  }

  const bool is_dimension_change = (angle_deg / 90) % 2 == 1;
  const bool are_dimensions_rotated =
      (buffer.dimension().width == output_buffer.dimension().height) &&
      (buffer.dimension().height == output_buffer.dimension().width);
  const bool are_dimensions_equal =
      buffer.dimension() == output_buffer.dimension();

  if (angle_deg >= 360 || angle_deg <= 0 || angle_deg % 90 != 0) {
    return absl::InvalidArgumentError(
        "Rotation angle must be between 0 and 360, in multiples of 90 "
        "degrees.");
  } else if ((is_dimension_change && !are_dimensions_rotated) ||
             (!is_dimension_change && !are_dimensions_equal)) {
    return absl::InvalidArgumentError(
        "Output buffer has invalid dimensions for rotation.");
  }
  return absl::OkStatus();
}

absl::Status ValidateCropBufferInputs(const FrameBuffer& buffer,
                                      const FrameBuffer& output_buffer, int x0,
                                      int y0, int x1, int y1) {
  if (!AreBufferFormatsCompatible(buffer, output_buffer)) {
    return absl::InvalidArgumentError(
        "Input and output buffer formats must match.");
  }

  bool is_buffer_size_valid =
      ((x1 < buffer.dimension().width) && y1 < buffer.dimension().height);
  bool are_points_valid = (x0 >= 0) && (y0 >= 0) && (x1 >= x0) && (y1 >= y0);

  if (!is_buffer_size_valid || !are_points_valid) {
    return absl::InvalidArgumentError("Invalid crop coordinates.");
  }
  return absl::OkStatus();
}

absl::Status ValidateFlipBufferInputs(const FrameBuffer& buffer,
                                      const FrameBuffer& output_buffer) {
  if (!AreBufferFormatsCompatible(buffer, output_buffer)) {
    return absl::InvalidArgumentError(
        "Input and output buffer formats must match.");
  }
  return AreBufferDimsEqual(buffer, output_buffer)
             ? absl::OkStatus()
             : absl::InvalidArgumentError(
                   "Input and output buffers must have the same dimensions.");
}

absl::Status ValidateConvertFormats(FrameBuffer::Format from_format,
                                    FrameBuffer::Format to_format) {
  if (from_format == to_format) {
    return absl::InvalidArgumentError("Formats must be different.");
  }

  switch (from_format) {
    case FrameBuffer::Format::kGRAY:
      return absl::InvalidArgumentError(
          "Grayscale format does not convert to other formats.");
    case FrameBuffer::Format::kRGB:
    case FrameBuffer::Format::kRGBA:
    case FrameBuffer::Format::kNV12:
    case FrameBuffer::Format::kNV21:
    case FrameBuffer::Format::kYV12:
    case FrameBuffer::Format::kYV21:
      return absl::OkStatus();
    default:
      return absl::InternalError(
          absl::StrFormat("Unsupported buffer format: %i.", from_format));
  }
}

absl::Status ValidateFloatTensorInputs(const FrameBuffer& buffer,
                                       const Tensor& tensor) {
  if (tensor.element_type() != Tensor::ElementType::kFloat32) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "Tensor type %i is not supported.", tensor.element_type()));
  }
  const auto& shape = tensor.shape();
  if (shape.dims.size() != 4 || shape.dims[0] != 1) {
    return absl::InvalidArgumentError("Expected tensor with batch size of 1.");
  }
  MP_ASSIGN_OR_RETURN(int channels, NumberOfChannels(buffer));
  if (shape.dims[2] != buffer.dimension().width ||
      shape.dims[1] != buffer.dimension().height || shape.dims[3] != channels) {
    return absl::InvalidArgumentError(
        "Input buffer and output tensor must have the same dimensions.");
  }
  return absl::OkStatus();
}

// Construct buffer helper functions.
//------------------------------------------------------------------------------

// Creates NV12 / NV21 / YV12 / YV21 YuvBuffer from the input `buffer`. The
// output YuvBuffer is agnostic to the YUV format since the YUV buffers are
// managed individually.
absl::StatusOr<YuvBuffer> CreateYuvBuffer(const FrameBuffer& buffer) {
  MP_ASSIGN_OR_RETURN(FrameBuffer::YuvData yuv_data,
                      FrameBuffer::GetYuvDataFromFrameBuffer(buffer));
  return YuvBuffer(const_cast<uint8_t*>(yuv_data.y_buffer),
                   const_cast<uint8_t*>(yuv_data.u_buffer),
                   const_cast<uint8_t*>(yuv_data.v_buffer),
                   buffer.dimension().width, buffer.dimension().height,
                   yuv_data.y_row_stride, yuv_data.uv_row_stride,
                   yuv_data.uv_pixel_stride);
}

absl::StatusOr<GrayBuffer> CreateGrayBuffer(const FrameBuffer& buffer) {
  if (buffer.plane_count() != 1) {
    return absl::InternalError("Unsupported grayscale planar format.");
  }
  return GrayBuffer(const_cast<uint8_t*>(buffer.plane(0).buffer()),
                    buffer.dimension().width, buffer.dimension().height);
}

absl::StatusOr<RgbBuffer> CreateRgbBuffer(const FrameBuffer& buffer) {
  if (buffer.plane_count() != 1) {
    return absl::InternalError("Unsupported rgb[a] planar format.");
  }
  bool alpha = buffer.format() == FrameBuffer::Format::kRGBA ? true : false;
  return RgbBuffer(const_cast<uint8_t*>(buffer.plane(0).buffer()),
                   buffer.dimension().width, buffer.dimension().height,
                   buffer.plane(0).stride().row_stride_bytes, alpha);
}

// Grayscale transformation functions.
//------------------------------------------------------------------------------

absl::Status CropGrayscale(const FrameBuffer& buffer, int x0, int y0, int x1,
                           int y1, FrameBuffer* output_buffer) {
  MP_ASSIGN_OR_RETURN(auto input, CreateGrayBuffer(buffer));
  MP_ASSIGN_OR_RETURN(auto output, CreateGrayBuffer(*output_buffer));
  bool success_crop = input.Crop(x0, y0, x1, y1);
  if (!success_crop) {
    return absl::UnknownError("Halide grayscale crop operation failed.");
  }
  bool success_resize = input.Resize(&output);
  if (!success_resize) {
    return absl::UnknownError("Halide grayscale resize operation failed.");
  }
  return absl::OkStatus();
}

absl::Status ResizeGrayscale(const FrameBuffer& buffer,
                             FrameBuffer* output_buffer) {
  MP_ASSIGN_OR_RETURN(auto input, CreateGrayBuffer(buffer));
  MP_ASSIGN_OR_RETURN(auto output, CreateGrayBuffer(*output_buffer));
  return input.Resize(&output)
             ? absl::OkStatus()
             : absl::UnknownError("Halide grayscale resize operation failed.");
}

absl::Status RotateGrayscale(const FrameBuffer& buffer, int angle_deg,
                             FrameBuffer* output_buffer) {
  MP_ASSIGN_OR_RETURN(auto input, CreateGrayBuffer(buffer));
  MP_ASSIGN_OR_RETURN(auto output, CreateGrayBuffer(*output_buffer));
  return input.Rotate(angle_deg % 360, &output)
             ? absl::OkStatus()
             : absl::UnknownError("Halide grayscale rotate operation failed.");
}

absl::Status FlipHorizontallyGrayscale(const FrameBuffer& buffer,
                                       FrameBuffer* output_buffer) {
  MP_ASSIGN_OR_RETURN(auto input, CreateGrayBuffer(buffer));
  MP_ASSIGN_OR_RETURN(auto output, CreateGrayBuffer(*output_buffer));
  return input.FlipHorizontally(&output)
             ? absl::OkStatus()
             : absl::UnknownError(
                   "Halide grayscale horizontal flip operation failed.");
}

absl::Status FlipVerticallyGrayscale(const FrameBuffer& buffer,
                                     FrameBuffer* output_buffer) {
  MP_ASSIGN_OR_RETURN(auto input, CreateGrayBuffer(buffer));
  MP_ASSIGN_OR_RETURN(auto output, CreateGrayBuffer(*output_buffer));
  return input.FlipVertically(&output)
             ? absl::OkStatus()
             : absl::UnknownError(
                   "Halide grayscale vertical flip operation failed.");
}

// Rgb transformation functions.
//------------------------------------------------------------------------------

absl::Status ResizeRgb(const FrameBuffer& buffer, FrameBuffer* output_buffer) {
  MP_ASSIGN_OR_RETURN(auto input, CreateRgbBuffer(buffer));
  MP_ASSIGN_OR_RETURN(auto output, CreateRgbBuffer(*output_buffer));
  return input.Resize(&output)
             ? absl::OkStatus()
             : absl::UnknownError("Halide rgb[a] resize operation failed.");
}

absl::Status ConvertRgb(const FrameBuffer& buffer, FrameBuffer* output_buffer) {
  MP_ASSIGN_OR_RETURN(auto input, CreateRgbBuffer(buffer));
  bool result = false;
  if (output_buffer->format() == FrameBuffer::Format::kGRAY) {
    MP_ASSIGN_OR_RETURN(auto output, CreateGrayBuffer(*output_buffer));
    result = input.Convert(&output);
  } else if (IsSupportedYuvBuffer(*output_buffer)) {
    MP_ASSIGN_OR_RETURN(auto output, CreateYuvBuffer(*output_buffer));
    result = input.Convert(&output);
  } else if (output_buffer->format() == FrameBuffer::Format::kRGBA ||
             output_buffer->format() == FrameBuffer::Format::kRGB) {
    MP_ASSIGN_OR_RETURN(auto output, CreateRgbBuffer(*output_buffer));
    result = input.Convert(&output);
  }
  return result ? absl::OkStatus()
                : absl::UnknownError("Halide rgb[a] convert operation failed.");
}

absl::Status CropRgb(const FrameBuffer& buffer, int x0, int y0, int x1, int y1,
                     FrameBuffer* output_buffer) {
  MP_ASSIGN_OR_RETURN(auto input, CreateRgbBuffer(buffer));
  MP_ASSIGN_OR_RETURN(auto output, CreateRgbBuffer(*output_buffer));
  bool success_crop = input.Crop(x0, y0, x1, y1);
  if (!success_crop) {
    return absl::UnknownError("Halide rgb[a] crop operation failed.");
  }
  bool success_resize = input.Resize(&output);
  if (!success_resize) {
    return absl::UnknownError("Halide rgb resize operation failed.");
  }
  return absl::OkStatus();
}

absl::Status FlipHorizontallyRgb(const FrameBuffer& buffer,
                                 FrameBuffer* output_buffer) {
  MP_ASSIGN_OR_RETURN(auto input, CreateRgbBuffer(buffer));
  MP_ASSIGN_OR_RETURN(auto output, CreateRgbBuffer(*output_buffer));
  return input.FlipHorizontally(&output)
             ? absl::OkStatus()
             : absl::UnknownError(
                   "Halide rgb[a] horizontal flip operation failed.");
}

absl::Status FlipVerticallyRgb(const FrameBuffer& buffer,
                               FrameBuffer* output_buffer) {
  MP_ASSIGN_OR_RETURN(auto input, CreateRgbBuffer(buffer));
  MP_ASSIGN_OR_RETURN(auto output, CreateRgbBuffer(*output_buffer));
  return input.FlipVertically(&output)
             ? absl::OkStatus()
             : absl::UnknownError(
                   "Halide rgb[a] vertical flip operation failed.");
}

absl::Status RotateRgb(const FrameBuffer& buffer, int angle,
                       FrameBuffer* output_buffer) {
  MP_ASSIGN_OR_RETURN(auto input, CreateRgbBuffer(buffer));
  MP_ASSIGN_OR_RETURN(auto output, CreateRgbBuffer(*output_buffer));
  return input.Rotate(angle % 360, &output)
             ? absl::OkStatus()
             : absl::UnknownError("Halide rgb[a] rotate operation failed.");
}

absl::Status ToFloatTensorRgb(const FrameBuffer& buffer, float scale,
                              float offset, Tensor& tensor) {
  MP_ASSIGN_OR_RETURN(auto input, CreateRgbBuffer(buffer));
  MP_ASSIGN_OR_RETURN(int channels, NumberOfChannels(buffer));
  auto view = tensor.GetCpuWriteView();
  float* data = view.buffer<float>();
  FloatBuffer output(data, buffer.dimension().width, buffer.dimension().height,
                     channels);
  return input.ToFloat(scale, offset, &output)
             ? absl::OkStatus()
             : absl::UnknownError("Halide rgb[a] to float conversion failed.");
}

// Yuv transformation functions.
//------------------------------------------------------------------------------

absl::Status CropYuv(const FrameBuffer& buffer, int x0, int y0, int x1, int y1,
                     FrameBuffer* output_buffer) {
  MP_ASSIGN_OR_RETURN(auto input, CreateYuvBuffer(buffer));
  MP_ASSIGN_OR_RETURN(auto output, CreateYuvBuffer(*output_buffer));
  bool success_crop = input.Crop(x0, y0, x1, y1);
  if (!success_crop) {
    return absl::UnknownError("Halide YUV crop operation failed.");
  }
  bool success_resize = input.Resize(&output);
  if (!success_resize) {
    return absl::UnknownError("Halide YUV resize operation failed.");
  }
  return absl::OkStatus();
}

absl::Status ResizeYuv(const FrameBuffer& buffer, FrameBuffer* output_buffer) {
  MP_ASSIGN_OR_RETURN(auto input, CreateYuvBuffer(buffer));
  MP_ASSIGN_OR_RETURN(auto output, CreateYuvBuffer(*output_buffer));
  return input.Resize(&output)
             ? absl::OkStatus()
             : absl::UnknownError("Halide YUV resize operation failed.");
}

absl::Status RotateYuv(const FrameBuffer& buffer, int angle_deg,
                       FrameBuffer* output_buffer) {
  MP_ASSIGN_OR_RETURN(auto input, CreateYuvBuffer(buffer));
  MP_ASSIGN_OR_RETURN(auto output, CreateYuvBuffer(*output_buffer));
  return input.Rotate(angle_deg % 360, &output)
             ? absl::OkStatus()
             : absl::UnknownError("Halide YUV rotate operation failed.");
}

absl::Status FlipHorizontallyYuv(const FrameBuffer& buffer,
                                 FrameBuffer* output_buffer) {
  MP_ASSIGN_OR_RETURN(auto input, CreateYuvBuffer(buffer));
  MP_ASSIGN_OR_RETURN(auto output, CreateYuvBuffer(*output_buffer));
  return input.FlipHorizontally(&output)
             ? absl::OkStatus()
             : absl::UnknownError(
                   "Halide YUV horizontal flip operation failed.");
}

absl::Status FlipVerticallyYuv(const FrameBuffer& buffer,
                               FrameBuffer* output_buffer) {
  MP_ASSIGN_OR_RETURN(auto input, CreateYuvBuffer(buffer));
  MP_ASSIGN_OR_RETURN(auto output, CreateYuvBuffer(*output_buffer));
  return input.FlipVertically(&output)
             ? absl::OkStatus()
             : absl::UnknownError("Halide YUV vertical flip operation failed.");
}

// Converts input YUV `buffer` into the `output_buffer` in RGB, RGBA or gray
// scale format.
absl::Status ConvertYuv(const FrameBuffer& buffer, FrameBuffer* output_buffer) {
  bool success_convert = false;
  MP_ASSIGN_OR_RETURN(auto input, CreateYuvBuffer(buffer));
  if (output_buffer->format() == FrameBuffer::Format::kRGBA ||
      output_buffer->format() == FrameBuffer::Format::kRGB) {
    MP_ASSIGN_OR_RETURN(auto output, CreateRgbBuffer(*output_buffer));
    bool half_sampling = false;
    if (buffer.dimension().width / 2 == output_buffer->dimension().width &&
        buffer.dimension().height / 2 == output_buffer->dimension().height) {
      half_sampling = true;
    }
    success_convert = input.Convert(half_sampling, &output);
  } else if (output_buffer->format() == FrameBuffer::Format::kGRAY) {
    if (buffer.plane(0).stride().row_stride_bytes == buffer.dimension().width) {
      std::copy(input.y_buffer()->host,
                input.y_buffer()->host + buffer.dimension().Size(),
                const_cast<uint8_t*>(output_buffer->plane(0).buffer()));
    } else {
      // The y_buffer is padded. The conversion removes the padding.
      uint8_t* gray_buffer =
          const_cast<uint8_t*>(output_buffer->plane(0).buffer());
      for (int i = 0; i < buffer.dimension().height; i++) {
        int src_address = i * buffer.plane(0).stride().row_stride_bytes;
        int dest_address = i * buffer.dimension().width;
        std::memcpy(&gray_buffer[dest_address],
                    &buffer.plane(0).buffer()[src_address],
                    buffer.dimension().width);
      }
    }
    success_convert = true;
  } else if (IsSupportedYuvBuffer(*output_buffer)) {
    MP_ASSIGN_OR_RETURN(auto output, CreateYuvBuffer(*output_buffer));
    success_convert = input.Resize(&output);
  }
  return success_convert
             ? absl::OkStatus()
             : absl::UnknownError("Halide YUV convert operation failed.");
}

}  // namespace

// Public methods.
//------------------------------------------------------------------------------

std::shared_ptr<FrameBuffer> CreateFromRgbaRawBuffer(
    uint8_t* input, FrameBuffer::Dimension dimension,
    FrameBuffer::Stride stride) {
  if (stride == kDefaultStride) {
    stride.row_stride_bytes = dimension.width * kRgbaChannels;
    stride.pixel_stride_bytes = kRgbaChannels;
  }
  FrameBuffer::Plane input_plane(/*buffer=*/input,
                                 /*stride=*/stride);
  std::vector<FrameBuffer::Plane> planes{input_plane};
  return std::make_shared<FrameBuffer>(planes, dimension,
                                       FrameBuffer::Format::kRGBA);
}

std::shared_ptr<FrameBuffer> CreateFromRgbRawBuffer(
    uint8_t* input, FrameBuffer::Dimension dimension,
    FrameBuffer::Stride stride) {
  if (stride == kDefaultStride) {
    stride.row_stride_bytes = dimension.width * kRgbChannels;
    stride.pixel_stride_bytes = kRgbChannels;
  }
  FrameBuffer::Plane input_plane(/*buffer=*/input,
                                 /*stride=*/stride);
  std::vector<FrameBuffer::Plane> planes{input_plane};
  return std::make_shared<FrameBuffer>(planes, dimension,
                                       FrameBuffer::Format::kRGB);
}

std::shared_ptr<FrameBuffer> CreateFromGrayRawBuffer(
    uint8_t* input, FrameBuffer::Dimension dimension,
    FrameBuffer::Stride stride) {
  if (stride == kDefaultStride) {
    stride.row_stride_bytes = dimension.width * kGrayChannel;
    stride.pixel_stride_bytes = kGrayChannel;
  }
  FrameBuffer::Plane input_plane(/*buffer=*/input,
                                 /*stride=*/stride);
  std::vector<FrameBuffer::Plane> planes{input_plane};
  return std::make_shared<FrameBuffer>(planes, dimension,
                                       FrameBuffer::Format::kGRAY);
}

absl::StatusOr<std::shared_ptr<FrameBuffer>> CreateFromYuvRawBuffer(
    uint8_t* y_plane, uint8_t* u_plane, uint8_t* v_plane,
    FrameBuffer::Format format, FrameBuffer::Dimension dimension,
    int row_stride_y, int row_stride_uv, int pixel_stride_uv) {
  const int pixel_stride_y = 1;
  std::vector<FrameBuffer::Plane> planes;
  if (format == FrameBuffer::Format::kNV21 ||
      format == FrameBuffer::Format::kYV12) {
    planes = {{y_plane, /*stride=*/{row_stride_y, pixel_stride_y}},
              {v_plane, /*stride=*/{row_stride_uv, pixel_stride_uv}},
              {u_plane, /*stride=*/{row_stride_uv, pixel_stride_uv}}};
  } else if (format == FrameBuffer::Format::kNV12 ||
             format == FrameBuffer::Format::kYV21) {
    planes = {{y_plane, /*stride=*/{row_stride_y, pixel_stride_y}},
              {u_plane, /*stride=*/{row_stride_uv, pixel_stride_uv}},
              {v_plane, /*stride=*/{row_stride_uv, pixel_stride_uv}}};
  } else {
    return absl::InvalidArgumentError(
        absl::StrFormat("Input format is not YUV-like: %i.", format));
  }
  return std::make_shared<FrameBuffer>(planes, dimension, format);
}

absl::StatusOr<std::shared_ptr<FrameBuffer>> CreateFromRawBuffer(
    uint8_t* buffer, FrameBuffer::Dimension dimension,
    const FrameBuffer::Format target_format) {
  switch (target_format) {
    case FrameBuffer::Format::kNV12:
    case FrameBuffer::Format::kNV21: {
      FrameBuffer::Plane plane(/*buffer=*/buffer,
                               /*stride=*/{dimension.width, kGrayChannel});
      std::vector<FrameBuffer::Plane> planes{plane};
      return std::make_shared<FrameBuffer>(planes, dimension, target_format);
    }
    case FrameBuffer::Format::kYV12: {
      MP_ASSIGN_OR_RETURN(const FrameBuffer::Dimension uv_dimension,
                          GetUvPlaneDimension(dimension, target_format));
      return CreateFromYuvRawBuffer(
          /*y_plane=*/buffer,
          /*u_plane=*/buffer + dimension.Size() + uv_dimension.Size(),
          /*v_plane=*/buffer + dimension.Size(), target_format, dimension,
          /*row_stride_y=*/dimension.width, uv_dimension.width,
          /*pixel_stride_uv=*/1);
    }
    case FrameBuffer::Format::kYV21: {
      MP_ASSIGN_OR_RETURN(const FrameBuffer::Dimension uv_dimension,
                          GetUvPlaneDimension(dimension, target_format));
      return CreateFromYuvRawBuffer(
          /*y_plane=*/buffer, /*u_plane=*/buffer + dimension.Size(),
          /*v_plane=*/buffer + dimension.Size() + uv_dimension.Size(),
          target_format, dimension, /*row_stride_y=*/dimension.width,
          uv_dimension.width,
          /*pixel_stride_uv=*/1);
    }
    case FrameBuffer::Format::kRGBA:
      return CreateFromRgbaRawBuffer(buffer, dimension);
    case FrameBuffer::Format::kRGB:
      return CreateFromRgbRawBuffer(buffer, dimension);
    case FrameBuffer::Format::kGRAY:
      return CreateFromGrayRawBuffer(buffer, dimension);
    default:
      return absl::InternalError(
          absl::StrFormat("Unsupported buffer format: %i.", target_format));
  }
}

absl::Status Crop(const FrameBuffer& buffer, int x0, int y0, int x1, int y1,
                  FrameBuffer* output_buffer) {
  MP_RETURN_IF_ERROR(
      ValidateCropBufferInputs(buffer, *output_buffer, x0, y0, x1, y1));
  MP_RETURN_IF_ERROR(ValidateBufferFormats(buffer, *output_buffer));

  switch (buffer.format()) {
    case FrameBuffer::Format::kGRAY:
      return CropGrayscale(buffer, x0, y0, x1, y1, output_buffer);
    case FrameBuffer::Format::kRGBA:
    case FrameBuffer::Format::kRGB:
      return CropRgb(buffer, x0, y0, x1, y1, output_buffer);
    case FrameBuffer::Format::kNV12:
    case FrameBuffer::Format::kNV21:
    case FrameBuffer::Format::kYV12:
    case FrameBuffer::Format::kYV21:
      return CropYuv(buffer, x0, y0, x1, y1, output_buffer);
    default:
      return absl::InternalError(
          absl::StrFormat("Format %i is not supported.", buffer.format()));
  }
}

absl::Status Resize(const FrameBuffer& buffer, FrameBuffer* output_buffer) {
  MP_RETURN_IF_ERROR(ValidateResizeBufferInputs(buffer, *output_buffer));

  switch (buffer.format()) {
    case FrameBuffer::Format::kGRAY:
      return ResizeGrayscale(buffer, output_buffer);
    case FrameBuffer::Format::kRGBA:
    case FrameBuffer::Format::kRGB:
      return ResizeRgb(buffer, output_buffer);
    case FrameBuffer::Format::kNV12:
    case FrameBuffer::Format::kNV21:
    case FrameBuffer::Format::kYV12:
    case FrameBuffer::Format::kYV21:
      return ResizeYuv(buffer, output_buffer);
    default:
      return absl::InternalError(
          absl::StrFormat("Format %i is not supported.", buffer.format()));
  }
}

absl::Status Rotate(const FrameBuffer& buffer, int angle_deg,
                    FrameBuffer* output_buffer) {
  MP_RETURN_IF_ERROR(
      ValidateRotateBufferInputs(buffer, *output_buffer, angle_deg));
  MP_RETURN_IF_ERROR(ValidateBufferFormats(buffer, *output_buffer));

  switch (buffer.format()) {
    case FrameBuffer::Format::kGRAY:
      return RotateGrayscale(buffer, angle_deg, output_buffer);
    case FrameBuffer::Format::kRGBA:
    case FrameBuffer::Format::kRGB:
      return RotateRgb(buffer, angle_deg, output_buffer);
    case FrameBuffer::Format::kNV12:
    case FrameBuffer::Format::kNV21:
    case FrameBuffer::Format::kYV12:
    case FrameBuffer::Format::kYV21:
      return RotateYuv(buffer, angle_deg, output_buffer);
    default:
      return absl::InternalError(
          absl::StrFormat("Format %i is not supported.", buffer.format()));
  }
}

absl::Status FlipHorizontally(const FrameBuffer& buffer,
                              FrameBuffer* output_buffer) {
  MP_RETURN_IF_ERROR(ValidateFlipBufferInputs(buffer, *output_buffer));
  MP_RETURN_IF_ERROR(ValidateBufferFormats(buffer, *output_buffer));

  switch (buffer.format()) {
    case FrameBuffer::Format::kGRAY:
      return FlipHorizontallyGrayscale(buffer, output_buffer);
    case FrameBuffer::Format::kRGBA:
    case FrameBuffer::Format::kRGB:
      return FlipHorizontallyRgb(buffer, output_buffer);
    case FrameBuffer::Format::kNV12:
    case FrameBuffer::Format::kNV21:
    case FrameBuffer::Format::kYV12:
    case FrameBuffer::Format::kYV21:
      return FlipHorizontallyYuv(buffer, output_buffer);
    default:
      return absl::InternalError(
          absl::StrFormat("Format %i is not supported.", buffer.format()));
  }
}

absl::Status FlipVertically(const FrameBuffer& buffer,
                            FrameBuffer* output_buffer) {
  MP_RETURN_IF_ERROR(ValidateFlipBufferInputs(buffer, *output_buffer));
  MP_RETURN_IF_ERROR(ValidateBufferFormats(buffer, *output_buffer));

  switch (buffer.format()) {
    case FrameBuffer::Format::kGRAY:
      return FlipVerticallyGrayscale(buffer, output_buffer);
    case FrameBuffer::Format::kRGBA:
    case FrameBuffer::Format::kRGB:
      return FlipVerticallyRgb(buffer, output_buffer);
    case FrameBuffer::Format::kNV12:
    case FrameBuffer::Format::kNV21:
    case FrameBuffer::Format::kYV12:
    case FrameBuffer::Format::kYV21:
      return FlipVerticallyYuv(buffer, output_buffer);
    default:
      return absl::InternalError(
          absl::StrFormat("Format %i is not supported.", buffer.format()));
  }
}

absl::Status Convert(const FrameBuffer& buffer, FrameBuffer* output_buffer) {
  MP_RETURN_IF_ERROR(
      ValidateConvertFormats(buffer.format(), output_buffer->format()));

  switch (buffer.format()) {
    case FrameBuffer::Format::kRGBA:
    case FrameBuffer::Format::kRGB:
      return ConvertRgb(buffer, output_buffer);
    case FrameBuffer::Format::kNV12:
    case FrameBuffer::Format::kNV21:
    case FrameBuffer::Format::kYV12:
    case FrameBuffer::Format::kYV21:
      return ConvertYuv(buffer, output_buffer);
    default:
      return absl::InternalError(
          absl::StrFormat("Format %i is not supported.", buffer.format()));
  }
}

absl::Status ToFloatTensor(const FrameBuffer& buffer, float scale, float offset,
                           Tensor& tensor) {
  MP_RETURN_IF_ERROR(ValidateFloatTensorInputs(buffer, tensor));
  switch (buffer.format()) {
    case FrameBuffer::Format::kRGB:
      return ToFloatTensorRgb(buffer, scale, offset, tensor);
    default:
      return absl::InvalidArgumentError(
          absl::StrFormat("Format %i is not supported.", buffer.format()));
  }
}

int GetFrameBufferByteSize(FrameBuffer::Dimension dimension,
                           FrameBuffer::Format format) {
  switch (format) {
    case FrameBuffer::Format::kNV12:
    case FrameBuffer::Format::kNV21:
    case FrameBuffer::Format::kYV12:
    case FrameBuffer::Format::kYV21:
      return /*y plane*/ dimension.Size() +
             /*uv plane*/ (dimension.width + 1) / 2 * (dimension.height + 1) /
                 2 * 2;
    case FrameBuffer::Format::kRGB:
      return dimension.Size() * kRgbPixelBytes;
    case FrameBuffer::Format::kRGBA:
      return dimension.Size() * kRgbaPixelBytes;
    case FrameBuffer::Format::kGRAY:
      return dimension.Size();
    default:
      return 0;
  }
}

absl::StatusOr<int> GetPixelStrides(FrameBuffer::Format format) {
  switch (format) {
    case FrameBuffer::Format::kGRAY:
      return kGrayPixelBytes;
    case FrameBuffer::Format::kRGB:
      return kRgbPixelBytes;
    case FrameBuffer::Format::kRGBA:
      return kRgbaPixelBytes;
    default:
      return absl::InvalidArgumentError(absl::StrFormat(
          "GetPixelStrides does not support format: %i.", format));
  }
}

absl::StatusOr<const uint8_t*> GetUvRawBuffer(const FrameBuffer& buffer) {
  if (buffer.format() != FrameBuffer::Format::kNV12 &&
      buffer.format() != FrameBuffer::Format::kNV21) {
    return absl::InvalidArgumentError(
        "Only support getting biplanar UV buffer from NV12/NV21 frame buffer.");
  }
  MP_ASSIGN_OR_RETURN(FrameBuffer::YuvData yuv_data,
                      FrameBuffer::GetYuvDataFromFrameBuffer(buffer));
  const uint8_t* uv_buffer = buffer.format() == FrameBuffer::Format::kNV12
                                 ? yuv_data.u_buffer
                                 : yuv_data.v_buffer;
  return uv_buffer;
}

absl::StatusOr<FrameBuffer::Dimension> GetUvPlaneDimension(
    FrameBuffer::Dimension dimension, FrameBuffer::Format format) {
  if (dimension.width <= 0 || dimension.height <= 0) {
    return absl::InvalidArgumentError(
        absl::StrFormat("Invalid input dimension: {%d, %d}.", dimension.width,
                        dimension.height));
  }
  switch (format) {
    case FrameBuffer::Format::kNV12:
    case FrameBuffer::Format::kNV21:
    case FrameBuffer::Format::kYV12:
    case FrameBuffer::Format::kYV21:
      return FrameBuffer::Dimension{(dimension.width + 1) / 2,
                                    (dimension.height + 1) / 2};
    default:
      return absl::InvalidArgumentError(
          absl::StrFormat("Input format is not YUV-like: %i.", format));
  }
}

FrameBuffer::Dimension GetCropDimension(int x0, int x1, int y0, int y1) {
  return {x1 - x0 + 1, y1 - y0 + 1};
}

}  // namespace frame_buffer
}  // namespace mediapipe
