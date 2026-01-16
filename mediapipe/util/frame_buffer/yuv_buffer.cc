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

#include "mediapipe/util/frame_buffer/yuv_buffer.h"

#include <utility>

#include "mediapipe/util/frame_buffer/buffer_common.h"
#include "mediapipe/util/frame_buffer/halide/yuv_flip_halide.h"
#include "mediapipe/util/frame_buffer/halide/yuv_resize_halide.h"
#include "mediapipe/util/frame_buffer/halide/yuv_rgb_halide.h"
#include "mediapipe/util/frame_buffer/halide/yuv_rotate_halide.h"
#include "mediapipe/util/frame_buffer/rgb_buffer.h"

namespace mediapipe {
namespace frame_buffer {

YuvBuffer::YuvBuffer(uint8_t* y_plane, uint8_t* u_plane, uint8_t* v_plane,
                     int width, int height, int row_stride_y, int row_stride_uv,
                     int pixel_stride_uv) {
  // Initialize the buffer shapes: {min, extent, stride} per dimension.
  // TODO: Ensure that width is less than or equal to row stride.
  const halide_dimension_t y_dimensions[2] = {
      {0, width, 1},
      {0, height, row_stride_y},
  };
  y_buffer_ = Halide::Runtime::Buffer<uint8_t>(y_plane, 2, y_dimensions);

  // Note that the Halide implementation expects the planes to be in VU
  // order, so we point at the V plane first.
  const halide_dimension_t uv_dimensions[3] = {
      {0, (width + 1) / 2, pixel_stride_uv},
      {0, (height + 1) / 2, row_stride_uv},
      {0, 2, static_cast<int32_t>(u_plane - v_plane)},
  };
  uv_buffer_ = Halide::Runtime::Buffer<uint8_t>(v_plane, 3, uv_dimensions);
}

YuvBuffer::YuvBuffer(uint8_t* data, int width, int height, Format format)
    : owned_buffer_(nullptr) {
  Initialize(data, width, height, format);
}

YuvBuffer::YuvBuffer(int width, int height, Format format)
    : owned_buffer_(new uint8_t[ByteSize(width, height)]) {
  Initialize(owned_buffer_.get(), width, height, format);
}

YuvBuffer::YuvBuffer(const YuvBuffer& other)
    : y_buffer_(other.y_buffer_), uv_buffer_(other.uv_buffer_) {
  // Never copy owned_buffer; ownership remains with the source of the copy.
}

YuvBuffer::YuvBuffer(YuvBuffer&& other) { *this = std::move(other); }

YuvBuffer& YuvBuffer::operator=(const YuvBuffer& other) {
  if (this != &other) {
    y_buffer_ = other.y_buffer_;
    uv_buffer_ = other.uv_buffer_;
  }
  return *this;
}
YuvBuffer& YuvBuffer::operator=(YuvBuffer&& other) {
  if (this != &other) {
    owned_buffer_ = std::move(other.owned_buffer_);
    y_buffer_ = other.y_buffer_;
    uv_buffer_ = other.uv_buffer_;
  }
  return *this;
}

YuvBuffer::~YuvBuffer() {}

void YuvBuffer::Initialize(uint8_t* data, int width, int height,
                           Format format) {
  y_buffer_ = Halide::Runtime::Buffer<uint8_t>(data, width, height);

  uint8_t* uv_data = data + (width * height);
  switch (format) {
    case NV21:
      // Interleaved UV (actually VU order).
      uv_buffer_ = Halide::Runtime::Buffer<uint8_t>::make_interleaved(
          uv_data, (width + 1) / 2, (height + 1) / 2, 2);
      break;
    case YV12:
      // Planar UV (actually VU order).
      uv_buffer_ = Halide::Runtime::Buffer<uint8_t>(uv_data, (width + 1) / 2,
                                                    (height + 1) / 2, 2);
      // NOTE: Halide operations have not been tested extensively in this
      // configuration.
      break;
  }
}

bool YuvBuffer::Crop(int x0, int y0, int x1, int y1) {
  if (x0 & 1 || y0 & 1) {
    // YUV images must be left-and top-aligned to even X/Y coordinates.
    return false;
  }

  // Twiddle the buffer start and extents for each plane to crop images.
  return (common::crop_buffer(x0, y0, x1, y1, y_buffer()) &&
          common::crop_buffer(x0 / 2, y0 / 2, x1 / 2, y1 / 2, uv_buffer()));
}

bool YuvBuffer::Resize(YuvBuffer* output) {
  const int result = yuv_resize_halide(
      y_buffer(), uv_buffer(), static_cast<float>(width()) / output->width(),
      static_cast<float>(height()) / output->height(), output->y_buffer(),
      output->uv_buffer());
  return result == 0;
}

bool YuvBuffer::Rotate(int angle, YuvBuffer* output) {
  const int result = yuv_rotate_halide(y_buffer(), uv_buffer(), angle,
                                       output->y_buffer(), output->uv_buffer());
  return result == 0;
}

bool YuvBuffer::FlipHorizontally(YuvBuffer* output) {
  const int result = yuv_flip_halide(y_buffer(), uv_buffer(),
                                     false,  // horizontal
                                     output->y_buffer(), output->uv_buffer());
  return result == 0;
}

bool YuvBuffer::FlipVertically(YuvBuffer* output) {
  const int result = yuv_flip_halide(y_buffer(), uv_buffer(),
                                     true,  // vertical
                                     output->y_buffer(), output->uv_buffer());
  return result == 0;
}

bool YuvBuffer::Convert(bool halve, RgbBuffer* output) {
  const int result =
      yuv_rgb_halide(y_buffer(), uv_buffer(), halve, output->buffer());
  return result == 0;
}

}  // namespace frame_buffer
}  // namespace mediapipe
