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

#include "mediapipe/util/frame_buffer/rgb_buffer.h"

#include <utility>

#include "mediapipe/util/frame_buffer/buffer_common.h"
#include "mediapipe/util/frame_buffer/float_buffer.h"
#include "mediapipe/util/frame_buffer/gray_buffer.h"
#include "mediapipe/util/frame_buffer/halide/rgb_flip_halide.h"
#include "mediapipe/util/frame_buffer/halide/rgb_float_halide.h"
#include "mediapipe/util/frame_buffer/halide/rgb_gray_halide.h"
#include "mediapipe/util/frame_buffer/halide/rgb_resize_halide.h"
#include "mediapipe/util/frame_buffer/halide/rgb_rgb_halide.h"
#include "mediapipe/util/frame_buffer/halide/rgb_rotate_halide.h"
#include "mediapipe/util/frame_buffer/halide/rgb_yuv_halide.h"
#include "mediapipe/util/frame_buffer/yuv_buffer.h"

namespace mediapipe {
namespace frame_buffer {

RgbBuffer::RgbBuffer(uint8_t* data, int width, int height, bool alpha)
    : owned_buffer_(nullptr) {
  Initialize(data, width, height, alpha);
}

RgbBuffer::RgbBuffer(uint8_t* data, int width, int height, int row_stride,
                     bool alpha) {
  const int channels = alpha ? 4 : 3;
  const halide_dimension_t dimensions[3] = {{/*m=*/0, width, channels},
                                            {/*m=*/0, height, row_stride},
                                            {/*m=*/0, channels, 1}};
  buffer_ = Halide::Runtime::Buffer<uint8_t>(data, /*d=*/3, dimensions);
}

RgbBuffer::RgbBuffer(int width, int height, bool alpha)
    : owned_buffer_(new uint8_t[ByteSize(width, height, alpha)]) {
  Initialize(owned_buffer_.get(), width, height, alpha);
}

RgbBuffer::RgbBuffer(const RgbBuffer& other) : buffer_(other.buffer_) {
  // Never copy owned_buffer; ownership remains with the source of the copy.
}

RgbBuffer::RgbBuffer(RgbBuffer&& other) { *this = std::move(other); }

RgbBuffer& RgbBuffer::operator=(const RgbBuffer& other) {
  if (this != &other) {
    buffer_ = other.buffer_;
  }
  return *this;
}
RgbBuffer& RgbBuffer::operator=(RgbBuffer&& other) {
  if (this != &other) {
    owned_buffer_ = std::move(other.owned_buffer_);
    buffer_ = other.buffer_;
  }
  return *this;
}

RgbBuffer::~RgbBuffer() {}

bool RgbBuffer::Crop(int x0, int y0, int x1, int y1) {
  // Twiddle the buffer start and extents to crop images.
  return common::crop_buffer(x0, y0, x1, y1, buffer());
}

bool RgbBuffer::Resize(RgbBuffer* output) {
  if (output->channels() > channels()) {
    // Fail fast; the Halide implementation would otherwise output garbage
    // alpha values (i.e. duplicate the blue channel into alpha).
    return false;
  }
  const int result = rgb_resize_halide(
      buffer(), static_cast<float>(width()) / output->width(),
      static_cast<float>(height()) / output->height(), output->buffer());
  return result == 0;
}

bool RgbBuffer::Rotate(int angle, RgbBuffer* output) {
  const int result = rgb_rotate_halide(buffer(), angle, output->buffer());
  return result == 0;
}

bool RgbBuffer::FlipHorizontally(RgbBuffer* output) {
  const int result = rgb_flip_halide(buffer(),
                                     false,  // horizontal
                                     output->buffer());
  return result == 0;
}

bool RgbBuffer::FlipVertically(RgbBuffer* output) {
  const int result = rgb_flip_halide(buffer(),
                                     true,  // vertical
                                     output->buffer());
  return result == 0;
}

bool RgbBuffer::Convert(YuvBuffer* output) {
  const int result =
      rgb_yuv_halide(buffer(), output->y_buffer(), output->uv_buffer());
  return result == 0;
}

bool RgbBuffer::Convert(GrayBuffer* output) {
  const int result = rgb_gray_halide(buffer(), output->buffer());
  return result == 0;
}

bool RgbBuffer::Convert(RgbBuffer* output) {
  const int result = rgb_rgb_halide(buffer(), output->buffer());
  return result == 0;
}

bool RgbBuffer::ToFloat(float scale, float offset, FloatBuffer* output) {
  const int result =
      rgb_float_halide(buffer(), scale, offset, output->buffer());
  return result == 0;
}

void RgbBuffer::Initialize(uint8_t* data, int width, int height, bool alpha) {
  const int channels = alpha ? 4 : 3;
  buffer_ = Halide::Runtime::Buffer<uint8_t>::make_interleaved(
      data, width, height, channels);
}

}  // namespace frame_buffer
}  // namespace mediapipe
