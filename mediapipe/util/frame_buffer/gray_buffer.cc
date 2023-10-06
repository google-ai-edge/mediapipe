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

#include "mediapipe/util/frame_buffer/gray_buffer.h"

#include <utility>

#include "mediapipe/util/frame_buffer/buffer_common.h"
#include "mediapipe/util/frame_buffer/halide/gray_flip_halide.h"
#include "mediapipe/util/frame_buffer/halide/gray_resize_halide.h"
#include "mediapipe/util/frame_buffer/halide/gray_rotate_halide.h"
#include "mediapipe/util/frame_buffer/yuv_buffer.h"

namespace mediapipe {
namespace frame_buffer {

GrayBuffer::GrayBuffer(uint8_t* buffer, int width, int height)
    : owned_buffer_(nullptr) {
  Initialize(buffer, width, height);
}

GrayBuffer::GrayBuffer(int width, int height)
    : owned_buffer_(new uint8_t[ByteSize(width, height)]) {
  Initialize(owned_buffer_.get(), width, height);
}

GrayBuffer::GrayBuffer(const GrayBuffer& other) : buffer_(other.buffer_) {}

GrayBuffer::GrayBuffer(GrayBuffer&& other) { *this = std::move(other); }

GrayBuffer& GrayBuffer::operator=(const GrayBuffer& other) {
  if (this != &other) {
    buffer_ = other.buffer_;
  }
  return *this;
}

GrayBuffer& GrayBuffer::operator=(GrayBuffer&& other) {
  if (this != &other) {
    owned_buffer_ = std::move(other.owned_buffer_);
    buffer_ = other.buffer_;
  }
  return *this;
}

GrayBuffer::~GrayBuffer() {}

void GrayBuffer::Initialize(uint8_t* data, int width, int height) {
  buffer_ = Halide::Runtime::Buffer<uint8_t>(data, width, height);
}

bool GrayBuffer::Crop(int x0, int y0, int x1, int y1) {
  // Twiddle the buffer start and extents to crop images.
  return common::crop_buffer(x0, y0, x1, y1, buffer());
}

bool GrayBuffer::Resize(GrayBuffer* output) {
  const int result = gray_resize_halide(
      buffer(), static_cast<float>(width()) / output->width(),
      static_cast<float>(height()) / output->height(), output->buffer());
  return result == 0;
}

bool GrayBuffer::Rotate(int angle, GrayBuffer* output) {
  const int result = gray_rotate_halide(buffer(), angle, output->buffer());
  return result == 0;
}

bool GrayBuffer::FlipHorizontally(GrayBuffer* output) {
  const int result = gray_flip_halide(buffer(),
                                      false,  // horizontal
                                      output->buffer());
  return result == 0;
}

bool GrayBuffer::FlipVertically(GrayBuffer* output) {
  const int result = gray_flip_halide(buffer(),
                                      true,  // vertical
                                      output->buffer());
  return result == 0;
}

}  // namespace frame_buffer
}  // namespace mediapipe
