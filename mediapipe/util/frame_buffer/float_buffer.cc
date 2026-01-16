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

#include "mediapipe/util/frame_buffer/float_buffer.h"

#include <memory>

namespace mediapipe {
namespace frame_buffer {

FloatBuffer::FloatBuffer(float* data, int width, int height, int channels)
    : owned_buffer_(nullptr) {
  Initialize(data, width, height, channels);
}

FloatBuffer::FloatBuffer(int width, int height, int channels) {
  owned_buffer_ = std::make_unique<float[]>(FloatSize(width, height, channels));
  Initialize(owned_buffer_.get(), width, height, channels);
}

FloatBuffer::FloatBuffer(const FloatBuffer& other) : buffer_(other.buffer_) {}

FloatBuffer::FloatBuffer(FloatBuffer&& other) { *this = std::move(other); }

FloatBuffer& FloatBuffer::operator=(const FloatBuffer& other) {
  if (this != &other) {
    buffer_ = other.buffer_;
  }
  return *this;
}
FloatBuffer& FloatBuffer::operator=(FloatBuffer&& other) {
  if (this != &other) {
    buffer_ = other.buffer_;
  }
  return *this;
}

FloatBuffer::~FloatBuffer() {}

void FloatBuffer::Initialize(float* data, int width, int height, int channels) {
  buffer_ = Halide::Runtime::Buffer<float>::make_interleaved(data, width,
                                                             height, channels);
}

}  // namespace frame_buffer
}  // namespace mediapipe
