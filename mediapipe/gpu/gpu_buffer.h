// Copyright 2019 The MediaPipe Authors.
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

#ifndef MEDIAPIPE_GPU_GPU_BUFFER_H_
#define MEDIAPIPE_GPU_GPU_BUFFER_H_

#include <utility>

#include "mediapipe/gpu/gl_base.h"
#include "mediapipe/gpu/gpu_buffer_format.h"

#if defined(__APPLE__)
#include <CoreVideo/CoreVideo.h>

#include "mediapipe/objc/CFHolder.h"
#if !TARGET_OS_OSX
#define MEDIAPIPE_GPU_BUFFER_USE_CV_PIXEL_BUFFER 1
#endif  // TARGET_OS_OSX
#endif  // defined(__APPLE__)

#if !MEDIAPIPE_GPU_BUFFER_USE_CV_PIXEL_BUFFER
#include "mediapipe/gpu/gl_texture_buffer.h"
#endif  // MEDIAPIPE_GPU_BUFFER_USE_CV_PIXEL_BUFFER

namespace mediapipe {

// This class wraps a platform-specific buffer of GPU data.
// An instance of GpuBuffer acts as an opaque reference to the underlying
// data object.
class GpuBuffer {
 public:
  // Default constructor creates invalid object.
  GpuBuffer() = default;

  // Copy and move constructors and assignment operators are supported.
  GpuBuffer(const GpuBuffer& other) = default;
  GpuBuffer(GpuBuffer&& other) = default;
  GpuBuffer& operator=(const GpuBuffer& other) = default;
  GpuBuffer& operator=(GpuBuffer&& other) = default;

  // Constructors from platform-specific representations, and accessors for the
  // underlying platform-specific representation. Use with caution, since they
  // are not portable. Applications and calculators should normally obtain
  // GpuBuffers in a portable way from the framework, e.g. using
  // GpuBufferMultiPool.
#if MEDIAPIPE_GPU_BUFFER_USE_CV_PIXEL_BUFFER
  explicit GpuBuffer(CFHolder<CVPixelBufferRef> pixel_buffer)
      : pixel_buffer_(std::move(pixel_buffer)) {}
  explicit GpuBuffer(CVPixelBufferRef pixel_buffer)
      : pixel_buffer_(pixel_buffer) {}

  CVPixelBufferRef GetCVPixelBufferRef() const { return *pixel_buffer_; }
#else
  explicit GpuBuffer(GlTextureBufferSharedPtr texture_buffer)
      : texture_buffer_(std::move(texture_buffer)) {}

  const GlTextureBufferSharedPtr& GetGlTextureBufferSharedPtr() const {
    return texture_buffer_;
  }
#endif  // MEDIAPIPE_GPU_BUFFER_USE_CV_PIXEL_BUFFER

  int width() const;
  int height() const;
  GpuBufferFormat format() const;

  // Converts to true iff valid.
  explicit operator bool() const { return operator!=(nullptr); }

  bool operator==(const GpuBuffer& other) const;
  bool operator!=(const GpuBuffer& other) const { return !operator==(other); }

  // Allow comparison with nullptr.
  bool operator==(std::nullptr_t other) const;
  bool operator!=(std::nullptr_t other) const { return !operator==(other); }

  // Allow assignment from nullptr.
  GpuBuffer& operator=(std::nullptr_t other);

 private:
#if MEDIAPIPE_GPU_BUFFER_USE_CV_PIXEL_BUFFER
  CFHolder<CVPixelBufferRef> pixel_buffer_;
#else
  GlTextureBufferSharedPtr texture_buffer_;
#endif  // MEDIAPIPE_GPU_BUFFER_USE_CV_PIXEL_BUFFER
};

#if MEDIAPIPE_GPU_BUFFER_USE_CV_PIXEL_BUFFER

inline int GpuBuffer::width() const {
  return static_cast<int>(CVPixelBufferGetWidth(*pixel_buffer_));
}

inline int GpuBuffer::height() const {
  return static_cast<int>(CVPixelBufferGetHeight(*pixel_buffer_));
}

inline GpuBufferFormat GpuBuffer::format() const {
  return GpuBufferFormatForCVPixelFormat(
      CVPixelBufferGetPixelFormatType(*pixel_buffer_));
}

inline bool GpuBuffer::operator==(std::nullptr_t other) const {
  return pixel_buffer_ == other;
}

inline bool GpuBuffer::operator==(const GpuBuffer& other) const {
  return pixel_buffer_ == other.pixel_buffer_;
}

inline GpuBuffer& GpuBuffer::operator=(std::nullptr_t other) {
  pixel_buffer_.reset(other);
  return *this;
}

#else

inline int GpuBuffer::width() const { return texture_buffer_->width(); }

inline int GpuBuffer::height() const { return texture_buffer_->height(); }

inline GpuBufferFormat GpuBuffer::format() const {
  return texture_buffer_->format();
}

inline bool GpuBuffer::operator==(std::nullptr_t other) const {
  return texture_buffer_ == other;
}

inline bool GpuBuffer::operator==(const GpuBuffer& other) const {
  return texture_buffer_ == other.texture_buffer_;
}

inline GpuBuffer& GpuBuffer::operator=(std::nullptr_t other) {
  texture_buffer_ = other;
  return *this;
}

#endif  // MEDIAPIPE_GPU_BUFFER_USE_CV_PIXEL_BUFFER

}  // namespace mediapipe

#endif  // MEDIAPIPE_GPU_GPU_BUFFER_H_
