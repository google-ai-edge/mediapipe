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

#include "mediapipe/framework/formats/image_frame.h"
#include "mediapipe/gpu/gl_base.h"
#include "mediapipe/gpu/gpu_buffer_format.h"

#if defined(__APPLE__)
#include <CoreVideo/CoreVideo.h>

#include "mediapipe/objc/CFHolder.h"
#endif  // defined(__APPLE__)

#if !MEDIAPIPE_GPU_BUFFER_USE_CV_PIXEL_BUFFER
#include "mediapipe/gpu/gl_texture_buffer.h"
#endif  // MEDIAPIPE_GPU_BUFFER_USE_CV_PIXEL_BUFFER

namespace mediapipe {

class GlContext;
class GlTextureView;

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

  // TODO: split into read and write, remove const from write.
  GlTextureView GetGlTextureView(int plane, bool for_reading) const;

  // Make a GpuBuffer copying the data from an ImageFrame.
  static GpuBuffer CopyingImageFrame(const ImageFrame& image_frame);

  // Make an ImageFrame, possibly sharing the same data. The data is shared if
  // the GpuBuffer's storage supports memory sharing; otherwise, it is copied.
  // In order to work correctly across platforms, callers should always treat
  // the returned ImageFrame as if it shares memory with the GpuBuffer, i.e.
  // treat it as immutable if the GpuBuffer must not be modified.
  std::unique_ptr<ImageFrame> AsImageFrame() const;

 private:
#if MEDIAPIPE_GPU_BUFFER_USE_CV_PIXEL_BUFFER
  CFHolder<CVPixelBufferRef> pixel_buffer_;
#else
  GlTextureBufferSharedPtr texture_buffer_;
#endif  // MEDIAPIPE_GPU_BUFFER_USE_CV_PIXEL_BUFFER
};

class GlTextureView {
 public:
  GlTextureView() {}
  ~GlTextureView() { Release(); }
  // TODO: make this class move-only.

  GlContext* gl_context() const { return gl_context_; }
  int width() const { return width_; }
  int height() const { return height_; }
  GLenum target() const { return target_; }
  GLuint name() const { return name_; }
  const GpuBuffer& gpu_buffer() const { return gpu_buffer_; }
  int plane() const { return plane_; }

 private:
  friend class GpuBuffer;
  using DetachFn = std::function<void(GlTextureView&)>;
  GlTextureView(GlContext* context, GLenum target, GLuint name, int width,
                int height, GpuBuffer gpu_buffer, int plane, DetachFn detach)
      : gl_context_(context),
        target_(target),
        name_(name),
        width_(width),
        height_(height),
        gpu_buffer_(std::move(gpu_buffer)),
        plane_(plane),
        detach_(std::move(detach)) {}

  // TODO: remove this friend declaration.
  friend class GlTexture;
  void Release();
  // TODO: make this non-const.
  void DoneWriting() const;

  GlContext* gl_context_ = nullptr;
  GLenum target_ = GL_TEXTURE_2D;
  GLuint name_ = 0;
  // Note: when scale is not 1, we still give the nominal size of the image.
  int width_ = 0;
  int height_ = 0;
  GpuBuffer gpu_buffer_;
  int plane_ = 0;
  DetachFn detach_;
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
