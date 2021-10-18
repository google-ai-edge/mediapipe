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
#include "mediapipe/gpu/gl_texture_view.h"
#include "mediapipe/gpu/gpu_buffer_format.h"
#include "mediapipe/gpu/gpu_buffer_storage.h"

#if defined(__APPLE__)
#include <CoreVideo/CoreVideo.h>

#include "mediapipe/objc/CFHolder.h"
#endif  // defined(__APPLE__)

#if MEDIAPIPE_GPU_BUFFER_USE_CV_PIXEL_BUFFER
#include "mediapipe/gpu/gpu_buffer_storage_cv_pixel_buffer.h"
#endif  // MEDIAPIPE_GPU_BUFFER_USE_CV_PIXEL_BUFFER

#if !MEDIAPIPE_GPU_BUFFER_USE_CV_PIXEL_BUFFER
#include "mediapipe/gpu/gl_texture_buffer.h"
#endif  // MEDIAPIPE_GPU_BUFFER_USE_CV_PIXEL_BUFFER

namespace mediapipe {

class GlContext;

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

  int width() const { return current_storage().width(); }
  int height() const { return current_storage().height(); }
  GpuBufferFormat format() const { return current_storage().format(); }

  // Converts to true iff valid.
  explicit operator bool() const { return operator!=(nullptr); }

  bool operator==(const GpuBuffer& other) const;
  bool operator!=(const GpuBuffer& other) const { return !operator==(other); }

  // Allow comparison with nullptr.
  bool operator==(std::nullptr_t other) const;
  bool operator!=(std::nullptr_t other) const { return !operator==(other); }

  // Allow assignment from nullptr.
  GpuBuffer& operator=(std::nullptr_t other);

  GlTextureView GetGlTextureReadView(int plane) const {
    return current_storage().GetGlTextureReadView(
        std::make_shared<GpuBuffer>(*this), plane);
  }

  GlTextureView GetGlTextureWriteView(int plane) {
    return current_storage().GetGlTextureWriteView(
        std::make_shared<GpuBuffer>(*this), plane);
  }

  // Make a GpuBuffer copying the data from an ImageFrame.
  static GpuBuffer CopyingImageFrame(const ImageFrame& image_frame);

  // Make an ImageFrame, possibly sharing the same data. The data is shared if
  // the GpuBuffer's storage supports memory sharing; otherwise, it is copied.
  // In order to work correctly across platforms, callers should always treat
  // the returned ImageFrame as if it shares memory with the GpuBuffer, i.e.
  // treat it as immutable if the GpuBuffer must not be modified.
  std::unique_ptr<ImageFrame> AsImageFrame() const {
    return current_storage().AsImageFrame();
  }

 private:
  class PlaceholderGpuBufferStorage
      : public mediapipe::internal::GpuBufferStorage {
   public:
    int width() const override { return 0; }
    int height() const override { return 0; }
    virtual GpuBufferFormat format() const override {
      return GpuBufferFormat::kUnknown;
    }
    GlTextureView GetGlTextureReadView(std::shared_ptr<GpuBuffer> gpu_buffer,
                                       int plane) const override {
      return {};
    }
    GlTextureView GetGlTextureWriteView(std::shared_ptr<GpuBuffer> gpu_buffer,
                                        int plane) override {
      return {};
    }
    void ViewDoneWriting(const GlTextureView& view) override{};
    std::unique_ptr<ImageFrame> AsImageFrame() const override {
      return nullptr;
    }
  };

  mediapipe::internal::GpuBufferStorage& no_storage() const {
    static PlaceholderGpuBufferStorage placeholder;
    return placeholder;
  }

  const mediapipe::internal::GpuBufferStorage& current_storage() const {
#if MEDIAPIPE_GPU_BUFFER_USE_CV_PIXEL_BUFFER
    if (pixel_buffer_ != nullptr) return pixel_buffer_;
#else
    if (texture_buffer_) return *texture_buffer_;
#endif  // MEDIAPIPE_GPU_BUFFER_USE_CV_PIXEL_BUFFER
    return no_storage();
  }

  mediapipe::internal::GpuBufferStorage& current_storage() {
#if MEDIAPIPE_GPU_BUFFER_USE_CV_PIXEL_BUFFER
    if (pixel_buffer_ != nullptr) return pixel_buffer_;
#else
    if (texture_buffer_) return *texture_buffer_;
#endif  // MEDIAPIPE_GPU_BUFFER_USE_CV_PIXEL_BUFFER
    return no_storage();
  }

#if MEDIAPIPE_GPU_BUFFER_USE_CV_PIXEL_BUFFER
  GpuBufferStorageCvPixelBuffer pixel_buffer_;
#else
  GlTextureBufferSharedPtr texture_buffer_;
#endif  // MEDIAPIPE_GPU_BUFFER_USE_CV_PIXEL_BUFFER
};

inline bool GpuBuffer::operator==(std::nullptr_t other) const {
  return &current_storage() == &no_storage();
}

inline bool GpuBuffer::operator==(const GpuBuffer& other) const {
#if MEDIAPIPE_GPU_BUFFER_USE_CV_PIXEL_BUFFER
  return pixel_buffer_ == other.pixel_buffer_;
#else
  return texture_buffer_ == other.texture_buffer_;
#endif  // MEDIAPIPE_GPU_BUFFER_USE_CV_PIXEL_BUFFER
}

inline GpuBuffer& GpuBuffer::operator=(std::nullptr_t other) {
#if MEDIAPIPE_GPU_BUFFER_USE_CV_PIXEL_BUFFER
  pixel_buffer_.reset(other);
#else
  texture_buffer_ = other;
#endif  // MEDIAPIPE_GPU_BUFFER_USE_CV_PIXEL_BUFFER
  return *this;
}

}  // namespace mediapipe

#endif  // MEDIAPIPE_GPU_GPU_BUFFER_H_
