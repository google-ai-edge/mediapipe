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

#include <memory>
#include <utility>

#include "mediapipe/framework/formats/image_frame.h"
#include "mediapipe/gpu/gl_base.h"
#include "mediapipe/gpu/gl_texture_view.h"
#include "mediapipe/gpu/gpu_buffer_format.h"
#include "mediapipe/gpu/gpu_buffer_storage.h"

// Note: these headers are needed for the legacy storage APIs. Do not add more
// storage-specific headers here. See WebGpuTextureBuffer/View for an example
// of adding a new storage and view.

#if defined(__APPLE__)
#include <CoreVideo/CoreVideo.h>

#include "mediapipe/objc/CFHolder.h"
#endif  // defined(__APPLE__)

#if MEDIAPIPE_GPU_BUFFER_USE_CV_PIXEL_BUFFER
#include "mediapipe/gpu/gpu_buffer_storage_cv_pixel_buffer.h"
#else
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
  explicit GpuBuffer(
      std::shared_ptr<mediapipe::internal::GpuBufferStorage> storage)
      : storage_(std::move(storage)) {}

  // Note: these constructors and accessors for specific storage types exist
  // for backwards compatibility reasons. Do not add new ones.
#if MEDIAPIPE_GPU_BUFFER_USE_CV_PIXEL_BUFFER
  explicit GpuBuffer(CFHolder<CVPixelBufferRef> pixel_buffer)
      : storage_(std::make_shared<GpuBufferStorageCvPixelBuffer>(
            std::move(pixel_buffer))) {}
  explicit GpuBuffer(CVPixelBufferRef pixel_buffer)
      : storage_(
            std::make_shared<GpuBufferStorageCvPixelBuffer>(pixel_buffer)) {}

  CVPixelBufferRef GetCVPixelBufferRef() const {
    auto p = storage_->down_cast<GpuBufferStorageCvPixelBuffer>();
    if (p) return **p;
    return nullptr;
  }
#else
  GlTextureBufferSharedPtr GetGlTextureBufferSharedPtr() const {
    return internal_storage<GlTextureBuffer>();
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

  // Gets a read view of the specified type. The arguments depend on the
  // specific view type; see the corresponding ViewProvider.
  template <class View, class... Args>
  auto GetReadView(Args... args) const {
    return current_storage()
        .down_cast<mediapipe::internal::ViewProvider<View>>()
        ->GetReadView(mediapipe::internal::types<View>{},
                      std::make_shared<GpuBuffer>(*this),
                      std::forward<Args>(args)...);
  }

  // Gets a write view of the specified type. The arguments depend on the
  // specific view type; see the corresponding ViewProvider.
  template <class View, class... Args>
  auto GetWriteView(Args... args) {
    return current_storage()
        .down_cast<mediapipe::internal::ViewProvider<View>>()
        ->GetWriteView(mediapipe::internal::types<View>{},
                       std::make_shared<GpuBuffer>(*this),
                       std::forward<Args>(args)...);
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

  // Attempts to access an underlying storage object of the specified type.
  // This method is meant for internal use: user code should access the contents
  // using views.
  template <class T>
  std::shared_ptr<T> internal_storage() const {
    if (storage_->down_cast<T>()) return std::static_pointer_cast<T>(storage_);
    return nullptr;
  }

 private:
  class PlaceholderGpuBufferStorage
      : public mediapipe::internal::GpuBufferStorageImpl<
            PlaceholderGpuBufferStorage> {
   public:
    int width() const override { return 0; }
    int height() const override { return 0; }
    virtual GpuBufferFormat format() const override {
      return GpuBufferFormat::kUnknown;
    }
    std::unique_ptr<ImageFrame> AsImageFrame() const override {
      return nullptr;
    }
  };

  std::shared_ptr<mediapipe::internal::GpuBufferStorage>& no_storage() const {
    static auto placeholder =
        std::static_pointer_cast<mediapipe::internal::GpuBufferStorage>(
            std::make_shared<PlaceholderGpuBufferStorage>());
    return placeholder;
  }

  const mediapipe::internal::GpuBufferStorage& current_storage() const {
    return *storage_;
  }

  mediapipe::internal::GpuBufferStorage& current_storage() { return *storage_; }

  std::shared_ptr<mediapipe::internal::GpuBufferStorage> storage_ =
      no_storage();
};

inline bool GpuBuffer::operator==(std::nullptr_t other) const {
  return storage_ == no_storage();
}

inline bool GpuBuffer::operator==(const GpuBuffer& other) const {
  return storage_ == other.storage_;
}

inline GpuBuffer& GpuBuffer::operator=(std::nullptr_t other) {
  storage_ = no_storage();
  return *this;
}

}  // namespace mediapipe

#endif  // MEDIAPIPE_GPU_GPU_BUFFER_H_
