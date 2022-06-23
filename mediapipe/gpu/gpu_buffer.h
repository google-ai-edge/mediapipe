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
#include "mediapipe/gpu/gpu_buffer_format.h"
#include "mediapipe/gpu/gpu_buffer_storage.h"

#if !MEDIAPIPE_DISABLE_GPU
#include "mediapipe/gpu/gl_texture_view.h"
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
#endif  // MEDIAPIPE_DISABLE_GPU

namespace mediapipe {

// This class wraps a platform-specific buffer of GPU data.
// An instance of GpuBuffer acts as an opaque reference to the underlying
// data object.
class GpuBuffer {
 public:
  using Format = GpuBufferFormat;

  // Default constructor creates invalid object.
  GpuBuffer() = default;

  // Creates an empty buffer of a given size and format. It will be allocated
  // when a view is requested.
  GpuBuffer(int width, int height, Format format)
      : GpuBuffer(std::make_shared<PlaceholderGpuBufferStorage>(width, height,
                                                                format)) {}

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
  explicit GpuBuffer(std::shared_ptr<internal::GpuBufferStorage> storage) {
    storages_.push_back(std::move(storage));
  }

#if !MEDIAPIPE_DISABLE_GPU && MEDIAPIPE_GPU_BUFFER_USE_CV_PIXEL_BUFFER
  // This is used to support backward-compatible construction of GpuBuffer from
  // some platform-specific types without having to make those types visible in
  // this header.
  template <class T, class = std::void_t<decltype(internal::AsGpuBufferStorage(
                         std::declval<T>()))>>
  explicit GpuBuffer(T&& storage_convertible)
      : GpuBuffer(internal::AsGpuBufferStorage(storage_convertible)) {}
#endif  // !MEDIAPIPE_DISABLE_GPU && MEDIAPIPE_GPU_BUFFER_USE_CV_PIXEL_BUFFER

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
  decltype(auto) GetReadView(Args... args) const {
    return GetViewProvider<View>(false)->GetReadView(
        internal::types<View>{}, std::make_shared<GpuBuffer>(*this),
        std::forward<Args>(args)...);
  }

  // Gets a write view of the specified type. The arguments depend on the
  // specific view type; see the corresponding ViewProvider.
  template <class View, class... Args>
  decltype(auto) GetWriteView(Args... args) {
    return GetViewProvider<View>(true)->GetWriteView(
        internal::types<View>{}, std::make_shared<GpuBuffer>(*this),
        std::forward<Args>(args)...);
  }

  // Attempts to access an underlying storage object of the specified type.
  // This method is meant for internal use: user code should access the contents
  // using views.
  template <class T>
  std::shared_ptr<T> internal_storage() const {
    for (const auto& s : storages_)
      if (s->down_cast<T>()) return std::static_pointer_cast<T>(s);
    return nullptr;
  }

 private:
  class PlaceholderGpuBufferStorage
      : public internal::GpuBufferStorageImpl<PlaceholderGpuBufferStorage> {
   public:
    PlaceholderGpuBufferStorage(int width, int height, Format format)
        : width_(width), height_(height), format_(format) {}
    int width() const override { return width_; }
    int height() const override { return height_; }
    GpuBufferFormat format() const override { return format_; }

   private:
    int width_ = 0;
    int height_ = 0;
    GpuBufferFormat format_ = GpuBufferFormat::kUnknown;
  };

  internal::GpuBufferStorage& GetStorageForView(TypeId view_provider_type,
                                                bool for_writing) const;

  template <class View>
  internal::ViewProvider<View>* GetViewProvider(bool for_writing) const {
    using VP = internal::ViewProvider<View>;
    return GetStorageForView(kTypeId<VP>, for_writing).template down_cast<VP>();
  }

  std::shared_ptr<internal::GpuBufferStorage>& no_storage() const {
    static auto placeholder =
        std::static_pointer_cast<internal::GpuBufferStorage>(
            std::make_shared<PlaceholderGpuBufferStorage>(
                0, 0, GpuBufferFormat::kUnknown));
    return placeholder;
  }

  const internal::GpuBufferStorage& current_storage() const {
    return storages_.empty() ? *no_storage() : *storages_[0];
  }

  internal::GpuBufferStorage& current_storage() {
    return storages_.empty() ? *no_storage() : *storages_[0];
  }

  // This is mutable because view methods that do not change the contents may
  // still need to allocate new storages.
  mutable std::vector<std::shared_ptr<internal::GpuBufferStorage>> storages_;
};

inline bool GpuBuffer::operator==(std::nullptr_t other) const {
  return storages_.empty();
}

inline bool GpuBuffer::operator==(const GpuBuffer& other) const {
  return storages_ == other.storages_;
}

inline GpuBuffer& GpuBuffer::operator=(std::nullptr_t other) {
  storages_.clear();
  return *this;
}

// Note: these constructors and accessors for specific storage types exist
// for backwards compatibility reasons. Do not add new ones.
#if MEDIAPIPE_GPU_BUFFER_USE_CV_PIXEL_BUFFER
CVPixelBufferRef GetCVPixelBufferRef(const GpuBuffer& buffer);
#endif  // MEDIAPIPE_GPU_BUFFER_USE_CV_PIXEL_BUFFER

}  // namespace mediapipe

#endif  // MEDIAPIPE_GPU_GPU_BUFFER_H_
