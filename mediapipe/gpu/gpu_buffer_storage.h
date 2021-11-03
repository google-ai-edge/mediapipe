#ifndef MEDIAPIPE_GPU_GPU_BUFFER_STORAGE_H_
#define MEDIAPIPE_GPU_GPU_BUFFER_STORAGE_H_

#include "mediapipe/framework/formats/image_frame.h"
#include "mediapipe/gpu/gpu_buffer_format.h"

namespace mediapipe {
class GlTextureView;
class GpuBuffer;
}  // namespace mediapipe

namespace mediapipe {
namespace internal {

template <class... T>
struct types {};

template <class V>
class ViewProvider;

// Note: this specialization temporarily lives here for backwards compatibility
// reasons. New specializations should be put in the same file as their view.
template <>
class ViewProvider<GlTextureView> {
 public:
  virtual ~ViewProvider() = default;
  // Note that the view type is encoded in an argument to allow overloading,
  // so a storage class can implement GetRead/WriteView for multiple view types.
  // We cannot use a template function because it cannot be virtual; we want to
  // have a virtual function here to enforce that different storages supporting
  // the same view implement the same signature.
  // Note that we allow different views to have custom signatures, providing
  // additional view-specific arguments that may be needed.
  virtual GlTextureView GetReadView(types<GlTextureView>,
                                    std::shared_ptr<GpuBuffer> gpu_buffer,
                                    int plane) const = 0;
  virtual GlTextureView GetWriteView(types<GlTextureView>,
                                     std::shared_ptr<GpuBuffer> gpu_buffer,
                                     int plane) = 0;
};

class GpuBufferStorage {
 public:
  virtual ~GpuBufferStorage() = default;
  virtual int width() const = 0;
  virtual int height() const = 0;
  virtual GpuBufferFormat format() const = 0;
  virtual std::unique_ptr<ImageFrame> AsImageFrame() const = 0;
  // We can't use dynamic_cast since we want to support building without RTTI.
  // The public methods delegate to the type-erased private virtual method.
  template <class T>
  T* down_cast() {
    return static_cast<T*>(
        const_cast<void*>(down_cast(tool::GetTypeHash<T>())));
  }
  template <class T>
  const T* down_cast() const {
    return static_cast<const T*>(down_cast(tool::GetTypeHash<T>()));
  }

 private:
  virtual const void* down_cast(size_t type_hash) const = 0;
  virtual size_t storage_type_hash() const = 0;
};

template <class T, class... U>
class GpuBufferStorageImpl : public GpuBufferStorage, public U... {
 private:
  virtual const void* down_cast(size_t type_hash) const override {
    return down_cast_impl(type_hash, types<T, U...>{});
  }
  size_t storage_type_hash() const override { return tool::GetTypeHash<T>(); }

  const void* down_cast_impl(size_t type_hash, types<>) const {
    return nullptr;
  }
  template <class V, class... W>
  const void* down_cast_impl(size_t type_hash, types<V, W...>) const {
    if (type_hash == tool::GetTypeHash<V>()) return static_cast<const V*>(this);
    return down_cast_impl(type_hash, types<W...>{});
  }
};

}  // namespace internal
}  // namespace mediapipe

#endif  // MEDIAPIPE_GPU_GPU_BUFFER_STORAGE_H_
