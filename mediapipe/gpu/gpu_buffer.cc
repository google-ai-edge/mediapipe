#include "mediapipe/gpu/gpu_buffer.h"

#include <memory>

#include "mediapipe/framework/port/logging.h"

#if MEDIAPIPE_GPU_BUFFER_USE_CV_PIXEL_BUFFER
#include "mediapipe/objc/util.h"
#endif  // MEDIAPIPE_GPU_BUFFER_USE_CV_PIXEL_BUFFER

namespace mediapipe {

internal::GpuBufferStorage& GpuBuffer::GetStorageForView(
    TypeId view_provider_type, bool for_writing) const {
  const std::shared_ptr<internal::GpuBufferStorage>* chosen_storage = nullptr;

  // First see if any current storage supports the view.
  for (const auto& s : storages_) {
    if (s->can_down_cast_to(view_provider_type)) {
      chosen_storage = &s;
      break;
    }
  }

  // Then try to convert existing storages to one that does.
  // TODO: choose best conversion.
  if (!chosen_storage) {
    for (const auto& s : storages_) {
      auto converter = internal::GpuBufferStorageRegistry::Get()
                           .StorageConverterForViewProvider(view_provider_type,
                                                            s->storage_type());
      if (converter) {
        storages_.push_back(converter(s));
        chosen_storage = &storages_.back();
      }
    }
  }

  if (for_writing) {
    if (!chosen_storage) {
      // Allocate a new storage supporting the requested view.
      auto factory = internal::GpuBufferStorageRegistry::Get()
                         .StorageFactoryForViewProvider(view_provider_type);
      if (factory) {
        storages_ = {factory(width(), height(), format())};
        chosen_storage = &storages_.back();
      }
    } else {
      // Discard all other storages.
      storages_ = {*chosen_storage};
      chosen_storage = &storages_.back();
    }
  }

  CHECK(chosen_storage) << "no view provider found";
  DCHECK((*chosen_storage)->can_down_cast_to(view_provider_type));
  return **chosen_storage;
}

#if !MEDIAPIPE_DISABLE_GPU
#if MEDIAPIPE_GPU_BUFFER_USE_CV_PIXEL_BUFFER
CVPixelBufferRef GetCVPixelBufferRef(const GpuBuffer& buffer) {
  auto p = buffer.internal_storage<GpuBufferStorageCvPixelBuffer>();
  if (p) return **p;
  return nullptr;
}
#endif  // MEDIAPIPE_GPU_BUFFER_USE_CV_PIXEL_BUFFER
#endif  // !MEDIAPIPE_DISABLE_GPU

}  // namespace mediapipe
