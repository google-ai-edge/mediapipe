#include "mediapipe/gpu/gpu_buffer.h"

#include <memory>
#include <utility>

#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "mediapipe/framework/port/logging.h"

#if MEDIAPIPE_GPU_BUFFER_USE_CV_PIXEL_BUFFER
#include "mediapipe/objc/util.h"
#endif  // MEDIAPIPE_GPU_BUFFER_USE_CV_PIXEL_BUFFER

namespace mediapipe {

namespace {

struct StorageTypeFormatter {
  void operator()(std::string* out,
                  const std::shared_ptr<internal::GpuBufferStorage>& s) const {
    absl::StrAppend(out, s->storage_type().name());
  }
};

}  // namespace

std::string GpuBuffer::DebugString() const {
  return absl::StrCat("GpuBuffer[",
                      absl::StrJoin(storages_, ", ", StorageTypeFormatter()),
                      "]");
}

internal::GpuBufferStorage* GpuBuffer::GetStorageForView(
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
      if (auto converter = internal::GpuBufferStorageRegistry::Get()
                               .StorageConverterForViewProvider(
                                   view_provider_type, s->storage_type())) {
        if (auto new_storage = converter(s)) {
          storages_.push_back(new_storage);
          chosen_storage = &storages_.back();
          break;
        }
      }
    }
  }

  if (for_writing) {
    if (chosen_storage) {
      // Discard all other storages.
      storages_ = {*chosen_storage};
      chosen_storage = &storages_.back();
    } else {
      // Allocate a new storage supporting the requested view.
      if (auto factory =
              internal::GpuBufferStorageRegistry::Get()
                  .StorageFactoryForViewProvider(view_provider_type)) {
        if (auto new_storage = factory(width(), height(), format())) {
          storages_ = {std::move(new_storage)};
          chosen_storage = &storages_.back();
        }
      }
    }
  }
  return chosen_storage ? chosen_storage->get() : nullptr;
}

internal::GpuBufferStorage& GpuBuffer::GetStorageForViewOrDie(
    TypeId view_provider_type, bool for_writing) const {
  auto* chosen_storage =
      GpuBuffer::GetStorageForView(view_provider_type, for_writing);
  CHECK(chosen_storage) << "no view provider found for requested view "
                        << view_provider_type.name() << "; storages available: "
                        << absl::StrJoin(storages_, ", ",
                                         StorageTypeFormatter());
  DCHECK(chosen_storage->can_down_cast_to(view_provider_type));
  return *chosen_storage;
}

#if !MEDIAPIPE_DISABLE_GPU
#if MEDIAPIPE_GPU_BUFFER_USE_CV_PIXEL_BUFFER
CVPixelBufferRef GetCVPixelBufferRef(const GpuBuffer& buffer) {
  if (buffer.GetStorageForView(
          kTypeId<internal::ViewProvider<CVPixelBufferRef>>,
          /*for_writing=*/false) != nullptr) {
    return *buffer.GetReadView<CVPixelBufferRef>();
  }
  return nullptr;
}
#endif  // MEDIAPIPE_GPU_BUFFER_USE_CV_PIXEL_BUFFER
#endif  // !MEDIAPIPE_DISABLE_GPU

}  // namespace mediapipe
