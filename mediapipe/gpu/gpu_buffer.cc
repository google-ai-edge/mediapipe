#include "mediapipe/gpu/gpu_buffer.h"

#include <memory>
#include <utility>

#include "absl/functional/bind_front.h"
#include "absl/log/absl_check.h"
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
  return holder_ ? absl::StrCat("GpuBuffer[", width(), "x", height(), " ",
                                format(), " as ", holder_->DebugString(), "]")
                 : "GpuBuffer[invalid]";
}

std::string GpuBuffer::StorageHolder::DebugString() const {
  absl::MutexLock lock(&mutex_);
  return absl::StrJoin(storages_, ", ", StorageTypeFormatter());
}

internal::GpuBufferStorage* GpuBuffer::StorageHolder::GetStorageForView(
    TypeId view_provider_type, bool for_writing) const {
  std::shared_ptr<internal::GpuBufferStorage> chosen_storage;
  std::function<std::shared_ptr<internal::GpuBufferStorage>()> conversion;

  {
    absl::MutexLock lock(&mutex_);
    // First see if any current storage supports the view.
    for (const auto& s : storages_) {
      if (s->can_down_cast_to(view_provider_type)) {
        chosen_storage = s;
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
          conversion = absl::bind_front(converter, s);
          break;
        }
      }
    }
  }

  // Avoid invoking a converter or factory while holding the mutex.
  // Two reasons:
  // 1. Readers that don't need a conversion will not be blocked.
  // 2. We use mutexes to make sure GL contexts are not used simultaneously on
  //    different threads, and we also rely on Mutex's deadlock detection
  //    heuristic, which enforces a consistent mutex acquisition order.
  //    This function is likely to be called within a GL context, and the
  //    conversion function may in turn use a GL context, and this may cause a
  //    false positive in the deadlock detector.
  //    TODO: we could use Mutex::ForgetDeadlockInfo instead.
  if (conversion) {
    auto new_storage = conversion();
    absl::MutexLock lock(&mutex_);
    // Another reader might have already completed and inserted the same
    // conversion. TODO: prevent this?
    for (const auto& s : storages_) {
      if (s->can_down_cast_to(view_provider_type)) {
        chosen_storage = s;
        break;
      }
    }
    if (!chosen_storage) {
      storages_.push_back(std::move(new_storage));
      chosen_storage = storages_.back();
    }
  }

  if (for_writing) {
    // This will temporarily hold storages to be released, and do so while the
    // lock is not held (see above).
    decltype(storages_) old_storages;
    using std::swap;
    if (chosen_storage) {
      // Discard all other storages.
      absl::MutexLock lock(&mutex_);
      swap(old_storages, storages_);
      storages_ = {chosen_storage};
    } else {
      // Allocate a new storage supporting the requested view.
      if (auto factory =
              internal::GpuBufferStorageRegistry::Get()
                  .StorageFactoryForViewProvider(view_provider_type)) {
        if (auto new_storage = factory(width_, height_, format_)) {
          absl::MutexLock lock(&mutex_);
          swap(old_storages, storages_);
          storages_ = {std::move(new_storage)};
          chosen_storage = storages_.back();
        }
      }
    }
  }

  // It is ok to return a non-owning storage pointer here because this object
  // ensures the storage's lifetime. Overwriting a GpuBuffer while readers are
  // active would violate this, but it's not allowed in MediaPipe.
  return chosen_storage ? chosen_storage.get() : nullptr;
}

internal::GpuBufferStorage& GpuBuffer::GetStorageForViewOrDie(
    TypeId view_provider_type, bool for_writing) const {
  auto* chosen_storage =
      GpuBuffer::GetStorageForView(view_provider_type, for_writing);
  ABSL_CHECK(chosen_storage)
      << "no view provider found for requested view "
      << view_provider_type.name() << "; storages available: "
      << (holder_ ? holder_->DebugString() : "invalid");
  ABSL_DCHECK(chosen_storage->can_down_cast_to(view_provider_type));
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
