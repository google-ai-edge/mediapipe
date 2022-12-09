#if !defined(MEDIAPIPE_NO_JNI) && \
    (__ANDROID_API__ >= 26 ||     \
     defined(__ANDROID_UNAVAILABLE_SYMBOLS_ARE_WEAK__))

#include <cstdint>

#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "mediapipe/framework/formats/tensor_backend.h"
#include "mediapipe/framework/formats/tensor_cpu_buffer.h"
#include "mediapipe/framework/formats/tensor_hardware_buffer.h"
#include "mediapipe/framework/formats/tensor_v2.h"
#include "util/task/status_macros.h"

namespace mediapipe {
namespace {

class TensorCpuViewImpl : public TensorCpuView {
 public:
  TensorCpuViewImpl(int access_capabilities, Tensor::View::Access access,
                    Tensor::View::State state,
                    const TensorCpuViewDescriptor& descriptor, void* pointer,
                    AHardwareBuffer* ahwb_handle)
      : TensorCpuView(access_capabilities, access, state, descriptor, pointer),
        ahwb_handle_(ahwb_handle) {}
  ~TensorCpuViewImpl() {
    // If handle_ is null then this view is constructed in GetViews with no
    // access.
    if (ahwb_handle_) {
      if (__builtin_available(android 26, *)) {
        AHardwareBuffer_unlock(ahwb_handle_, nullptr);
      }
    }
  }

 private:
  AHardwareBuffer* ahwb_handle_;
};

class TensorHardwareBufferViewImpl : public TensorHardwareBufferView {
 public:
  TensorHardwareBufferViewImpl(
      int access_capability, Tensor::View::Access access,
      Tensor::View::State state,
      const TensorHardwareBufferViewDescriptor& descriptor,
      AHardwareBuffer* handle)
      : TensorHardwareBufferView(access_capability, access, state, descriptor,
                                 handle) {}
  ~TensorHardwareBufferViewImpl() = default;
};

class HardwareBufferCpuStorage : public TensorStorage {
 public:
  ~HardwareBufferCpuStorage() {
    if (!ahwb_handle_) return;
    if (__builtin_available(android 26, *)) {
      AHardwareBuffer_release(ahwb_handle_);
    }
  }

  static absl::Status CanProvide(
      int access_capability, const Tensor::Shape& shape, uint64_t view_type_id,
      const Tensor::ViewDescriptor& base_descriptor) {
    // TODO: use AHardwareBuffer_isSupported for API >= 29.
    static const bool is_ahwb_supported = [] {
      if (__builtin_available(android 26, *)) {
        AHardwareBuffer_Desc desc = {};
        // Aligned to the largest possible virtual memory page size.
        constexpr uint32_t kPageSize = 16384;
        desc.width = kPageSize;
        desc.height = 1;
        desc.layers = 1;
        desc.format = AHARDWAREBUFFER_FORMAT_BLOB;
        desc.usage = AHARDWAREBUFFER_USAGE_CPU_READ_OFTEN |
                     AHARDWAREBUFFER_USAGE_CPU_WRITE_OFTEN;
        AHardwareBuffer* handle;
        if (AHardwareBuffer_allocate(&desc, &handle) != 0) return false;
        AHardwareBuffer_release(handle);
        return true;
      }
      return false;
    }();
    if (!is_ahwb_supported) {
      return absl::UnavailableError(
          "AHardwareBuffer is not supported on the platform.");
    }

    if (view_type_id != TensorCpuView::kId &&
        view_type_id != TensorHardwareBufferView::kId) {
      return absl::InvalidArgumentError(
          "A view type is not supported by this storage.");
    }
    return absl::OkStatus();
  }

  std::vector<std::unique_ptr<Tensor::View>> GetViews(uint64_t latest_version) {
    std::vector<std::unique_ptr<Tensor::View>> result;
    auto update_state = latest_version == version_
                            ? Tensor::View::State::kUpToDate
                            : Tensor::View::State::kOutdated;
    if (ahwb_handle_) {
      result.push_back(
          std::unique_ptr<Tensor::View>(new TensorHardwareBufferViewImpl(
              kAccessCapability, Tensor::View::Access::kNoAccess, update_state,
              hw_descriptor_, ahwb_handle_)));

      result.push_back(std::unique_ptr<Tensor::View>(new TensorCpuViewImpl(
          kAccessCapability, Tensor::View::Access::kNoAccess, update_state,
          cpu_descriptor_, nullptr, nullptr)));
    }
    return result;
  }

  absl::StatusOr<std::unique_ptr<Tensor::View>> GetView(
      Tensor::View::Access access, const Tensor::Shape& shape,
      uint64_t latest_version, uint64_t view_type_id,
      const Tensor::ViewDescriptor& base_descriptor, int access_capability) {
    MP_RETURN_IF_ERROR(
        CanProvide(access_capability, shape, view_type_id, base_descriptor));
    const auto& buffer_descriptor =
        view_type_id == TensorHardwareBufferView::kId
            ? static_cast<const TensorHardwareBufferViewDescriptor&>(
                  base_descriptor)
                  .buffer
            : static_cast<const TensorCpuViewDescriptor&>(base_descriptor)
                  .buffer;
    if (!ahwb_handle_) {
      if (__builtin_available(android 26, *)) {
        AHardwareBuffer_Desc desc = {};
        desc.width = TensorBufferSize(buffer_descriptor, shape);
        desc.height = 1;
        desc.layers = 1;
        desc.format = AHARDWAREBUFFER_FORMAT_BLOB;
        // TODO: Use access capabilities to set hints.
        desc.usage = AHARDWAREBUFFER_USAGE_CPU_READ_OFTEN |
                     AHARDWAREBUFFER_USAGE_CPU_WRITE_OFTEN;
        auto error = AHardwareBuffer_allocate(&desc, &ahwb_handle_);
        if (error != 0) {
          return absl::UnknownError(
              absl::StrCat("Error allocating hardware buffer: ", error));
        }
        // Fill all possible views to provide it as proto views.
        hw_descriptor_.buffer = buffer_descriptor;
        cpu_descriptor_.buffer = buffer_descriptor;
      }
    }
    if (buffer_descriptor.format != hw_descriptor_.buffer.format ||
        buffer_descriptor.size_alignment >
            hw_descriptor_.buffer.size_alignment ||
        hw_descriptor_.buffer.size_alignment %
                buffer_descriptor.size_alignment >
            0) {
      return absl::AlreadyExistsError(
          "A view with different params is already allocated with this "
          "storage");
    }

    absl::StatusOr<std::unique_ptr<Tensor::View>> result;
    if (view_type_id == TensorHardwareBufferView::kId) {
      result = GetAhwbView(access, shape, base_descriptor);
    } else {
      result = GetCpuView(access, shape, base_descriptor);
    }
    if (result.ok()) version_ = latest_version;
    return result;
  }

 private:
  absl::StatusOr<std::unique_ptr<Tensor::View>> GetAhwbView(
      Tensor::View::Access access, const Tensor::Shape& shape,
      const Tensor::ViewDescriptor& base_descriptor) {
    return std::unique_ptr<Tensor::View>(new TensorHardwareBufferViewImpl(
        kAccessCapability, access, Tensor::View::State::kUpToDate,
        hw_descriptor_, ahwb_handle_));
  }

  absl::StatusOr<std::unique_ptr<Tensor::View>> GetCpuView(
      Tensor::View::Access access, const Tensor::Shape& shape,
      const Tensor::ViewDescriptor& base_descriptor) {
    void* pointer = nullptr;
    if (__builtin_available(android 26, *)) {
      int error =
          AHardwareBuffer_lock(ahwb_handle_,
                               access == Tensor::View::Access::kWriteOnly
                                   ? AHARDWAREBUFFER_USAGE_CPU_WRITE_OFTEN
                                   : AHARDWAREBUFFER_USAGE_CPU_READ_OFTEN,
                               -1, nullptr, &pointer);
      if (error != 0) {
        return absl::UnknownError(
            absl::StrCat("Error locking hardware buffer: ", error));
      }
    }
    return std::unique_ptr<Tensor::View>(
        new TensorCpuViewImpl(access == Tensor::View::Access::kWriteOnly
                                  ? Tensor::View::AccessCapability::kWrite
                                  : Tensor::View::AccessCapability::kRead,
                              access, Tensor::View::State::kUpToDate,
                              cpu_descriptor_, pointer, ahwb_handle_));
  }

  static constexpr int kAccessCapability =
      Tensor::View::AccessCapability::kRead |
      Tensor::View::AccessCapability::kWrite;
  TensorHardwareBufferViewDescriptor hw_descriptor_;
  AHardwareBuffer* ahwb_handle_ = nullptr;

  TensorCpuViewDescriptor cpu_descriptor_;
  uint64_t version_ = 0;
};
TENSOR_REGISTER_STORAGE(HardwareBufferCpuStorage);

}  // namespace
}  // namespace mediapipe

#endif  // !defined(MEDIAPIPE_NO_JNI) && (__ANDROID_API__ >= 26 ||
        // defined(__ANDROID_UNAVAILABLE_SYMBOLS_ARE_WEAK__))
