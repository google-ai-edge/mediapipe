#include "absl/status/status.h"
#include "mediapipe/framework/formats/tensor.h"

#ifdef MEDIAPIPE_TENSOR_USE_AHWB
#include <EGL/egl.h>
#include <EGL/eglext.h>

#include <cstdint>
#include <list>
#include <utility>
#include <vector>

#include "absl/base/const_init.h"
#include "absl/base/log_severity.h"
#include "absl/functional/any_invocable.h"
#include "absl/log/absl_check.h"
#include "absl/log/absl_log.h"
#include "absl/synchronization/mutex.h"
#include "mediapipe/framework/deps/no_destructor.h"
#include "mediapipe/framework/formats/ahwb_gpu_releaser.h"
#include "mediapipe/framework/formats/hardware_buffer.h"
#include "mediapipe/gpu/gl_base.h"
#endif  // MEDIAPIPE_TENSOR_USE_AHWB

namespace mediapipe {
#ifdef MEDIAPIPE_TENSOR_USE_AHWB

namespace {
PFNGLBUFFERSTORAGEEXTERNALEXTPROC glBufferStorageExternalEXT;
PFNEGLGETNATIVECLIENTBUFFERANDROIDPROC eglGetNativeClientBufferANDROID;
PFNEGLDUPNATIVEFENCEFDANDROIDPROC eglDupNativeFenceFDANDROID;
PFNEGLCREATESYNCKHRPROC eglCreateSyncKHR;
PFNEGLWAITSYNCKHRPROC eglWaitSyncKHR;
PFNEGLCLIENTWAITSYNCKHRPROC eglClientWaitSyncKHR;
PFNEGLDESTROYSYNCKHRPROC eglDestroySyncKHR;

bool IsGlSupported() {
  static const bool extensions_allowed = [] {
    eglGetNativeClientBufferANDROID =
        reinterpret_cast<PFNEGLGETNATIVECLIENTBUFFERANDROIDPROC>(
            eglGetProcAddress("eglGetNativeClientBufferANDROID"));
    glBufferStorageExternalEXT =
        reinterpret_cast<PFNGLBUFFERSTORAGEEXTERNALEXTPROC>(
            eglGetProcAddress("glBufferStorageExternalEXT"));
    eglDupNativeFenceFDANDROID =
        reinterpret_cast<PFNEGLDUPNATIVEFENCEFDANDROIDPROC>(
            eglGetProcAddress("eglDupNativeFenceFDANDROID"));
    eglCreateSyncKHR = reinterpret_cast<PFNEGLCREATESYNCKHRPROC>(
        eglGetProcAddress("eglCreateSyncKHR"));
    eglWaitSyncKHR = reinterpret_cast<PFNEGLWAITSYNCKHRPROC>(
        eglGetProcAddress("eglWaitSyncKHR"));
    eglClientWaitSyncKHR = reinterpret_cast<PFNEGLCLIENTWAITSYNCKHRPROC>(
        eglGetProcAddress("eglClientWaitSyncKHR"));
    eglDestroySyncKHR = reinterpret_cast<PFNEGLDESTROYSYNCKHRPROC>(
        eglGetProcAddress("eglDestroySyncKHR"));
    return eglClientWaitSyncKHR && eglWaitSyncKHR &&
           eglGetNativeClientBufferANDROID && glBufferStorageExternalEXT &&
           eglCreateSyncKHR && eglDupNativeFenceFDANDROID && eglDestroySyncKHR;
  }();
  return extensions_allowed;
}

// Expects the target SSBO to be already bound.
absl::Status MapAHardwareBufferToGlBuffer(AHardwareBuffer* handle,
                                          size_t size) {
  if (!IsGlSupported()) {
    return absl::UnknownError(
        "No GL extension functions found to bind AHardwareBuffer and "
        "OpenGL buffer");
  }
  EGLClientBuffer native_buffer = eglGetNativeClientBufferANDROID(handle);
  if (!native_buffer) {
    return absl::UnknownError("Can't get native buffer");
  }
  glBufferStorageExternalEXT(GL_SHADER_STORAGE_BUFFER, 0, size, native_buffer,
                             GL_MAP_READ_BIT | GL_MAP_WRITE_BIT |
                                 GL_MAP_COHERENT_BIT_EXT |
                                 GL_MAP_PERSISTENT_BIT_EXT);
  if (glGetError() == GL_NO_ERROR) {
    return absl::OkStatus();
  } else {
    return absl::InternalError("Error in glBufferStorageExternalEXT");
  }
}

static inline int AlignedToPowerOf2(int value, int alignment) {
  // alignment must be a power of 2
  return ((value - 1) | (alignment - 1)) + 1;
}

class AhwbUsageTrack {
 public:
  static void Insert(uint64_t id) {
    absl::MutexLock lock(&mutex_);
    ahwb_usage_track_->insert(id);
  }
  static bool Contains(uint64_t id) {
    absl::MutexLock lock(&mutex_);
    return ahwb_usage_track_->contains(id);
  }

 private:
  AhwbUsageTrack() = default;

  ABSL_CONST_INIT static absl::Mutex mutex_;

  // TODO: Tracks all unique tensors. Can grow to a large number. LRU
  // (Least Recently Used) can be more predicted.
  static NoDestructor<absl::flat_hash_set<uint64_t>> ahwb_usage_track_
      ABSL_GUARDED_BY(mutex_);
};

ABSL_CONST_INIT absl::Mutex AhwbUsageTrack::mutex_(absl::kConstInit);
NoDestructor<absl::flat_hash_set<uint64_t>> AhwbUsageTrack::ahwb_usage_track_;

}  // namespace

Tensor::AHardwareBufferView Tensor::GetAHardwareBufferReadView() const {
  auto lock(std::make_unique<absl::MutexLock>(&view_mutex_));
  ABSL_CHECK(valid_ != kValidNone)
      << "Tensor must be written prior to read from.";
  ABSL_CHECK(!(valid_ & kValidOpenGlTexture2d))
      << "Tensor conversion between OpenGL texture and AHardwareBuffer is not "
         "supported.";
  bool transfer = ahwb_ == nullptr;
  ABSL_CHECK_OK(AllocateAHardwareBuffer())
      << "AHardwareBuffer is not supported on the target system.";
  valid_ |= kValidAHardwareBuffer;
  if (transfer) {
    MoveCpuOrSsboToAhwb();
  } else {
    if (valid_ & kValidOpenGlBuffer) CreateEglSyncAndFd();
  }

  EraseCompletedUsages(ahwb_usages_);
  ahwb_usages_.push_back(TensorAhwbUsage());
  return {ahwb_.get(),
          &write_complete_fence_fd_,  // Returned by GetWriteCompleteFenceFd.
          &ahwb_usages_.back(), std::move(lock),
          /*is_write_view=*/false};
}

void Tensor::CreateEglSyncAndFd() const {
  gl_context_->Run([this]() {
    if (IsGlSupported()) {
      auto egl_display = eglGetDisplay(EGL_DEFAULT_DISPLAY);
      if (egl_display != EGL_NO_DISPLAY) {
        fence_sync_ = eglCreateSyncKHR(egl_display,
                                       EGL_SYNC_NATIVE_FENCE_ANDROID, nullptr);
        if (fence_sync_ != EGL_NO_SYNC_KHR) {
          // TODO: Ensure we don't leak GL sync objects and fd
          // fences. This can happen if write_complete_fence_fd_ is already
          // valid here.
          write_complete_fence_fd_ =
              UniqueFd(eglDupNativeFenceFDANDROID(egl_display, fence_sync_));
          if (!write_complete_fence_fd_.IsValid()) {
            eglDestroySyncKHR(egl_display, fence_sync_);
            fence_sync_ = EGL_NO_SYNC_KHR;
          }
        }
      }
    }
    // Can't use Sync object.
    if (fence_sync_ == EGL_NO_SYNC_KHR) glFinish();
  });
}

Tensor::AHardwareBufferView Tensor::GetAHardwareBufferWriteView() const {
  auto lock(std::make_unique<absl::MutexLock>(&view_mutex_));
  ABSL_CHECK_OK(AllocateAHardwareBuffer())
      << "AHardwareBuffer is not supported on the target system.";
  if (valid_ != 0) {
    ABSL_LOG(ERROR)
        << "Tensors are designed for single writes. Multiple writes to a "
           "Tensor instance are not supported and may lead to undefined "
           "behavior due to lack of synchronization.";
  }
  valid_ = kValidAHardwareBuffer;

  EraseCompletedUsages(ahwb_usages_);
  if (!ahwb_usages_.empty()) {
    ABSL_LOG(DFATAL) << absl::StrFormat(
        "Write attempt while reading or writing AHWB (num usages: %d).",
        ahwb_usages_.size());
  }
  ahwb_usages_.push_back(TensorAhwbUsage());
  return {ahwb_.get(),
          &write_complete_fence_fd_,  // Set by SetWritingFinishedFD.
          &ahwb_usages_.back(), std::move(lock),
          /*is_write_view=*/true};
}

absl::Status Tensor::AllocateAHardwareBuffer() const {
  // Mark current tracking key as Ahwb-use.
  AhwbUsageTrack::Insert(ahwb_tracking_key_);
  use_ahwb_ = true;

  if (ahwb_ == nullptr) {
    HardwareBufferSpec spec = {};
    if (memory_alignment_ == 0) {
      spec.width = bytes();
    } else {
      // We expect allocations to be page-aligned, implicitly satisfying any
      // requirements from Edge TPU. No need to add a check for this,
      // since Edge TPU will check for us.
      spec.width = AlignedToPowerOf2(bytes(), memory_alignment_);
    }
    spec.height = 1;
    spec.layers = 1;
    spec.format = HardwareBufferSpec::AHARDWAREBUFFER_FORMAT_BLOB;
    spec.usage = HardwareBufferSpec::AHARDWAREBUFFER_USAGE_CPU_WRITE_OFTEN |
                 HardwareBufferSpec::AHARDWAREBUFFER_USAGE_CPU_READ_OFTEN |
                 HardwareBufferSpec::AHARDWAREBUFFER_USAGE_GPU_DATA_BUFFER;
    if (hardware_buffer_pool_ == nullptr) {
      MP_ASSIGN_OR_RETURN(auto new_ahwb, HardwareBuffer::Create(spec));
      ahwb_ = std::make_shared<HardwareBuffer>(std::move(new_ahwb));
    } else {
      MP_ASSIGN_OR_RETURN(ahwb_, hardware_buffer_pool_->GetBuffer(spec));
    }
  }
  return absl::OkStatus();
}

bool Tensor::AllocateAhwbMapToSsbo() const {
  if (__builtin_available(android 26, *)) {
    if (AllocateAHardwareBuffer().ok()) {
      if (MapAHardwareBufferToGlBuffer(ahwb_->GetAHardwareBuffer(), bytes())
              .ok()) {
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
        return true;
      }
      // Unable to make OpenGL <-> AHWB binding. Use regular SSBO instead.
      ahwb_.reset();
    }
  }
  return false;
}

// Moves Cpu/Ssbo resource under the Ahwb backed memory.
void Tensor::MoveCpuOrSsboToAhwb() const {
  auto dest =
      ahwb_->Lock(HardwareBufferSpec::AHARDWAREBUFFER_USAGE_CPU_WRITE_RARELY);
  ABSL_CHECK_OK(dest) << "Lock of AHWB failed";
  if (valid_ & kValidCpu) {
    std::memcpy(*dest, cpu_buffer_, bytes());
    // Free CPU memory because next time AHWB is mapped instead.
    FreeCpuBuffer();
    valid_ &= ~kValidCpu;
  } else if (valid_ & kValidOpenGlBuffer) {
    gl_context_->Run([this, dest]() {
      glBindBuffer(GL_SHADER_STORAGE_BUFFER, opengl_buffer_);
      const void* src = glMapBufferRange(GL_SHADER_STORAGE_BUFFER, 0, bytes(),
                                         GL_MAP_READ_BIT);
      std::memcpy(*dest, src, bytes());
      glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);
      glDeleteBuffers(1, &opengl_buffer_);
    });
    opengl_buffer_ = GL_INVALID_INDEX;
    gl_context_ = nullptr;
    // Reset OpenGL Buffer validness. The OpenGL buffer will be allocated on top
    // of the Ahwb at the next request to the OpenGlBufferView.
    valid_ &= ~kValidOpenGlBuffer;
  } else {
    ABSL_LOG(FATAL) << "Can't convert tensor with mask " << valid_
                    << " into AHWB.";
  }
  ABSL_CHECK_OK(ahwb_->Unlock()) << "Unlock of AHWB failed";
}

// SSBO is created on top of AHWB. A fence is inserted into the GPU queue before
// the GPU task that is going to read from the SSBO. When the writing into AHWB
// is finished then the GPU reads from the SSBO.
bool Tensor::InsertAhwbToSsboFence() const {
  if (!ahwb_) return false;
  if (write_complete_fence_fd_.IsValid()) {
    // Can't wait for FD to be signaled on GPU.
    // TODO: wait on CPU instead.
    if (!IsGlSupported()) return true;

    // Server-side fence.
    auto egl_display = eglGetDisplay(EGL_DEFAULT_DISPLAY);
    if (egl_display == EGL_NO_DISPLAY) return true;

    // EGL will take ownership of the passed fd if eglCreateSyncKHR is
    // successful.
    int fd_for_egl = dup(write_complete_fence_fd_.Get());

    EGLint sync_attribs[] = {EGL_SYNC_NATIVE_FENCE_FD_ANDROID,
                             (EGLint)fd_for_egl, EGL_NONE};
    fence_sync_ = eglCreateSyncKHR(egl_display, EGL_SYNC_NATIVE_FENCE_ANDROID,
                                   sync_attribs);
    if (fence_sync_ != EGL_NO_SYNC_KHR) {
      eglWaitSyncKHR(egl_display, fence_sync_, 0);
    } else {
      close(fd_for_egl);
    }
  }
  return true;
}

void Tensor::MoveAhwbStuff(Tensor* src) {
  // TODO: verify move/cleanup is done correctly.
  hardware_buffer_pool_ = std::exchange(src->hardware_buffer_pool_, nullptr);
  ahwb_ = std::exchange(src->ahwb_, nullptr);
  fence_sync_ = std::exchange(src->fence_sync_, EGL_NO_SYNC_KHR);
  ssbo_read_ = std::exchange(src->ssbo_read_, static_cast<GLsync>(0));
  write_complete_fence_fd_ = std::move(src->write_complete_fence_fd_);
  ahwb_usages_ = std::move(src->ahwb_usages_);
  use_ahwb_ = std::exchange(src->use_ahwb_, false);
}

absl::Status Tensor::ReleaseAhwbStuff() {
  write_complete_fence_fd_.Reset();
  if (__builtin_available(android 26, *)) {
    if (ahwb_) {
      const bool gl_operation_maybe_pending =
          ssbo_read_ != 0 || fence_sync_ != EGL_NO_SYNC_KHR;
      if (gl_operation_maybe_pending && gl_context_ == nullptr) {
        ABSL_LOG(DFATAL)
            << "Pending GL operations without captured GL context.";
      }
      if ((gl_operation_maybe_pending || HasIncompleteUsages(ahwb_usages_)) &&
          gl_context_ != nullptr) {
        // Delay release until the GPU usage is finished.
        MP_RETURN_IF_ERROR(gl_context_->Run([this]() -> absl::Status {
          auto& releaser = gl_context_->GetCachedAttachment(kAhwbGpuReleaser);
          return releaser.AddAndFreeUnusedResources(ahwb_, opengl_buffer_,
                                                    fence_sync_, ssbo_read_,
                                                    std::move(ahwb_usages_));
        }));
        opengl_buffer_ = GL_INVALID_INDEX;
      } else {
        CompleteAndEraseUsages(ahwb_usages_);
        ahwb_.reset();
      }
    }
  }
  return absl::OkStatus();
}

void* Tensor::MapAhwbToCpuRead() const {
  if (ahwb_ != nullptr) {
    if (!(valid_ & kValidCpu)) {
      if ((valid_ & kValidOpenGlBuffer) &&
          !write_complete_fence_fd_.IsValid()) {
        // EGLSync is failed. Use another synchronization method.
        // TODO: Use tflite::gpu::GlBufferSync and GlActiveSync.
        gl_context_->Run([]() { glFinish(); });
      } else if (valid_ & kValidAHardwareBuffer) {
        ABSL_CHECK_GT(ahwb_usages_.size(), 0);
        // ahwb_usages_.size() can be greater than one and can contain reader
        // and writer usages, see b/349348621.

        CompleteAndEraseUsages(ahwb_usages_);
      }
    }
    auto ptr =
        ahwb_->Lock(HardwareBufferSpec::AHARDWAREBUFFER_USAGE_CPU_READ_OFTEN,
                    write_complete_fence_fd_.Release());
    ABSL_CHECK_OK(ptr) << "Lock of AHWB failed";
    return *ptr;
  }
  return nullptr;
}

void* Tensor::MapAhwbToCpuWrite() const {
  if (ahwb_ != nullptr) {
    // TODO: If previously acquired view is GPU write view then need
    // to be sure that writing is finished. That's a warning: two consequent
    // write views should be interleaved with read view.
    auto locked_ptr =
        ahwb_->Lock(HardwareBufferSpec::AHARDWAREBUFFER_USAGE_CPU_WRITE_OFTEN);
    ABSL_CHECK_OK(locked_ptr) << "Lock of AHWB failed";
    return *locked_ptr;
  }
  return nullptr;
}

void Tensor::TrackAhwbUsage(uint64_t source_location_hash) const {
  if (ahwb_tracking_key_ == 0) {
    ahwb_tracking_key_ = source_location_hash;
    for (int dim : shape_.dims) {
      ahwb_tracking_key_ = tensor_internal::FnvHash64(ahwb_tracking_key_, dim);
    }
    ahwb_tracking_key_ =
        tensor_internal::FnvHash64(ahwb_tracking_key_, memory_alignment_);
  }
  // Keep flag value if it was set previously.
  use_ahwb_ = use_ahwb_ || AhwbUsageTrack::Contains(ahwb_tracking_key_);
}

#else  // MEDIAPIPE_TENSOR_USE_AHWB

bool Tensor::AllocateAhwbMapToSsbo() const { return false; }
bool Tensor::InsertAhwbToSsboFence() const { return false; }
void Tensor::MoveAhwbStuff(Tensor* src) {}
absl::Status Tensor::ReleaseAhwbStuff() { return absl::OkStatus(); }
void* Tensor::MapAhwbToCpuRead() const { return nullptr; }
void* Tensor::MapAhwbToCpuWrite() const { return nullptr; }
void Tensor::TrackAhwbUsage(uint64_t key) const {}

#endif  // MEDIAPIPE_TENSOR_USE_AHWB

}  // namespace mediapipe
