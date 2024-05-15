
#include "mediapipe/framework/formats/tensor.h"

#ifdef MEDIAPIPE_TENSOR_USE_AHWB
#include <EGL/egl.h>
#include <EGL/eglext.h>

#include <cstdint>
#include <list>
#include <utility>
#include <vector>

#include "absl/functional/any_invocable.h"
#include "absl/log/absl_check.h"
#include "absl/log/absl_log.h"
#include "absl/synchronization/mutex.h"
#include "mediapipe/framework/deps/no_destructor.h"
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

void EraseCompletedUsages(std::list<Tensor::AhwbUsage>& ahwb_usages) {
  for (auto it = ahwb_usages.begin(); it != ahwb_usages.end();) {
    bool is_ready = true;
    if (it->is_complete_fn) {
      if (!it->is_complete_fn(/*force_completion=*/false)) {
        is_ready = false;
      }
    } else {
      LOG(ERROR) << "Usage is missing completion function.";
    }

    if (is_ready) {
      for (auto& release_callback : it->release_callbacks) {
        release_callback();
      }
      it = ahwb_usages.erase(it);
    } else {
      ++it;
    }
  }
}

void CompleteAndEraseUsages(std::list<Tensor::AhwbUsage>& ahwb_usages) {
  for (auto& ahwb_usage : ahwb_usages) {
    if (ahwb_usage.is_complete_fn &&
        !ahwb_usage.is_complete_fn(/*force_completion=*/false)) {
      if (ahwb_usage.is_complete_fn(/*force_completion=*/true)) {
        ABSL_LOG(DFATAL) << "Failed to force-complete AHWB usage.";
      }
    }

    for (auto& release_callback : ahwb_usage.release_callbacks) {
      release_callback();
    }
  }
  ahwb_usages.clear();
}

bool HasIncompleteUsages(const std::list<Tensor::AhwbUsage>& ahwb_usages) {
  for (auto& ahwb_usage : ahwb_usages) {
    if (ahwb_usage.is_complete_fn &&
        !ahwb_usage.is_complete_fn(/*force_completion=*/false)) {
      return true;
    }
  }
  return false;
}

// This class keeps tensor's resources while the tensor is in use on GPU or TPU
// but is already released on CPU. When a regular OpenGL buffer is bound to the
// GPU queue for execution and released on client side then the buffer is still
// not released because is being used by GPU. OpenGL driver keeps traking of
// that. When OpenGL buffer is build on top of AHWB then the traking is done
// with the DeleyedRelease which, actually, keeps record of all AHWBs allocated
// and releases each of them if already used. EGL/GL fences are used to check
// the status of a buffer.
class DelayedReleaser {
 public:
  // Non-copyable
  DelayedReleaser(const DelayedReleaser&) = delete;
  DelayedReleaser& operator=(const DelayedReleaser&) = delete;
  // Non-movable
  DelayedReleaser(DelayedReleaser&&) = delete;
  DelayedReleaser& operator=(DelayedReleaser&&) = delete;

  static void Add(std::shared_ptr<HardwareBuffer> ahwb, GLuint opengl_buffer,
                  EGLSyncKHR ssbo_sync, GLsync ssbo_read,
                  std::list<Tensor::AhwbUsage>&& ahwb_usages,
                  std::shared_ptr<mediapipe::GlContext> gl_context) {
    static absl::Mutex mutex;
    std::deque<std::unique_ptr<DelayedReleaser>> to_release_local;
    using std::swap;

    // IsSignaled will grab other mutexes, so we don't want to call it while
    // holding the deque mutex.
    {
      absl::MutexLock lock(&mutex);
      swap(to_release_local, *to_release_);
    }

    // Using `new` to access a non-public constructor.
    to_release_local.emplace_back(absl::WrapUnique(
        new DelayedReleaser(std::move(ahwb), opengl_buffer, ssbo_sync,
                            ssbo_read, std::move(ahwb_usages), gl_context)));
    for (auto it = to_release_local.begin(); it != to_release_local.end();) {
      if ((*it)->IsSignaled()) {
        it = to_release_local.erase(it);
      } else {
        ++it;
      }
    }

    {
      absl::MutexLock lock(&mutex);
      to_release_->insert(to_release_->end(),
                          std::make_move_iterator(to_release_local.begin()),
                          std::make_move_iterator(to_release_local.end()));
      to_release_local.clear();
    }
  }

  ~DelayedReleaser() { CompleteAndEraseUsages(ahwb_usages_); }

  bool IsSignaled() {
    bool ready = !HasIncompleteUsages(ahwb_usages_);

    if (ssbo_read_ != 0) {
      gl_context_->Run([this, &ready]() {
        GLenum status = glClientWaitSync(ssbo_read_, 0,
                                         /* timeout ns = */ 0);
        if (status != GL_CONDITION_SATISFIED && status != GL_ALREADY_SIGNALED) {
          ready = false;
          return;
        }
        glDeleteSync(ssbo_read_);
        ssbo_read_ = 0;
      });
    }

    if (ready && gl_context_) {
      gl_context_->Run([this]() {
        if (fence_sync_ != EGL_NO_SYNC_KHR && IsGlSupported()) {
          auto egl_display = eglGetDisplay(EGL_DEFAULT_DISPLAY);
          if (egl_display != EGL_NO_DISPLAY) {
            eglDestroySyncKHR(egl_display, fence_sync_);
          }
          fence_sync_ = EGL_NO_SYNC_KHR;
        }
        glDeleteBuffers(1, &opengl_buffer_);
        opengl_buffer_ = GL_INVALID_INDEX;
      });
    }

    return ready;
  }

 protected:
  std::shared_ptr<HardwareBuffer> ahwb_;
  GLuint opengl_buffer_;
  // TODO: use wrapper instead.
  EGLSyncKHR fence_sync_;
  // TODO: use wrapper instead.
  GLsync ssbo_read_;
  std::list<Tensor::AhwbUsage> ahwb_usages_;
  std::shared_ptr<mediapipe::GlContext> gl_context_;

  static inline NoDestructor<std::deque<std::unique_ptr<DelayedReleaser>>>
      to_release_;

  DelayedReleaser(std::shared_ptr<HardwareBuffer> ahwb, GLuint opengl_buffer,
                  EGLSyncKHR fence_sync, GLsync ssbo_read,
                  std::list<Tensor::AhwbUsage>&& ahwb_usages,
                  std::shared_ptr<mediapipe::GlContext> gl_context)
      : ahwb_(std::move(ahwb)),
        opengl_buffer_(opengl_buffer),
        fence_sync_(fence_sync),
        ssbo_read_(ssbo_read),
        ahwb_usages_(std::move(ahwb_usages)),
        gl_context_(gl_context) {}
};

}  // namespace

Tensor::AHardwareBufferView Tensor::GetAHardwareBufferReadView() const {
  auto lock(absl::make_unique<absl::MutexLock>(&view_mutex_));
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
  ahwb_usages_.push_back(AhwbUsage());
  auto& ahwb_usage = ahwb_usages_.back();
  return {ahwb_.get(),
          ssbo_written_,
          &fence_fd_,  // The FD is created for SSBO -> AHWB synchronization.
          &ahwb_usage.is_complete_fn,  // Filled by SetReadingFinishedFunc.
          &ahwb_usage.release_callbacks,
          std::move(lock)};
}

void Tensor::CreateEglSyncAndFd() const {
  gl_context_->Run([this]() {
    if (IsGlSupported()) {
      auto egl_display = eglGetDisplay(EGL_DEFAULT_DISPLAY);
      if (egl_display != EGL_NO_DISPLAY) {
        fence_sync_ = eglCreateSyncKHR(egl_display,
                                       EGL_SYNC_NATIVE_FENCE_ANDROID, nullptr);
        if (fence_sync_ != EGL_NO_SYNC_KHR) {
          ssbo_written_ = eglDupNativeFenceFDANDROID(egl_display, fence_sync_);
          if (ssbo_written_ == -1) {
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
  auto lock(absl::make_unique<absl::MutexLock>(&view_mutex_));
  ABSL_CHECK_OK(AllocateAHardwareBuffer())
      << "AHardwareBuffer is not supported on the target system.";
  valid_ = kValidAHardwareBuffer;

  EraseCompletedUsages(ahwb_usages_);
  if (!ahwb_usages_.empty()) {
    ABSL_LOG(DFATAL) << absl::StrFormat(
        "Write attempt while reading or writing AHWB (num usages: %d).",
        ahwb_usages_.size());
  }
  ahwb_usages_.push_back(AhwbUsage());
  auto& ahwb_usage = ahwb_usages_.back();
  return {ahwb_.get(),
          /*ssbo_written=*/-1,
          &fence_fd_,                  // For SetWritingFinishedFD.
          &ahwb_usage.is_complete_fn,  // Filled by SetWritingFinishedFunc.
          &ahwb_usage.release_callbacks,
          std::move(lock)};
}

absl::Status Tensor::AllocateAHardwareBuffer() const {
  // Mark current tracking key as Ahwb-use.
  ahwb_usage_track_.insert(ahwb_tracking_key_);
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
  if (fence_fd_ != -1) {
    // Can't wait for FD to be signaled on GPU.
    // TODO: wait on CPU instead.
    if (!IsGlSupported()) return true;

    // Server-side fence.
    auto egl_display = eglGetDisplay(EGL_DEFAULT_DISPLAY);
    if (egl_display == EGL_NO_DISPLAY) return true;

    // EGL will take ownership of the passed fd if eglCreateSyncKHR is
    // successful.
    int fd_for_egl = dup(fence_fd_);

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
  ssbo_written_ = std::exchange(src->ssbo_written_, -1);
  fence_fd_ = std::exchange(src->fence_fd_, -1);
  ahwb_usages_ = std::move(src->ahwb_usages_);
}

void Tensor::ReleaseAhwbStuff() {
  if (fence_fd_ != -1) {
    close(fence_fd_);
    fence_fd_ = -1;
  }
  if (__builtin_available(android 26, *)) {
    if (ahwb_) {
      if (ssbo_read_ != 0 || fence_sync_ != EGL_NO_SYNC_KHR ||
          HasIncompleteUsages(ahwb_usages_)) {
        if (ssbo_written_ != -1) close(ssbo_written_);
        DelayedReleaser::Add(std::move(ahwb_), opengl_buffer_, fence_sync_,
                             ssbo_read_, std::move(ahwb_usages_), gl_context_);
        opengl_buffer_ = GL_INVALID_INDEX;
      } else {
        CompleteAndEraseUsages(ahwb_usages_);
        ahwb_.reset();
      }
    }
  }
}

void* Tensor::MapAhwbToCpuRead() const {
  if (ahwb_ != nullptr) {
    if (!(valid_ & kValidCpu)) {
      if ((valid_ & kValidOpenGlBuffer) && ssbo_written_ == -1) {
        // EGLSync is failed. Use another synchronization method.
        // TODO: Use tflite::gpu::GlBufferSync and GlActiveSync.
        gl_context_->Run([]() { glFinish(); });
      } else if (valid_ & kValidAHardwareBuffer) {
        ABSL_CHECK_GT(ahwb_usages_.size(), 0);
        if (ahwb_usages_.size() != 1) {
          // Theoretically that could be the case with previous implementation,
          // and we don't want to crash in such cases.
          ABSL_LOG(DFATAL)
              << "Multiple writes into the same AHWB tensor detected.";
        }

        auto& ahwb_usage = ahwb_usages_.back();
        ABSL_CHECK(ahwb_usage.is_complete_fn)
            << "Ahwb-to-Cpu synchronization requires the "
               "completion function to be set";
        ABSL_CHECK(ahwb_usage.is_complete_fn(/*force_completion=*/true))
            << "An error occurred while waiting for the buffer to be written.";

        CompleteAndEraseUsages(ahwb_usages_);
      }
    }
    auto ptr =
        ahwb_->Lock(HardwareBufferSpec::AHARDWAREBUFFER_USAGE_CPU_READ_OFTEN,
                    ssbo_written_);
    ABSL_CHECK_OK(ptr) << "Lock of AHWB failed";
    close(ssbo_written_);
    ssbo_written_ = -1;
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
  use_ahwb_ = use_ahwb_ || ahwb_usage_track_.contains(ahwb_tracking_key_);
}

#else  // MEDIAPIPE_TENSOR_USE_AHWB

bool Tensor::AllocateAhwbMapToSsbo() const { return false; }
bool Tensor::InsertAhwbToSsboFence() const { return false; }
void Tensor::MoveAhwbStuff(Tensor* src) {}
void Tensor::ReleaseAhwbStuff() {}
void* Tensor::MapAhwbToCpuRead() const { return nullptr; }
void* Tensor::MapAhwbToCpuWrite() const { return nullptr; }
void Tensor::TrackAhwbUsage(uint64_t key) const {}

#endif  // MEDIAPIPE_TENSOR_USE_AHWB

}  // namespace mediapipe
