#include <cstdint>
#include <utility>

#include "mediapipe/framework/formats/tensor.h"

#ifdef MEDIAPIPE_TENSOR_USE_AHWB
#include "absl/synchronization/mutex.h"
#include "mediapipe/framework/port.h"
#include "mediapipe/framework/port/logging.h"
#include "mediapipe/gpu/gl_base.h"
#include "third_party/GL/gl/include/EGL/egl.h"
#include "third_party/GL/gl/include/EGL/eglext.h"
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

  static void Add(AHardwareBuffer* ahwb, GLuint opengl_buffer,
                  EGLSyncKHR ssbo_sync, GLsync ssbo_read,
                  Tensor::FinishingFunc&& ahwb_written,
                  std::shared_ptr<mediapipe::GlContext> gl_context,
                  std::function<void()>&& callback) {
    static absl::Mutex mutex;
    std::deque<std::unique_ptr<DelayedReleaser>> to_release_local;
    using std::swap;

    // IsSignaled will grab other mutexes, so we don't want to call it while
    // holding the deque mutex.
    {
      absl::MutexLock lock(&mutex);
      swap(to_release_local, to_release_);
    }

    // Using `new` to access a non-public constructor.
    to_release_local.emplace_back(absl::WrapUnique(new DelayedReleaser(
        ahwb, opengl_buffer, ssbo_sync, ssbo_read, std::move(ahwb_written),
        gl_context, std::move(callback))));
    for (auto it = to_release_local.begin(); it != to_release_local.end();) {
      if ((*it)->IsSignaled()) {
        it = to_release_local.erase(it);
      } else {
        ++it;
      }
    }

    {
      absl::MutexLock lock(&mutex);
      to_release_.insert(to_release_.end(),
                         std::make_move_iterator(to_release_local.begin()),
                         std::make_move_iterator(to_release_local.end()));
      to_release_local.clear();
    }
  }

  ~DelayedReleaser() {
    if (release_callback_) release_callback_();
    if (__builtin_available(android 26, *)) {
      AHardwareBuffer_release(ahwb_);
    }
  }

  bool IsSignaled() {
    bool ready = true;

    if (ahwb_written_) {
      if (!ahwb_written_(false)) {
        ready = false;
      }
    }

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
  AHardwareBuffer* ahwb_;
  GLuint opengl_buffer_;
  // TODO: use wrapper instead.
  EGLSyncKHR fence_sync_;
  // TODO: use wrapper instead.
  GLsync ssbo_read_;
  Tensor::FinishingFunc ahwb_written_;
  std::shared_ptr<mediapipe::GlContext> gl_context_;
  std::function<void()> release_callback_;
  static inline std::deque<std::unique_ptr<DelayedReleaser>> to_release_;

  DelayedReleaser(AHardwareBuffer* ahwb, GLuint opengl_buffer,
                  EGLSyncKHR fence_sync, GLsync ssbo_read,
                  Tensor::FinishingFunc&& ahwb_written,
                  std::shared_ptr<mediapipe::GlContext> gl_context,
                  std::function<void()>&& callback)
      : ahwb_(ahwb),
        opengl_buffer_(opengl_buffer),
        fence_sync_(fence_sync),
        ssbo_read_(ssbo_read),
        ahwb_written_(std::move(ahwb_written)),
        gl_context_(gl_context),
        release_callback_(std::move(callback)) {}
};
}  // namespace

Tensor::AHardwareBufferView Tensor::GetAHardwareBufferReadView() const {
  auto lock(absl::make_unique<absl::MutexLock>(&view_mutex_));
  CHECK(valid_ != kValidNone) << "Tensor must be written prior to read from.";
  CHECK(!(valid_ & kValidOpenGlTexture2d))
      << "Tensor conversion between OpenGL texture and AHardwareBuffer is not "
         "supported.";
  CHECK(ahwb_ || !(valid_ & kValidOpenGlBuffer))
      << "Interoperability bettween OpenGL buffer and AHardwareBuffer is not "
         "supported on targe system.";
  CHECK(AllocateAHardwareBuffer())
      << "AHardwareBuffer is not supported on the target system.";
  valid_ |= kValidAHardwareBuffer;
  if (valid_ & kValidOpenGlBuffer) CreateEglSyncAndFd();
  return {ahwb_,
          ssbo_written_,
          &fence_fd_,  // The FD is created for SSBO -> AHWB synchronization.
          &ahwb_written_,  // Filled by SetReadingFinishedFunc.
          &release_callback_,
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

Tensor::AHardwareBufferView Tensor::GetAHardwareBufferWriteView(
    int size_alignment) const {
  auto lock(absl::make_unique<absl::MutexLock>(&view_mutex_));
  CHECK(AllocateAHardwareBuffer(size_alignment))
      << "AHardwareBuffer is not supported on the target system.";
  valid_ = kValidAHardwareBuffer;
  return {ahwb_,
          /*ssbo_written=*/-1,
          &fence_fd_,  // For SetWritingFinishedFD.
          &ahwb_written_,
          &release_callback_,
          std::move(lock)};
}

bool Tensor::AllocateAHardwareBuffer(int size_alignment) const {
  if (!use_ahwb_) return false;
  if (__builtin_available(android 26, *)) {
    if (ahwb_ == nullptr) {
      AHardwareBuffer_Desc desc = {};
      if (size_alignment == 0) {
        desc.width = bytes();
      } else {
        // We expect allocations to be page-aligned, implicitly satisfying any
        // requirements from Edge TPU. No need to add a check for this,
        // since Edge TPU will check for us.
        desc.width = AlignedToPowerOf2(bytes(), size_alignment);
      }
      desc.height = 1;
      desc.layers = 1;
      desc.format = AHARDWAREBUFFER_FORMAT_BLOB;
      desc.usage = AHARDWAREBUFFER_USAGE_CPU_WRITE_OFTEN |
                   AHARDWAREBUFFER_USAGE_CPU_READ_OFTEN |
                   AHARDWAREBUFFER_USAGE_GPU_DATA_BUFFER;
      return AHardwareBuffer_allocate(&desc, &ahwb_) == 0;
    }
    return true;
  }
  return false;
}

bool Tensor::AllocateAhwbMapToSsbo() const {
  if (__builtin_available(android 26, *)) {
    if (AllocateAHardwareBuffer()) {
      if (MapAHardwareBufferToGlBuffer(ahwb_, bytes()).ok()) {
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
        return true;
      }
      // Unable to make OpenGL <-> AHWB binding. Use regular SSBO instead.
      AHardwareBuffer_release(ahwb_);
      ahwb_ = nullptr;
    }
  }
  return false;
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
  ahwb_ = std::exchange(src->ahwb_, nullptr);
  fence_sync_ = std::exchange(src->fence_sync_, EGL_NO_SYNC_KHR);
  ssbo_read_ = std::exchange(src->ssbo_read_, static_cast<GLsync>(0));
  ssbo_written_ = std::exchange(src->ssbo_written_, -1);
  fence_fd_ = std::exchange(src->fence_fd_, -1);
  ahwb_written_ = std::move(src->ahwb_written_);
  release_callback_ = std::move(src->release_callback_);
}

void Tensor::ReleaseAhwbStuff() {
  if (fence_fd_ != -1) {
    close(fence_fd_);
    fence_fd_ = -1;
  }
  if (__builtin_available(android 26, *)) {
    if (ahwb_) {
      if (ssbo_read_ != 0 || fence_sync_ != EGL_NO_SYNC_KHR || ahwb_written_) {
        if (ssbo_written_ != -1) close(ssbo_written_);
        DelayedReleaser::Add(ahwb_, opengl_buffer_, fence_sync_, ssbo_read_,
                             std::move(ahwb_written_), gl_context_,
                             std::move(release_callback_));
        opengl_buffer_ = GL_INVALID_INDEX;
      } else {
        if (release_callback_) release_callback_();
        AHardwareBuffer_release(ahwb_);
      }
    }
  }
}

void* Tensor::MapAhwbToCpuRead() const {
  if (__builtin_available(android 26, *)) {
    if (ahwb_) {
      if (!(valid_ & kValidCpu)) {
        if ((valid_ & kValidOpenGlBuffer) && ssbo_written_ == -1) {
          // EGLSync is failed. Use another synchronization method.
          // TODO: Use tflite::gpu::GlBufferSync and GlActiveSync.
          glFinish();
        } else if (valid_ & kValidAHardwareBuffer) {
          CHECK(ahwb_written_) << "Ahwb-to-Cpu synchronization requires the "
                                  "completion function to be set";
          CHECK(ahwb_written_(true))
              << "An error oqcured while waiting for the buffer to be written";
        }
      }
      void* ptr;
      auto error =
          AHardwareBuffer_lock(ahwb_, AHARDWAREBUFFER_USAGE_CPU_READ_OFTEN,
                               ssbo_written_, nullptr, &ptr);
      CHECK(error == 0) << "AHardwareBuffer_lock " << error;
      close(ssbo_written_);
      ssbo_written_ = -1;
      return ptr;
    }
  }
  return nullptr;
}

void* Tensor::MapAhwbToCpuWrite() const {
  if (__builtin_available(android 26, *)) {
    if (ahwb_) {
      // TODO: If previously acquired view is GPU write view then need
      // to be sure that writing is finished. That's a warning: two consequent
      // write views should be interleaved with read view.
      void* ptr;
      auto error = AHardwareBuffer_lock(
          ahwb_, AHARDWAREBUFFER_USAGE_CPU_WRITE_OFTEN, -1, nullptr, &ptr);
      CHECK(error == 0) << "AHardwareBuffer_lock " << error;
      return ptr;
    }
  }
  return nullptr;
}

#else  // MEDIAPIPE_TENSOR_USE_AHWB

bool Tensor::AllocateAhwbMapToSsbo() const { return false; }
bool Tensor::InsertAhwbToSsboFence() const { return false; }
void Tensor::MoveAhwbStuff(Tensor* src) {}
void Tensor::ReleaseAhwbStuff() {}
void* Tensor::MapAhwbToCpuRead() const { return nullptr; }
void* Tensor::MapAhwbToCpuWrite() const { return nullptr; }

#endif  // MEDIAPIPE_TENSOR_USE_AHWB

}  // namespace mediapipe
