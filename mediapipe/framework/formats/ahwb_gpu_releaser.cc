#include "mediapipe/framework/formats/ahwb_gpu_releaser.h"

#include "mediapipe/framework/port.h"  // IWYU pragma: keep

#if MEDIAPIPE_TENSOR_USE_AHWB

#include <EGL/egl.h>
#include <EGL/eglext.h>

#include "mediapipe/framework/port/ret_check.h"
#include "mediapipe/gpu/gl_base.h"

namespace mediapipe {

namespace {

PFNEGLDESTROYSYNCKHRPROC eglDestroySyncKHR;

bool IsGlSupported() {
  static const bool extensions_allowed = [] {
    eglDestroySyncKHR = reinterpret_cast<PFNEGLDESTROYSYNCKHRPROC>(
        eglGetProcAddress("eglDestroySyncKHR"));
    return eglDestroySyncKHR;
  }();
  return extensions_allowed;
}
}  // namespace

AhwbGpuReleaser::~AhwbGpuReleaser() {
  {
    absl::MutexLock lock(&mutex_);
    to_release_.clear();
  }
  absl::MutexLock cache_lock(&ssbo_cache_->mutex);
  const bool is_owning_context_current = gl_context_ != nullptr
                                             ? gl_context_->IsCurrent()
                                             : GlContext::IsAnyContextCurrent();
  if (is_owning_context_current) {
    for (const auto& [ahwb, opengl_buffer] : ssbo_cache_->entries) {
      if (opengl_buffer != GL_INVALID_INDEX) {
        glDeleteBuffers(1, &opengl_buffer);
      }
    }
  }
  ssbo_cache_->entries.clear();
}

void AhwbGpuReleaser::RegisterSsbo(HardwareBuffer& ahwb, GLuint opengl_buffer) {
  AHardwareBuffer* raw_ahwb = ahwb.GetAHardwareBuffer();
  {
    absl::MutexLock lock(&ssbo_cache_->mutex);
    if (!ssbo_cache_->entries.try_emplace(raw_ahwb, opengl_buffer).second) {
      return;
    }
  }
  std::weak_ptr<GlContext> weak_gl_context = gl_context_ != nullptr
                                                 ? gl_context_->weak_from_this()
                                                 : std::weak_ptr<GlContext>();
  ahwb.AddReleaseCallback([weak_cache = std::weak_ptr<SsboCache>(ssbo_cache_),
                           weak_gl_context = std::move(weak_gl_context),
                           raw_ahwb]() {
    GLuint ssbo_to_delete = GL_INVALID_INDEX;
    if (auto cache = weak_cache.lock()) {
      absl::MutexLock lock(&cache->mutex);
      if (auto node = cache->entries.extract(raw_ahwb); !node.empty()) {
        ssbo_to_delete = node.mapped();
      }
    }
    if (ssbo_to_delete != GL_INVALID_INDEX) {
      if (auto gl_ctx = weak_gl_context.lock()) {
        gl_ctx->RunWithoutWaiting(
            [ssbo_to_delete]() { glDeleteBuffers(1, &ssbo_to_delete); });
      }
    }
  });
}

AhwbGpuReleaser::AhwbGpuResources::~AhwbGpuResources() {
  CompleteAndEraseUsages(ahwb_usages_);
  if (ssbo_read_ != nullptr) {
    glDeleteSync(ssbo_read_);
    ssbo_read_ = nullptr;
  }
  if (fence_sync_ != EGL_NO_SYNC_KHR && IsGlSupported()) {
    auto egl_display = eglGetDisplay(EGL_DEFAULT_DISPLAY);
    if (egl_display != EGL_NO_DISPLAY) {
      eglDestroySyncKHR(egl_display, fence_sync_);
    }
    fence_sync_ = EGL_NO_SYNC_KHR;
  }
  if (opengl_buffer_ != GL_INVALID_INDEX) {
    glDeleteBuffers(1, &opengl_buffer_);
    opengl_buffer_ = GL_INVALID_INDEX;
  }
}

absl::Status AhwbGpuReleaser::AddAndFreeUnusedResources(
    std::unique_ptr<AhwbGpuResources> ahwb_gpu_resources) {
  RET_CHECK(IsGlSupported()) << "AHWB GPU releaser requires OpenGL support.";
  std::deque<std::unique_ptr<AhwbGpuResources>> to_release_local;
  using std::swap;

  // IsSignaled will grab other mutexes, so we don't want to call it while
  // holding the deque mutex.
  {
    absl::MutexLock lock(&mutex_);
    swap(to_release_local, to_release_);
  }

  // Using `new` to access a non-public constructor.
  to_release_local.emplace_back(std::move(ahwb_gpu_resources));
  for (auto it = to_release_local.begin(); it != to_release_local.end();) {
    if ((*it)->IsSignalled()) {
      it = to_release_local.erase(it);
    } else {
      ++it;
    }
  }

  {
    absl::MutexLock lock(&mutex_);
    to_release_.insert(to_release_.end(),
                       std::make_move_iterator(to_release_local.begin()),
                       std::make_move_iterator(to_release_local.end()));
    to_release_local.clear();
  }
  return absl::OkStatus();
}

bool AhwbGpuReleaser::AhwbGpuResources::IsSignalled() {
  if (!GlContext::IsAnyContextCurrent()) {
    ABSL_LOG(DFATAL) << "Must be called on GPU thread.";
  }

  if (ssbo_read_ != nullptr) {
    GLenum status = glClientWaitSync(ssbo_read_, 0,
                                     /* timeout ns = */ 0);
    if (status != GL_CONDITION_SATISFIED && status != GL_ALREADY_SIGNALED) {
      return false;
    }
    glDeleteSync(ssbo_read_);
    ssbo_read_ = nullptr;
  }

  if (HasIncompleteUsages(ahwb_usages_)) {
    return false;
  }

  if (fence_sync_ != EGL_NO_SYNC_KHR && IsGlSupported()) {
    auto egl_display = eglGetDisplay(EGL_DEFAULT_DISPLAY);
    if (egl_display != EGL_NO_DISPLAY) {
      eglDestroySyncKHR(egl_display, fence_sync_);
    }
    fence_sync_ = EGL_NO_SYNC_KHR;
  }
  glDeleteBuffers(1, &opengl_buffer_);
  opengl_buffer_ = GL_INVALID_INDEX;
  return true;
}

}  // namespace mediapipe

#endif  // MEDIAPIPE_TENSOR_USE_AHWB
