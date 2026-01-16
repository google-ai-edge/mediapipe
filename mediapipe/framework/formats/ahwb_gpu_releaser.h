#ifndef MEDIAPIPE_FRAMEWORK_FORMATS_HARDWARE_BUFFER_DELAYED_RELEASER_H_
#define MEDIAPIPE_FRAMEWORK_FORMATS_HARDWARE_BUFFER_DELAYED_RELEASER_H_

#include "mediapipe/framework/port.h"

#if MEDIAPIPE_TENSOR_USE_AHWB
#include <EGL/egl.h>
#include <EGL/eglext.h>

#include <deque>
#include <list>
#include <memory>
#include <utility>

#include "absl/status/status.h"
#include "absl/synchronization/mutex.h"
#include "mediapipe/framework/formats/hardware_buffer.h"
#include "mediapipe/framework/formats/tensor_ahwb_usage.h"
#include "mediapipe/gpu/gl_base.h"
#include "mediapipe/gpu/gl_context.h"

namespace mediapipe {

// This class keeps tensor's resources while the tensor is in use on GPU or TPU
// but is already released on CPU. When a regular OpenGL buffer is bound to the
// GPU queue for execution and released on client side then the buffer is still
// not released because is being used by GPU. OpenGL driver keeps traking of
// that. When OpenGL buffer is build on top of AHWB then the traking is done
// with the DeleyedRelease which, actually, keeps record of all AHWBs allocated
// and releases each of them if already used. EGL/GL fences are used to check
// the status of a buffer.
class AhwbGpuReleaser {
 public:
  // Note: This method must be called on GPU thread.
  absl::Status AddAndFreeUnusedResources(
      std::shared_ptr<HardwareBuffer> ahwb, GLuint opengl_buffer,
      EGLSyncKHR fence_sync, GLsync ssbo_read,
      std::list<TensorAhwbUsage>&& ahwb_usages) {
    return AddAndFreeUnusedResources(std::make_unique<AhwbGpuResources>(
        std::move(ahwb), opengl_buffer, fence_sync, ssbo_read,
        std::move(ahwb_usages)));
  }

 private:
  class AhwbGpuResources {
   public:
    AhwbGpuResources(std::shared_ptr<HardwareBuffer> ahwb, GLuint opengl_buffer,
                     EGLSyncKHR fence_sync, GLsync ssbo_read,
                     std::list<TensorAhwbUsage>&& ahwb_usages)
        : ahwb_(std::move(ahwb)),
          opengl_buffer_(opengl_buffer),
          fence_sync_(fence_sync),
          ssbo_read_(ssbo_read),
          ahwb_usages_(std::move(ahwb_usages)) {}

    // Destructor syncs on ahwb_usages
    ~AhwbGpuResources();

    // This method must be called on GPU thread.
    bool IsSignalled();

    // Non-copyable
    AhwbGpuResources(const AhwbGpuResources&) = delete;
    AhwbGpuResources& operator=(const AhwbGpuResources&) = delete;

   private:
    std::shared_ptr<HardwareBuffer> ahwb_;
    GLuint opengl_buffer_;
    // TODO: use wrapper instead.
    EGLSyncKHR fence_sync_;
    // TODO: use wrapper instead.
    GLsync ssbo_read_;
    std::list<TensorAhwbUsage> ahwb_usages_;
  };

  absl::Status AddAndFreeUnusedResources(
      std::unique_ptr<AhwbGpuResources> ahwb_gpu_resources);

  absl::Mutex mutex_;
  std::deque<std::unique_ptr<AhwbGpuResources>> to_release_
      ABSL_GUARDED_BY(mutex_);
};

inline constexpr GlContext::Attachment<AhwbGpuReleaser> kAhwbGpuReleaser(
    [](GlContext&) -> GlContext::Attachment<AhwbGpuReleaser>::Ptr {
      return GlContext::Attachment<AhwbGpuReleaser>::MakePtr();
    });

}  // namespace mediapipe

#endif  // MEDIAPIPE_TENSOR_USE_AHWB

#endif  // MEDIAPIPE_FRAMEWORK_FORMATS_HARDWARE_BUFFER_DELAYED_RELEASER_H_
