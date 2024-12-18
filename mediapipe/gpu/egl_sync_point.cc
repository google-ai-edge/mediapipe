#include "mediapipe/gpu/egl_sync_point.h"

#include <memory>
#include <utility>

#include "absl/log/absl_log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "mediapipe/gpu/egl_sync.h"
#include "mediapipe/gpu/gl_context.h"

namespace mediapipe {

namespace {

class EglFenceSyncPoint : public GlSyncPoint {
 public:
  explicit EglFenceSyncPoint(std::shared_ptr<GlContext> gl_context,
                             EglSync egl_sync)
      : GlSyncPoint(std::move(gl_context)), egl_sync_(std::move(egl_sync)) {}

  ~EglFenceSyncPoint() override {
    gl_context_->RunWithoutWaiting(
        [ptr = new EglSync(std::move(egl_sync_))]() { delete ptr; });
  }

  EglFenceSyncPoint(const EglFenceSyncPoint&) = delete;
  EglFenceSyncPoint& operator=(const EglFenceSyncPoint&) = delete;

  void Wait() override {
    if (GlContext::IsAnyContextCurrent()) {
      WaitInternal();
    }
    // Fall back to GL context used during sync creation.
    gl_context_->Run([this] { WaitInternal(); });
  }

  void WaitInternal() {
    absl::Status result = egl_sync_.Wait();
    if (!result.ok()) {
      ABSL_LOG(DFATAL) << "EGL sync Wait failed: " << result;
    }
  }

  void WaitOnGpu() override {
    if (!GlContext::IsAnyContextCurrent()) {
      ABSL_LOG(DFATAL) << "WaitOnGpu without current context.";
    }

    absl::Status result = egl_sync_.WaitOnGpu();
    if (!result.ok()) {
      ABSL_LOG(DFATAL) << "EGL sync WaitOnGpu failed: " << result;
    }
  }

  bool IsReady() override {
    if (GlContext::IsAnyContextCurrent()) {
      return IsReadyInternal();
    }

    // Fall back to GL context used during sync creation.
    bool ready = false;
    gl_context_->Run([this, &ready] { ready = IsReadyInternal(); });
    return ready;
  }

  bool IsReadyInternal() {
    absl::StatusOr<bool> is_ready = egl_sync_.IsSignaled();
    if (!is_ready.ok()) {
      ABSL_LOG(DFATAL) << "EGL sync IsSignaled failed: " << is_ready.status();
      return false;
    }
    return *is_ready;
  }

 private:
  EglSync egl_sync_;
};

}  // namespace

absl::StatusOr<std::unique_ptr<GlSyncPoint>> CreateEglSyncPoint(
    std::shared_ptr<GlContext> gl_context, EglSync egl_sync) {
  return std::make_unique<EglFenceSyncPoint>(std::move(gl_context),
                                             std::move(egl_sync));
}

}  // namespace mediapipe
