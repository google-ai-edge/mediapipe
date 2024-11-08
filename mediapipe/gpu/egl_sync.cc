#include "mediapipe/gpu/egl_sync.h"

#include <unistd.h>

#include <cstring>
#include <utility>

#include "absl/cleanup/cleanup.h"
#include "absl/log/absl_log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "mediapipe/framework/deps/no_destructor.h"
#include "mediapipe/framework/formats/unique_fd.h"
#include "mediapipe/framework/port/ret_check.h"
#include "mediapipe/framework/port/status_macros.h"
#include "mediapipe/framework/profiler/perfetto_profiling.h"
#include "mediapipe/gpu/egl_base.h"
#include "mediapipe/gpu/egl_errors.h"

namespace mediapipe {

namespace {

PFNEGLCREATESYNCKHRPROC eglCreateSyncKHR;
PFNEGLWAITSYNCKHRPROC eglWaitSyncKHR;
PFNEGLCLIENTWAITSYNCKHRPROC eglClientWaitSyncKHR;
PFNEGLDESTROYSYNCKHRPROC eglDestroySyncKHR;
PFNEGLDUPNATIVEFENCEFDANDROIDPROC eglDupNativeFenceFDANDROID;
PFNEGLGETSYNCATTRIBKHRPROC eglGetSyncAttribKHR;

bool HasExtension(EGLDisplay display, const char* extension) {
  const char* extensions = eglQueryString(display, EGL_EXTENSIONS);
  return extensions && std::strstr(extensions, extension);
}

absl::Status CheckEglFenceSyncSupported(EGLDisplay display) {
  static bool supported = HasExtension(display, "EGL_KHR_fence_sync");
  if (supported) {
    return absl::OkStatus();
  }
  return absl::UnavailableError("EGL_KHR_fence_sync unavailable.");
}

absl::Status CheckEglWaitSyncSupported(EGLDisplay display) {
  static bool supported = HasExtension(display, "EGL_KHR_wait_sync");
  if (supported) {
    return absl::OkStatus();
  }
  return absl::UnavailableError("EGL_KHR_wait_sync unavailable.");
}

absl::Status CheckEglAndroidNativeSyncSupported(EGLDisplay display) {
  static bool supported =
      HasExtension(display, "EGL_ANDROID_native_fence_sync");
  if (supported) {
    return absl::OkStatus();
  }
  return absl::UnavailableError("EGL_ANDROID_native_fence_sync unavailable.");
}

absl::Status CheckEglSyncSupported(EGLDisplay egl_display) {
  static NoDestructor<absl::Status> support_status([&]() -> absl::Status {
    MP_RETURN_IF_ERROR(CheckEglFenceSyncSupported(egl_display));
    MP_RETURN_IF_ERROR(CheckEglWaitSyncSupported(egl_display));

    RET_CHECK(eglCreateSyncKHR = reinterpret_cast<PFNEGLCREATESYNCKHRPROC>(
                  eglGetProcAddress("eglCreateSyncKHR")));
    RET_CHECK(eglWaitSyncKHR = reinterpret_cast<PFNEGLWAITSYNCKHRPROC>(
                  eglGetProcAddress("eglWaitSyncKHR")));
    RET_CHECK(eglClientWaitSyncKHR =
                  reinterpret_cast<PFNEGLCLIENTWAITSYNCKHRPROC>(
                      eglGetProcAddress("eglClientWaitSyncKHR")));
    RET_CHECK(eglDestroySyncKHR = reinterpret_cast<PFNEGLDESTROYSYNCKHRPROC>(
                  eglGetProcAddress("eglDestroySyncKHR")));
    RET_CHECK(eglGetSyncAttribKHR =
                  reinterpret_cast<PFNEGLGETSYNCATTRIBKHRPROC>(
                      eglGetProcAddress("eglGetSyncAttribKHR")));
    return absl::OkStatus();
  }());
  return *support_status;
}

absl::Status CheckEglNativeSyncSupported(EGLDisplay egl_display) {
  static NoDestructor<absl::Status> support_status([&]() -> absl::Status {
    MP_RETURN_IF_ERROR(CheckEglAndroidNativeSyncSupported(egl_display));
    RET_CHECK(eglDupNativeFenceFDANDROID =
                  reinterpret_cast<PFNEGLDUPNATIVEFENCEFDANDROIDPROC>(
                      eglGetProcAddress("eglDupNativeFenceFDANDROID")));
    return absl::OkStatus();
  }());
  return *support_status;
}

}  // namespace

absl::StatusOr<EglSync> EglSync::Create(EGLDisplay display) {
  MP_RETURN_IF_ERROR(CheckEglSyncSupported(display));

  const EGLSyncKHR egl_sync =
      eglCreateSyncKHR(display, EGL_SYNC_FENCE_KHR, nullptr);
  RET_CHECK_NE(egl_sync, EGL_NO_SYNC_KHR)
      << "Create/eglCreateSyncKHR failed: " << GetEglError();
  return EglSync(display, egl_sync);
}

absl::StatusOr<EglSync> EglSync::CreateNative(EGLDisplay display) {
  MEDIAPIPE_PERFETTO_TRACE_EVENT("EglSync::CreateNative");

  MP_RETURN_IF_ERROR(CheckEglSyncSupported(display));
  MP_RETURN_IF_ERROR(CheckEglNativeSyncSupported(display));

  const EGLSyncKHR egl_sync =
      eglCreateSyncKHR(display, EGL_SYNC_NATIVE_FENCE_ANDROID, nullptr);
  RET_CHECK_NE(egl_sync, EGL_NO_SYNC_KHR)
      << "CreateNative/eglCreateSyncKHR failed: " << GetEglError();
  return EglSync(display, egl_sync);
}

absl::StatusOr<EglSync> EglSync::CreateNative(EGLDisplay display,
                                              const UniqueFd& native_fence_fd) {
  MEDIAPIPE_PERFETTO_TRACE_EVENT(
      absl::StrCat("EglSync::CreateNative for FD: ", native_fence_fd.Get()));

  RET_CHECK(native_fence_fd.IsValid());
  MP_RETURN_IF_ERROR(CheckEglSyncSupported(display));
  MP_RETURN_IF_ERROR(CheckEglNativeSyncSupported(display));

  MP_ASSIGN_OR_RETURN(UniqueFd fd_for_egl, native_fence_fd.Dup());
  // NOTE: it looks like one could rely on `UniqueFd` for the cleanup, but
  // there's some clashing on ownership of the FD when passing it to
  // eglCreateSyncKHR and then using `UniqueFd::Release`, hence relying on
  // absl::Cleanup.
  const int fd = fd_for_egl.Release();
  absl::Cleanup fd_cleanup = [fd]() { close(fd); };
  const EGLint sync_attribs[] = {EGL_SYNC_NATIVE_FENCE_FD_ANDROID,
                                 static_cast<EGLint>(fd), EGL_NONE};
  const EGLSyncKHR egl_sync =
      eglCreateSyncKHR(display, EGL_SYNC_NATIVE_FENCE_ANDROID, sync_attribs);
  RET_CHECK_NE(egl_sync, EGL_NO_SYNC_KHR) << absl::StrCat(
      "CreateNative/eglCreateSyncKHR with original FD: ", native_fence_fd.Get(),
      " and dup FD: ", fd, " - failed: ", GetEglError());
  // EGL took ownership of the passed FD as eglCreateSyncKHR succeeded, so
  // cancelling the cleanup.
  std::move(fd_cleanup).Cancel();

  return EglSync(display, egl_sync);
}

bool EglSync::IsSupported(EGLDisplay display) {
  return CheckEglSyncSupported(display).ok();
}

bool EglSync::IsNativeSupported(EGLDisplay display) {
  return CheckEglNativeSyncSupported(display).ok();
}

EglSync::EglSync(EglSync&& sync) { *this = std::move(sync); }

EglSync& EglSync::operator=(EglSync&& sync) {
  if (this != &sync) {
    Invalidate();

    using std::swap;
    sync_ = std::exchange(sync.sync_, EGL_NO_SYNC_KHR);
    display_ = std::exchange(sync.display_, EGL_NO_DISPLAY);
  }
  return *this;
}

void EglSync::Invalidate() {
  if (sync_ == EGL_NO_SYNC_KHR || display_ == EGL_NO_DISPLAY) {
    return;
  }

  const absl::Status egl_sync_support = CheckEglSyncSupported(display_);
  if (!egl_sync_support.ok()) {
    ABSL_LOG(DFATAL) << "Attempt to destroy an EGL sync: " << egl_sync_support;
    return;
  }

  // Needs extension: EGL_KHR_fence_sync (EGL) / GL_OES_EGL_sync (OpenGL ES).
  // Note: we're doing nothing when the function pointer is nullptr, or the
  // call returns EGL_FALSE.
  const EGLBoolean result = eglDestroySyncKHR(display_, sync_);
  if (result == EGL_FALSE) {
    ABSL_LOG(DFATAL) << "eglDestroySyncKHR failed: " << GetEglError();
  }
  sync_ = EGL_NO_SYNC_KHR;
}

absl::Status EglSync::WaitOnGpu() {
  MEDIAPIPE_PERFETTO_TRACE_EVENT("EglSync::WaitOnGpu");

  MP_RETURN_IF_ERROR(CheckEglSyncSupported(display_));

  const EGLint result = eglWaitSyncKHR(display_, sync_, 0);
  RET_CHECK_EQ(result, EGL_TRUE) << "eglWaitSyncKHR failed: " << GetEglError();
  return absl::OkStatus();
}

absl::Status EglSync::Wait() {
  MEDIAPIPE_PERFETTO_TRACE_EVENT("EglSync::Wait");

  MP_RETURN_IF_ERROR(CheckEglSyncSupported(display_));

  const EGLint result = eglClientWaitSyncKHR(
      display_, sync_, EGL_SYNC_FLUSH_COMMANDS_BIT_KHR, EGL_FOREVER_KHR);
  RET_CHECK_EQ(result, EGL_CONDITION_SATISFIED_KHR)
      << "eglClientWaitSyncKHR failed: " << GetEglError();
  return absl::OkStatus();
}

absl::StatusOr<UniqueFd> EglSync::DupNativeFd() {
  MEDIAPIPE_PERFETTO_TRACE_EVENT("EglSync::DupNativeFd");

  MP_RETURN_IF_ERROR(CheckEglNativeSyncSupported(display_));

  const int fd = eglDupNativeFenceFDANDROID(display_, sync_);
  RET_CHECK_NE(fd, EGL_NO_NATIVE_FENCE_FD_ANDROID)
      << "eglDupNativeFenceFDANDROID failed: " << GetEglError();
  return UniqueFd(fd);
}

absl::StatusOr<bool> EglSync::IsSignaled() {
  MEDIAPIPE_PERFETTO_TRACE_EVENT("EglSync::IsSignaled");

  EGLint status;
  const EGLBoolean success =
      eglGetSyncAttribKHR(display_, sync_, EGL_SYNC_STATUS_KHR, &status);
  RET_CHECK_EQ(success, EGL_TRUE)
      << "eglGetSyncAttribKHR failed: " << GetEglError();
  return status == EGL_SIGNALED_KHR;
}

}  // namespace mediapipe
