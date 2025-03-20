#include "mediapipe/gpu/egl_errors.h"

#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "mediapipe/gpu/egl_base.h"

namespace mediapipe {

absl::Status GetEglError() {
  EGLint error = eglGetError();
  switch (error) {
    case EGL_SUCCESS:
      return absl::OkStatus();
    case EGL_NOT_INITIALIZED:
      return absl::InternalError(
          "EGL is not initialized, or could not be initialized, for the "
          "specified EGL display connection.");
    case EGL_BAD_ACCESS:
      return absl::InternalError(
          "EGL cannot access a requested resource (for example a context is "
          "bound in another thread).");
    case EGL_BAD_ALLOC:
      return absl::InternalError(
          "EGL failed to allocate resources for the requested operation.");
    case EGL_BAD_ATTRIBUTE:
      return absl::InternalError(
          "An unrecognized attribute or attribute value was passed in the "
          "attribute list.");
    case EGL_BAD_CONTEXT:
      return absl::InternalError(
          "An EGLContext argument does not name a valid EGL rendering "
          "context.");
    case EGL_BAD_CONFIG:
      return absl::InternalError(
          "An EGLConfig argument does not name a valid EGL frame buffer "
          "configuration.");
    case EGL_BAD_CURRENT_SURFACE:
      return absl::InternalError(
          "The current surface of the calling thread is a window, pixel buffer "
          "or pixmap that is no longer valid.");
    case EGL_BAD_DISPLAY:
      return absl::InternalError(
          "An EGLDisplay argument does not name a valid EGL display "
          "connection.");
    case EGL_BAD_SURFACE:
      return absl::InternalError(
          "An EGLSurface argument does not name a valid surface (window, pixel "
          "buffer or pixmap) configured for GL rendering.");
    case EGL_BAD_MATCH:
      return absl::InternalError(
          "Arguments are inconsistent (for example, a valid context requires "
          "buffers not supplied by a valid surface).");
    case EGL_BAD_PARAMETER:
      return absl::InternalError("One or more argument values are invalid.");
    case EGL_BAD_NATIVE_PIXMAP:
      return absl::InternalError(
          "A NativePixmapType argument does not refer to a valid native "
          "pixmap.");
    case EGL_BAD_NATIVE_WINDOW:
      return absl::InternalError(
          "A NativeWindowType argument does not refer to a valid native "
          "window.");
    case EGL_CONTEXT_LOST:
      return absl::InternalError(
          "A power management event has occurred. The application must destroy "
          "all contexts and reinitialize OpenGL ES state and objects to "
          "continue rendering.");
  }
  return absl::UnknownError(absl::StrCat("EGL error: ", error));
}

}  // namespace mediapipe
