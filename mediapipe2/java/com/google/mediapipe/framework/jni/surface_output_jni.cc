// Copyright 2019 The MediaPipe Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "absl/synchronization/mutex.h"
#ifdef __ANDROID__
#include <android/native_window_jni.h>
#endif  // __ANDROID__

#include "mediapipe/framework/port/ret_check.h"
#include "mediapipe/framework/port/status.h"
#include "mediapipe/gpu/egl_surface_holder.h"
#include "mediapipe/gpu/gpu_shared_data_internal.h"
#include "mediapipe/java/com/google/mediapipe/framework/jni/graph.h"
#include "mediapipe/java/com/google/mediapipe/framework/jni/jni_util.h"
#include "mediapipe/java/com/google/mediapipe/framework/jni/surface_output_jni.h"

// TODO: CHECK in JNI does not work. Raise exception instead.

namespace {
mediapipe::EglSurfaceHolder* GetSurfaceHolder(jlong packet) {
  return mediapipe::android::Graph::GetPacketFromHandle(packet)
      .Get<std::unique_ptr<mediapipe::EglSurfaceHolder>>()
      .get();
}

mediapipe::GlContext* GetGlContext(jlong context) {
  auto mediapipe_graph = reinterpret_cast<mediapipe::android::Graph*>(context);
  mediapipe::GpuResources* gpu_resources = mediapipe_graph->GetGpuResources();
  return gpu_resources ? gpu_resources->gl_context().get() : nullptr;
}
}  // namespace

JNIEXPORT void JNICALL MEDIAPIPE_SURFACE_OUTPUT_METHOD(nativeSetFlipY)(
    JNIEnv* env, jobject thiz, jlong packet, jboolean flip) {
  mediapipe::EglSurfaceHolder* surface_holder = GetSurfaceHolder(packet);
  surface_holder->flip_y = flip;
}

JNIEXPORT void JNICALL MEDIAPIPE_SURFACE_OUTPUT_METHOD(nativeSetSurface)(
    JNIEnv* env, jobject thiz, jlong context, jlong packet, jobject surface) {
#ifdef __ANDROID__
  mediapipe::GlContext* gl_context = GetGlContext(context);
  CHECK(gl_context) << "GPU shared data not created";
  mediapipe::EglSurfaceHolder* surface_holder = GetSurfaceHolder(packet);

  // ANativeWindow_fromSurface must not be called on the GL thread, it is a
  // JNI call.
  ANativeWindow* window = nullptr;
  if (surface) {
    window = ANativeWindow_fromSurface(env, surface);
  }

  auto status = gl_context->Run(
      [gl_context, surface_holder, surface, window]() -> absl::Status {
        absl::MutexLock lock(&surface_holder->mutex);
        // Must destroy old surface first in case we are assigning the same
        // surface.
        // TODO: keep a ref to Java object and short-circuit if same?
        if (surface_holder->owned) {
          // NOTE: according to the eglDestroySurface documentation, the surface
          // is destroyed immediately "if it is not current on any thread". This
          // surface is only made current by the SurfaceSinkCalculator while it
          // holds the surface_holder->mutex, so at this point we know it is not
          // current on any thread, and we can rely on it being destroyed
          // immediately.
          RET_CHECK(eglDestroySurface(gl_context->egl_display(),
                                      surface_holder->surface))
              << "eglDestroySurface failed:" << eglGetError();
        }
        EGLSurface egl_surface = EGL_NO_SURFACE;
        if (surface) {
          EGLint surface_attr[] = {EGL_NONE};

          egl_surface = eglCreateWindowSurface(gl_context->egl_display(),
                                               gl_context->egl_config(), window,
                                               surface_attr);
          RET_CHECK(egl_surface != EGL_NO_SURFACE)
              << "eglCreateWindowSurface() returned error:" << eglGetError();
        }
        surface_holder->surface = egl_surface;
        surface_holder->owned = egl_surface != EGL_NO_SURFACE;
        return absl::OkStatus();
      });
  MEDIAPIPE_CHECK_OK(status);

  if (window) {
    VLOG(2) << "releasing window";
    ANativeWindow_release(window);
  }
#else
  LOG(FATAL) << "setSurface is only supported on Android";
#endif  // __ANDROID__
}

JNIEXPORT void JNICALL MEDIAPIPE_SURFACE_OUTPUT_METHOD(nativeSetEglSurface)(
    JNIEnv* env, jobject thiz, jlong context, jlong packet, jlong surface) {
  mediapipe::GlContext* gl_context = GetGlContext(context);
  CHECK(gl_context) << "GPU shared data not created";
  auto egl_surface = reinterpret_cast<EGLSurface>(surface);
  mediapipe::EglSurfaceHolder* surface_holder = GetSurfaceHolder(packet);
  EGLSurface old_surface = EGL_NO_SURFACE;

  {
    absl::MutexLock lock(&surface_holder->mutex);
    if (surface_holder->owned) {
      old_surface = surface_holder->surface;
    }
    surface_holder->surface = egl_surface;
    surface_holder->owned = false;
  }

  if (old_surface != EGL_NO_SURFACE) {
    MEDIAPIPE_CHECK_OK(
        gl_context->Run([gl_context, old_surface]() -> absl::Status {
          RET_CHECK(eglDestroySurface(gl_context->egl_display(), old_surface))
              << "eglDestroySurface failed:" << eglGetError();
          return absl::OkStatus();
        }));
  }
}
