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

#include <array>
#include <cstdint>
#include <cstring>
#include <string>
#include <utility>

#include "absl/base/no_destructor.h"
#include "absl/container/node_hash_map.h"
#include "absl/log/absl_check.h"
#include "absl/log/absl_log.h"
#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/escaping.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
#include "mediapipe/framework/port/ret_check.h"
#include "mediapipe/framework/port/status.h"
#include "mediapipe/framework/port/status_builder.h"
#include "mediapipe/gpu/gl_context.h"
#include "mediapipe/gpu/gl_context_internal.h"

#ifndef EGL_OPENGL_ES3_BIT_KHR
#define EGL_OPENGL_ES3_BIT_KHR 0x00000040
#endif

// ANGLE-specific EGL extension tokens used for Vulkan device selection.
// Defined locally because system EGL headers frequently lag behind the Khronos
// and ANGLE extension registries.
#ifndef EGL_PLATFORM_ANGLE_ANGLE
#define EGL_PLATFORM_ANGLE_ANGLE 0x3202
#endif
#ifndef EGL_PLATFORM_ANGLE_TYPE_ANGLE
#define EGL_PLATFORM_ANGLE_TYPE_ANGLE 0x3203
#endif
#ifndef EGL_PLATFORM_ANGLE_TYPE_VULKAN_ANGLE
#define EGL_PLATFORM_ANGLE_TYPE_VULKAN_ANGLE 0x3450
#endif
#ifndef EGL_PLATFORM_ANGLE_VULKAN_DEVICE_UUID_ANGLE
#define EGL_PLATFORM_ANGLE_VULKAN_DEVICE_UUID_ANGLE 0x34F0
#endif

#if HAS_EGL

namespace mediapipe {

namespace {

static pthread_key_t egl_release_thread_key;
static pthread_once_t egl_release_key_once = PTHREAD_ONCE_INIT;

static void EglThreadExitCallback(void* key_value) {
  EGLDisplay current_display = eglGetCurrentDisplay();
  if (current_display != EGL_NO_DISPLAY) {
    // Some implementations have chosen to allow EGL_NO_DISPLAY as a valid
    // display parameter for eglMakeCurrent. This behavior is not portable to
    // all EGL implementations, and should be considered as an undocumented
    // vendor extension.
    // https://www.khronos.org/registry/EGL/sdk/docs/man/html/eglMakeCurrent.xhtml
    // Instead, to release the current context, we pass the current display.
    // If the current display is already EGL_NO_DISPLAY, no context is current.
    eglMakeCurrent(current_display, EGL_NO_SURFACE, EGL_NO_SURFACE,
                   EGL_NO_CONTEXT);
  }
  eglReleaseThread();
}

// If a key has a destructor callback, and a thread has a non-NULL value for
// that key, then the destructor is called when the thread exits.
static void MakeEglReleaseThreadKey() {
  int err = pthread_key_create(&egl_release_thread_key, EglThreadExitCallback);
  if (err) {
    ABSL_LOG(ERROR) << "cannot create pthread key: " << err;
  }
}

// This function can be called any number of times. For any thread on which it
// was called at least once, the EglThreadExitCallback will be called (once)
// when the thread exits.
static void EnsureEglThreadRelease() {
  pthread_once(&egl_release_key_once, MakeEglReleaseThreadKey);
  pthread_setspecific(egl_release_thread_key,
                      reinterpret_cast<void*>(0xDEADBEEF));
}

static absl::StatusOr<EGLDisplay> GetInitializedDefaultEglDisplay() {
  EGLDisplay display = eglGetDisplay(EGL_DEFAULT_DISPLAY);
  RET_CHECK(display != EGL_NO_DISPLAY)
      << "eglGetDisplay() returned error " << std::showbase << std::hex
      << eglGetError();

  EGLint major = 0;
  EGLint minor = 0;
  EGLBoolean egl_initialized = eglInitialize(display, &major, &minor);
  RET_CHECK(egl_initialized) << "Unable to initialize EGL";
  ABSL_LOG(INFO) << "Successfully initialized EGL. Major : " << major
                 << " Minor: " << minor;

  return display;
}

// Formats a GPU device UUID for logs and error messages.
static std::string DeviceUuidToHexString(
    absl::Span<const uint8_t> device_uuid) {
  return absl::BytesToHexString(absl::string_view(
      reinterpret_cast<const char*>(device_uuid.data()), device_uuid.size()));
}

// Returns an initialized ANGLE EGLDisplay backed by the Vulkan physical device
// identified by `device_uuid`. EGLDisplays are process-global, so successfully
// initialized displays are cached and reused for later requests for the same
// device.
static absl::StatusOr<EGLDisplay> GetInitializedAngleDisplayForDeviceUuid(
    absl::Span<const uint8_t> device_uuid) {
  using DeviceUuid = std::array<uint8_t, GlContext::kDeviceUuidSize>;
  using DisplayCache = absl::node_hash_map<DeviceUuid, EGLDisplay>;

  if (device_uuid.size() != GlContext::kDeviceUuidSize) {
    return absl::InvalidArgumentError(
        absl::StrFormat("Device UUID must be %d bytes, got %d.",
                        GlContext::kDeviceUuidSize, device_uuid.size()));
  }
  DeviceUuid uuid_key;
  std::memcpy(uuid_key.data(), device_uuid.data(), uuid_key.size());

  // Serializes first-time device initialization for the whole process. The
  // lock is held across eglGetPlatformDisplay() and eglInitialize() so that
  // concurrent requests for the same device cannot both initialize it; ANGLE
  // does not call back into this code, so there is no re-entrancy risk.
  static absl::Mutex display_mutex(absl::kConstInit);
  // ANGLE retains the UUID pointer passed through EGLAttrib beyond the
  // eglGetPlatformDisplay() call, so the pointer must outlive the display. The
  // cache key doubles as that storage: absl::node_hash_map keeps its keys at
  // stable addresses, entries are never erased, and the cache is never
  // destroyed, so the keys stay valid until process exit.
  static absl::NoDestructor<DisplayCache> display_cache;

  absl::MutexLock lock(&display_mutex);
  const DisplayCache::iterator it =
      display_cache->try_emplace(uuid_key, EGL_NO_DISPLAY).first;
  // node_hash_map stabilizes references, not iterators, so bind what we need
  // out of the entry rather than keeping the iterator live below.
  const uint8_t* const uuid_storage = it->first.data();
  EGLDisplay& cached_display = it->second;
  if (cached_display != EGL_NO_DISPLAY) {
    return cached_display;
  }

  PFNEGLGETPLATFORMDISPLAYPROC eglGetPlatformDisplay =
      reinterpret_cast<PFNEGLGETPLATFORMDISPLAYPROC>(
          eglGetProcAddress("eglGetPlatformDisplay"));
  if (eglGetPlatformDisplay == nullptr) {
    return absl::UnavailableError("eglGetPlatformDisplay not supported");
  }

  EGLAttrib attribs[] = {
      EGL_PLATFORM_ANGLE_TYPE_ANGLE,
      EGL_PLATFORM_ANGLE_TYPE_VULKAN_ANGLE,
      EGL_PLATFORM_ANGLE_VULKAN_DEVICE_UUID_ANGLE,
      reinterpret_cast<EGLAttrib>(uuid_storage),
      EGL_NONE,
  };

  EGLDisplay display = eglGetPlatformDisplay(
      EGL_PLATFORM_ANGLE_ANGLE, reinterpret_cast<void*>(EGL_DEFAULT_DISPLAY),
      attribs);
  if (display == EGL_NO_DISPLAY) {
    return absl::InternalError(absl::StrFormat(
        "eglGetPlatformDisplay returned EGL_NO_DISPLAY for device UUID %s, "
        "error: 0x%X",
        DeviceUuidToHexString(device_uuid), eglGetError()));
  }

  EGLint major = 0;
  EGLint minor = 0;
  if (!eglInitialize(display, &major, &minor)) {
    // Deliberately no eglTerminate(): ANGLE owns this handle in its own
    // display cache keyed by the attributes above, so a later retry for the
    // same UUID gets the same handle back and re-runs eglInitialize().
    return absl::InternalError(absl::StrFormat(
        "eglInitialize failed for ANGLE display for device UUID %s, "
        "error: 0x%X",
        DeviceUuidToHexString(device_uuid), eglGetError()));
  }
  ABSL_LOG(INFO) << "Successfully initialized ANGLE EGL display for device "
                    "UUID "
                 << DeviceUuidToHexString(device_uuid) << ". Major: " << major
                 << " Minor: " << minor;
  cached_display = display;
  return display;
}

static absl::StatusOr<EGLDisplay> GetInitializedEglDisplay() {
  auto status_or_display = GetInitializedDefaultEglDisplay();
  return status_or_display;
}

}  // namespace

GlContext::StatusOrGlContext GlContext::Create(std::nullptr_t nullp,
                                               bool create_thread) {
  return Create(EGL_NO_CONTEXT, create_thread);
}

GlContext::StatusOrGlContext GlContext::Create(const GlContext& share_context,
                                               bool create_thread) {
  std::shared_ptr<GlContext> context(new GlContext());
  if (share_context.display_ != EGL_NO_DISPLAY) {
    ABSL_RETURN_IF_ERROR(
        context->CreateContext(share_context.display_, share_context.context_));
  } else {
    ABSL_RETURN_IF_ERROR(context->CreateContext(share_context.context_));
  }
  ABSL_RETURN_IF_ERROR(context->FinishInitialization(create_thread));
  return std::move(context);
}

GlContext::StatusOrGlContext GlContext::Create(EGLContext share_context,
                                               bool create_thread) {
  std::shared_ptr<GlContext> context(new GlContext());
  ABSL_RETURN_IF_ERROR(context->CreateContext(share_context));
  ABSL_RETURN_IF_ERROR(context->FinishInitialization(create_thread));
  return std::move(context);
}

GlContext::StatusOrGlContext GlContext::CreateForDeviceUuid(
    absl::Span<const uint8_t> device_uuid, bool create_thread) {
  std::shared_ptr<GlContext> context(new GlContext());
  ABSL_RETURN_IF_ERROR(
      context->CreateContextForDeviceUuid(device_uuid, EGL_NO_CONTEXT));
  ABSL_RETURN_IF_ERROR(context->FinishInitialization(create_thread));
  return std::move(context);
}

absl::Status GlContext::CreateContextInternal(EGLContext share_context,
                                              int gl_version) {
  ABSL_CHECK(gl_version == 2 || gl_version == 3);

  const EGLint config_attr[] = {
      // clang-format off
      EGL_RENDERABLE_TYPE, gl_version == 3 ? EGL_OPENGL_ES3_BIT_KHR
                                           : EGL_OPENGL_ES2_BIT,
      // Allow rendering to pixel buffers or directly to windows.
      EGL_SURFACE_TYPE,
#ifdef MEDIAPIPE_OMIT_EGL_WINDOW_BIT
      EGL_PBUFFER_BIT,
#else
      EGL_PBUFFER_BIT | EGL_WINDOW_BIT,
#endif
      EGL_RED_SIZE, 8,
      EGL_GREEN_SIZE, 8,
      EGL_BLUE_SIZE, 8,
      EGL_ALPHA_SIZE, 8,  // if you need the alpha channel
      EGL_DEPTH_SIZE, 16,  // if you need the depth buffer
      EGL_NONE
      // clang-format on
  };

  // TODO: improve config selection.
  EGLint num_configs;
  EGLBoolean success =
      eglChooseConfig(display_, config_attr, &config_, 1, &num_configs);
  if (!success) {
    return ::mediapipe::UnknownErrorBuilder(MEDIAPIPE_LOC)
           << "eglChooseConfig() returned error " << std::showbase << std::hex
           << eglGetError();
  }
  if (!num_configs) {
    return mediapipe::UnknownErrorBuilder(MEDIAPIPE_LOC)
           << "eglChooseConfig() returned no matching EGL configuration for "
           << "RGBA8888 D16 ES" << gl_version << " request. ";
  }

  const EGLint context_attr[] = {
      // clang-format off
      EGL_CONTEXT_CLIENT_VERSION, gl_version,
      EGL_NONE
      // clang-format on
  };

  context_ = eglCreateContext(display_, config_, share_context, context_attr);
  int error = eglGetError();
  RET_CHECK(context_ != EGL_NO_CONTEXT)
      << "Could not create GLES " << gl_version << " context; "
      << "eglCreateContext() returned error " << std::showbase << std::hex
      << error
      << (error == EGL_BAD_CONTEXT
              ? ": external context uses a different version of OpenGL"
              : "");

  // We can't always rely on GL_MAJOR_VERSION and GL_MINOR_VERSION, since
  // GLES 2 does not have them, so let's set the major version here at least.
  gl_major_version_ = gl_version;

  return absl::OkStatus();
}

absl::Status GlContext::CreateContext(EGLContext share_context) {
  ABSL_ASSIGN_OR_RETURN(display_, GetInitializedEglDisplay());
  return CreateContext(display_, share_context);
}

absl::Status GlContext::CreateContextForDeviceUuid(
    absl::Span<const uint8_t> device_uuid, EGLContext share_context) {
  ABSL_ASSIGN_OR_RETURN(display_,
                        GetInitializedAngleDisplayForDeviceUuid(device_uuid));
  return CreateContext(display_, share_context);
}

absl::Status GlContext::CreateContext(EGLDisplay display,
                                      EGLContext share_context) {
  display_ = display;
  auto status = CreateContextInternal(share_context, 3);
  if (!status.ok()) {
    ABSL_LOG(WARNING) << "Creating a context with OpenGL ES 3 failed: "
                      << status;
    ABSL_LOG(WARNING) << "Fall back on OpenGL ES 2.";
    status = CreateContextInternal(share_context, 2);
  }
  ABSL_RETURN_IF_ERROR(status);

  EGLint pbuffer_attr[] = {EGL_WIDTH, 1, EGL_HEIGHT, 1, EGL_NONE};

  surface_ = eglCreatePbufferSurface(display_, config_, pbuffer_attr);
  RET_CHECK(surface_ != EGL_NO_SURFACE)
      << "eglCreatePbufferSurface() returned error " << std::showbase
      << std::hex << eglGetError();

  return absl::OkStatus();
}

void GlContext::DestroyContext() {
#ifdef __ANDROID__
  if (HasContext()) {
    // Detach the current program to work around b/166322604.
    auto detach_program = [this] {
      GlContext::ContextBinding saved_context;
      GetCurrentContextBinding(&saved_context);
      // Note: cannot use ThisContextBinding because it calls shared_from_this,
      // which is not available during destruction.
      if (eglMakeCurrent(display_, surface_, surface_, context_)) {
        glUseProgram(0);
      } else {
        ABSL_LOG(ERROR) << "eglMakeCurrent() returned error " << std::showbase
                        << std::hex << eglGetError();
      }
      return SetCurrentContextBinding(saved_context);
    };
    auto status = thread_ ? thread_->Run(detach_program) : detach_program();
    ABSL_LOG_IF(ERROR, !status.ok()) << status;
  }
#endif  // __ANDROID__

  if (thread_) {
    // Delete thread-local storage.
    // TODO: in theory our EglThreadExitCallback should suffice for
    // this; however, heapcheck still reports a leak without this call here
    // when using SwiftShader.
    // Perhaps heapcheck misses the thread destructors?
    thread_
        ->Run([] {
          eglReleaseThread();
          return absl::OkStatus();
        })
        .IgnoreError();
  }

  // Destroy the context and surface.
  if (IsCurrent()) {
    if (!eglMakeCurrent(display_, EGL_NO_SURFACE, EGL_NO_SURFACE,
                        EGL_NO_CONTEXT)) {
      ABSL_LOG(ERROR) << "eglMakeCurrent() returned error " << std::showbase
                      << std::hex << eglGetError();
    }
  }
  if (surface_ != EGL_NO_SURFACE) {
    if (!eglDestroySurface(display_, surface_)) {
      ABSL_LOG(ERROR) << "eglDestroySurface() returned error " << std::showbase
                      << std::hex << eglGetError();
    }
    surface_ = EGL_NO_SURFACE;
  }
  if (context_ != EGL_NO_CONTEXT) {
    if (!eglDestroyContext(display_, context_)) {
      ABSL_LOG(ERROR) << "eglDestroyContext() returned error " << std::showbase
                      << std::hex << eglGetError();
    }
    context_ = EGL_NO_CONTEXT;
  }

  // Under standard EGL, eglTerminate will terminate the display connection
  // for the entire process, no matter how many times eglInitialize has been
  // called. So we do not want to terminate it here, in case someone else is
  // using it.
  // However, Android implements non-standard reference-counted semantics for
  // eglInitialize/eglTerminate, so we should call it on that platform.
#ifdef __ANDROID__
  // TODO: this is removed for now since it caused issues on
  // YouTube. But in theory we _should_ be calling it. Needs more
  // investigation.
  // eglTerminate(display_);
#endif  // __ANDROID__
}

GlContext::ContextBinding GlContext::ThisContextBindingPlatform() {
  GlContext::ContextBinding result;
  result.display = display_;
  result.draw_surface = surface_;
  result.read_surface = surface_;
  result.context = context_;
  return result;
}

void GlContext::GetCurrentContextBinding(GlContext::ContextBinding* binding) {
  binding->display = eglGetCurrentDisplay();
  binding->draw_surface = eglGetCurrentSurface(EGL_DRAW);
  binding->read_surface = eglGetCurrentSurface(EGL_READ);
  binding->context = eglGetCurrentContext();
}

absl::Status GlContext::SetCurrentContextBinding(
    const ContextBinding& new_binding) {
  EnsureEglThreadRelease();
  EGLDisplay display = new_binding.display;
  if (display == EGL_NO_DISPLAY) {
    display = eglGetCurrentDisplay();
  }
  if (display == EGL_NO_DISPLAY) {
    display = eglGetDisplay(EGL_DEFAULT_DISPLAY);
  }
  EGLBoolean success =
      eglMakeCurrent(display, new_binding.draw_surface,
                     new_binding.read_surface, new_binding.context);
  RET_CHECK(success) << "eglMakeCurrent() returned error " << std::showbase
                     << std::hex << eglGetError();
  return absl::OkStatus();
}

bool GlContext::HasContext() const { return context_ != EGL_NO_CONTEXT; }

bool GlContext::IsCurrent() const {
  return HasContext() && (eglGetCurrentContext() == context_);
}

}  // namespace mediapipe

#endif  // HAS_EGL
