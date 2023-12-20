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

#include <utility>

#include "absl/log/absl_check.h"
#include "absl/log/absl_log.h"
#include "absl/memory/memory.h"
#include "mediapipe/framework/port/logging.h"
#include "mediapipe/framework/port/ret_check.h"
#include "mediapipe/framework/port/status.h"
#include "mediapipe/framework/port/status_builder.h"
#include "mediapipe/gpu/gl_context.h"
#include "mediapipe/gpu/gl_context_internal.h"

#if defined(__EMSCRIPTEN__)
#include <emscripten.h>

namespace mediapipe {

// TODO: Handle webGL "context lost" and "context restored" events.
GlContext::StatusOrGlContext GlContext::Create(std::nullptr_t nullp,
                                               bool create_thread) {
  return Create(static_cast<EMSCRIPTEN_WEBGL_CONTEXT_HANDLE>(0), create_thread);
}

GlContext::StatusOrGlContext GlContext::Create(const GlContext& share_context,
                                               bool create_thread) {
  return Create(share_context.context_, create_thread);
}

GlContext::StatusOrGlContext GlContext::Create(
    EMSCRIPTEN_WEBGL_CONTEXT_HANDLE share_context, bool create_thread) {
  std::shared_ptr<GlContext> context(new GlContext());
  MP_RETURN_IF_ERROR(context->CreateContext(share_context));
  MP_RETURN_IF_ERROR(context->FinishInitialization(create_thread));
  return std::move(context);
}

absl::Status GlContext::CreateContextInternal(
    EMSCRIPTEN_WEBGL_CONTEXT_HANDLE external_context, int webgl_version) {
  ABSL_CHECK(webgl_version == 1 || webgl_version == 2);

  EmscriptenWebGLContextAttributes attrs;
  emscripten_webgl_init_context_attributes(&attrs);
  attrs.explicitSwapControl = 0;
  attrs.depth = 1;
  attrs.stencil = 0;
  attrs.antialias = 0;
  attrs.majorVersion = webgl_version;
  attrs.minorVersion = 0;

  // This flag tells the page compositor that the image written to the canvas
  // uses premultiplied alpha, and so can be used directly for compositing.
  // Without this, it needs to make an additional full-canvas rendering pass.
  attrs.premultipliedAlpha = 1;

  // TODO: Investigate this option in more detail, esp. on Safari.
  attrs.preserveDrawingBuffer = 0;

  // Quick patch for -s DISABLE_DEPRECATED_FIND_EVENT_TARGET_BEHAVIOR so it also
  // looks for our #canvas target in Module.canvas, where we expect it to be.
  // -s OFFSCREENCANVAS_SUPPORT=1 will no longer work with this under the new
  // event target behavior, but it was never supposed to be tapping into our
  // canvas anyways. See b/278155946 for more background.
  EM_ASM({ specialHTMLTargets["#canvas"] = Module.canvas; });
  EMSCRIPTEN_WEBGL_CONTEXT_HANDLE context_handle =
      emscripten_webgl_create_context("#canvas", &attrs);

  // Check for failure
  if (context_handle <= 0) {
    ABSL_LOG(INFO) << "Couldn't create webGL " << webgl_version << " context.";
    return ::mediapipe::UnknownErrorBuilder(MEDIAPIPE_LOC)
           << "emscripten_webgl_create_context() returned error "
           << context_handle;
  } else {
    emscripten_webgl_get_context_attributes(context_handle, &attrs);
    webgl_version = attrs.majorVersion;
  }
  context_ = context_handle;
  attrs_ = attrs;
  // We can't always rely on GL_MAJOR_VERSION and GL_MINOR_VERSION, since
  // GLES 2 does not have them, so let's set the major version here at least.
  // WebGL 1.0 maps to GLES 2.0 and WebGL 2.0 maps to GLES 3.0, so we add 1.
  gl_major_version_ = webgl_version + 1;
  return absl::OkStatus();
}

absl::Status GlContext::CreateContext(
    EMSCRIPTEN_WEBGL_CONTEXT_HANDLE external_context) {
  // TODO: If we're given a non-0 external_context, could try to use
  // that directly, since we're assuming a single-threaded single-context
  // environment anyways now.

  auto status = CreateContextInternal(external_context, 2);
  if (!status.ok()) {
    ABSL_LOG(WARNING) << "Creating a context with WebGL 2 failed: " << status;
    ABSL_LOG(WARNING) << "Fall back on WebGL 1.";
    status = CreateContextInternal(external_context, 1);
  }
  MP_RETURN_IF_ERROR(status);

  VLOG(1) << "Successfully created a WebGL context with major version "
          << gl_major_version_ << " and handle " << context_;
  return absl::OkStatus();
}

void GlContext::DestroyContext() {
  if (thread_) {
    // For now, we force web MediaPipe to be single-threaded, so error here.
    ABSL_LOG(ERROR) << "thread_ should not exist in DestroyContext() on web.";
  }

  // Destroy the context and surface.
  if (context_ != 0) {
    EMSCRIPTEN_RESULT res = emscripten_webgl_destroy_context(context_);
    if (res != EMSCRIPTEN_RESULT_SUCCESS) {
      ABSL_LOG(ERROR) << "emscripten_webgl_destroy_context() returned error "
                      << res;
    } else {
      ABSL_LOG(INFO) << "Successfully destroyed WebGL context with handle "
                     << context_;
    }
    context_ = 0;
  }
}

GlContext::ContextBinding GlContext::ThisContextBindingPlatform() {
  GlContext::ContextBinding result;
  result.context = context_;
  return result;
}

void GlContext::GetCurrentContextBinding(GlContext::ContextBinding* binding) {
  binding->context = emscripten_webgl_get_current_context();
}

absl::Status GlContext::SetCurrentContextBinding(
    const ContextBinding& new_binding) {
  if (new_binding.context == 0) {
    // Calling emscripten_webgl_make_context_current(0) is resulting in an error
    // so don't remove context for now, only replace!  In the future, can
    // perhaps create a separate "do-nothing" context for this.
    return absl::OkStatus();
  }
  // TODO: See if setting the same context to current multiple times
  // comes with a performance cost, and fix if so.
  EMSCRIPTEN_RESULT res =
      emscripten_webgl_make_context_current(new_binding.context);
  RET_CHECK(res == EMSCRIPTEN_RESULT_SUCCESS)
      << "emscripten_webgl_make_context_current() returned error " << res;
  return absl::OkStatus();
}

bool GlContext::HasContext() const { return context_ != 0; }

bool GlContext::IsCurrent() const {
  return HasContext() && (emscripten_webgl_get_current_context() == context_);
}

}  // namespace mediapipe

#endif  // defined(__EMSCRIPTEN__)
