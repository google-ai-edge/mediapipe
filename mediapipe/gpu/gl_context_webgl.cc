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
  return Create(0, create_thread);
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

::mediapipe::Status GlContext::CreateContextInternal(
    EMSCRIPTEN_WEBGL_CONTEXT_HANDLE external_context, int webgl_version) {
  CHECK(webgl_version == 1 || webgl_version == 2);

  EmscriptenWebGLContextAttributes attrs;
  attrs.explicitSwapControl = 0;
  attrs.depth = 1;
  attrs.stencil = 0;
  attrs.antialias = 0;
  attrs.majorVersion = webgl_version;
  attrs.minorVersion = 0;

  attrs.premultipliedAlpha = 0;
  // New one to try out...  TODO: see if actually necessary for
  // pushing resulting texture through MediaPipe pipeline.
  attrs.preserveDrawingBuffer = 0;

  // Since the Emscripten canvas target finding function is visible from here,
  // we hijack findCanvasEventTarget directly for enforcing old Module.canvas
  // behavior if the user desires, falling back to the new DOM element CSS
  // selector behavior next if that is specified, and finally just allowing the
  // lookup to proceed on a null target.
  // TODO: Ensure this works with all options (in particular,
  //   multithreading options, like the special-case combination of USE_PTHREADS
  //   and OFFSCREEN_FRAMEBUFFER)
  EM_ASM(let init_once = true; if (init_once) {
    const __cachedFindCanvasEventTarget = __findCanvasEventTarget;

    if (typeof __cachedFindCanvasEventTarget != = 'function') {
      if (typeof console != = 'undefined') {
        console.error(
            'Expected Emscripten global function ' +
            '"__findCanvasEventTarget" not found. WebGL context creation ' +
            'may fail.');
      }
      return;
    }

    __findCanvasEventTarget = function(target) {
      if (Module && Module.canvas) {
        return Module.canvas;
      } else if (Module && Module.canvasCssSelector) {
        return __cachedFindCanvasEventTarget(Module.canvasCssSelector);
      } else {
        if (typeof console != = 'undefined') {
          console.warn('Module properties canvas and canvasCssSelector not ' +
                       'found during WebGL context creation.');
        }
        // We still go through with the find attempt, although for most use
        // cases it will not succeed, just in case the user does want to fall-
        // back.
        return __cachedFindCanvasEventTarget(target);
      }
    };  // NOLINT: Necessary semicolon.
    init_once = false;
  });

  // Note: below id parameter is only actually used if both Module.canvas and
  // Module.canvasCssSelector are undefined.
  EMSCRIPTEN_WEBGL_CONTEXT_HANDLE context_handle =
      emscripten_webgl_create_context(0 /* id */, &attrs);

  // Check for failure
  if (context_handle <= 0) {
    LOG(INFO) << "Couldn't create webGL " << webgl_version << " context.";
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
  return ::mediapipe::OkStatus();
}

::mediapipe::Status GlContext::CreateContext(
    EMSCRIPTEN_WEBGL_CONTEXT_HANDLE external_context) {
  // TODO: If we're given a non-0 external_context, could try to use
  // that directly, since we're assuming a single-threaded single-context
  // environment anyways now.

  auto status = CreateContextInternal(external_context, 2);
  if (!status.ok()) {
    LOG(WARNING) << "Creating a context with WebGL 2 failed: " << status;
    LOG(WARNING) << "Fall back on WebGL 1.";
    status = CreateContextInternal(external_context, 1);
  }
  MP_RETURN_IF_ERROR(status);

  LOG(INFO) << "Successfully created a WebGL Context with major version "
            << gl_major_version_ << " and context " << context_;

  return ::mediapipe::OkStatus();
}

void GlContext::DestroyContext() {
  if (thread_) {
    // For now, we force web MediaPipe to be single-threaded, so error here.
    LOG(ERROR) << "thread_ should not exist in DestroyContext() on web.";
  }

  // Destroy the context and surface.
  if (context_ != 0) {
    EMSCRIPTEN_RESULT res = emscripten_webgl_destroy_context(context_);
    if (res != EMSCRIPTEN_RESULT_SUCCESS) {
      LOG(ERROR) << "emscripten_webgl_destroy_context() returned error " << res;
    }
    context_ = 0;
  }
}

GlContext::ContextBinding GlContext::ThisContextBinding() {
  GlContext::ContextBinding result;
  result.context_object = shared_from_this();
  result.context = context_;
  return result;
}

void GlContext::GetCurrentContextBinding(GlContext::ContextBinding* binding) {
  binding->context = emscripten_webgl_get_current_context();
}

::mediapipe::Status GlContext::SetCurrentContextBinding(
    const ContextBinding& new_binding) {
  if (new_binding.context == 0) {
    // Calling emscripten_webgl_make_context_current(0) is resulting in an error
    // so don't remove context for now, only replace!  In the future, can
    // perhaps create a separate "do-nothing" context for this.
    return ::mediapipe::OkStatus();
  }
  // TODO: See if setting the same context to current multiple times
  // comes with a performance cost, and fix if so.
  EMSCRIPTEN_RESULT res =
      emscripten_webgl_make_context_current(new_binding.context);
  RET_CHECK(res == EMSCRIPTEN_RESULT_SUCCESS)
      << "emscripten_webgl_make_context_current() returned error " << res;
  return ::mediapipe::OkStatus();
}

bool GlContext::HasContext() const { return context_ != 0; }

bool GlContext::IsCurrent() const {
  return HasContext() && (emscripten_webgl_get_current_context() == context_);
}

}  // namespace mediapipe

#endif  // defined(__EMSCRIPTEN__)
