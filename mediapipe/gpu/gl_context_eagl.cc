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

#if HAS_EAGL

#if !__has_feature(objc_arc)
#error This file must be built with ARC.
#endif

namespace mediapipe {

GlContext::StatusOrGlContext GlContext::Create(std::nullptr_t nullp,
                                               bool create_thread) {
  return Create(static_cast<EAGLSharegroup*>(nil), create_thread);
}

GlContext::StatusOrGlContext GlContext::Create(const GlContext& share_context,
                                               bool create_thread) {
  return Create(share_context.context_.sharegroup, create_thread);
}

GlContext::StatusOrGlContext GlContext::Create(EAGLContext* share_context,
                                               bool create_thread) {
  return Create(share_context.sharegroup, create_thread);
}

GlContext::StatusOrGlContext GlContext::Create(EAGLSharegroup* sharegroup,
                                               bool create_thread) {
  std::shared_ptr<GlContext> context(new GlContext());
  MP_RETURN_IF_ERROR(context->CreateContext(sharegroup));
  MP_RETURN_IF_ERROR(context->FinishInitialization(create_thread));
  return std::move(context);
}

::mediapipe::Status GlContext::CreateContext(EAGLSharegroup* sharegroup) {
  context_ = [[EAGLContext alloc] initWithAPI:kEAGLRenderingAPIOpenGLES3
                                   sharegroup:sharegroup];
  if (context_) {
    gl_major_version_ = 3;
  } else {
    context_ = [[EAGLContext alloc] initWithAPI:kEAGLRenderingAPIOpenGLES2
                                     sharegroup:sharegroup];
    gl_major_version_ = 2;
  }
  RET_CHECK(context_) << "Could not create an EAGLContext";

  CVOpenGLESTextureCacheRef cache;
  CVReturn err = CVOpenGLESTextureCacheCreate(kCFAllocatorDefault, NULL,
                                              context_, NULL, &cache);
  RET_CHECK_EQ(err, kCVReturnSuccess)
      << "Error at CVOpenGLESTextureCacheCreate";
  texture_cache_.adopt(cache);

  return ::mediapipe::OkStatus();
}

void GlContext::DestroyContext() {
  if (*texture_cache_) {
    // The texture cache must be flushed on tear down, otherwise we potentially
    // leak pixel buffers whose textures have pending GL operations after the
    // CVOpenGLESTextureRef is released in GlTexture::Release.
    CVOpenGLESTextureCacheFlush(*texture_cache_, 0);
  }
}

GlContext::ContextBinding GlContext::ThisContextBinding() {
  GlContext::ContextBinding result;
  result.context_object = shared_from_this();
  result.context = context_;
  return result;
}

void GlContext::GetCurrentContextBinding(GlContext::ContextBinding* binding) {
  binding->context = [EAGLContext currentContext];
}

::mediapipe::Status GlContext::SetCurrentContextBinding(
    const ContextBinding& new_binding) {
  BOOL success = [EAGLContext setCurrentContext:new_binding.context];
  RET_CHECK(success) << "Cannot set OpenGL context";
  return ::mediapipe::OkStatus();
}

bool GlContext::HasContext() const { return context_ != nil; }

bool GlContext::IsCurrent() const {
  return HasContext() && ([EAGLContext currentContext] == context_);
}

}  // namespace mediapipe

#endif  // HAS_EAGL
