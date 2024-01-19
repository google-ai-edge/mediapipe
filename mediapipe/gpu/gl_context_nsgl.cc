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

#include "absl/log/absl_log.h"
#include "absl/memory/memory.h"
#include "mediapipe/framework/port/ret_check.h"
#include "mediapipe/framework/port/status.h"
#include "mediapipe/framework/port/status_builder.h"
#include "mediapipe/gpu/gl_context.h"
#include "mediapipe/gpu/gl_context_internal.h"

#if HAS_NSGL

namespace mediapipe {

GlContext::StatusOrGlContext GlContext::Create(std::nullptr_t nullp,
                                               bool create_thread) {
  return Create(static_cast<NSOpenGLContext*>(nil), create_thread);
}

GlContext::StatusOrGlContext GlContext::Create(const GlContext& share_context,
                                               bool create_thread) {
  return Create(share_context.context_, create_thread);
}

GlContext::StatusOrGlContext GlContext::Create(NSOpenGLContext* share_context,
                                               bool create_thread) {
  std::shared_ptr<GlContext> context(new GlContext());
  MP_RETURN_IF_ERROR(context->CreateContext(share_context));
  MP_RETURN_IF_ERROR(context->FinishInitialization(create_thread));
  return std::move(context);
}

absl::Status GlContext::CreateContext(NSOpenGLContext* share_context) {
  // TODO: choose a better list?
  NSOpenGLPixelFormatAttribute attrs[] = {
  // This is required to get any OpenGL version 3.2 or higher. Note that
  // once this is enabled up to version 4.1 can be supported (depending on
  // hardware).
  // TODO: Remove the need for the OSX_ENABLE_3_2_CORE if this
  // proves to be safe in general.
#if defined(TARGET_OS_OSX) && defined(OSX_ENABLE_3_2_CORE)
      NSOpenGLPFAOpenGLProfile,
      NSOpenGLProfileVersion3_2Core,
#endif
      NSOpenGLPFAAccelerated,
      NSOpenGLPFAColorSize,
      24,
      NSOpenGLPFAAlphaSize,
      8,
      NSOpenGLPFADepthSize,
      16,
      0};

  pixel_format_ = [[NSOpenGLPixelFormat alloc] initWithAttributes:attrs];
  // If OpenGL 3.2 Core does not work, try again without it.
  if (!pixel_format_) {
    NSOpenGLPixelFormatAttribute attrs_2_1[] = {NSOpenGLPFAAccelerated,
                                                NSOpenGLPFAColorSize,
                                                24,
                                                NSOpenGLPFAAlphaSize,
                                                8,
                                                NSOpenGLPFADepthSize,
                                                16,
                                                0};

    pixel_format_ = [[NSOpenGLPixelFormat alloc] initWithAttributes:attrs_2_1];
  }
  if (!pixel_format_) {
    // On several Forge machines, the default config fails. For now let's do
    // this.
    ABSL_LOG(WARNING)
        << "failed to create pixel format; trying without acceleration";
    NSOpenGLPixelFormatAttribute attrs_no_accel[] = {NSOpenGLPFAColorSize,
                                                     24,
                                                     NSOpenGLPFAAlphaSize,
                                                     8,
                                                     NSOpenGLPFADepthSize,
                                                     16,
                                                     0};
    pixel_format_ =
        [[NSOpenGLPixelFormat alloc] initWithAttributes:attrs_no_accel];
  }
  if (!pixel_format_)
    return absl::InternalError("Could not create an NSOpenGLPixelFormat");
  context_ = [[NSOpenGLContext alloc] initWithFormat:pixel_format_
                                        shareContext:share_context];

  // Try to query pixel format from shared context.
  if (!context_) {
    ABSL_LOG(WARNING)
        << "Requested context not created, using queried context.";
    CGLContextObj cgl_ctx =
        static_cast<CGLContextObj>([share_context CGLContextObj]);
    CGLPixelFormatObj cgl_fmt =
        static_cast<CGLPixelFormatObj>(CGLGetPixelFormat(cgl_ctx));
    pixel_format_ =
        [[NSOpenGLPixelFormat alloc] initWithCGLPixelFormatObj:cgl_fmt];
    context_ = [[NSOpenGLContext alloc] initWithFormat:pixel_format_
                                          shareContext:share_context];
  }

  RET_CHECK(context_) << "Could not create an NSOpenGLContext";

  CVOpenGLTextureCacheRef cache;
  CVReturn err = CVOpenGLTextureCacheCreate(
      kCFAllocatorDefault, NULL, context_.CGLContextObj,
      pixel_format_.CGLPixelFormatObj, NULL, &cache);
  RET_CHECK_EQ(err, kCVReturnSuccess) << "Error at CVOpenGLTextureCacheCreate";
  texture_cache_.adopt(cache);

  return absl::OkStatus();
}

void GlContext::DestroyContext() {
  if (*texture_cache_) {
    // The texture cache must be flushed on tear down, otherwise we potentially
    // leak pixel buffers whose textures have pending GL operations after the
    // CVOpenGLTextureRef is released in GlTexture::Release.
    CVOpenGLTextureCacheFlush(*texture_cache_, 0);
  }
}

GlContext::ContextBinding GlContext::ThisContextBindingPlatform() {
  GlContext::ContextBinding result;
  result.context = context_;
  return result;
}

void GlContext::GetCurrentContextBinding(GlContext::ContextBinding* binding) {
  binding->context = [NSOpenGLContext currentContext];
}

absl::Status GlContext::SetCurrentContextBinding(
    const ContextBinding& new_binding) {
  if (new_binding.context) {
    [new_binding.context makeCurrentContext];
  } else {
    [NSOpenGLContext clearCurrentContext];
  }
  return absl::OkStatus();
}

bool GlContext::HasContext() const { return context_ != nil; }

bool GlContext::IsCurrent() const {
  return HasContext() && ([NSOpenGLContext currentContext] == context_);
}

}  // namespace mediapipe

#endif  // HAS_NSGL
