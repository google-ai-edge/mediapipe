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

#include "mediapipe/gpu/gl_calculator_helper_impl.h"
#include "mediapipe/gpu/gpu_buffer_format.h"
#include "mediapipe/gpu/gpu_shared_data_internal.h"

namespace mediapipe {

GlCalculatorHelperImpl::GlCalculatorHelperImpl(CalculatorContext* cc,
                                               GpuResources* gpu_resources)
    : gpu_resources_(*gpu_resources) {
  gl_context_ = gpu_resources_.gl_context(cc);
// GL_ES_VERSION_2_0 and up (at least through ES 3.2) may contain the extension.
// Checking against one also checks against higher ES versions. So this checks
// against GLES >= 2.0.
#if GL_ES_VERSION_2_0
  // No linear float filtering by default, check extensions.
  can_linear_filter_float_textures_ =
      gl_context_->HasGlExtension("OES_texture_float_linear");
#else
  // Any float32 texture we create should automatically have linear filtering.
  can_linear_filter_float_textures_ = true;
#endif  // GL_ES_VERSION_2_0
}

GlCalculatorHelperImpl::~GlCalculatorHelperImpl() {
  RunInGlContext(
      [this] {
        if (framebuffer_) {
          glDeleteFramebuffers(1, &framebuffer_);
          framebuffer_ = 0;
        }
        return ::mediapipe::OkStatus();
      },
      /*calculator_context=*/nullptr)
      .IgnoreError();
}

GlContext& GlCalculatorHelperImpl::GetGlContext() const { return *gl_context_; }

::mediapipe::Status GlCalculatorHelperImpl::RunInGlContext(
    std::function<::mediapipe::Status(void)> gl_func,
    CalculatorContext* calculator_context) {
  if (calculator_context) {
    return gl_context_->Run(std::move(gl_func), calculator_context->NodeId(),
                            calculator_context->InputTimestamp());
  } else {
    return gl_context_->Run(std::move(gl_func));
  }
}

void GlCalculatorHelperImpl::CreateFramebuffer() {
  // Our framebuffer will have a color attachment but no depth attachment,
  // so it's important that the depth test be off. It is disabled by default,
  // but we wanted to be explicit.
  // TODO: move this to glBindFramebuffer?
  glDisable(GL_DEPTH_TEST);
  glGenFramebuffers(1, &framebuffer_);
}

void GlCalculatorHelperImpl::BindFramebuffer(const GlTexture& dst) {
#ifdef __ANDROID__
  // On (some?) Android devices, attaching a new texture to the frame buffer
  // does not seem to detach the old one. As a result, using that texture
  // for texturing can produce incorrect output. See b/32091368 for details.
  // To fix this, we have to call either glBindFramebuffer with a FBO id of 0
  // or glFramebufferTexture2D with a texture ID of 0.
  glBindFramebuffer(GL_FRAMEBUFFER, 0);
#endif
  if (!framebuffer_) {
    CreateFramebuffer();
  }
  glBindFramebuffer(GL_FRAMEBUFFER, framebuffer_);
  glViewport(0, 0, dst.width(), dst.height());

  glActiveTexture(GL_TEXTURE0);
  glBindTexture(dst.target(), dst.name());
  glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, dst.target(),
                         dst.name(), 0);

#ifndef NDEBUG
  GLenum status = glCheckFramebufferStatus(GL_FRAMEBUFFER);
  if (status != GL_FRAMEBUFFER_COMPLETE) {
    VLOG(2) << "incomplete framebuffer: " << status;
  }
#endif
}

void GlCalculatorHelperImpl::SetStandardTextureParams(GLenum target,
                                                      GLint internal_format) {
  // Default to using linear filter everywhere. For float32 textures, fall back
  // to GL_NEAREST if linear filtering unsupported.
  GLint filter;
  switch (internal_format) {
    case GL_R32F:
    case GL_RG32F:
    case GL_RGBA32F:
      // 32F (unlike 16f) textures do not always support texture filtering
      // (According to OpenGL ES specification [TEXTURE IMAGE SPECIFICATION])
      filter = can_linear_filter_float_textures_ ? GL_LINEAR : GL_NEAREST;
      break;
    default:
      filter = GL_LINEAR;
  }
  glTexParameteri(target, GL_TEXTURE_MIN_FILTER, filter);
  glTexParameteri(target, GL_TEXTURE_MAG_FILTER, filter);
  glTexParameteri(target, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
  glTexParameteri(target, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
}

#if !MEDIAPIPE_GPU_BUFFER_USE_CV_PIXEL_BUFFER
GlTexture GlCalculatorHelperImpl::CreateSourceTexture(
    const ImageFrame& image_frame) {
  GlTexture texture = MapGlTextureBuffer(MakeGlTextureBuffer(image_frame));
  texture.for_reading_ = true;
  return texture;
}

GlTexture GlCalculatorHelperImpl::CreateSourceTexture(
    const GpuBuffer& gpu_buffer) {
  GlTexture texture = MapGpuBuffer(gpu_buffer, 0);
  texture.for_reading_ = true;
  return texture;
}

GlTexture GlCalculatorHelperImpl::CreateSourceTexture(
    const GpuBuffer& gpu_buffer, int plane) {
  GlTexture texture = MapGpuBuffer(gpu_buffer, plane);
  texture.for_reading_ = true;
  return texture;
}

GlTexture GlCalculatorHelperImpl::MapGpuBuffer(const GpuBuffer& gpu_buffer,
                                               int plane) {
  CHECK_EQ(plane, 0);
  return MapGlTextureBuffer(gpu_buffer.GetGlTextureBufferSharedPtr());
}

GlTexture GlCalculatorHelperImpl::MapGlTextureBuffer(
    const GlTextureBufferSharedPtr& texture_buffer) {
  // Insert wait call to sync with the producer.
  texture_buffer->WaitOnGpu();
  GlTexture texture;
  texture.helper_impl_ = this;
  texture.gpu_buffer_ = GpuBuffer(texture_buffer);
  texture.plane_ = 0;
  texture.width_ = texture_buffer->width_;
  texture.height_ = texture_buffer->height_;
  texture.target_ = texture_buffer->target_;
  texture.name_ = texture_buffer->name_;

  // TODO: do the params need to be reset here??
  glBindTexture(texture.target(), texture.name());
  GlTextureInfo info =
      GlTextureInfoForGpuBufferFormat(texture_buffer->format(), texture.plane_);
  SetStandardTextureParams(texture.target(), info.gl_internal_format);
  glBindTexture(texture.target(), 0);

  return texture;
}

GlTextureBufferSharedPtr GlCalculatorHelperImpl::MakeGlTextureBuffer(
    const ImageFrame& image_frame) {
  CHECK(gl_context_->IsCurrent());
  auto buffer = GlTextureBuffer::Create(
      image_frame.Width(), image_frame.Height(),
      GpuBufferFormatForImageFormat(image_frame.Format()),
      image_frame.PixelData());
  glBindTexture(GL_TEXTURE_2D, buffer->name_);
  GlTextureInfo info =
      GlTextureInfoForGpuBufferFormat(buffer->format_, /*plane=*/0);
  SetStandardTextureParams(buffer->target_, info.gl_internal_format);
  glBindTexture(GL_TEXTURE_2D, 0);

  return buffer;
}
#endif  // !MEDIAPIPE_GPU_BUFFER_USE_CV_PIXEL_BUFFER

GlTexture GlCalculatorHelperImpl::CreateDestinationTexture(
    int width, int height, GpuBufferFormat format) {
  if (!framebuffer_) {
    CreateFramebuffer();
  }

  GpuBuffer buffer =
      gpu_resources_.gpu_buffer_pool().GetBuffer(width, height, format);
  GlTexture texture = MapGpuBuffer(buffer, 0);

  return texture;
}

void GlCalculatorHelperImpl::ReadTexture(const GlTexture& texture, void* output,
                                         size_t size) {
  CHECK_GE(size, texture.width_ * texture.height_ * 4);

  GLint current_fbo;
  glGetIntegerv(GL_FRAMEBUFFER_BINDING, &current_fbo);
  CHECK_NE(current_fbo, 0);

  GLint color_attachment_name;
  glGetFramebufferAttachmentParameteriv(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                                        GL_FRAMEBUFFER_ATTACHMENT_OBJECT_NAME,
                                        &color_attachment_name);
  if (color_attachment_name != texture.name_) {
    // Save the viewport. Note that we assume that the color attachment is a
    // GL_TEXTURE_2D texture.
    GLint viewport[4];
    glGetIntegerv(GL_VIEWPORT, viewport);

    // Set the data from GLTexture object.
    glViewport(0, 0, texture.width_, texture.height_);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                           texture.target_, texture.name_, 0);
    glReadPixels(0, 0, texture.width_, texture.height_, GL_RGBA,
                 GL_UNSIGNED_BYTE, output);

    // Restore from the saved viewport and color attachment name.
    glViewport(viewport[0], viewport[1], viewport[2], viewport[3]);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D,
                           color_attachment_name, 0);
  } else {
    glReadPixels(0, 0, texture.width_, texture.height_, GL_RGBA,
                 GL_UNSIGNED_BYTE, output);
  }
}

}  // namespace mediapipe
