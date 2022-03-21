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

#include <memory>

#include "absl/memory/memory.h"
#include "mediapipe/framework/formats/image_frame.h"
#include "mediapipe/gpu/gl_calculator_helper_impl.h"
#include "mediapipe/gpu/gpu_buffer_format.h"
#include "mediapipe/gpu/gpu_shared_data_internal.h"
#include "mediapipe/gpu/image_frame_view.h"

namespace mediapipe {

GlCalculatorHelperImpl::GlCalculatorHelperImpl(CalculatorContext* cc,
                                               GpuResources* gpu_resources)
    : gpu_resources_(*gpu_resources) {
  gl_context_ = gpu_resources_.gl_context(cc);
}

GlCalculatorHelperImpl::~GlCalculatorHelperImpl() {
  RunInGlContext(
      [this] {
        if (framebuffer_) {
          glDeleteFramebuffers(1, &framebuffer_);
          framebuffer_ = 0;
        }
        return absl::OkStatus();
      },
      /*calculator_context=*/nullptr)
      .IgnoreError();
}

GlContext& GlCalculatorHelperImpl::GetGlContext() const { return *gl_context_; }

absl::Status GlCalculatorHelperImpl::RunInGlContext(
    std::function<absl::Status(void)> gl_func,
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

GlTexture GlCalculatorHelperImpl::MapGpuBuffer(const GpuBuffer& gpu_buffer,
                                               GlTextureView view) {
  if (gpu_buffer.format() != GpuBufferFormat::kUnknown) {
    // TODO: do the params need to be reset here??
    glBindTexture(view.target(), view.name());
    GlTextureInfo info = GlTextureInfoForGpuBufferFormat(
        gpu_buffer.format(), view.plane(), GetGlVersion());
    gl_context_->SetStandardTextureParams(view.target(),
                                          info.gl_internal_format);
    glBindTexture(view.target(), 0);
  }

  return GlTexture(std::move(view));
}

GlTexture GlCalculatorHelperImpl::CreateSourceTexture(
    const GpuBuffer& gpu_buffer) {
  return CreateSourceTexture(gpu_buffer, 0);
}

GlTexture GlCalculatorHelperImpl::CreateSourceTexture(
    const GpuBuffer& gpu_buffer, int plane) {
  return MapGpuBuffer(gpu_buffer, gpu_buffer.GetReadView<GlTextureView>(plane));
}

GlTexture GlCalculatorHelperImpl::CreateSourceTexture(
    const ImageFrame& image_frame) {
  auto gpu_buffer = GpuBufferCopyingImageFrame(image_frame);
  return MapGpuBuffer(gpu_buffer, gpu_buffer.GetReadView<GlTextureView>(0));
}

GpuBuffer GlCalculatorHelperImpl::GpuBufferWithImageFrame(
    std::shared_ptr<ImageFrame> image_frame) {
  return GpuBuffer(
      std::make_shared<GpuBufferStorageImageFrame>(std::move(image_frame)));
}

GpuBuffer GlCalculatorHelperImpl::GpuBufferCopyingImageFrame(
    const ImageFrame& image_frame) {
#if MEDIAPIPE_GPU_BUFFER_USE_CV_PIXEL_BUFFER
  auto maybe_buffer = CreateCVPixelBufferCopyingImageFrame(image_frame);
  // Converts absl::StatusOr to absl::Status since CHECK_OK() currently only
  // deals with absl::Status in MediaPipe OSS.
  CHECK_OK(maybe_buffer.status());
  return GpuBuffer(std::move(maybe_buffer).value());
#else
  return GpuBuffer(GlTextureBuffer::Create(image_frame));
#endif  // !MEDIAPIPE_GPU_BUFFER_USE_CV_PIXEL_BUFFER
}

template <>
std::unique_ptr<ImageFrame> GlTexture::GetFrame<ImageFrame>() const {
  view_->DoneWriting();
  std::shared_ptr<const ImageFrame> view =
      view_->gpu_buffer().GetReadView<ImageFrame>();
  auto copy = absl::make_unique<ImageFrame>();
  copy->CopyFrom(*view, ImageFrame::kDefaultAlignmentBoundary);
  return copy;
}

template <>
std::unique_ptr<GpuBuffer> GlTexture::GetFrame<GpuBuffer>() const {
  auto gpu_buffer = view_->gpu_buffer();
#ifdef __EMSCRIPTEN__
  // When WebGL is used, the GL context may be spontaneously lost which can
  // cause GpuBuffer allocations to fail. In that case, return a dummy buffer
  // to allow processing of the current frame complete.
  if (!gpu_buffer) {
    return std::make_unique<GpuBuffer>();
  }
#endif  // __EMSCRIPTEN__
  view_->DoneWriting();
  return absl::make_unique<GpuBuffer>(gpu_buffer);
}

GlTexture GlCalculatorHelperImpl::CreateDestinationTexture(
    int width, int height, GpuBufferFormat format) {
  if (!framebuffer_) {
    CreateFramebuffer();
  }

  GpuBuffer gpu_buffer =
      gpu_resources_.gpu_buffer_pool().GetBuffer(width, height, format);
  return MapGpuBuffer(gpu_buffer, gpu_buffer.GetWriteView<GlTextureView>(0));
}

}  // namespace mediapipe
