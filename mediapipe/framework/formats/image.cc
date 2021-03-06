// Copyright 2020 The MediaPipe Authors.
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

#include "mediapipe/framework/formats/image.h"

namespace mediapipe {

// TODO Refactor common code from GpuBufferToImageFrameCalculator
bool Image::ConvertToCpu() const {
  if (!use_gpu_) return true;  // Already on CPU.
#if !MEDIAPIPE_DISABLE_GPU
#if MEDIAPIPE_GPU_BUFFER_USE_CV_PIXEL_BUFFER
  image_frame_ = CreateImageFrameForCVPixelBuffer(GetCVPixelBufferRef());
#else
  auto gl_texture = gpu_buffer_.GetGlTextureBufferSharedPtr();
  if (!gl_texture->GetProducerContext()) return false;
  gl_texture->GetProducerContext()->Run([this, &gl_texture]() {
    gl_texture->WaitOnGpu();
    const auto gpu_buf = mediapipe::GpuBuffer(GetGlTextureBufferSharedPtr());
#ifdef __ANDROID__
    glBindFramebuffer(GL_FRAMEBUFFER, 0);  // b/32091368
#endif
    GLuint fb = 0;
    glDisable(GL_DEPTH_TEST);
    // TODO Re-use a shared framebuffer.
    glGenFramebuffers(1, &fb);
    glBindFramebuffer(GL_FRAMEBUFFER, fb);
    glViewport(0, 0, gpu_buf.width(), gpu_buf.height());
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(gl_texture->target(), gl_texture->name());
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                           gl_texture->target(), gl_texture->name(), 0);
    auto frame = std::make_shared<ImageFrame>(
        mediapipe::ImageFormatForGpuBufferFormat(gpu_buf.format()),
        gpu_buf.width(), gpu_buf.height(),
        ImageFrame::kGlDefaultAlignmentBoundary);
    const auto info = GlTextureInfoForGpuBufferFormat(
        gpu_buf.format(), 0, gl_texture->GetProducerContext()->GetGlVersion());
    glReadPixels(0, 0, gpu_buf.width(), gpu_buf.height(), info.gl_format,
                 info.gl_type, frame->MutablePixelData());
    glDeleteFramebuffers(1, &fb);
    // Cleanup
    gl_texture->DidRead(gl_texture->GetProducerContext()->CreateSyncToken());
    image_frame_ = frame;
  });
#endif  //  MEDIAPIPE_GPU_BUFFER_USE_CV_PIXEL_BUFFER
#endif  // !MEDIAPIPE_DISABLE_GPU
  use_gpu_ = false;
  return true;
}

// TODO Refactor common code from ImageFrameToGpuBufferCalculator
bool Image::ConvertToGpu() const {
#if MEDIAPIPE_DISABLE_GPU
  return false;
#else
  if (use_gpu_) return true;  // Already on GPU.
#if MEDIAPIPE_GPU_BUFFER_USE_CV_PIXEL_BUFFER
  auto packet = MakePacket<ImageFrame>(std::move(*image_frame_));
  image_frame_ = nullptr;
  CFHolder<CVPixelBufferRef> buffer;
  auto status = CreateCVPixelBufferForImageFramePacket(packet, true, &buffer);
  CHECK_OK(status);
  gpu_buffer_ = mediapipe::GpuBuffer(std::move(buffer));
#else
  // GlCalculatorHelperImpl::MakeGlTextureBuffer (CreateSourceTexture)
  auto buffer = mediapipe::GlTextureBuffer::Create(
      image_frame_->Width(), image_frame_->Height(),
      mediapipe::GpuBufferFormatForImageFormat(image_frame_->Format()),
      image_frame_->PixelData());
  glBindTexture(GL_TEXTURE_2D, buffer->name());
  // See GlCalculatorHelperImpl::SetStandardTextureParams
  glTexParameteri(buffer->target(), GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  glTexParameteri(buffer->target(), GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  glTexParameteri(buffer->target(), GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
  glTexParameteri(buffer->target(), GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
  glBindTexture(GL_TEXTURE_2D, 0);
  glFlush();
  gpu_buffer_ = mediapipe::GpuBuffer(std::move(buffer));
#endif  //  MEDIAPIPE_GPU_BUFFER_USE_CV_PIXEL_BUFFER
  use_gpu_ = true;
  return true;
#endif  // MEDIAPIPE_DISABLE_GPU
}

}  // namespace mediapipe
