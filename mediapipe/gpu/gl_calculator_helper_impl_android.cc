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

#include "mediapipe/gpu/gl_calculator_helper_impl.h"
#include "mediapipe/gpu/gpu_shared_data_internal.h"

namespace mediapipe {

// TODO: move this method to GlCalculatorHelper, then we can
// access its framebuffer instead of requiring that one is already set.
template <>
std::unique_ptr<ImageFrame> GlTexture::GetFrame<ImageFrame>() const {
  auto output =
      absl::make_unique<ImageFrame>(ImageFormat::SRGBA, width_, height_,
                                    ImageFrame::kGlDefaultAlignmentBoundary);

  CHECK(helper_impl_);
  helper_impl_->ReadTexture(*this, output->MutablePixelData(),
                            output->PixelDataSize());

  return output;
}

template <>
std::unique_ptr<GpuBuffer> GlTexture::GetFrame<GpuBuffer>() const {
  CHECK(gpu_buffer_);
  // Inform the GlTextureBuffer that we have produced new content, and create
  // a producer sync point.
  gpu_buffer_.GetGlTextureBufferSharedPtr()->Updated(
      helper_impl_->GetGlContext().CreateSyncToken());

#ifdef __ANDROID__
  // On (some?) Android devices, the texture may need to be explicitly
  // detached from the current framebuffer.
  // TODO: is this necessary even with the unbind in BindFramebuffer?
  // It is not clear if this affected other contexts too, but let's keep it
  // while in doubt.
  GLint type = GL_NONE;
  glGetFramebufferAttachmentParameteriv(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                                        GL_FRAMEBUFFER_ATTACHMENT_OBJECT_TYPE,
                                        &type);
  if (type == GL_TEXTURE) {
    GLint color_attachment = 0;
    glGetFramebufferAttachmentParameteriv(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                                          GL_FRAMEBUFFER_ATTACHMENT_OBJECT_NAME,
                                          &color_attachment);
    if (color_attachment == name_) {
      glBindFramebuffer(GL_FRAMEBUFFER, 0);
    }
  }

  // Some Android drivers log a GL_INVALID_ENUM error after the first
  // glGetFramebufferAttachmentParameteriv call if there is no bound object,
  // even though it should be ok to ask for the type and get back GL_NONE.
  // Let's just ignore any pending errors here.
  GLenum error;
  while ((error = glGetError()) != GL_NO_ERROR) {
  }

#endif  // __ANDROID__
  return absl::make_unique<GpuBuffer>(gpu_buffer_);
}

void GlTexture::Release() {
  if (for_reading_ && gpu_buffer_) {
    // Inform the GlTextureBuffer that we have finished accessing its contents,
    // and create a consumer sync point.
    gpu_buffer_.GetGlTextureBufferSharedPtr()->DidRead(
        helper_impl_->GetGlContext().CreateSyncToken());
  }
  helper_impl_ = nullptr;
  for_reading_ = false;
  gpu_buffer_ = nullptr;
  plane_ = 0;
  name_ = 0;
  width_ = 0;
  height_ = 0;
}

}  // namespace mediapipe
