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

#include "mediapipe/gpu/gl_calculator_helper.h"

#include "absl/log/absl_check.h"
#include "absl/log/absl_log.h"
#include "absl/status/statusor.h"
#include "mediapipe/framework/formats/image.h"
#include "mediapipe/framework/formats/image_frame.h"
#include "mediapipe/framework/legacy_calculator_support.h"
#include "mediapipe/framework/port/canonical_errors.h"
#include "mediapipe/framework/port/ret_check.h"
#include "mediapipe/framework/port/status.h"
#include "mediapipe/gpu/gpu_buffer.h"
#include "mediapipe/gpu/gpu_service.h"

namespace mediapipe {

GlCalculatorHelper::GlCalculatorHelper() {}

GlCalculatorHelper::~GlCalculatorHelper() {}

void GlCalculatorHelper::InitializeInternal(CalculatorContext* cc,
                                            GpuResources* gpu_resources) {
  gpu_resources_ = gpu_resources;
  gl_context_ = gpu_resources_->gl_context(cc);
}

absl::Status GlCalculatorHelper::Open(CalculatorContext* cc) {
  ABSL_CHECK(cc);
  auto gpu_service = cc->Service(kGpuService);
  RET_CHECK(gpu_service.IsAvailable())
      << "GPU service not available. Did you forget to call "
         "GlCalculatorHelper::UpdateContract?";
  InitializeInternal(cc, &gpu_service.GetObject());
  return absl::OkStatus();
}

void GlCalculatorHelper::InitializeForTest(GpuSharedData* gpu_shared) {
  InitializeInternal(nullptr, gpu_shared->gpu_resources.get());
}

void GlCalculatorHelper::InitializeForTest(GpuResources* gpu_resources) {
  InitializeInternal(nullptr, gpu_resources);
}

// static
absl::Status GlCalculatorHelper::UpdateContract(CalculatorContract* cc,
                                                bool request_gpu_as_optional) {
  if (request_gpu_as_optional) {
    cc->UseService(kGpuService).Optional();
  } else {
    cc->UseService(kGpuService);
  }
  // Allow the legacy side packet to be provided, too, for backwards
  // compatibility with existing graphs. It will just be ignored.
  auto& input_side_packets = cc->InputSidePackets();
  auto id = input_side_packets.GetId(kGpuSharedTagName, 0);
  if (id.IsValid()) {
    input_side_packets.Get(id).Set<GpuSharedData*>();
  }
  return absl::OkStatus();
}

// static
absl::Status GlCalculatorHelper::SetupInputSidePackets(
    PacketTypeSet* input_side_packets) {
  auto cc = LegacyCalculatorSupport::Scoped<CalculatorContract>::current();
  if (cc) {
    ABSL_CHECK_EQ(input_side_packets, &cc->InputSidePackets());
    return UpdateContract(cc);
  }

  // TODO: remove when we can.
  ABSL_LOG(WARNING)
      << "CalculatorContract not available. If you're calling this "
         "from a GetContract method, call GlCalculatorHelper::UpdateContract "
         "instead.";
  auto id = input_side_packets->GetId(kGpuSharedTagName, 0);
  RET_CHECK(id.IsValid()) << "A " << mediapipe::kGpuSharedTagName
                          << " input side packet is required here.";
  input_side_packets->Get(id).Set<GpuSharedData*>();
  return absl::OkStatus();
}

absl::Status GlCalculatorHelper::RunInGlContext(
    std::function<absl::Status(void)> gl_func,
    CalculatorContext* calculator_context) {
  if (calculator_context) {
    return gl_context_->Run(std::move(gl_func), calculator_context->NodeId(),
                            calculator_context->InputTimestamp());
  } else {
    return gl_context_->Run(std::move(gl_func));
  }
}

absl::Status GlCalculatorHelper::RunInGlContext(
    std::function<absl::Status(void)> gl_func) {
  if (!Initialized()) return absl::InternalError("helper not initialized");
  // TODO: Remove LegacyCalculatorSupport from MediaPipe OSS.
  auto calculator_context =
      LegacyCalculatorSupport::Scoped<CalculatorContext>::current();
  return RunInGlContext(gl_func, calculator_context);
}

GLuint GlCalculatorHelper::framebuffer() const { return framebuffer_; }

void GlCalculatorHelper::CreateFramebuffer() {
  // Our framebuffer will have a color attachment but no depth attachment,
  // so it's important that the depth test be off. It is disabled by default,
  // but we wanted to be explicit.
  // TODO: move this to glBindFramebuffer? Or just remove.
  glDisable(GL_DEPTH_TEST);
  framebuffer_ = kUtilityFramebuffer.Get(*gl_context_);
}

void GlCalculatorHelper::BindFramebuffer(const GlTexture& dst) {
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
  glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, dst.target(),
                         dst.name(), 0);

#ifndef NDEBUG
  GLenum status = glCheckFramebufferStatus(GL_FRAMEBUFFER);
  if (status != GL_FRAMEBUFFER_COMPLETE) {
    VLOG(2) << "incomplete framebuffer: " << status;
  }
#endif
}

GlTexture GlCalculatorHelper::MapGpuBuffer(const GpuBuffer& gpu_buffer,
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

  return GlTexture(std::move(view), gpu_buffer);
}

GlTexture GlCalculatorHelper::CreateSourceTexture(const GpuBuffer& gpu_buffer) {
  return CreateSourceTexture(gpu_buffer, 0);
}

GlTexture GlCalculatorHelper::CreateSourceTexture(const GpuBuffer& gpu_buffer,
                                                  int plane) {
  return MapGpuBuffer(gpu_buffer, gpu_buffer.GetReadView<GlTextureView>(plane));
}

GlTexture GlCalculatorHelper::CreateSourceTexture(
    const ImageFrame& image_frame) {
  auto gpu_buffer = GpuBufferCopyingImageFrame(image_frame);
  return MapGpuBuffer(gpu_buffer, gpu_buffer.GetReadView<GlTextureView>(0));
}

GpuBuffer GlCalculatorHelper::GpuBufferWithImageFrame(
    std::shared_ptr<ImageFrame> image_frame) {
  return GpuBuffer(
      std::make_shared<GpuBufferStorageImageFrame>(std::move(image_frame)));
}

GpuBuffer GlCalculatorHelper::GpuBufferCopyingImageFrame(
    const ImageFrame& image_frame) {
#if MEDIAPIPE_GPU_BUFFER_USE_CV_PIXEL_BUFFER
  auto maybe_buffer = CreateCVPixelBufferCopyingImageFrame(image_frame);
  // Converts absl::StatusOr to absl::Status since ABSL_CHECK_OK() currently
  // only deals with absl::Status in MediaPipe OSS.
  ABSL_CHECK_OK(maybe_buffer);
  return GpuBuffer(std::move(maybe_buffer).value());
#else
  return GpuBuffer(GlTextureBuffer::Create(image_frame));
#endif  // !MEDIAPIPE_GPU_BUFFER_USE_CV_PIXEL_BUFFER
}

void GlCalculatorHelper::GetGpuBufferDimensions(const GpuBuffer& pixel_buffer,
                                                int* width, int* height) {
  ABSL_CHECK(width);
  ABSL_CHECK(height);
  *width = pixel_buffer.width();
  *height = pixel_buffer.height();
}

GlTexture GlCalculatorHelper::CreateDestinationTexture(int width, int height,
                                                       GpuBufferFormat format) {
  if (!framebuffer_) {
    CreateFramebuffer();
  }

  auto gpu_buffer =
      gpu_resources_->gpu_buffer_pool().GetBuffer(width, height, format);
  ABSL_CHECK_OK(gpu_buffer);
  return MapGpuBuffer(*gpu_buffer, gpu_buffer->GetWriteView<GlTextureView>(0));
}

GlTexture GlCalculatorHelper::CreateDestinationTexture(
    const ImageFrame& image_frame) {
  // TODO: ensure buffer pool is used when creating textures out of
  // ImageFrame.
  GpuBuffer gpu_buffer = GpuBufferCopyingImageFrame(image_frame);
  return MapGpuBuffer(gpu_buffer, gpu_buffer.GetWriteView<GlTextureView>(0));
}

GlTexture GlCalculatorHelper::CreateDestinationTexture(GpuBuffer& gpu_buffer) {
  return MapGpuBuffer(gpu_buffer, gpu_buffer.GetWriteView<GlTextureView>(0));
}

GlTexture GlCalculatorHelper::CreateSourceTexture(
    const mediapipe::Image& image) {
  return CreateSourceTexture(image.GetGpuBuffer());
}

template <>
std::unique_ptr<ImageFrame> GlTexture::GetFrame<ImageFrame>() const {
  view_->DoneWriting();
  std::shared_ptr<const ImageFrame> view =
      gpu_buffer_.GetReadView<ImageFrame>();
  auto copy = absl::make_unique<ImageFrame>();
  copy->CopyFrom(*view, ImageFrame::kDefaultAlignmentBoundary);
  return copy;
}

template <>
std::unique_ptr<GpuBuffer> GlTexture::GetFrame<GpuBuffer>() const {
  view_->DoneWriting();
  return absl::make_unique<GpuBuffer>(gpu_buffer_);
}

template <>
std::unique_ptr<mediapipe::Image> GlTexture::GetFrame<mediapipe::Image>()
    const {
  std::unique_ptr<GpuBuffer> buf = GetFrame<GpuBuffer>();
  auto output = absl::make_unique<mediapipe::Image>(*buf);
  return output;
}

}  // namespace mediapipe
