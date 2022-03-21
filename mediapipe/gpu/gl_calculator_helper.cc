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

#include "mediapipe/framework/formats/image.h"
#include "mediapipe/framework/formats/image_frame.h"
#include "mediapipe/framework/legacy_calculator_support.h"
#include "mediapipe/framework/port/canonical_errors.h"
#include "mediapipe/framework/port/ret_check.h"
#include "mediapipe/framework/port/status.h"
#include "mediapipe/gpu/gl_calculator_helper_impl.h"
#include "mediapipe/gpu/gpu_buffer.h"
#include "mediapipe/gpu/gpu_service.h"

namespace mediapipe {

// The constructor and destructor need to be defined here so that
// std::unique_ptr can see the full definition of GlCalculatorHelperImpl.
// In the header, it is an incomplete type.
GlCalculatorHelper::GlCalculatorHelper() {}

GlCalculatorHelper::~GlCalculatorHelper() {}

absl::Status GlCalculatorHelper::Open(CalculatorContext* cc) {
  CHECK(cc);
  auto gpu_service = cc->Service(kGpuService);
  RET_CHECK(gpu_service.IsAvailable())
      << "GPU service not available. Did you forget to call "
         "GlCalculatorHelper::UpdateContract?";
  // TODO return error from impl_ (needs two-stage init)
  impl_ =
      absl::make_unique<GlCalculatorHelperImpl>(cc, &gpu_service.GetObject());
  return absl::OkStatus();
}

void GlCalculatorHelper::InitializeForTest(GpuSharedData* gpu_shared) {
  impl_ = absl::make_unique<GlCalculatorHelperImpl>(
      nullptr, gpu_shared->gpu_resources.get());
}

void GlCalculatorHelper::InitializeForTest(GpuResources* gpu_resources) {
  impl_ = absl::make_unique<GlCalculatorHelperImpl>(nullptr, gpu_resources);
}

// static
absl::Status GlCalculatorHelper::UpdateContract(CalculatorContract* cc) {
  cc->UseService(kGpuService);
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
    CHECK_EQ(input_side_packets, &cc->InputSidePackets());
    return UpdateContract(cc);
  }

  // TODO: remove when we can.
  LOG(WARNING)
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
    std::function<absl::Status(void)> gl_func) {
  if (!impl_) return absl::InternalError("helper not initialized");
  // TODO: Remove LegacyCalculatorSupport from MediaPipe OSS.
  auto calculator_context =
      LegacyCalculatorSupport::Scoped<CalculatorContext>::current();
  return impl_->RunInGlContext(gl_func, calculator_context);
}

GLuint GlCalculatorHelper::framebuffer() const { return impl_->framebuffer(); }

void GlCalculatorHelper::BindFramebuffer(const GlTexture& dst) {
  return impl_->BindFramebuffer(dst);
}

GlTexture GlCalculatorHelper::CreateSourceTexture(
    const GpuBuffer& pixel_buffer) {
  return impl_->CreateSourceTexture(pixel_buffer);
}

GlTexture GlCalculatorHelper::CreateSourceTexture(
    const ImageFrame& image_frame) {
  return impl_->CreateSourceTexture(image_frame);
}

GlTexture GlCalculatorHelper::CreateSourceTexture(const GpuBuffer& pixel_buffer,
                                                  int plane) {
  return impl_->CreateSourceTexture(pixel_buffer, plane);
}

GpuBuffer GlCalculatorHelper::GpuBufferWithImageFrame(
    std::shared_ptr<ImageFrame> image_frame) {
  return impl_->GpuBufferWithImageFrame(std::move(image_frame));
}

GpuBuffer GlCalculatorHelper::GpuBufferCopyingImageFrame(
    const ImageFrame& image_frame) {
  return impl_->GpuBufferCopyingImageFrame(image_frame);
}

void GlCalculatorHelper::GetGpuBufferDimensions(const GpuBuffer& pixel_buffer,
                                                int* width, int* height) {
  CHECK(width);
  CHECK(height);
  *width = pixel_buffer.width();
  *height = pixel_buffer.height();
}

GlTexture GlCalculatorHelper::CreateDestinationTexture(int output_width,
                                                       int output_height,
                                                       GpuBufferFormat format) {
  return impl_->CreateDestinationTexture(output_width, output_height, format);
}

GlContext& GlCalculatorHelper::GetGlContext() const {
  return impl_->GetGlContext();
}

GlVersion GlCalculatorHelper::GetGlVersion() const {
  return impl_->GetGlVersion();
}

GlTexture GlCalculatorHelper::CreateSourceTexture(
    const mediapipe::Image& image) {
  return impl_->CreateSourceTexture(image.GetGpuBuffer());
}

template <>
std::unique_ptr<mediapipe::Image> GlTexture::GetFrame<mediapipe::Image>()
    const {
  std::unique_ptr<GpuBuffer> buf = GetFrame<GpuBuffer>();
  auto output = absl::make_unique<mediapipe::Image>(*buf);
  return output;
}

}  // namespace mediapipe
