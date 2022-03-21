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

#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/image_frame.h"
#include "mediapipe/framework/port/ret_check.h"
#include "mediapipe/framework/port/status.h"

#define HAVE_GPU_BUFFER
#ifdef __APPLE__
#include "mediapipe/objc/util.h"
#endif

#include "mediapipe/gpu/gl_calculator_helper.h"

namespace mediapipe {

// Convert an input image (GpuBuffer or ImageFrame) to ImageFrame.
class GpuBufferToImageFrameCalculator : public CalculatorBase {
 public:
  GpuBufferToImageFrameCalculator() {}

  static absl::Status GetContract(CalculatorContract* cc);

  absl::Status Open(CalculatorContext* cc) override;
  absl::Status Process(CalculatorContext* cc) override;

 private:
#if !MEDIAPIPE_GPU_BUFFER_USE_CV_PIXEL_BUFFER
  GlCalculatorHelper helper_;
#endif  // MEDIAPIPE_GPU_BUFFER_USE_CV_PIXEL_BUFFER
};
REGISTER_CALCULATOR(GpuBufferToImageFrameCalculator);

// static
absl::Status GpuBufferToImageFrameCalculator::GetContract(
    CalculatorContract* cc) {
  cc->Inputs().Index(0).SetAny();
  cc->Outputs().Index(0).Set<ImageFrame>();
  // Note: we call this method even on platforms where we don't use the helper,
  // to ensure the calculator's contract is the same. In particular, the helper
  // enables support for the legacy side packet, which several graphs still use.
  MP_RETURN_IF_ERROR(GlCalculatorHelper::UpdateContract(cc));
  return absl::OkStatus();
}

absl::Status GpuBufferToImageFrameCalculator::Open(CalculatorContext* cc) {
  // Inform the framework that we always output at the same timestamp
  // as we receive a packet at.
  cc->SetOffset(TimestampDiff(0));
#if !MEDIAPIPE_GPU_BUFFER_USE_CV_PIXEL_BUFFER
  MP_RETURN_IF_ERROR(helper_.Open(cc));
#endif  // MEDIAPIPE_GPU_BUFFER_USE_CV_PIXEL_BUFFER
  return absl::OkStatus();
}

absl::Status GpuBufferToImageFrameCalculator::Process(CalculatorContext* cc) {
  if (cc->Inputs().Index(0).Value().ValidateAsType<ImageFrame>().ok()) {
    cc->Outputs().Index(0).AddPacket(cc->Inputs().Index(0).Value());
    return absl::OkStatus();
  }

#ifdef HAVE_GPU_BUFFER
  if (cc->Inputs().Index(0).Value().ValidateAsType<GpuBuffer>().ok()) {
    const auto& input = cc->Inputs().Index(0).Get<GpuBuffer>();
#if MEDIAPIPE_GPU_BUFFER_USE_CV_PIXEL_BUFFER
    std::unique_ptr<ImageFrame> frame =
        CreateImageFrameForCVPixelBuffer(GetCVPixelBufferRef(input));
    cc->Outputs().Index(0).Add(frame.release(), cc->InputTimestamp());
#else
    helper_.RunInGlContext([this, &input, &cc]() {
      auto src = helper_.CreateSourceTexture(input);
      std::unique_ptr<ImageFrame> frame = absl::make_unique<ImageFrame>(
          ImageFormatForGpuBufferFormat(input.format()), src.width(),
          src.height(), ImageFrame::kGlDefaultAlignmentBoundary);
      helper_.BindFramebuffer(src);
      const auto info = GlTextureInfoForGpuBufferFormat(input.format(), 0,
                                                        helper_.GetGlVersion());
      glReadPixels(0, 0, src.width(), src.height(), info.gl_format,
                   info.gl_type, frame->MutablePixelData());
      glFlush();
      cc->Outputs().Index(0).Add(frame.release(), cc->InputTimestamp());
      src.Release();
    });
#endif  // MEDIAPIPE_GPU_BUFFER_USE_CV_PIXEL_BUFFER
    return absl::OkStatus();
  }
#endif  // defined(HAVE_GPU_BUFFER)

  return absl::Status(absl::StatusCode::kInvalidArgument,
                      "Input packets must be ImageFrame or GpuBuffer.");
}

}  // namespace mediapipe
