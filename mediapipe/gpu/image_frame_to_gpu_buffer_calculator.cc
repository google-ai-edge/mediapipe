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
#include "mediapipe/framework/port/status.h"
#include "mediapipe/gpu/gl_calculator_helper.h"

#ifdef __APPLE__
#include "mediapipe/objc/util.h"
#endif

namespace mediapipe {

// Convert ImageFrame to GpuBuffer.
class ImageFrameToGpuBufferCalculator : public CalculatorBase {
 public:
  ImageFrameToGpuBufferCalculator() {}

  static ::mediapipe::Status GetContract(CalculatorContract* cc);

  ::mediapipe::Status Open(CalculatorContext* cc) override;
  ::mediapipe::Status Process(CalculatorContext* cc) override;

 private:
#if !MEDIAPIPE_GPU_BUFFER_USE_CV_PIXEL_BUFFER
  GlCalculatorHelper helper_;
#endif  // !MEDIAPIPE_GPU_BUFFER_USE_CV_PIXEL_BUFFER
};
REGISTER_CALCULATOR(ImageFrameToGpuBufferCalculator);

// static
::mediapipe::Status ImageFrameToGpuBufferCalculator::GetContract(
    CalculatorContract* cc) {
  cc->Inputs().Index(0).Set<ImageFrame>();
  cc->Outputs().Index(0).Set<GpuBuffer>();
  // Note: we call this method even on platforms where we don't use the helper,
  // to ensure the calculator's contract is the same. In particular, the helper
  // enables support for the legacy side packet, which several graphs still use.
  MP_RETURN_IF_ERROR(GlCalculatorHelper::UpdateContract(cc));
  return ::mediapipe::OkStatus();
}

::mediapipe::Status ImageFrameToGpuBufferCalculator::Open(
    CalculatorContext* cc) {
  // Inform the framework that we always output at the same timestamp
  // as we receive a packet at.
  cc->SetOffset(TimestampDiff(0));
#if !MEDIAPIPE_GPU_BUFFER_USE_CV_PIXEL_BUFFER
  MP_RETURN_IF_ERROR(helper_.Open(cc));
#endif  // !MEDIAPIPE_GPU_BUFFER_USE_CV_PIXEL_BUFFER
  return ::mediapipe::OkStatus();
}

::mediapipe::Status ImageFrameToGpuBufferCalculator::Process(
    CalculatorContext* cc) {
#if MEDIAPIPE_GPU_BUFFER_USE_CV_PIXEL_BUFFER
  CFHolder<CVPixelBufferRef> buffer;
  MP_RETURN_IF_ERROR(CreateCVPixelBufferForImageFramePacket(
      cc->Inputs().Index(0).Value(), &buffer));
  cc->Outputs().Index(0).Add(new GpuBuffer(buffer), cc->InputTimestamp());
#else
  const auto& input = cc->Inputs().Index(0).Get<ImageFrame>();
  helper_.RunInGlContext([this, &input, &cc]() {
    auto src = helper_.CreateSourceTexture(input);
    auto output = src.GetFrame<GpuBuffer>();
    glFlush();
    cc->Outputs().Index(0).Add(output.release(), cc->InputTimestamp());
    src.Release();
  });
#endif  // MEDIAPIPE_GPU_BUFFER_USE_CV_PIXEL_BUFFER
  return ::mediapipe::OkStatus();
}

}  // namespace mediapipe
