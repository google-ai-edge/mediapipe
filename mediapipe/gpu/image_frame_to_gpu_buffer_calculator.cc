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

#include "mediapipe/gpu/image_frame_to_gpu_buffer_calculator.h"

#include "absl/status/status.h"
#include "mediapipe/framework/api3/calculator.h"
#include "mediapipe/framework/api3/calculator_context.h"
#include "mediapipe/framework/api3/calculator_contract.h"
#include "mediapipe/framework/api3/contract.h"
#include "mediapipe/framework/port/ret_check.h"
#include "mediapipe/framework/port/status_macros.h"
#include "mediapipe/gpu/gl_calculator_helper.h"
#include "mediapipe/gpu/gpu_buffer.h"

#ifdef __APPLE__
#include "mediapipe/objc/util.h"
#endif

namespace mediapipe::api3 {

class ImageFrameToGpuBufferCalculator
    : public Calculator<ImageFrameToGpuBufferNode,
                        ImageFrameToGpuBufferCalculator> {
 public:
  static absl::Status UpdateContract(
      CalculatorContract<ImageFrameToGpuBufferNode>& cc);
  absl::Status Open(CalculatorContext<ImageFrameToGpuBufferNode>& cc) override;
  absl::Status Process(
      CalculatorContext<ImageFrameToGpuBufferNode>& cc) override;

 private:
#if !MEDIAPIPE_GPU_BUFFER_USE_CV_PIXEL_BUFFER
  mediapipe::GlCalculatorHelper helper_;
#endif  // !MEDIAPIPE_GPU_BUFFER_USE_CV_PIXEL_BUFFER
};

// static
absl::Status ImageFrameToGpuBufferCalculator::UpdateContract(
    CalculatorContract<ImageFrameToGpuBufferNode>& cc) {
  // Note: we call this method even on platforms where we don't use the helper,
  // to ensure the calculator's contract is the same. In particular, the helper
  // enables support for the legacy side packet, which several graphs still use.
  return GlCalculatorHelper::UpdateContract(&cc.GetGenericContract());
}

absl::Status ImageFrameToGpuBufferCalculator::Open(
    CalculatorContext<ImageFrameToGpuBufferNode>& cc) {
#if !MEDIAPIPE_GPU_BUFFER_USE_CV_PIXEL_BUFFER
  MP_RETURN_IF_ERROR(helper_.Open(&cc.GetGenericContext()));
#endif  // !MEDIAPIPE_GPU_BUFFER_USE_CV_PIXEL_BUFFER
  return absl::OkStatus();
}

absl::Status ImageFrameToGpuBufferCalculator::Process(
    CalculatorContext<ImageFrameToGpuBufferNode>& cc) {
  RET_CHECK(cc.image_frame);

#if MEDIAPIPE_GPU_BUFFER_USE_CV_PIXEL_BUFFER
  CFHolder<CVPixelBufferRef> buffer;
  MP_RETURN_IF_ERROR(CreateCVPixelBufferForImageFramePacket(
      cc.image_frame.Packet().AsLegacyPacket(), &buffer));
  cc.gpu_buffer.Send(GpuBuffer(buffer));
#else
  const auto& input = cc.image_frame.GetOrDie();
  helper_.RunInGlContext([this, &input, &cc]() {
    GlTexture dst = helper_.CreateDestinationTexture(input);
    cc.gpu_buffer.Send(dst.GetFrame<GpuBuffer>());
    dst.Release();
  });
#endif  // MEDIAPIPE_GPU_BUFFER_USE_CV_PIXEL_BUFFER
  return absl::OkStatus();
}

}  // namespace mediapipe::api3
