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
#include "mediapipe/gpu/gpu_buffer_storage_image_frame.h"

namespace mediapipe {

// Convert ImageFrame to GpuBuffer.
class ImageFrameToGpuBufferCalculator : public CalculatorBase {
 public:
  ImageFrameToGpuBufferCalculator() {}

  static absl::Status GetContract(CalculatorContract* cc);

  absl::Status Open(CalculatorContext* cc) override;
  absl::Status Process(CalculatorContext* cc) override;

 private:
  GlCalculatorHelper helper_;
};
REGISTER_CALCULATOR(ImageFrameToGpuBufferCalculator);

// static
absl::Status ImageFrameToGpuBufferCalculator::GetContract(
    CalculatorContract* cc) {
  cc->Inputs().Index(0).Set<ImageFrame>();
  cc->Outputs().Index(0).Set<GpuBuffer>();
  // Note: we call this method even on platforms where we don't use the helper,
  // to ensure the calculator's contract is the same. In particular, the helper
  // enables support for the legacy side packet, which several graphs still use.
  MP_RETURN_IF_ERROR(GlCalculatorHelper::UpdateContract(cc));
  return absl::OkStatus();
}

absl::Status ImageFrameToGpuBufferCalculator::Open(CalculatorContext* cc) {
  // Inform the framework that we always output at the same timestamp
  // as we receive a packet at.
  cc->SetOffset(TimestampDiff(0));
  MP_RETURN_IF_ERROR(helper_.Open(cc));
  return absl::OkStatus();
}

absl::Status ImageFrameToGpuBufferCalculator::Process(CalculatorContext* cc) {
  auto image_frame = std::const_pointer_cast<ImageFrame>(
      mediapipe::SharedPtrWithPacket<ImageFrame>(
          cc->Inputs().Index(0).Value()));
  auto gpu_buffer = MakePacket<GpuBuffer>(
                        std::make_shared<mediapipe::GpuBufferStorageImageFrame>(
                            std::move(image_frame)))
                        .At(cc->InputTimestamp());
  // Request GPU access to ensure the data is available to the GPU.
  // TODO: have a better way to do this, or defer until later.
  helper_.RunInGlContext([&gpu_buffer] {
    auto view = gpu_buffer.Get<GpuBuffer>().GetReadView<GlTextureView>(0);
  });
  cc->Outputs().Index(0).AddPacket(std::move(gpu_buffer));

  return absl::OkStatus();
}

}  // namespace mediapipe
