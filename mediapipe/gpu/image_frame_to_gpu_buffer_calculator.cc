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

#include "mediapipe/framework/api2/node.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/image_frame.h"
#include "mediapipe/framework/port/status.h"
#include "mediapipe/gpu/gl_calculator_helper.h"

namespace mediapipe {
namespace api2 {

class ImageFrameToGpuBufferCalculator
    : public RegisteredNode<ImageFrameToGpuBufferCalculator> {
 public:
  static constexpr Input<ImageFrame> kIn{""};
  static constexpr Output<GpuBuffer> kOut{""};

  MEDIAPIPE_NODE_INTERFACE(ImageFrameToGpuBufferCalculator, kIn, kOut);

  static absl::Status UpdateContract(CalculatorContract* cc);

  absl::Status Open(CalculatorContext* cc) override;
  absl::Status Process(CalculatorContext* cc) override;

 private:
  GlCalculatorHelper helper_;
};

// static
absl::Status ImageFrameToGpuBufferCalculator::UpdateContract(
    CalculatorContract* cc) {
  // Note: we call this method even on platforms where we don't use the helper,
  // to ensure the calculator's contract is the same. In particular, the helper
  // enables support for the legacy side packet, which several graphs still use.
  return GlCalculatorHelper::UpdateContract(cc);
}

absl::Status ImageFrameToGpuBufferCalculator::Open(CalculatorContext* cc) {
  MP_RETURN_IF_ERROR(helper_.Open(cc));
  return absl::OkStatus();
}

absl::Status ImageFrameToGpuBufferCalculator::Process(CalculatorContext* cc) {
  auto image_frame = std::const_pointer_cast<ImageFrame>(
      mediapipe::SharedPtrWithPacket<ImageFrame>(kIn(cc).packet()));
  auto gpu_buffer = api2::MakePacket<GpuBuffer>(
                        std::make_shared<mediapipe::GpuBufferStorageImageFrame>(
                            std::move(image_frame)))
                        .At(cc->InputTimestamp());
  // This calculator's behavior has been to do the texture upload eagerly, and
  // some graphs may rely on running this on a separate GL context to avoid
  // blocking another context with the read operation. So let's request GPU
  // access here to ensure that the behavior stays the same.
  // TODO: have a better way to do this, or defer until later.
  helper_.RunInGlContext(
      [&gpu_buffer] { auto view = gpu_buffer->GetReadView<GlTextureView>(0); });
  kOut(cc).Send(std::move(gpu_buffer));
  return absl::OkStatus();
}

}  // namespace api2
}  // namespace mediapipe
