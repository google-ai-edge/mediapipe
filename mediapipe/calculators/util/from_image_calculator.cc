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

#include "mediapipe/calculators/util/from_image_calculator.h"

#include <cstdint>
#include <memory>
#include <utility>

#include "absl/status/status.h"
#include "mediapipe/framework/api3/calculator.h"
#include "mediapipe/framework/api3/calculator_context.h"
#include "mediapipe/framework/api3/calculator_contract.h"
#include "mediapipe/framework/api3/packet.h"
#include "mediapipe/framework/formats/image.h"
#include "mediapipe/framework/formats/image_frame.h"
#include "mediapipe/framework/port/status_macros.h"

#if !MEDIAPIPE_DISABLE_GPU
#include "mediapipe/gpu/gl_calculator_helper.h"
#include "mediapipe/gpu/gpu_service.h"
#endif  // !MEDIAPIPE_DISABLE_GPU

namespace mediapipe::api3 {

class FromImageNodeImpl : public Calculator<FromImageNode, FromImageNodeImpl> {
 public:
  static absl::Status UpdateContract(CalculatorContract<FromImageNode>& cc);
  absl::Status Open(CalculatorContext<FromImageNode>& cc) override;
  absl::Status Process(CalculatorContext<FromImageNode>& cc) override;

 private:
  bool check_image_source_ = false;
  bool gpu_output_ = false;
#if !MEDIAPIPE_DISABLE_GPU
  mediapipe::GlCalculatorHelper gpu_helper_;
#endif  // !MEDIAPIPE_DISABLE_GPU
};

absl::Status FromImageNodeImpl::UpdateContract(
    CalculatorContract<FromImageNode>& cc) {
  if (cc.out_image_cpu.IsConnected() && cc.out_image_gpu.IsConnected()) {
    return absl::InternalError("Cannot have multiple outputs.");
  }
  if (cc.out_image_gpu.IsConnected()) {
#if MEDIAPIPE_DISABLE_GPU
    return absl::InternalError("GPU is disabled. Cannot use IMAGE_GPU stream.");
#else
    cc.UseService(kGpuService);
#endif
  }
  return absl::OkStatus();
}

absl::Status FromImageNodeImpl::Open(CalculatorContext<FromImageNode>& cc) {
  gpu_output_ = cc.out_image_gpu.IsConnected();
  check_image_source_ = cc.out_source_on_gpu.IsConnected();
  if (gpu_output_) {
#if !MEDIAPIPE_DISABLE_GPU
    MP_RETURN_IF_ERROR(gpu_helper_.Open(&cc.GetGenericContext()));
#endif  // !MEDIAPIPE_DISABLE_GPU
  }

  return absl::OkStatus();
}

absl::Status FromImageNodeImpl::Process(CalculatorContext<FromImageNode>& cc) {
  if (check_image_source_) {
    const Image& input = cc.in_image.GetOrDie();
    cc.out_source_on_gpu.Send(input.UsesGpu());
  }

  if (gpu_output_) {
#if !MEDIAPIPE_DISABLE_GPU
    MP_RETURN_IF_ERROR(gpu_helper_.RunInGlContext([&cc]() -> absl::Status {
      const Image& input = cc.in_image.GetOrDie();
      // Unwrap texture pointer; shallow copy.
      cc.out_image_gpu.Send(input.GetGpuBuffer());
      return absl::OkStatus();
    }));
#endif  // !MEDIAPIPE_DISABLE_GPU
  } else {
    // The input Image.
    const Image& input = cc.in_image.GetOrDie();
    // Make a copy of the input packet to co-own the input Image.
    auto* packet_copy_ptr =
        new mediapipe::api3::Packet<Image>(cc.in_image.Packet());
    // Create an output ImageFrame that points to the same pixel data as the
    // input Image and also owns the packet copy. As a result, the output
    // ImageFrame indirectly co-owns the input Image. This ensures a correct
    // life span of the shared pixel data.
    auto output = std::make_unique<mediapipe::ImageFrame>(
        input.image_format(), input.width(), input.height(), input.step(),
        const_cast<uint8_t*>(input.GetImageFrameSharedPtr()->PixelData()),
        [packet_copy_ptr](uint8_t*) { delete packet_copy_ptr; });
    cc.out_image_cpu.Send(std::move(output));
  }

  return absl::OkStatus();
}

}  // namespace mediapipe::api3
