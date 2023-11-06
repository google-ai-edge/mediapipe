// Copyright 2021 The MediaPipe Authors.
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

#include "mediapipe/calculators/image/image_clone_calculator.pb.h"
#include "mediapipe/framework/api2/node.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/image.h"
#include "mediapipe/framework/port/status.h"

#if !MEDIAPIPE_DISABLE_GPU
#include "mediapipe/gpu/gl_calculator_helper.h"
#endif  // !MEDIAPIPE_DISABLE_GPU

namespace mediapipe {
namespace api2 {

#if MEDIAPIPE_DISABLE_GPU
// Just a placeholder to not have to depend on mediapipe::GpuBuffer.
using GpuBuffer = AnyType;
#else
using GpuBuffer = mediapipe::GpuBuffer;
#endif  // MEDIAPIPE_DISABLE_GPU

// Clones an input image and makes sure in the output clone the pixel data are
// stored on the target storage (CPU vs GPU) specified in the calculator option.
//
// The clone shares ownership of the input pixel data on the existing storage.
// If the target storage is different from the existing one, then the data is
// further copied there.
//
// Example usage:
// node {
//   calculator: "ImageCloneCalculator"
//   input_stream: "input"
//   output_stream: "output"
//   options: {
//     [mediapipe.ImageCloneCalculatorOptions.ext] {
//       output_on_gpu: true
//     }
//   }
// }
class ImageCloneCalculator : public Node {
 public:
  static constexpr Input<Image> kIn{""};
  static constexpr Output<Image> kOut{""};

  MEDIAPIPE_NODE_CONTRACT(kIn, kOut);

  static absl::Status UpdateContract(CalculatorContract* cc) {
#if MEDIAPIPE_DISABLE_GPU
    if (cc->Options<mediapipe::ImageCloneCalculatorOptions>().output_on_gpu()) {
      return absl::UnimplementedError(
          "GPU processing is disabled in build flags");
    }
#else
    MP_RETURN_IF_ERROR(mediapipe::GlCalculatorHelper::UpdateContract(
        cc, /*request_gpu_as_optional=*/true));
#endif  // MEDIAPIPE_DISABLE_GPU
    return absl::OkStatus();
  }

  absl::Status Open(CalculatorContext* cc) override {
    const auto& options = cc->Options<mediapipe::ImageCloneCalculatorOptions>();
    output_on_gpu_ = options.output_on_gpu();
    return absl::OkStatus();
  }

  absl::Status Process(CalculatorContext* cc) override {
    std::unique_ptr<Image> output;
    const auto& input = *kIn(cc);
    bool input_on_gpu = input.UsesGpu();
    if (input_on_gpu) {
#if !MEDIAPIPE_DISABLE_GPU
      // Create an output Image that co-owns the underlying texture buffer as
      // the input Image.
      output = std::make_unique<Image>(input.GetGpuBuffer());
#endif  // !MEDIAPIPE_DISABLE_GPU
    } else {
      // Make a copy of the input packet to co-own the input Image.
      mediapipe::Packet* packet_copy_ptr =
          new mediapipe::Packet(kIn(cc).packet());
      // Create an output Image that (co-)owns a new ImageFrame that points to
      // the same pixel data as the input Image and also owns the packet
      // copy. As a result, the output Image indirectly co-owns the input
      // Image. This ensures a correct life span of the shared pixel data.
      output = std::make_unique<Image>(std::make_unique<mediapipe::ImageFrame>(
          input.image_format(), input.width(), input.height(), input.step(),
          const_cast<uint8_t*>(input.GetImageFrameSharedPtr()->PixelData()),
          [packet_copy_ptr](uint8_t*) { delete packet_copy_ptr; }));
    }

    if (output_on_gpu_ && !input_on_gpu) {
#if !MEDIAPIPE_DISABLE_GPU
      if (!gpu_initialized_) {
        MP_RETURN_IF_ERROR(gpu_helper_.Open(cc));
        gpu_initialized_ = true;
      }
      gpu_helper_.RunInGlContext([&output]() { output->ConvertToGpu(); });
#endif  // !MEDIAPIPE_DISABLE_GPU
    } else if (!output_on_gpu_ && input_on_gpu) {
      output->ConvertToCpu();
    }
    kOut(cc).Send(std::move(output));

    return absl::OkStatus();
  }

 private:
  bool output_on_gpu_;
#if !MEDIAPIPE_DISABLE_GPU
  mediapipe::GlCalculatorHelper gpu_helper_;
  bool gpu_initialized_ = false;
#endif  // !MEDIAPIPE_DISABLE_GPU
};
MEDIAPIPE_REGISTER_NODE(ImageCloneCalculator);

}  // namespace api2
}  // namespace mediapipe
