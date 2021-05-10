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
#include "mediapipe/framework/formats/image.h"
#include "mediapipe/framework/formats/image_frame.h"

#if !MEDIAPIPE_DISABLE_GPU
#include "mediapipe/gpu/gpu_buffer.h"
#endif  // !MEDIAPIPE_DISABLE_GPU

namespace mediapipe {
namespace api2 {

#if MEDIAPIPE_DISABLE_GPU
// Just a placeholder to not have to depend on mediapipe::GpuBuffer.
using GpuBuffer = AnyType;
#else
using GpuBuffer = mediapipe::GpuBuffer;
#endif  // MEDIAPIPE_DISABLE_GPU

// Extracts image properties from the input image and outputs the properties.
// Currently only supports image size.
// Input:
//   One of the following:
//   IMAGE: An Image or ImageFrame (for backward compatibility with existing
//          graphs that use IMAGE for ImageFrame input)
//   IMAGE_CPU: An ImageFrame
//   IMAGE_GPU: A GpuBuffer
//
// Output:
//   SIZE: Size (as a std::pair<int, int>) of the input image.
//
// Example usage:
// node {
//   calculator: "ImagePropertiesCalculator"
//   input_stream: "IMAGE:image"
//   output_stream: "SIZE:size"
// }
class ImagePropertiesCalculator : public Node {
 public:
  static constexpr Input<
      OneOf<mediapipe::Image, mediapipe::ImageFrame>>::Optional kIn{"IMAGE"};
  // IMAGE_CPU, dedicated to ImageFrame input, is only needed in some top-level
  // graphs for the Python Solution APIs to figure out the type of input stream
  // without running into ambiguities from IMAGE.
  // TODO: Remove IMAGE_CPU once Python Solution APIs adopt Image.
  static constexpr Input<mediapipe::ImageFrame>::Optional kInCpu{"IMAGE_CPU"};
  static constexpr Input<GpuBuffer>::Optional kInGpu{"IMAGE_GPU"};
  static constexpr Output<std::pair<int, int>> kOut{"SIZE"};

  MEDIAPIPE_NODE_CONTRACT(kIn, kInCpu, kInGpu, kOut);

  static absl::Status UpdateContract(CalculatorContract* cc) {
    RET_CHECK_EQ(kIn(cc).IsConnected() + kInCpu(cc).IsConnected() +
                     kInGpu(cc).IsConnected(),
                 1)
        << "One and only one of IMAGE, IMAGE_CPU and IMAGE_GPU input is "
           "expected.";

    return absl::OkStatus();
  }

  absl::Status Process(CalculatorContext* cc) override {
    std::pair<int, int> size;

    if (kIn(cc).IsConnected()) {
      kIn(cc).Visit(
          [&size](const mediapipe::Image& value) {
            size.first = value.width();
            size.second = value.height();
          },
          [&size](const mediapipe::ImageFrame& value) {
            size.first = value.Width();
            size.second = value.Height();
          });
    }
    if (kInCpu(cc).IsConnected()) {
      const auto& image = *kInCpu(cc);
      size.first = image.Width();
      size.second = image.Height();
    }
#if !MEDIAPIPE_DISABLE_GPU
    if (kInGpu(cc).IsConnected()) {
      const auto& image = *kInGpu(cc);
      size.first = image.width();
      size.second = image.height();
    }
#endif  // !MEDIAPIPE_DISABLE_GPU

    kOut(cc).Send(size);

    return absl::OkStatus();
  }
};

MEDIAPIPE_REGISTER_NODE(ImagePropertiesCalculator);

}  // namespace api2
}  // namespace mediapipe
