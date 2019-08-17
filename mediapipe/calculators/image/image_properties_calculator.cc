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

#if defined(__ANDROID__) || (defined(__APPLE__) && !TARGET_OS_OSX)
#include "mediapipe/gpu/gpu_buffer.h"
#endif  // __ANDROID__ or iOS

namespace mediapipe {

// Extracts image properties from the input image and outputs the properties.
// Currently only supports image size.
// Input:
//   One of the following:
//   IMAGE: An ImageFrame
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
class ImagePropertiesCalculator : public CalculatorBase {
 public:
  static ::mediapipe::Status GetContract(CalculatorContract* cc) {
    RET_CHECK(cc->Inputs().HasTag("IMAGE") ^ cc->Inputs().HasTag("IMAGE_GPU"));
    if (cc->Inputs().HasTag("IMAGE")) {
      cc->Inputs().Tag("IMAGE").Set<ImageFrame>();
    }
#if defined(__ANDROID__) || (defined(__APPLE__) && !TARGET_OS_OSX)
    if (cc->Inputs().HasTag("IMAGE_GPU")) {
      cc->Inputs().Tag("IMAGE_GPU").Set<::mediapipe::GpuBuffer>();
    }
#endif  // __ANDROID__ or iOS

    if (cc->Outputs().HasTag("SIZE")) {
      cc->Outputs().Tag("SIZE").Set<std::pair<int, int>>();
    }

    return ::mediapipe::OkStatus();
  }

  ::mediapipe::Status Open(CalculatorContext* cc) override {
    cc->SetOffset(TimestampDiff(0));
    return ::mediapipe::OkStatus();
  }

  ::mediapipe::Status Process(CalculatorContext* cc) override {
    int width;
    int height;

    if (cc->Inputs().HasTag("IMAGE") && !cc->Inputs().Tag("IMAGE").IsEmpty()) {
      const auto& image = cc->Inputs().Tag("IMAGE").Get<ImageFrame>();
      width = image.Width();
      height = image.Height();
    }
#if defined(__ANDROID__) || (defined(__APPLE__) && !TARGET_OS_OSX)
    if (cc->Inputs().HasTag("IMAGE_GPU") &&
        !cc->Inputs().Tag("IMAGE_GPU").IsEmpty()) {
      const auto& image =
          cc->Inputs().Tag("IMAGE_GPU").Get<mediapipe::GpuBuffer>();
      width = image.width();
      height = image.height();
    }
#endif  // __ANDROID__ or iOS

    cc->Outputs().Tag("SIZE").AddPacket(
        MakePacket<std::pair<int, int>>(width, height)
            .At(cc->InputTimestamp()));

    return ::mediapipe::OkStatus();
  }
};
REGISTER_CALCULATOR(ImagePropertiesCalculator);

}  // namespace mediapipe
