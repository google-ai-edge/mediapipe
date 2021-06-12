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

#include <memory>

#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/calculator_options.pb.h"
#include "mediapipe/framework/formats/image.h"
#include "mediapipe/framework/formats/image_format.pb.h"
#include "mediapipe/framework/formats/image_frame.h"
#include "mediapipe/framework/port/logging.h"
#include "mediapipe/framework/port/status.h"
#include "mediapipe/framework/port/vector.h"

#if !MEDIAPIPE_DISABLE_GPU
#include "mediapipe/gpu/gl_calculator_helper.h"
#endif  // !MEDIAPIPE_DISABLE_GPU

namespace mediapipe {

namespace {
constexpr char kImageFrameTag[] = "IMAGE_CPU";
constexpr char kGpuBufferTag[] = "IMAGE_GPU";
constexpr char kImageTag[] = "IMAGE";
}  // namespace

// A calculator for converting from legacy MediaPipe datatypes into a
// unified image container.
//
// Inputs:
//   One of the following two tags:
//   IMAGE_CPU:  An ImageFrame containing input image.
//   IMAGE_GPU:  A GpuBuffer containing input image.
//
// Output:
//   IMAGE:  An Image containing output image.
//
// Note:
//   No CPU/GPU conversion is done.
//
class ToImageCalculator : public CalculatorBase {
 public:
  ToImageCalculator() = default;
  ~ToImageCalculator() override = default;

  static absl::Status GetContract(CalculatorContract* cc);

  // From Calculator.
  absl::Status Open(CalculatorContext* cc) override;
  absl::Status Process(CalculatorContext* cc) override;
  absl::Status Close(CalculatorContext* cc) override;

 private:
  absl::Status RenderGpu(CalculatorContext* cc);
  absl::Status RenderCpu(CalculatorContext* cc);

  bool gpu_input_ = false;
  bool gpu_initialized_ = false;
#if !MEDIAPIPE_DISABLE_GPU
  mediapipe::GlCalculatorHelper gpu_helper_;
#endif  // !MEDIAPIPE_DISABLE_GPU
};
REGISTER_CALCULATOR(ToImageCalculator);

absl::Status ToImageCalculator::GetContract(CalculatorContract* cc) {
  cc->Outputs().Tag(kImageTag).Set<mediapipe::Image>();

  bool gpu_input = false;

  if (cc->Inputs().HasTag(kImageFrameTag) &&
      cc->Inputs().HasTag(kGpuBufferTag)) {
    return absl::InternalError("Cannot have multiple inputs.");
  }

  if (cc->Inputs().HasTag(kGpuBufferTag)) {
#if !MEDIAPIPE_DISABLE_GPU
    cc->Inputs().Tag(kGpuBufferTag).Set<mediapipe::GpuBuffer>();
    gpu_input = true;
#else
    RET_CHECK_FAIL() << "GPU is disabled. Cannot use IMAGE_GPU stream.";
#endif  // !MEDIAPIPE_DISABLE_GPU
  }
  if (cc->Inputs().HasTag(kImageFrameTag)) {
    cc->Inputs().Tag(kImageFrameTag).Set<mediapipe::ImageFrame>();
  }

  if (gpu_input) {
#if !MEDIAPIPE_DISABLE_GPU
    MP_RETURN_IF_ERROR(mediapipe::GlCalculatorHelper::UpdateContract(cc));
#endif  // !MEDIAPIPE_DISABLE_GPU
  }

  return absl::OkStatus();
}

absl::Status ToImageCalculator::Open(CalculatorContext* cc) {
  cc->SetOffset(TimestampDiff(0));

  if (cc->Inputs().HasTag(kGpuBufferTag)) {
    gpu_input_ = true;
  }

  if (gpu_input_) {
#if !MEDIAPIPE_DISABLE_GPU
    MP_RETURN_IF_ERROR(gpu_helper_.Open(cc));
#endif
  }  //  !MEDIAPIPE_DISABLE_GPU

  return absl::OkStatus();
}

absl::Status ToImageCalculator::Process(CalculatorContext* cc) {
  if (gpu_input_) {
#if !MEDIAPIPE_DISABLE_GPU
    MP_RETURN_IF_ERROR(gpu_helper_.RunInGlContext([&cc]() -> absl::Status {
      auto& input = cc->Inputs().Tag(kGpuBufferTag).Get<mediapipe::GpuBuffer>();
      // Wrap texture pointer; shallow copy.
      auto output = std::make_unique<mediapipe::Image>(input);
      cc->Outputs().Tag(kImageTag).Add(output.release(), cc->InputTimestamp());
      return absl::OkStatus();
    }));
#endif  // !MEDIAPIPE_DISABLE_GPU
  } else {
    // The input ImageFrame.
    auto& input = cc->Inputs().Tag(kImageFrameTag).Get<mediapipe::ImageFrame>();
    // Make a copy of the input packet to co-own the input ImageFrame.
    Packet* packet_copy_ptr =
        new Packet(cc->Inputs().Tag(kImageFrameTag).Value());
    // Create an output Image that (co-)owns a new ImageFrame that points to
    // the same pixel data as the input ImageFrame and also owns the packet
    // copy. As a result, the output Image indirectly co-owns the input
    // ImageFrame. This ensures a correct life span of the shared pixel data.
    std::unique_ptr<mediapipe::Image> output =
        std::make_unique<mediapipe::Image>(
            std::make_shared<mediapipe::ImageFrame>(
                input.Format(), input.Width(), input.Height(),
                input.WidthStep(), const_cast<uint8*>(input.PixelData()),
                [packet_copy_ptr](uint8*) { delete packet_copy_ptr; }));
    cc->Outputs().Tag(kImageTag).Add(output.release(), cc->InputTimestamp());
  }

  return absl::OkStatus();
}

absl::Status ToImageCalculator::Close(CalculatorContext* cc) {
  return absl::OkStatus();
}

}  // namespace mediapipe
