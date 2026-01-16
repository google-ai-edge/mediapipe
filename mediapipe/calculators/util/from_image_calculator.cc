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
constexpr char kSourceOnGpuTag[] = "SOURCE_ON_GPU";
}  // namespace

// A calculator for converting the unified image container into
// legacy MediaPipe datatypes.
//
// Inputs:
//   IMAGE:  An Image containing input image.
//
// Output:
//   One of the following two tags:
//   IMAGE_CPU:  An ImageFrame containing output image.
//   IMAGE_GPU:  A GpuBuffer containing output image.
//
//   SOURCE_ON_GPU: The source Image is stored on GPU or CPU.
//
// Note:
//   Data is automatically transferred to/from the CPU or GPU
//   depending on output type.
//
class FromImageCalculator : public CalculatorBase {
 public:
  FromImageCalculator() = default;
  ~FromImageCalculator() override = default;

  static absl::Status GetContract(CalculatorContract* cc);

  // From Calculator.
  absl::Status Open(CalculatorContext* cc) override;
  absl::Status Process(CalculatorContext* cc) override;
  absl::Status Close(CalculatorContext* cc) override;

 private:
  absl::Status RenderGpu(CalculatorContext* cc);
  absl::Status RenderCpu(CalculatorContext* cc);

  bool check_image_source_ = false;
  bool gpu_output_ = false;
  bool gpu_initialized_ = false;
#if !MEDIAPIPE_DISABLE_GPU
  mediapipe::GlCalculatorHelper gpu_helper_;
#endif  // !MEDIAPIPE_DISABLE_GPU
};
REGISTER_CALCULATOR(FromImageCalculator);

absl::Status FromImageCalculator::GetContract(CalculatorContract* cc) {
  cc->Inputs().Tag(kImageTag).Set<mediapipe::Image>();

  bool gpu_output = false;

  if (cc->Outputs().HasTag(kImageFrameTag) &&
      cc->Outputs().HasTag(kGpuBufferTag)) {
    return absl::InternalError("Cannot have multiple outputs.");
  }

  if (cc->Outputs().HasTag(kGpuBufferTag)) {
#if !MEDIAPIPE_DISABLE_GPU
    cc->Outputs().Tag(kGpuBufferTag).Set<mediapipe::GpuBuffer>();
    gpu_output = true;
#else
    RET_CHECK_FAIL() << "GPU is disabled. Cannot use IMAGE_GPU stream.";
#endif  // !MEDIAPIPE_DISABLE_GPU
  }
  if (cc->Outputs().HasTag(kImageFrameTag)) {
    cc->Outputs().Tag(kImageFrameTag).Set<mediapipe::ImageFrame>();
  }

  if (gpu_output) {
#if !MEDIAPIPE_DISABLE_GPU
    MP_RETURN_IF_ERROR(mediapipe::GlCalculatorHelper::UpdateContract(cc));
#endif  // !MEDIAPIPE_DISABLE_GPU
  }

  if (cc->Outputs().HasTag(kSourceOnGpuTag)) {
    cc->Outputs().Tag(kSourceOnGpuTag).Set<bool>();
  }
  return absl::OkStatus();
}

absl::Status FromImageCalculator::Open(CalculatorContext* cc) {
  cc->SetOffset(TimestampDiff(0));

  if (cc->Outputs().HasTag(kGpuBufferTag)) {
    gpu_output_ = true;
  }
  if (cc->Outputs().HasTag(kSourceOnGpuTag)) {
    check_image_source_ = true;
  }
  if (gpu_output_) {
#if !MEDIAPIPE_DISABLE_GPU
    MP_RETURN_IF_ERROR(gpu_helper_.Open(cc));
#endif
  }  //  !MEDIAPIPE_DISABLE_GPU

  return absl::OkStatus();
}

absl::Status FromImageCalculator::Process(CalculatorContext* cc) {
  if (check_image_source_) {
    auto& input = cc->Inputs().Tag(kImageTag).Get<mediapipe::Image>();
    cc->Outputs()
        .Tag(kSourceOnGpuTag)
        .AddPacket(MakePacket<bool>(input.UsesGpu()).At(cc->InputTimestamp()));
  }

  if (gpu_output_) {
#if !MEDIAPIPE_DISABLE_GPU
    MP_RETURN_IF_ERROR(gpu_helper_.RunInGlContext([&cc]() -> absl::Status {
      auto& input = cc->Inputs().Tag(kImageTag).Get<mediapipe::Image>();
      // Unwrap texture pointer; shallow copy.
      auto output =
          std::make_unique<mediapipe::GpuBuffer>(input.GetGpuBuffer());
      cc->Outputs()
          .Tag(kGpuBufferTag)
          .Add(output.release(), cc->InputTimestamp());
      return absl::OkStatus();
    }));
#endif  // !MEDIAPIPE_DISABLE_GPU
  } else {
    // The input Image.
    auto& input = cc->Inputs().Tag(kImageTag).Get<mediapipe::Image>();
    // Make a copy of the input packet to co-own the input Image.
    Packet* packet_copy_ptr = new Packet(cc->Inputs().Tag(kImageTag).Value());
    // Create an output ImageFrame that points to the same pixel data as the
    // input Image and also owns the packet copy. As a result, the output
    // ImageFrame indirectly co-owns the input Image. This ensures a correct
    // life span of the shared pixel data.
    std::unique_ptr<mediapipe::ImageFrame> output =
        std::make_unique<mediapipe::ImageFrame>(
            input.image_format(), input.width(), input.height(), input.step(),
            const_cast<uint8_t*>(input.GetImageFrameSharedPtr()->PixelData()),
            [packet_copy_ptr](uint8_t*) { delete packet_copy_ptr; });
    cc->Outputs()
        .Tag(kImageFrameTag)
        .Add(output.release(), cc->InputTimestamp());
  }

  return absl::OkStatus();
}

absl::Status FromImageCalculator::Close(CalculatorContext* cc) {
  return absl::OkStatus();
}

}  // namespace mediapipe
