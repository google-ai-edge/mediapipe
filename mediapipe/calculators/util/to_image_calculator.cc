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
#include <utility>

#include "absl/status/status.h"
#include "mediapipe/framework/api2/node.h"
#include "mediapipe/framework/api2/packet.h"
#include "mediapipe/framework/api2/port.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/calculator_options.pb.h"
#include "mediapipe/framework/formats/image.h"
#include "mediapipe/framework/formats/image_format.pb.h"
#include "mediapipe/framework/formats/image_frame.h"
#include "mediapipe/framework/port/logging.h"
#include "mediapipe/framework/port/status.h"
#include "mediapipe/framework/port/status_macros.h"
#include "mediapipe/framework/port/vector.h"
#if !MEDIAPIPE_DISABLE_GPU
#include "mediapipe/gpu/gpu_buffer.h"
#endif  // !MEDIAPIPE_DISABLE_GPU
namespace mediapipe {
namespace api2 {

#if MEDIAPIPE_DISABLE_GPU
// Just a placeholder to not have to depend on mediapipe::GpuBuffer.
class Nothing {};
using GpuBuffer = Nothing;
#else
using GpuBuffer = mediapipe::GpuBuffer;
#endif  // MEDIAPIPE_DISABLE_GPU

// A calculator for converting from legacy MediaPipe datatypes into a
// unified image container.
//
// Inputs:
//   One of the following two tags:
//   IMAGE:  An Image, ImageFrame, or GpuBuffer containing input image.
//   IMAGE_CPU:  An ImageFrame containing input image.
//   IMAGE_GPU:  A GpuBuffer containing input image.
//
// Output:
//   IMAGE:  An Image containing output image.
//
// Note:
//   No CPU/GPU conversion is done.
//
class ToImageCalculator : public Node {
 public:
  ToImageCalculator() = default;
  ~ToImageCalculator() override = default;

  static constexpr Input<
      OneOf<mediapipe::Image, mediapipe::ImageFrame, GpuBuffer>>::Optional kIn{
      "IMAGE"};
  static constexpr Input<mediapipe::ImageFrame>::Optional kInCpu{"IMAGE_CPU"};
  static constexpr Input<GpuBuffer>::Optional kInGpu{"IMAGE_GPU"};
  static constexpr Output<mediapipe::Image> kOut{"IMAGE"};
  MEDIAPIPE_NODE_CONTRACT(kIn, kInCpu, kInGpu, kOut);

  static absl::Status UpdateContract(CalculatorContract* cc);

  // From Calculator.
  absl::Status Process(CalculatorContext* cc) override;
  absl::Status Close(CalculatorContext* cc) override;

 private:
  absl::StatusOr<Packet<Image>> GetInputImage(CalculatorContext* cc);
};
MEDIAPIPE_REGISTER_NODE(ToImageCalculator);

absl::Status ToImageCalculator::UpdateContract(CalculatorContract* cc) {
  int num_inputs = static_cast<int>(kIn(cc).IsConnected()) +
                   static_cast<int>(kInCpu(cc).IsConnected()) +
                   static_cast<int>(kInGpu(cc).IsConnected());
  if (num_inputs != 1) {
    return absl::InternalError("Cannot have multiple inputs.");
  }

  return absl::OkStatus();
}

absl::Status ToImageCalculator::Process(CalculatorContext* cc) {
  MP_ASSIGN_OR_RETURN(auto output, GetInputImage(cc));
  kOut(cc).Send(output.At(cc->InputTimestamp()));
  return absl::OkStatus();
}

absl::Status ToImageCalculator::Close(CalculatorContext* cc) {
  return absl::OkStatus();
}

// Wrap ImageFrameSharedPtr; shallow copy.
absl::StatusOr<Packet<Image>> FromImageFrame(Packet<ImageFrame> packet) {
  MP_ASSIGN_OR_RETURN(auto shared_ptr, packet.Share());
  return MakePacket<Image, std::shared_ptr<mediapipe::ImageFrame>>(
      std::const_pointer_cast<mediapipe::ImageFrame>(std::move(shared_ptr)));
}

// Wrap texture pointer; shallow copy.
absl::StatusOr<Packet<Image>> FromGpuBuffer(Packet<GpuBuffer> packet) {
#if !MEDIAPIPE_DISABLE_GPU
  const GpuBuffer& buffer = *packet;
  return MakePacket<Image, const GpuBuffer&>(buffer);
#else
  return absl::UnimplementedError("GPU processing is disabled in build flags");
#endif  // !MEDIAPIPE_DISABLE_GPU
}

absl::StatusOr<Packet<Image>> ToImageCalculator::GetInputImage(
    CalculatorContext* cc) {
  if (kIn(cc).IsConnected()) {
    return kIn(cc).Visit(
        [&](const mediapipe::Image&) {
          return absl::StatusOr<Packet<Image>>(kIn(cc).As<Image>());
        },
        [&](const mediapipe::ImageFrame&) {
          return FromImageFrame(kIn(cc).As<ImageFrame>());
        },
        [&](const GpuBuffer&) {
          return FromGpuBuffer(kIn(cc).As<GpuBuffer>());
        });
  } else if (kInCpu(cc).IsConnected()) {
    return FromImageFrame(kInCpu(cc).As<ImageFrame>());
  } else if (kInGpu(cc).IsConnected()) {
    return FromGpuBuffer(kInGpu(cc).As<GpuBuffer>());
  }
  return absl::InvalidArgumentError("No input found.");
}

}  // namespace api2
}  // namespace mediapipe
