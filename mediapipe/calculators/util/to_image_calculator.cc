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

#include "mediapipe/calculators/util/to_image_calculator.h"

#include <memory>
#include <utility>

#include "absl/functional/overload.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "mediapipe/framework/api3/calculator.h"
#include "mediapipe/framework/api3/calculator_context.h"
#include "mediapipe/framework/api3/calculator_contract.h"
#include "mediapipe/framework/api3/packet.h"
#include "mediapipe/framework/formats/image.h"
#include "mediapipe/framework/formats/image_frame.h"
#include "mediapipe/framework/port/status_macros.h"

namespace mediapipe::api3 {

using GpuBuffer = ToImageNode::GpuBuffer;

class ToImageNodeImpl
    : public mediapipe::api3::Calculator<ToImageNode, ToImageNodeImpl> {
 public:
  static absl::Status UpdateContract(CalculatorContract<ToImageNode>& cc);
  absl::Status Process(CalculatorContext<ToImageNode>& cc) override;

 private:
  absl::StatusOr<Packet<Image>> GetInputImage(
      CalculatorContext<ToImageNode>& cc);
};

absl::Status ToImageNodeImpl::UpdateContract(
    CalculatorContract<ToImageNode>& cc) {
  int num_inputs = static_cast<int>(cc.in_image.IsConnected()) +
                   static_cast<int>(cc.in_image_cpu.IsConnected()) +
                   static_cast<int>(cc.in_image_gpu.IsConnected());
  if (num_inputs != 1) {
    return absl::InternalError("Cannot have multiple inputs.");
  }
  return absl::OkStatus();
}

absl::Status ToImageNodeImpl::Process(CalculatorContext<ToImageNode>& cc) {
  MP_ASSIGN_OR_RETURN(auto output, GetInputImage(cc));
  cc.out_image.Send(std::move(output).At(cc.InputTimestamp()));
  return absl::OkStatus();
}

namespace {

// Wrap ImageFrameSharedPtr; shallow copy.
absl::StatusOr<Packet<Image>> FromImageFrame(Packet<ImageFrame> packet) {
  MP_ASSIGN_OR_RETURN(auto shared_ptr, packet.Share());
  return MakePacket<Image, std::shared_ptr<mediapipe::ImageFrame>>(
      std::const_pointer_cast<mediapipe::ImageFrame>(std::move(shared_ptr)));
}

// Wrap texture pointer; shallow copy.
absl::StatusOr<Packet<Image>> FromGpuBuffer(Packet<GpuBuffer> packet) {
#if !MEDIAPIPE_DISABLE_GPU
  const GpuBuffer& buffer = packet.GetOrDie();
  return MakePacket<Image, const GpuBuffer&>(buffer);
#else
  return absl::UnimplementedError("GPU processing is disabled in build flags");
#endif  // !MEDIAPIPE_DISABLE_GPU
}

}  // namespace

absl::StatusOr<Packet<Image>> ToImageNodeImpl::GetInputImage(
    CalculatorContext<ToImageNode>& cc) {
  if (cc.in_image.IsConnected()) {
    return cc.in_image.VisitAsPacketOrDie(absl::Overload(
        [&](Packet<mediapipe::Image> packet) {
          return absl::StatusOr<Packet<Image>>(std::move(packet));
        },
        [&](Packet<mediapipe::ImageFrame> packet) {
          return FromImageFrame(std::move(packet));
        },
        [&](Packet<GpuBuffer> packet) {
          return FromGpuBuffer(std::move(packet));
        }));
  } else if (cc.in_image_cpu.IsConnected()) {
    return FromImageFrame(cc.in_image_cpu.Packet());
  } else if (cc.in_image_gpu.IsConnected()) {
    return FromGpuBuffer(cc.in_image_gpu.Packet());
  }
  return absl::InvalidArgumentError("No input found.");
}

}  // namespace mediapipe::api3
