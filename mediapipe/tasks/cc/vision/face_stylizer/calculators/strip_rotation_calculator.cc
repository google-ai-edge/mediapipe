/* Copyright 2023 The MediaPipe Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "mediapipe/framework/api2/node.h"
#include "mediapipe/framework/api2/packet.h"
#include "mediapipe/framework/api2/port.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/rect.pb.h"

namespace mediapipe {
namespace tasks {
namespace {
using ::mediapipe::api2::Input;
using ::mediapipe::api2::Node;
using ::mediapipe::api2::Output;
}  // namespace

// A calculator to strip the rotation information from the NormalizedRect.
class StripRotationCalculator : public Node {
 public:
  static constexpr Input<NormalizedRect> kInNormRect{"NORM_RECT"};
  static constexpr Output<NormalizedRect> kOutNormRect{"NORM_RECT"};
  MEDIAPIPE_NODE_CONTRACT(kInNormRect, kOutNormRect);

  absl::Status Process(CalculatorContext* cc) {
    if (!kInNormRect(cc).IsEmpty()) {
      NormalizedRect rect = kInNormRect(cc).Get();
      rect.clear_rotation();
      kOutNormRect(cc).Send(rect);
    }
    return absl::OkStatus();
  }
};

MEDIAPIPE_REGISTER_NODE(::mediapipe::tasks::StripRotationCalculator);

}  // namespace tasks
}  // namespace mediapipe
