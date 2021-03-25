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

#include "mediapipe/framework/api2/node.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/port/ret_check.h"

namespace mediapipe {
namespace api2 {

// A Calculator that returns 0 if INPUT is 0, and 1 otherwise.
class NonZeroCalculator : public Node {
 public:
  static constexpr Input<int>::SideFallback kIn{"INPUT"};
  static constexpr Output<int> kOut{"OUTPUT"};

  MEDIAPIPE_NODE_CONTRACT(kIn, kOut);

  absl::Status Process(CalculatorContext* cc) final {
    if (!kIn(cc).IsEmpty()) {
      auto output = std::make_unique<int>((*kIn(cc) != 0) ? 1 : 0);
      kOut(cc).Send(std::move(output));
    }
    return absl::OkStatus();
  }
};

MEDIAPIPE_REGISTER_NODE(NonZeroCalculator);

}  // namespace api2
}  // namespace mediapipe
