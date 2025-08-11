// Copyright 2025 The MediaPipe Authors.
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

#ifndef MEDIAPIPE_CALCULATORS_CORE_PASS_THROUGH_CALCULATOR_H_
#define MEDIAPIPE_CALCULATORS_CORE_PASS_THROUGH_CALCULATOR_H_

#include "absl/strings/string_view.h"
#include "mediapipe/framework/api3/any.h"
#include "mediapipe/framework/api3/contract.h"
#include "mediapipe/framework/api3/node.h"

namespace mediapipe::api3 {

inline constexpr absl::string_view kPassThroughNodeName =
    "PassThroughCalculator";

// Calculator that simply passes its input Packets through unchanged.
//
// Indices for corresponding inputs and outputs must match.
struct PassThroughNode : Node<kPassThroughNodeName> {
  template <typename S>
  struct Contract {
    Repeated<Input<S, Any>> in{""};
    Repeated<Output<S, Any>> out{""};
  };
};

}  // namespace mediapipe::api3

#endif  // MEDIAPIPE_CALCULATORS_CORE_PASS_THROUGH_CALCULATOR_H_
