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

#ifndef MEDIAPIPE_CALCULATORS_CORE_GET_VECTOR_ITEM_CALCULATOR_H_
#define MEDIAPIPE_CALCULATORS_CORE_GET_VECTOR_ITEM_CALCULATOR_H_

#include <optional>

#include "mediapipe/framework/api2/node.h"
#include "mediapipe/framework/api2/port.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/packet.h"
#include "mediapipe/framework/port/ret_check.h"
#include "mediapipe/framework/port/status.h"

namespace mediapipe {
namespace api2 {

// A calcutlator to return an item from the vector by its index.
//
// Inputs:
//   VECTOR - std::vector<T>
//     Vector to take an item from.
//   INDEX - int
//     Index of the item to return.
//
// Outputs:
//   ITEM - T
//     Item from the vector at given index.
//
// Example config:
//   node {
//     calculator: "Get{SpecificType}VectorItemCalculator"
//     input_stream: "VECTOR:vector"
//     input_stream: "INDEX:index"
//     input_stream: "ITEM:item"
//   }
//
template <typename T>
class GetVectorItemCalculator : public Node {
 public:
  static constexpr Input<std::vector<T>> kIn{"VECTOR"};
  static constexpr Input<int> kIdx{"INDEX"};
  static constexpr Output<T> kOut{"ITEM"};

  MEDIAPIPE_NODE_CONTRACT(kIn, kIdx, kOut);

  absl::Status Process(CalculatorContext* cc) final {
    if (kIn(cc).IsEmpty() || kIdx(cc).IsEmpty()) {
      return absl::OkStatus();
    }

    const std::vector<T>& items = kIn(cc).Get();
    const int idx = kIdx(cc).Get();

    RET_CHECK_LT(idx, items.size());
    kOut(cc).Send(items[idx]);

    return absl::OkStatus();
  }
};

}  // namespace api2
}  // namespace mediapipe

#endif  // MEDIAPIPE_CALCULATORS_CORE_GET_VECTOR_ITEM_CALCULATOR_H_
