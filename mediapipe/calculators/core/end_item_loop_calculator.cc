// Copyright 2024 The MediaPipe Authors.
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

#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "mediapipe/framework/api2/contract.h"
#include "mediapipe/framework/api2/node.h"
#include "mediapipe/framework/api2/packet.h"
#include "mediapipe/framework/api2/port.h"
#include "mediapipe/framework/calculator_context.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/port/ret_check.h"

namespace mediapipe {

using ::mediapipe::api2::AnyType;
using ::mediapipe::api2::Input;
using ::mediapipe::api2::Node;
using ::mediapipe::api2::Output;
using ::mediapipe::api2::PacketBase;
using ::mediapipe::api2::SameType;
using ::mediapipe::api2::TimestampChange;

// Calculator for completing the processing of items loops inside a MediaPipe
// graph. The EndLoopCalculator collects all input packets from ITEM
// input_stream into a collection and upon receiving the flush signal from the
// "BATCH_END" tagged input stream, it emits the aggregated results at the
// original timestamp contained in the "BATCH_END" input stream.
//
// See BeginItemLoopCalculator for a usage example.

class EndItemLoopCalculator : public Node {
 public:
  static constexpr Input<AnyType> kItemIn{"ITEM"};
  static constexpr Input<Timestamp> kBatchEndIn{"BATCH_END"};

  static constexpr Output<SameType<kItemIn>>::Multiple kItemsOut{"ITEM"};

  MEDIAPIPE_NODE_CONTRACT(kItemIn, kBatchEndIn, kItemsOut,
                          TimestampChange::Arbitrary());

  absl::Status Process(CalculatorContext* cc) override {
    if (!kItemIn(cc).IsEmpty()) {
      items_.push_back(kItemIn(cc));
    }

    if (!kBatchEndIn(cc).IsEmpty()) {  // flush signal
      const Timestamp output_ts = kBatchEndIn(cc).Get();

      if (!items_.empty()) {
        RET_CHECK_EQ(items_.size(), kItemsOut(cc).Count())
            << "Number of input items must match number of outputs";
        for (int n = 0; n < items_.size(); ++n) {
          kItemsOut(cc)[n].Send(std::move(items_[n]).At(output_ts));
        }
        items_.clear();
      } else {
        // Propagate timestamp bounds.
        Timestamp next_ts = output_ts.NextAllowedInStream();
        for (int n = 0; n < kItemsOut(cc).Count(); ++n) {
          kItemsOut(cc)[n].SetNextTimestampBound(next_ts);
        }
      }
    }

    return absl::OkStatus();
  }

 private:
  std::vector<PacketBase> items_;
};

MEDIAPIPE_REGISTER_NODE(EndItemLoopCalculator);

}  // namespace mediapipe
