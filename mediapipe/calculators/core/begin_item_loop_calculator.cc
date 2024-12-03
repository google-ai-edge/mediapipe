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

#include <algorithm>

#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "mediapipe/framework/api2/contract.h"
#include "mediapipe/framework/api2/node.h"
#include "mediapipe/framework/api2/packet.h"
#include "mediapipe/framework/api2/port.h"
#include "mediapipe/framework/calculator_context.h"
#include "mediapipe/framework/calculator_contract.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/port/ret_check.h"

namespace mediapipe {

using ::mediapipe::api2::AnyType;
using ::mediapipe::api2::Input;
using ::mediapipe::api2::Node;
using ::mediapipe::api2::Output;
using ::mediapipe::api2::SameType;

// Calculator for implementing loops on fixed-sized sets of items inside a
// MediaPipe graph. Given a set of inputs of type InputT, the following graph
// transforms all inputs to outputs of type OutputIterT by applying
// InputToOutputConverter to every element (in this example 2 elements):
//
// node {                                        # Type        @timestamp
//   calculator:    "BeginItemLoopCalculator"
//   input_stream:  "ITEM:0:input_item_0"        # InputT      @input_ts
//   input_stream:  "ITEM:1:input_item_1"        # InputT      @input_ts
//   input_stream:  "CLONE:extra_input"          # ExtraT      @extra_ts
//   output_stream: "ITEM:input_iterator"        # InputT      @loop_internal_ts
//   output_stream: "CLONE:cloned_extra_input"   # ExtraT      @loop_internal_ts
//   output_stream: "BATCH_END:batch_end_ts"     # Timestamp   @loop_internal_ts
// }
//
// node {
//   calculator:    "InputToOutputConverter"
//   input_stream:  "INPUT:input_iterator"       # InputT      @loop_internal_ts
//   input_stream:  "EXTRA:cloned_extra_input"   # ExtraT      @loop_internal_ts
//   output_stream: "OUTPUT:output_iterator"     # OutputT     @loop_internal_ts
// }
//
// node {
//   calculator:    "EndItemLoopCalculator"
//   input_stream:  "ITEM:output_iterator"       # OutputT     @loop_internal_ts
//   input_stream:  "BATCH_END:batch_end_ts"     # Timestamp   @loop_internal_ts
//   output_stream: "ITEM:0:output_item_0"       # OutputT     @input_ts
//   output_stream: "ITEM:1:output_item_1"       # OutputT     @input_ts
// }
//
// The resulting output items have the same timestamp as the input items.
// The output packets of this calculator are part of the loop body and have
// loop-internal timestamps that are unrelated to the input iterator timestamp.
//
// It is not possible to mix empty and non-empty ITEM packets. If one input ITEM
// packet is set, they all must be set.
//
// Input streams tagged with "CLONE" are cloned to the corresponding output
// streams at loop-internal timestamps. This ensures that a MediaPipe graph or
// sub-graph can run multiple times, once per input item for each packet clone
// of the packets in the "CLONE" input streams. Think of CLONEd inputs as
// loop-wide constants.
//
// Compared to Begin/EndLoopCalculator, this calculator has several advantages:
//   - It works for all item types without instantiating type-specific variants.
//   - It does not require (de-)vectorization of items.
//   - It does not have restrictions to copyable types or consumable packets.
// However, this calculator requires you to know an upper bound for the number
// of items. Use Begin/EndLoopCalculator only if items are already vectorize.

class BeginItemLoopCalculator : public Node {
 public:
  static constexpr Input<AnyType>::Multiple kItemsIn{"ITEM"};
  static constexpr Input<AnyType>::Multiple kCloneIn{"CLONE"};

  static constexpr Output<Timestamp> kBatchEndOut{"BATCH_END"};
  static constexpr Output<SameType<kItemsIn>> kItemOut{"ITEM"};
  static constexpr Output<AnyType>::Multiple kCloneOut{"CLONE"};

  MEDIAPIPE_NODE_CONTRACT(kItemsIn, kCloneIn, kBatchEndOut, kItemOut, kCloneOut,
                          mediapipe::api2::TimestampChange::Arbitrary());

  static absl::Status UpdateContract(CalculatorContract* cc) {
    // The below enables processing of timestamp bound updates, and that enables
    // correct timestamp propagation by the companion EndItemLoopCalculator.
    //
    // For instance, Process() function will be still invoked even if upstream
    // calculator has updated timestamp bound for all ITEM inputs instead of
    // providing actual value.
    cc->SetProcessTimestampBounds(true);

    RET_CHECK_GT(kItemsIn(cc).Count(), 0)
        << "Must have at least one ITEM input";
    RET_CHECK_EQ(kCloneIn(cc).Count(), kCloneOut(cc).Count())
        << "Number of CLONE inputs and outputs must match";
    for (int n = 0; n < kCloneOut(cc).Count(); ++n) {
      cc->Outputs().Get("CLONE", n).SetSameAs(&cc->Inputs().Get("CLONE", n));
    }
    return absl::OkStatus();
  }

  absl::Status Process(CalculatorContext* cc) override {
    constexpr absl::string_view kMixEmptyError =
        "Cannot mix non-empty input ITEMs with empty input ITEMs";
    if (!kItemsIn(cc)[0].IsEmpty()) {
      for (const auto& item : kItemsIn(cc)) {
        RET_CHECK(!item.IsEmpty()) << kMixEmptyError;
        kItemOut(cc).Send(item.At(loop_internal_timestamp_));
        ForwardClonePackets(cc, loop_internal_timestamp_);
        ++loop_internal_timestamp_;
      }
    } else {
      // Items may be empty in case of a timestamp bounds update. But then they
      // must all be empty.
      RET_CHECK(std::all_of(kItemsIn(cc).begin(), kItemsIn(cc).end(),
                            [](const auto& item) { return item.IsEmpty(); }))
          << kMixEmptyError;

      // Increment loop_internal_timestamp_ because we send BATCH_END below.
      // Otherwise, it could keep using the same timestamp.
      ++loop_internal_timestamp_;
      for (auto it = cc->Outputs().begin(); it < cc->Outputs().end(); ++it) {
        it->SetNextTimestampBound(loop_internal_timestamp_);
      }
    }

    // Send BATCH_END packet along with the last input item.
    kBatchEndOut(cc).Send(api2::MakePacket<Timestamp>(cc->InputTimestamp())
                              .At(loop_internal_timestamp_ - 1));
    return absl::OkStatus();
  }

 private:
  void ForwardClonePackets(CalculatorContext* cc, Timestamp output_timestamp) {
    for (int n = 0; n < kCloneIn(cc).Count(); ++n) {
      kCloneOut(cc)[n].Send(kCloneIn(cc)[n].At(output_timestamp));
    }
  }

  // Fake timestamps generated per element in collection.
  Timestamp loop_internal_timestamp_ = Timestamp(0);
};

MEDIAPIPE_REGISTER_NODE(BeginItemLoopCalculator);

}  // namespace mediapipe
