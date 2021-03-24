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

#ifndef MEDIAPIPE_CALCULATORS_CORE_BEGIN_LOOP_CALCULATOR_H_
#define MEDIAPIPE_CALCULATORS_CORE_BEGIN_LOOP_CALCULATOR_H_

#include "absl/memory/memory.h"
#include "mediapipe/framework/calculator_context.h"
#include "mediapipe/framework/calculator_contract.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/collection_item_id.h"
#include "mediapipe/framework/packet.h"
#include "mediapipe/framework/port/integral_types.h"
#include "mediapipe/framework/port/ret_check.h"
#include "mediapipe/framework/port/status.h"

namespace mediapipe {

// Calculator for implementing loops on iterable collections inside a MediaPipe
// graph.
//
// It is designed to be used like:
//
// node {
//   calculator:    "BeginLoopWithIterableCalculator"
//   input_stream:  "ITERABLE:input_iterable"      # IterableT @ext_ts
//   output_stream: "ITEM:input_element"           # ItemT     @loop_internal_ts
//   output_stream: "BATCH_END:ext_ts"             # Timestamp @loop_internal_ts
// }
//
// node {
//   calculator:    "ElementToBlaConverterSubgraph"
//   input_stream:  "ITEM:input_to_loop_body"      # ItemT     @loop_internal_ts
//   output_stream: "BLA:output_of_loop_body"      # ItemU     @loop_internal_ts
// }
//
// node {
//   calculator:    "EndLoopWithOutputCalculator"
//   input_stream:  "ITEM:output_of_loop_body"     # ItemU     @loop_internal_ts
//   input_stream:  "BATCH_END:ext_ts"             # Timestamp @loop_internal_ts
//   output_stream: "OUTPUT:aggregated_result"     # IterableU @ext_ts
// }
//
// Input streams tagged with "CLONE" are cloned to the corresponding output
// streams at loop timestamps. This ensures that a MediaPipe graph or sub-graph
// can run multiple times, once per element in the "ITERABLE" for each pakcet
// clone of the packets in the "CLONE" input streams.
template <typename IterableT>
class BeginLoopCalculator : public CalculatorBase {
  using ItemT = typename IterableT::value_type;

 public:
  static absl::Status GetContract(CalculatorContract* cc) {
    // The below enables processing of timestamp bound updates, and that enables
    // correct timestamp propagation by the companion EndLoopCalculator.
    //
    // For instance, Process() function will be still invoked even if upstream
    // calculator has updated timestamp bound for ITERABLE input instead of
    // providing actual value.
    cc->SetProcessTimestampBounds(true);

    // A non-empty packet in the optional "TICK" input stream wakes up the
    // calculator.
    // DEPRECATED as timestamp bound updates are processed by default in this
    // calculator.
    if (cc->Inputs().HasTag("TICK")) {
      cc->Inputs().Tag("TICK").SetAny();
    }

    // An iterable collection in the input stream.
    RET_CHECK(cc->Inputs().HasTag("ITERABLE"));
    cc->Inputs().Tag("ITERABLE").Set<IterableT>();

    // An element from the collection.
    RET_CHECK(cc->Outputs().HasTag("ITEM"));
    cc->Outputs().Tag("ITEM").Set<ItemT>();

    RET_CHECK(cc->Outputs().HasTag("BATCH_END"));
    cc->Outputs()
        .Tag("BATCH_END")
        .Set<Timestamp>(
            // A flush signal to the corresponding EndLoopCalculator for it to
            // emit the aggregated result with the timestamp contained in this
            // flush signal packet.
        );

    // Input streams tagged with "CLONE" are cloned to the corresponding
    // "CLONE" output streams at loop timestamps.
    RET_CHECK(cc->Inputs().NumEntries("CLONE") ==
              cc->Outputs().NumEntries("CLONE"));
    if (cc->Inputs().NumEntries("CLONE") > 0) {
      for (int i = 0; i < cc->Inputs().NumEntries("CLONE"); ++i) {
        cc->Inputs().Get("CLONE", i).SetAny();
        cc->Outputs().Get("CLONE", i).SetSameAs(&cc->Inputs().Get("CLONE", i));
      }
    }

    return absl::OkStatus();
  }

  absl::Status Process(CalculatorContext* cc) final {
    Timestamp last_timestamp = loop_internal_timestamp_;
    if (!cc->Inputs().Tag("ITERABLE").IsEmpty()) {
      const IterableT& collection =
          cc->Inputs().Tag("ITERABLE").template Get<IterableT>();
      for (const auto& item : collection) {
        cc->Outputs().Tag("ITEM").AddPacket(
            MakePacket<ItemT>(item).At(loop_internal_timestamp_));
        ForwardClonePackets(cc, loop_internal_timestamp_);
        ++loop_internal_timestamp_;
      }
    }

    // The collection was empty and nothing was processed.
    if (last_timestamp == loop_internal_timestamp_) {
      // Increment loop_internal_timestamp_ because it is used up now.
      ++loop_internal_timestamp_;
      for (auto it = cc->Outputs().begin(); it < cc->Outputs().end(); ++it) {
        it->SetNextTimestampBound(loop_internal_timestamp_);
      }
    }

    // The for loop processing the input collection already incremented
    // loop_internal_timestamp_. To emit BATCH_END packet along the last
    // non-BATCH_END packet, decrement by one.
    cc->Outputs()
        .Tag("BATCH_END")
        .AddPacket(MakePacket<Timestamp>(cc->InputTimestamp())
                       .At(Timestamp(loop_internal_timestamp_ - 1)));

    return absl::OkStatus();
  }

 private:
  void ForwardClonePackets(CalculatorContext* cc, Timestamp output_timestamp) {
    if (cc->Inputs().NumEntries("CLONE") > 0) {
      for (int i = 0; i < cc->Inputs().NumEntries("CLONE"); ++i) {
        if (!cc->Inputs().Get("CLONE", i).IsEmpty()) {
          auto input_packet = cc->Inputs().Get("CLONE", i).Value();
          cc->Outputs()
              .Get("CLONE", i)
              .AddPacket(std::move(input_packet).At(output_timestamp));
        }
      }
    }
  }

  // Fake timestamps generated per element in collection.
  Timestamp loop_internal_timestamp_ = Timestamp(0);
};

}  // namespace mediapipe

#endif  // MEDIAPIPE_CALCULATORS_CORE_BEGIN_LOOP_CALCULATOR_H_
