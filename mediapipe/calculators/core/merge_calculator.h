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

#ifndef MEDIAPIPE_CALCULATORS_CORE_MERGE_CALCULATOR_H_
#define MEDIAPIPE_CALCULATORS_CORE_MERGE_CALCULATOR_H_

#include "absl/strings/string_view.h"
#include "mediapipe/framework/api3/any.h"
#include "mediapipe/framework/api3/contract.h"
#include "mediapipe/framework/api3/node.h"

namespace mediapipe::api3 {

inline constexpr absl::string_view kMergeNodeName = "MergeCalculator";

// This calculator takes a set of input streams and combines them into a single
// output stream. The packets from different streams do not need to contain the
// same type. If there are packets arriving at the same time from two or more
// input streams, the packet corresponding to the input stream with the smallest
// index is passed to the output and the rest are ignored.
//
// Example use-case:
// Suppose we have two (or more) different algorithms for detecting shot
// boundaries and we need to merge their packets into a single stream. The
// algorithms may emit shot boundaries at the same time and their output types
// may not be compatible. Subsequent calculators that process the merged stream
// may be interested only in the timestamps of the shot boundary packets and so
// it may not even need to inspect the values stored inside the packets.
//
// Example config:
// node {
//   calculator: "MergeCalculator"
//   input_stream: "shot_info1"
//   input_stream: "shot_info2"
//   input_stream: "shot_info3"
//   output_stream: "merged_shot_infos"
// }
struct MergeNode : Node<kMergeNodeName> {
  template <typename S>
  struct Contract {
    Repeated<Input<S, Any>> in{""};
    Output<S, Any> out{""};
  };
};

}  // namespace mediapipe::api3

#endif  // MEDIAPIPE_CALCULATORS_CORE_MERGE_CALCULATOR_H_
