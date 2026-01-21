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

#ifndef MEDIAPIPE_FRAMEWORK_API3_SIDE_PACKET_H_
#define MEDIAPIPE_FRAMEWORK_API3_SIDE_PACKET_H_

#include <string>
#include <utility>

#include "mediapipe/framework/api3/any.h"
#include "mediapipe/framework/api3/internal/graph_builder.h"

namespace mediapipe::api3 {

// SidePacket represents graph input side packet
// (`CalculatorGraphConfig::input_side_packet`)
// or node output side packet
// (`CalculatorGraphConfig::Node::output_side_packet`).
//
// NOTE: Always valid when returned by graph/node.
// NOTE: Should be passed around by copy.
//
// RECOMMENDATION: when having optional nodes in the graph, represent optional
// side packets using `std::optional`:
// ```
//   std::optional<SidePacket<float>> extra_input;
//   if (generate_extra_input) {
//     ...
//     extra_input = ...;
//   }
//
//   if (extra_input) {
//     ...
//   }
// ```
template <typename T>
class /*ABSL_ATTRIBUTE_VIEW*/ SidePacket {
 public:
  explicit SidePacket(builder::SideSource& side_packet)
      : side_source_(&side_packet) {}

  SidePacket<T> SetName(std::string name) {
    side_source_->name = std::move(name);
    return SidePacket(*side_source_);
  }

  const std::string& Name() const { return side_source_->name; }

  template <typename CastT>
  SidePacket<CastT> Cast() {
    static_assert(std::is_same_v<T, Any> || std::is_same_v<CastT, Any>,
                  "Cast is only allowed if either T or CastT is Any.");
    return SidePacket<CastT>(*side_source_);
  }

 protected:
  builder::SideSource* GetBase() const { return side_source_; }

  // Never nullptr.
  builder::SideSource* side_source_;

  template <typename S, typename PayloadT>
  friend class SideInput;
  template <typename S, typename PayloadT>
  friend class SideOutput;
};

}  // namespace mediapipe::api3

#endif  // MEDIAPIPE_FRAMEWORK_API3_SIDE_PACKET_H_
