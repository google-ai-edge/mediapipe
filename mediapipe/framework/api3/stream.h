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

#ifndef MEDIAPIPE_FRAMEWORK_API3_STREAM_H_
#define MEDIAPIPE_FRAMEWORK_API3_STREAM_H_

#include <string>
#include <utility>

#include "mediapipe/framework/api3/any.h"
#include "mediapipe/framework/api3/internal/graph_builder.h"

namespace mediapipe::api3 {

// Stream represents graph input stream (`CalculatorGraphConfig::input_stream`)
// or node output stream (`CalculatorGraphConfig::Node::output_stream`).
//
// NOTE: Always valid when returned by graph/node.
// NOTE: Should be passed around by copy.
//
// RECOMMENDATION: when having optional nodes in the graph, you can represent
// their output streams using `std::optional`:
// ```
//   std::optional<Stream<Tensor>> extra_input;
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
class /*ABSL_ATTRIBUTE_VIEW*/ Stream {
 public:
  explicit Stream(builder::Source& source) : source_(&source) {}

  Stream<T> SetName(std::string name) {
    source_->name = std::move(name);
    return Stream(*source_);
  }

  const std::string& Name() const { return source_->name; }

  template <typename CastT>
  Stream<CastT> Cast() {
    static_assert(std::is_same_v<T, Any> || std::is_same_v<CastT, Any>,
                  "Cast is only allowed if either T or CastT is Any.");
    return Stream<CastT>(*source_);
  }

 protected:
  builder::Source* GetBase() const { return source_; }

  // Never nullptr.
  builder::Source* source_;

  template <typename S, typename PayloadT>
  friend class Input;
  template <typename S, typename PayloadT>
  friend class Output;
};

}  // namespace mediapipe::api3

#endif  // MEDIAPIPE_FRAMEWORK_API3_STREAM_H_
