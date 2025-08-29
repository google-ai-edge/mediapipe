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

#ifndef MEDIAPIPE_FRAMEWORK_API3_ONE_OF_H_
#define MEDIAPIPE_FRAMEWORK_API3_ONE_OF_H_

#include <type_traits>

namespace mediapipe::api3 {

// OneOf type can be useful for nodes willing to support multiple input types.
//
// For example:
// ```
//   struct MultiTypeInputNode : Node<...> {
//     template <typename S>
//     struct Contract {
//       Input<S, OneOf<int, float>> input{"IN"};
//       ...
//     };
//   };
// ```
//
// This node interface allows clients to send either `int` or `float` packets
// into the calculator.
//
// In node calculator implementations, you can check the underlying type as
// following:
// ```
//   cc.input.Has<int>();
//   cc.input.GetOrDie<int>();
//   ...
//   cc.input.VisitOrDie(
//       [](const Type1& value) { ... },
//       ...
//       [](const TypeN& value) { ... }
//   );
// ```
//
// Graph construction is the same as for regular types.
// ```
//   Stream<int> in = ...;
//   node.input.Set(in);
//
//   Stream<float> in = ...;
//   node.input.Set(in);
// ```
template <typename... T>
struct OneOf {};

template <class T>
struct IsOneOf : std::false_type {};

template <class... T>
struct IsOneOf<OneOf<T...>> : std::true_type {};

}  // namespace mediapipe::api3

#endif  // MEDIAPIPE_FRAMEWORK_API3_ONE_OF_H_
