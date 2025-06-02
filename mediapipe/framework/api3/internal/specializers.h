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

#ifndef MEDIAPIPE_FRAMEWORK_API3_INTERNAL_SPECIALIZERS_H_
#define MEDIAPIPE_FRAMEWORK_API3_INTERNAL_SPECIALIZERS_H_

namespace mediapipe::api3 {

// Specializers used by MediaPipe framework internally.
//
// Visible to users, but users are not supposed to use them explicitly.
//
// How specializers work?
//
// Imagine, user defined a node:
// ```
// struct FooNode : Node... {
//   template <typename S>
//   struct Contract {
//     Input<S, int> in{"IN"};
//     Output<S, std::string> out{"OUT"};
//   };
// };
// ```
//
// Now, MediaPipe framework, can specialize node's contract for various
// scenarios differently.
//
// E.g. `FooNode::Contract<ContextSpecializer>` results in:
// - `Input<ContextSpecializer, int> in{"IN"};`
// - `Output<ContextSpecializer, std::string> out{"OUT"};`
//
// Where `in` & `out` are specialized to work with `CalculatorContext` - you
// can access `in` & `out` through `CalculatorContext<FooNode>` object and they
// have particular methods tailored specifically for `CalculatorContext` use
// case:
// - `if (context.in) { ...context.in.GetOrDie()... }`
// - `context.out.Send(...)`
//
// And if `ContractSpecializer` is used by the framework, then `in` & `out` are
// implemented differently, tailored specifically for `CalculatorContract` use
// case.

// Specializer for contract fields used for `CalculatorContract`.
struct ContractSpecializer {};

// Specializer for contract fields used for `CalculatorContext`.
struct ContextSpecializer {};

// Specializer for contract fields used for `Graph`.
struct GraphSpecializer {};

// Specializer for contract fields used for `Graph` packet generators returned
// by `Graph::AddLegacyPacketGenerator<...>()`
struct GraphGeneratorSpecializer {};

// Specializer for contract fields used for `Graph` nodes returned by
// `Graph::AddNode<...>()`
struct GraphNodeSpecializer {};

}  // namespace mediapipe::api3

#endif  // MEDIAPIPE_FRAMEWORK_API3_INTERNAL_SPECIALIZERS_H_
