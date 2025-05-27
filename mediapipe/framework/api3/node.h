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

#ifndef MEDIAPIPE_FRAMEWORK_API3_NODE_H_
#define MEDIAPIPE_FRAMEWORK_API3_NODE_H_

#include "absl/strings/string_view.h"

namespace mediapipe::api3 {

// `Node` allows to define, implement and automatically register a MediaPipe
// node. See below patterns to use `Node` depending on various use cases.
//
// +----------------------------------------------------------------------+
// |                                                                      |
// |   1. Define a simple node and implement it as a calculator           |
// |                                                                      |
// +----------------------------------------------------------------------+
//
// foo_node.h:
// ```
//   inline constexpr absl::string_view kFooNodeName = "Foo";
//   struct FooNode : Node<kFooNodeName> {
//     template <typename S>
//     struct Contract {
//       Input<S, int> input{"INPUT"};
//       Output<S, std::string> output{"OUTPUT"};
//       Options<S, FooOptions> options;
//     };
//   };
// ```
// IMPORTANT: `Contract` defines ports (input/output streams, input/output side
//   packets) and options - the name should be exactly `Contract` and it must be
//   a template: `template <typename S>` where `S` stands for "specializier" and
//   will be used to specialize ports for different use cases:
//   `CalculatorContext`, `CalculatorContract`, `Graph`, `Runner`. More details
//   about contract provided in `contract.h`.
//
// OPTIONAL: you can update contract (e.g. offset, input stream handler) for
//   calculator implementations by adding `UpdateContract`as shown below:
//
// foo_node.h
// ```
//   inline constexpr absl::string_view kFooNodeName = "Foo";
//   struct FooNode : Node<kFooName> {
//     template <typename S>
//     struct Contract {
//       // ...
//       static absl::Status UpdateContract(CalculatorContract<FooNode>& cc) {
//         cc.UseService(...);
//         return absl::OkStatus();
//       }
//     };
//   };
// ```
//
// To implement your node as a calculator:
//
// foo_node.cc:
// ```
//   class FooNodeImpl : public Calculator<FooNode, FooNodeImpl> {
//    public:
//     absl::Status Open(CalculatorContext<FooNode>& cc) final ...
//     absl::Status Process(CalculatorContext<FooNode>& cc) final ...
//     absl::Status Close(CalculatorContext<FooNode>& cc) final ...
//   };
// ```
// More details on implementing a calculator provided in `calculator.h`.
//
// +----------------------------------------------------------------------+
// |                                                                      |
// |   2. Define a node that has a template parameter (e.g. often needed  |
// |      for calculators accepting vectors)                              |
// |                                                                      |
// +----------------------------------------------------------------------+
//
// foo_node.h:
// ```
//   template <typename T, typename V>
//   inline constexpr absl::string_view kFooNodeName;
//
//   template <typename T, typename V>
//   struct FooNode : Node<kFooNodeName<T, V>> {
//     template <typename S>
//     struct Contract {
//       Input<S, std::vector<T>> input{"INPUT"};
//       Output<S, std:vector<V>> output{"OUTPUT"};
//     };
//   };
//
//  template <>
//  inline constexpr absl::string_view kFooNodeName<int, float> = "FooIntFloat";
// ```
//
// To implement your node as a calculator:
//
// foo_node.cc:
// ```
//   template <typename T, typename V>
//   class FooNodeImpl : public Calculator<FooNode<T, V>, FooNodeImpl> {
//    public:
//     absl::Status Open(CalculatorContext<FooNode<T, V>>& cc) final ...
//     absl::Status Process(CalculatorContext<FooNode<T, V>>& cc) final ...
//     absl::Status Close(CalculatorContext<FooNode<T, V>>& cc) final ...
//   };
//
//   template class FooNodeImpl<int, float>;
// ```
//
// IMPORTANT: pay attention to node name specialization in foo_node.h and
//   explicit `FooNodeImpl` instantiation for `int` and `float` in foo_node.cc.
//
//
// +----------------------------------------------------------------------+
// |                                                                      |
// |   3. Define multiple nodes that share the same contract (a.k.a       |
// |      split contract)                                                 |
// |                                                                      |
// +----------------------------------------------------------------------+
//
// First, define your external contract.
//
// foo.h:
// ```
//   template <typename S>
//   struct Foo {
//     Input<S, int> input{"INPUT"};
//     Output<S, std::string> output{"OUTPUT"};
//     Options<S, FooOptions> options;
//   };
// ```
//
// OPTIONAL: you can set other contract defaults (e.g. offset, input stream
//   handler) for calculator implementations by adding static `UpdateContract`as
//   shown below:
//
// foo.h
// ```
//   template <typename S>
//   struct Foo {
//     ...
//     template <typename N>
//     static absl::Status UpdateContract(CalculatorContract<N>& cc) {
//       cc.UseService(...);
//       return absl::OkStatus();
//     }
//   };
// ```
//
// Then define your nodes using the same contract available from foo.h:
//
// foo_a_node.h:
// ```
//   inline constexpr absl::string_view kFooANodeName = "FooA";
//   struct FooANode : Node<kFooANodeName> {
//     template <typename S>
//     using Contract = Foo<S>;
//   };
// ```
//
// foo_b_node.h
// ```
//   inline constexpr absl::string_view kFooBNodeName = "FooB";
//   struct FooBNode : Node<kFooBNodeName> {
//     template <typename S>
//     using Contract = Foo<S>;
//   };
// ```
//
// And to implement your nodes as calculators:
//
// foo_a_node.cc:
// ```
//   class FooANodeImpl : public Calculator<FooANode, FooANodeImpl> {
//    public:
//     absl::Status Open(CalculatorContext<FooANode>& cc) final ...
//     absl::Status Process(CalculatorContext<FooANode>& cc) final ...
//     absl::Status Close(CalculatorContext<FooANode>& cc) final ...
//   };
// ```
//
// foo_b_node.cc
// ```
//   class FooBNodeImpl : public Calculator<FooBNode, FooBNodeImpl> {
//    public:
//     absl::Status Open(CalculatorContext<FooBNode>& cc) final ...
//     absl::Status Process(CalculatorContext<FooBNode>& cc) final ...
//     absl::Status Close(CalculatorContext<FooBNode>& cc) final ...
//   };
// ```
template <const absl::string_view& kRegistrationName>
struct Node {
  static constexpr absl::string_view GetRegistrationName() {
    return kRegistrationName;
  }
};

}  // namespace mediapipe::api3

#endif  // MEDIAPIPE_FRAMEWORK_API3_NODE_H_
