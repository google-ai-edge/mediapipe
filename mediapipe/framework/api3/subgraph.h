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

#ifndef MEDIAPIPE_FRAMEWORK_API3_SUBGRAPH_H_
#define MEDIAPIPE_FRAMEWORK_API3_SUBGRAPH_H_

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "mediapipe/framework/api2/node.h"
#include "mediapipe/framework/api3/graph.h"
#include "mediapipe/framework/api3/internal/contract_validator.h"
#include "mediapipe/framework/api3/subgraph_context.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/port/status_macros.h"
#include "mediapipe/framework/tool/options_map.h"

namespace mediapipe::api3 {

// Base class for defining MediaPipe subgraphs.
//
// A subgraph allows you to encapsulate a portion of a MediaPipe graph into a
// single node-like component, which can be reused in different graphs. This
// class provides the main interface for API3-based subgraphs.
//
// To define a subgraph, inherit from this class and implement the `Expand`
// method.
//
// NodeT: a structure defining the node's (subgraph's) interface (e.g. inputs
//        and outputs). See node.h for more details.
// ImplT: the actual user-defined subgraph implementation class that inherits
//        from `Subgraph`.
//
// Example:
// ```
//   struct MyNode : Node<"MyNode"> {
//     template <typename S>
//     struct Contract {
//       Input<S, Image> in{"IN"};
//       Output<S, Image> out{"OUT"};
//     };
//   };
//
//   // Implementing the above node as a subgraph.
//   class MyNodeImpl : Subgraph<MyNode, MyNodeImpl> {
//    public:
//     absl::Status Expand(GenericGraph& graph,
//                         SubgraphContext<MyNode>& sc) final {
//       Stream<Image> in = sc.in.Get();
//
//       Stream<Image> out = [&] {
//           auto& node = graph.AddNode<...>();
//           node.input_image.Set(in);
//           return node.output_image.Get();
//       }();
//
//       sc.out.Set(out);
//       return absl::OkStatus();
//     }
//   };
// ```
template <typename NodeT, typename ImplT>
class Subgraph : public mediapipe::Subgraph,
                 private api2::internal::SubgraphRegistrator<ImplT>,
                 private ContractValidator<NodeT::template Contract> {
 public:
  // Implementations should use `graph.AddNode(...)` to add nodes and connect
  // them to each other and to the subgraph's inputs and outputs available
  // through `sc` (e.g. sc.in.Get()).
  //
  // `graph` - the generic graph used to construct the subgraph's topology.
  // `sc` - the subgraph context, providing access to subgraph inputs, outputs,
  //        options and services.
  virtual absl::Status Expand(GenericGraph& graph,
                              SubgraphContext<NodeT>& sc) = 0;

  absl::StatusOr<CalculatorGraphConfig> GetConfig(
      mediapipe::SubgraphContext* sc) final {
    mediapipe::CalculatorContract contract;
    MP_RETURN_IF_ERROR(contract.Initialize(sc->OriginalNode()));
    GenericGraph graph;
    SubgraphContext<NodeT> context(*sc, contract, graph);
    MP_RETURN_IF_ERROR(Expand(graph, context));
    return graph.GetConfig();
  }

  static constexpr auto kCalculatorName = NodeT::GetRegistrationName();
};

}  // namespace mediapipe::api3

#endif  // MEDIAPIPE_FRAMEWORK_API3_SUBGRAPH_H_
