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

#ifndef MEDIAPIPE_FRAMEWORK_API3_GRAPH_H_
#define MEDIAPIPE_FRAMEWORK_API3_GRAPH_H_

#include <memory>
#include <type_traits>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/log/absl_log.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "mediapipe/framework/api3/contract.h"
#include "mediapipe/framework/api3/internal/contract_fields.h"
#include "mediapipe/framework/api3/internal/contract_to_tuple.h"
#include "mediapipe/framework/api3/internal/contract_validator.h"
#include "mediapipe/framework/api3/internal/graph_builder.h"
#include "mediapipe/framework/api3/internal/port_base.h"
#include "mediapipe/framework/api3/internal/specializers.h"
#include "mediapipe/framework/api3/side_packet.h"
#include "mediapipe/framework/api3/stream.h"
#include "mediapipe/framework/calculator.pb.h"

namespace mediapipe::api3 {

// `Graph` must be used whenever you want to build a MediaPipe graph in C++.
//
// `Graph` allows you to:
// - Construct and maintain complex graphs
// - Parametrize your graphs
// - Exclude/include parts of the graphs
// - Have compile time validation for proper node usage
//
// Here's the common pattern for using `Graph`:
//
// 1. Define a contract for your graph (as described in `contract.h`)
// ```
//   template <typename S>
//   struct ObjectDetection {
//     Input<S, Image> image{"IMAGE"};
//     Output<S, std::vector<Detection>> detections{"DETECTIONS"};
//   };
// ```
//
// 2. Construct your graph.
// ```
//   Graph<ObjectDetection> graph;
//
//   Stream<Image> input_image = graph.image.Get();
//
//   Stream<Tensor> input_tensor = [&]() {
//     auto& node = graph.AddNode<ImageToTensorNode>();
//     node.image.Set(input_image);
//     return node.tensor.Get();
//   }();
//
//   ...
//
//   Stream<std::vector<Detection>> detections = ...;
//
//   graph.detections.Set(detections);
// ```
template <template <typename, typename...> typename ContractT, typename... Ts>
class Graph;

// Every `Graph<...>` derives from `GenericGraph`, so if you want to write a
// common utility for graph construction, you can do this using `GenericGraph`:
//
// 1. Write a utility function to add a node for image to tensor conversion:
// ```
//   Stream<Tensor> ConvertImageToTensor(GenericGraph& graph,
//                                       Stream<Image> image) {
//     auto& node = graph.AddNode<ImageToTensorNode>();
//     node.image.Set(input_image);
//     return node.tensor.Get();
//   }
// ```
//
// 2. Use it and other utilities with your specific `Graph<...>`:
// ```
//   Graph<ObjectDetection> graph;
//
//   Stream<Image> input_image = graph.image.Get();
//
//   Stream<Tensor> input_tensor = ConvertImageToTensor(graph, input_image);
//   auto [boxes, scores] = RunDetectionInference(graph, input_tensor);
//   Stream<std::vector<Detection>> detections
//       = DecodeDetections(graph, boxes, scores);
//
//   graph.detections.Set(detections);
// ```
class GenericGraph;

// `GraphNode` is returned by `Graph<...>::AddNode<...>()` function.
//
// Common pattern to use it:
// ```
//   auto& node = graph.AddNode<...Node>();
//   node.input.Set(input);
//   Stream<...> output = node.output.Get();
// ```
//
// See more details in `AddNode` documentation.
//
// NOTE: `template <typename, typename...> typename ContractT` is a contract
//   type as described in contract.h
template <template <typename, typename...> typename ContractT, typename... Ts>
class GraphNode;

// NOTE: For backward compatibility only.
//
// Avoid using `PacketGenerator`, instead you should simply use a `Node`
// implemented as a `Calculator` that has `Open` function only which receives
// and sends side packets.
template <template <typename, typename...> typename ContractT, typename... Ts>
class LegacyPacketGenerator;

namespace internal_graph {

class GraphNodeBase {
 public:
  virtual ~GraphNodeBase() = default;
};

class GraphLegacyPacketGeneratorBase {
 public:
  virtual ~GraphLegacyPacketGeneratorBase() = default;
};

}  // namespace internal_graph

// See documentation in forward declaration above.
template <template <typename, typename...> typename ContractT, typename... Ts>
class GraphNode : public ContractT<GraphNodeSpecializer>,
                  public internal_graph::GraphNodeBase,
                  private ContractValidator<ContractT, Ts...> {
 public:
  explicit GraphNode(builder::GraphBuilder& graph, absl::string_view name)
      : generic_node_(graph.AddNode(name)) {
    ContractT<GraphNodeSpecializer>* ptr = this;
    auto field_ptrs = ContractToFieldPtrTuple(*ptr);
    std::apply(
        [&](auto&... args) {
          ((internal_port::SetNode(*args, generic_node_)), ...);
        },
        field_ptrs);
  }

  void SetLegacyExecutor(builder::Executor& executor) {
    generic_node_.SetExecutor(executor);
  }

  builder::InputStreamHandler& SetLegacyInputStreamHandler(
      absl::string_view type) {
    return generic_node_.SetInputStreamHandler(type);
  }

  builder::OutputStreamHandler& SetLegacyOutputStreamHandler(
      absl::string_view type) {
    return generic_node_.SetOutputStreamHandler(type);
  }

  void SetSourceLayer(int source_layer) {
    generic_node_.SetSourceLayer(source_layer);
  }

 private:
  builder::NodeBuilder& generic_node_;
};

template <template <typename, typename...> typename ContractT, typename... Ts>
class GraphLegacyPacketGenerator
    : public ContractT<GraphGeneratorSpecializer>,
      public internal_graph::GraphLegacyPacketGeneratorBase,
      private ContractValidator<ContractT, Ts...> {
 public:
  explicit GraphLegacyPacketGenerator(builder::GraphBuilder& graph,
                                      absl::string_view name)
      : generator_builder_(graph.AddPacketGenerator(name)) {
    ContractT<GraphGeneratorSpecializer>* ptr = this;
    auto field_ptrs = ContractToFieldPtrTuple(*ptr);
    std::apply(
        [&](auto&... args) {
          ((internal_port::SetPacketGenerator(*args, generator_builder_)), ...);
        },
        field_ptrs);
  }

 private:
  builder::PacketGeneratorBuilder& generator_builder_;
};

// See documentation in forward declaration above.
class GenericGraph {
 public:
  GenericGraph() = default;
  ~GenericGraph() = default;
  GenericGraph(GenericGraph&&) = default;
  GenericGraph& operator=(GenericGraph&&) = default;
  // Explicitly delete copies to improve error messages.
  GenericGraph(const GenericGraph&) = delete;
  GenericGraph& operator=(const GenericGraph&) = delete;

  // Adds a node of a specific type (NodeT as described per node.h) to the
  // graph.
  //
  // Common usage pattern:
  // ```
  //   auto& node = graph.AddNode<...Node>();
  //   node.input.Set(input);
  //   Stream<...> output = node.output.Get();
  // ```
  // RECOMMENDATION: to make your graphs more readable - avoid mixing multiple
  // nodes construction by using the following techniques:
  //
  // - Extract construction into utility functions as shown below:
  //   ```
  //     Stream<Tensor> ConvertImageToTensor(GenericGraph& graph,
  //                                         Stream<Image> image) {
  //       auto& node = graph.AddNode<ImageToTensorNode>();
  //       node.image.Set(image);
  //       return node.tensor.Get();
  //     }
  //   ```
  // - Use lambdas:
  //   ```
  //     Stream<Tensor> input_tensor = [&]() {
  //       auto& node = graph.AddNode<ImageToTensorNode>();
  //       node.image.Set(image);
  //       return node.tensor.Get();
  //     }();
  //  ```
  // - Localize individual nodes construction and usage to a single place and
  //   pass streams and side packets around:
  //   ```
  //     auto& image_to_tensor = graph.AddNode<ImageToTensorNode>();
  //     image_to_tensor.image.Set(input_image);
  //     Stream<Tensor> input_tensor = image_to_tensor.tensor.Get();
  //
  //     auto& inference = graph.AddNode<InferenceNode>();
  //     inference.input_tensor.Add(input_tensor);
  //     Stream<Tensor> boxes_tensor = inference.output_tensor.Add();
  //     Stream<Tensor> scores_tensor = inference.output_tensor.Add();
  //
  //     ... other nodes
  //   ```
  template <typename NodeT>
  GraphNode<NodeT::template Contract>& AddNode() {
    return AddNodeByContract<NodeT::template Contract>(
        NodeT::GetRegistrationName());
  }

  // Adds a node by contract, passing a custom name.
  //
  // NOTE: this function is designed for rare scenarios, only when
  //   `AddNode<NodeType>` cannot be used.
  // NOTE: `template <typename, typename...> typename NodeContractT` is a
  //   contract type as described in contract.h
  template <template <typename, typename...> typename NodeContractT,
            typename... NodeContractTs>
  GraphNode<NodeContractT, NodeContractTs...>& AddNodeByContract(
      absl::string_view name) {
    auto node = std::make_unique<GraphNode<NodeContractT, NodeContractTs...>>(
        graph_, name);
    GraphNode<NodeContractT, NodeContractTs...>* result = node.get();
    nodes_.emplace_back(std::move(node));
    return *result;
  }

  // Add a legacy packet generator.
  //
  // NOTE: For backward compatibility only.
  //
  // Avoid using `PacketGenerator` - instead you should simply use a `Node`
  // implemented as a `Calculator` that has `Open` function only which receives
  // and sends side packets.
  template <typename NodeT>
  GraphLegacyPacketGenerator<NodeT::template Contract>&
  AddLegacyPacketGenerator() {
    auto node =
        std::make_unique<GraphLegacyPacketGenerator<NodeT::template Contract>>(
            graph_, NodeT::GetRegistrationName());
    GraphLegacyPacketGenerator<NodeT::template Contract>* result = node.get();
    generators_.emplace_back(std::move(node));
    return *result;
  }

  builder::Executor& AddLegacyExecutor(absl::string_view name) {
    return graph_.AddExecutor(name);
  }

  absl::StatusOr<mediapipe::CalculatorGraphConfig> GetConfig() {
    return graph_.GetConfig();
  }

 protected:
  builder::GraphBuilder graph_;
  std::vector<std::unique_ptr<internal_graph::GraphNodeBase>> nodes_;
  std::vector<std::unique_ptr<internal_graph::GraphLegacyPacketGeneratorBase>>
      generators_;
};

template <template <typename, typename...> typename ContractT, typename... Ts>
class Graph : public ContractT<GraphSpecializer, Ts...>,
              public GenericGraph,
              private ContractValidator<ContractT, Ts...> {
 public:
  explicit Graph() : GenericGraph() {
    ContractT<GraphSpecializer, Ts...>* ptr = this;
    auto field_ptrs = ContractToFieldPtrTuple(*ptr);
    std::apply(
        [&](auto&... args) { ((internal_port::SetGraph(*args, graph_)), ...); },
        field_ptrs);
  }
};

// +----------------------------------------------------------------------+
// |                                                                      |
// |   Specializations of (Side)Input/Output, Options for                 |
// |   GraphSpecializer.                                                  |
// |                                                                      |
// +----------------------------------------------------------------------+

template <typename PayloadT>
class Input<GraphSpecializer, PayloadT>
    : public internal_port::Port<GraphSpecializer, InputStreamField> {
 public:
  using Payload = PayloadT;
  using internal_port::Port<GraphSpecializer, InputStreamField>::Port;

  // Returns an object representing this input as a stream - can be passed
  // around to be set on graph node inputs or graph outputs.
  Stream<PayloadT> Get() const {
    return Stream<PayloadT>(graph_builder_->In(Tag()).At(Index()));
  }
};

template <typename PayloadT>
class SideInput<GraphSpecializer, PayloadT>
    : public internal_port::Port<GraphSpecializer, InputSidePacketField> {
 public:
  using Payload = PayloadT;
  using internal_port::Port<GraphSpecializer, InputSidePacketField>::Port;

  // Returns an object representing this input side packet - can be passed
  // around to be set on graph node input side packets or graph output side
  // packets.
  SidePacket<PayloadT> Get() const {
    return SidePacket<PayloadT>(graph_builder_->SideIn(Tag()).At(Index()));
  }
};

template <typename PayloadT>
class Output<GraphSpecializer, PayloadT>
    : public internal_port::Port<GraphSpecializer, OutputStreamField> {
 public:
  using Payload = PayloadT;
  using internal_port::Port<GraphSpecializer, OutputStreamField>::Port;

  void Set(Stream<PayloadT> stream) const {
    stream.source_->ConnectTo(graph_builder_->Out(Tag()).At(Index()));
  }
};

template <typename PayloadT>
class SideOutput<GraphSpecializer, PayloadT>
    : public internal_port::Port<GraphSpecializer, OutputSidePacketField> {
 public:
  using Payload = PayloadT;
  using internal_port::Port<GraphSpecializer, OutputSidePacketField>::Port;

  void Set(SidePacket<PayloadT> side_packet) const {
    side_packet.side_source_->ConnectTo(
        graph_builder_->SideOut(Tag()).At(Index()));
  }
};

namespace internal_graph {

template <typename P>
class BuilderRepeated
    : public internal_port::RepeatedBase<typename P::Specializer,
                                         typename P::Field> {
 public:
  static_assert(!std::is_same_v<typename P::Field, OptionalField>,
                "`Repeated` doesn't accept `Optional` ports (`Repeated` is "
                "already optional.)");

  using Field = RepeatedField;
  using Contained = P;
  using internal_port::RepeatedBase<typename P::Specializer,
                                    typename P::Field>::RepeatedBase;

 protected:
  P& InternalAdd() {
    int index = repeated_ports_.size();
    auto port = std::unique_ptr<P>(
        new P(std::make_unique<internal_port::StrViewTag>(
                  internal_port::RepeatedBase<typename P::Specializer,
                                              typename P::Field>::Tag()),
              index));
    internal_port::RepeatedBase<typename P::Specializer,
                                typename P::Field>::InitPort(*port);
    P* repated_port_ptr = port.get();
    if (auto [it, inserted] = repeated_ports_.insert({index, std::move(port)});
        !inserted) {
      ABSL_LOG(DFATAL) << "Unexpected port at index: " << index;
    }
    return *repated_port_ptr;
  }

  absl::flat_hash_map<int, std::unique_ptr<P>> repeated_ports_;
};

}  // namespace internal_graph

template <typename P>
class Repeated<P,
               std::enable_if_t<
                   std::is_same_v<typename P::Specializer, GraphSpecializer> &&
                   std::is_same_v<typename P::Field, OutputStreamField>>>
    : public internal_graph::BuilderRepeated<P> {
 public:
  using internal_graph::BuilderRepeated<P>::BuilderRepeated;

  void Add(Stream<typename P::Payload> stream) {
    internal_graph::BuilderRepeated<P>::InternalAdd().Set(stream);
  }
};

template <typename P>
class Repeated<P,
               std::enable_if_t<
                   std::is_same_v<typename P::Specializer, GraphSpecializer> &&
                   std::is_same_v<typename P::Field, InputStreamField>>>
    : public internal_graph::BuilderRepeated<P> {
 public:
  using internal_graph::BuilderRepeated<P>::BuilderRepeated;

  Stream<typename P::Payload> Add() {
    return internal_graph::BuilderRepeated<P>::InternalAdd().Get();
  }
};

template <typename P>
class Repeated<P,
               std::enable_if_t<
                   std::is_same_v<typename P::Specializer, GraphSpecializer> &&
                   std::is_same_v<typename P::Field, OutputSidePacketField>>>
    : public internal_graph::BuilderRepeated<P> {
 public:
  using internal_graph::BuilderRepeated<P>::BuilderRepeated;

  void Add(SidePacket<typename P::Payload> side_packet) {
    internal_graph::BuilderRepeated<P>::InternalAdd().Set(side_packet);
  }
};

template <typename P>
class Repeated<P,
               std::enable_if_t<
                   std::is_same_v<typename P::Specializer, GraphSpecializer> &&
                   std::is_same_v<typename P::Field, InputSidePacketField>>>
    : public internal_graph::BuilderRepeated<P> {
 public:
  using internal_graph::BuilderRepeated<P>::BuilderRepeated;

  SidePacket<typename P::Payload> Add() {
    return internal_graph::BuilderRepeated<P>::InternalAdd().Get();
  }
};

// +----------------------------------------------------------------------+
// |                                                                      |
// |   Specializations of (Side)Input/Output, Options & Repeated for      |
// |   GraphNodeSpecializer.                                              |
// |                                                                      |
// +----------------------------------------------------------------------+

template <typename PayloadT>
class Input<GraphNodeSpecializer, PayloadT>
    : public internal_port::Port<GraphNodeSpecializer, InputStreamField> {
 public:
  using Payload = PayloadT;
  using internal_port::Port<GraphNodeSpecializer, InputStreamField>::Port;

  void Set(Stream<PayloadT> stream, bool back_edge = false) {
    auto& in = node_builder_->In(Tag()).At(Index());
    stream.source_->ConnectTo(back_edge ? in.AsBackEdge() : in);
  }
};

template <typename PayloadT>
class SideInput<GraphNodeSpecializer, PayloadT>
    : public internal_port::Port<GraphNodeSpecializer, InputSidePacketField> {
 public:
  using Payload = PayloadT;
  using internal_port::Port<GraphNodeSpecializer, InputSidePacketField>::Port;

  void Set(SidePacket<PayloadT> side_packet) {
    side_packet.side_source_->ConnectTo(
        node_builder_->SideIn(Tag()).At(Index()));
  }
};

template <typename PayloadT>
class Output<GraphNodeSpecializer, PayloadT>
    : public internal_port::Port<GraphNodeSpecializer, OutputStreamField> {
 public:
  using Payload = PayloadT;
  using internal_port::Port<GraphNodeSpecializer, OutputStreamField>::Port;

  // Returns an object representing this output as a stream - can be passed
  // around to be set as an input to graph nodes or output to the graph.
  Stream<PayloadT> Get() const {
    return Stream<PayloadT>(node_builder_->Out(Tag()).At(Index()));
  }
};

template <typename PayloadT>
class SideOutput<GraphNodeSpecializer, PayloadT>
    : public internal_port::Port<GraphNodeSpecializer, OutputSidePacketField> {
 public:
  using Payload = PayloadT;
  using internal_port::Port<GraphNodeSpecializer, OutputSidePacketField>::Port;

  // Returns an object representing this side output as a stream - can be passed
  // around to be set as a side input to graph nodes or side output to the
  // graph.
  SidePacket<PayloadT> Get() const {
    return SidePacket<PayloadT>(node_builder_->SideOut(Tag()).At(Index()));
  }
};

template <typename ProtoT>
class Options<GraphNodeSpecializer, ProtoT> {
 public:
  using Field = OptionsField;
  using Specializer = GraphNodeSpecializer;
  using Payload = ProtoT;

  // Returns node options pointer (never null) for population.
  ProtoT* Mutable() { return &node_builder_->GetOptions<ProtoT>(); }

 protected:
  template <typename V, typename N>
  friend void internal_port::SetNode(V& v, N& node);

  // Not owned, set by the framework.
  builder::NodeBuilder* node_builder_ = nullptr;
};

template <typename P>
class Repeated<
    P, std::enable_if_t<
           std::is_same_v<typename P::Specializer, GraphNodeSpecializer> &&
           std::is_same_v<typename P::Field, InputStreamField>>>
    : public internal_graph::BuilderRepeated<P> {
 public:
  using internal_graph::BuilderRepeated<P>::BuilderRepeated;

  void Add(Stream<typename P::Payload> stream, bool back_edge = false) {
    internal_graph::BuilderRepeated<P>::InternalAdd().Set(stream, back_edge);
  }
};

template <typename P>
class Repeated<
    P, std::enable_if_t<
           std::is_same_v<typename P::Specializer, GraphNodeSpecializer> &&
           std::is_same_v<typename P::Field, OutputStreamField>>>
    : public internal_graph::BuilderRepeated<P> {
 public:
  using internal_graph::BuilderRepeated<P>::BuilderRepeated;

  Stream<typename P::Payload> Add() {
    return internal_graph::BuilderRepeated<P>::InternalAdd().Get();
  }
};

template <typename P>
class Repeated<
    P, std::enable_if_t<
           std::is_same_v<typename P::Specializer, GraphNodeSpecializer> &&
           std::is_same_v<typename P::Field, InputSidePacketField>>>
    : public internal_graph::BuilderRepeated<P> {
 public:
  using internal_graph::BuilderRepeated<P>::BuilderRepeated;

  void Add(SidePacket<typename P::Payload> side_packet) {
    internal_graph::BuilderRepeated<P>::InternalAdd().Set(side_packet);
  }
};

template <typename P>
class Repeated<
    P, std::enable_if_t<
           std::is_same_v<typename P::Specializer, GraphNodeSpecializer> &&
           std::is_same_v<typename P::Field, OutputSidePacketField>>>
    : public internal_graph::BuilderRepeated<P> {
 public:
  using internal_graph::BuilderRepeated<P>::BuilderRepeated;

  SidePacket<typename P::Payload> Add() {
    return internal_graph::BuilderRepeated<P>::InternalAdd().Get();
  }
};

// +----------------------------------------------------------------------+
// |                                                                      |
// |   Specializations of SideInput/Output, Options & Repeated for        |
// |   GraphGeneratorSpecializer.                                         |
// |                                                                      |
// +----------------------------------------------------------------------+

template <typename PayloadT>
class SideInput<GraphGeneratorSpecializer, PayloadT>
    : public internal_port::Port<GraphGeneratorSpecializer,
                                 InputSidePacketField> {
 public:
  using Payload = PayloadT;
  using internal_port::Port<GraphGeneratorSpecializer,
                            InputSidePacketField>::Port;

  void Set(SidePacket<PayloadT> side_packet) const {
    side_packet.side_source_->ConnectTo(
        generator_builder_->SideIn(Tag()).At(Index()));
  }
};

template <typename PayloadT>
class SideOutput<GraphGeneratorSpecializer, PayloadT>
    : public internal_port::Port<GraphGeneratorSpecializer,
                                 OutputSidePacketField> {
 public:
  using Payload = PayloadT;
  using internal_port::Port<GraphGeneratorSpecializer,
                            OutputSidePacketField>::Port;

  // Returns an object representing this side output as a stream - can be passed
  // around to be set as a side input to graph nodes or side output to the
  // graph.
  SidePacket<PayloadT> Get() const {
    return SidePacket<PayloadT>(generator_builder_->SideOut(Tag()).At(Index()));
  }
};

template <typename ProtoT>
class Options<GraphGeneratorSpecializer, ProtoT> {
 public:
  using Field = OptionsField;
  using Specializer = GraphGeneratorSpecializer;
  using Payload = ProtoT;

  // Returns generator options pointer (never null) for population.
  ProtoT* Mutable() { return &generator_builder_->GetOptions<ProtoT>(); }

 protected:
  template <typename V, typename N>
  friend void internal_port::SetPacketGenerator(V& v, N& node);

  // Not owned, set by the framework.
  builder::PacketGeneratorBuilder* generator_builder_ = nullptr;
};

template <typename P>
class Repeated<
    P, std::enable_if_t<
           std::is_same_v<typename P::Specializer, GraphGeneratorSpecializer> &&
           std::is_same_v<typename P::Field, InputSidePacketField>>>
    : public internal_graph::BuilderRepeated<P> {
 public:
  using internal_graph::BuilderRepeated<P>::BuilderRepeated;

  void Add(SidePacket<typename P::Payload> side_packet) {
    internal_graph::BuilderRepeated<P>::InternalAdd().Set(side_packet);
  }
};

template <typename P>
class Repeated<
    P, std::enable_if_t<
           std::is_same_v<typename P::Specializer, GraphGeneratorSpecializer> &&
           std::is_same_v<typename P::Field, OutputSidePacketField>>>
    : public internal_graph::BuilderRepeated<P> {
 public:
  using internal_graph::BuilderRepeated<P>::BuilderRepeated;

  auto Add() { return internal_graph::BuilderRepeated<P>::InternalAdd().Get(); }
};

}  // namespace mediapipe::api3

#endif  // MEDIAPIPE_FRAMEWORK_API3_GRAPH_H_
