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

#ifndef MEDIAPIPE_FRAMEWORK_API3_SUBGRAPH_CONTEXT_H_
#define MEDIAPIPE_FRAMEWORK_API3_SUBGRAPH_CONTEXT_H_

#include <type_traits>

#include "mediapipe/framework/api3/contract.h"
#include "mediapipe/framework/api3/graph.h"
#include "mediapipe/framework/api3/internal/contract_fields.h"
#include "mediapipe/framework/api3/internal/graph_builder.h"
#include "mediapipe/framework/api3/internal/port_base.h"
#include "mediapipe/framework/api3/internal/specializers.h"
#include "mediapipe/framework/api3/side_packet.h"
#include "mediapipe/framework/api3/stream.h"
#include "mediapipe/framework/calculator.pb.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/graph_service.h"
#include "mediapipe/framework/resources.h"

namespace mediapipe::api3 {

// In API3, `Subgraph::Expand` receives `GenericGraph` and `SubgraphContext` in
// order to expand the subgraph. Where `SubgraphContext` object provides access
// to inputs, outputs, options, services, resources and original node.
//
// Inputs, outputs and options are accessed as member variables inherited from
// contract definition, for example:
// ```
//   Stream<int> in = sc.in.Get();
//   const auto& options = sc.options.Get();
// ```
template <typename NodeT>
class SubgraphContext : public NodeT::template Contract<SubgraphSpecializer> {
 public:
  explicit SubgraphContext(mediapipe::SubgraphContext& generic_context,
                           mediapipe::CalculatorContract& contract,
                           GenericGraph& graph)
      : generic_context_(generic_context) {
    typename NodeT::template Contract<SubgraphSpecializer>* ptr = this;
    auto field_ptrs = ContractToFieldPtrTuple(*ptr);
    std::apply(
        [&](auto&... args) {
          (
              [&]() {
                if constexpr (std::is_same_v<
                                  typename std::remove_pointer_t<
                                      std::decay_t<decltype(args)>>::Field,
                                  OptionsField>) {
                  internal_port::SetSubgraphContext(*args, generic_context);
                } else {
                  internal_port::SetSubgraphContextAndExtras(
                      *args, generic_context, contract, graph.builder_);
                }
              }(),
              ...);
        },
        field_ptrs);
  };

  const mediapipe::CalculatorGraphConfig::Node& OriginalNode() {
    return generic_context_.OriginalNode();
  }

  const Resources& GetResources() { return generic_context_.GetResources(); }

  template <typename T>
  ServiceBinding<T> Service(const GraphService<T>& service) const {
    return generic_context_.Service(service);
  }

 private:
  mediapipe::SubgraphContext& generic_context_;
};

// +-------------------------------------------------------------------------+
// |                                                                         |
// |   Specializations of (Side)Input/Output, Options for SubgraphContext.   |
// |                                                                         |
// +-------------------------------------------------------------------------+

template <typename PayloadT>
class Input<SubgraphSpecializer, PayloadT>
    : public internal_port::Port<SubgraphSpecializer, InputStreamField> {
 public:
  using Payload = PayloadT;
  using internal_port::Port<SubgraphSpecializer, InputStreamField>::Port;

  Stream<PayloadT> Get() const {
    return Stream<PayloadT>(graph_builder_->In(Tag()).At(Index()));
  }

  Stream<PayloadT> operator()() const { return Get(); }
};

template <typename PayloadT>
class SideInput<SubgraphSpecializer, PayloadT>
    : public internal_port::Port<SubgraphSpecializer, InputSidePacketField> {
 public:
  using Payload = PayloadT;
  using internal_port::Port<SubgraphSpecializer, InputSidePacketField>::Port;

  SidePacket<PayloadT> Get() const {
    return SidePacket<PayloadT>(graph_builder_->SideIn(Tag()).At(Index()));
  }
};

template <typename PayloadT>
class Output<SubgraphSpecializer, PayloadT>
    : public internal_port::Port<SubgraphSpecializer, OutputStreamField> {
 public:
  using Payload = PayloadT;
  using internal_port::Port<SubgraphSpecializer, OutputStreamField>::Port;

  void Set(Stream<PayloadT> stream) const {
    stream.source_->ConnectTo(graph_builder_->Out(Tag()).At(Index()));
  }
};

template <typename PayloadT>
class SideOutput<SubgraphSpecializer, PayloadT>
    : public internal_port::Port<SubgraphSpecializer, OutputSidePacketField> {
 public:
  using Payload = PayloadT;
  using internal_port::Port<SubgraphSpecializer, OutputSidePacketField>::Port;

  void Set(SidePacket<PayloadT> side_packet) const {
    side_packet.side_source_->ConnectTo(
        graph_builder_->SideOut(Tag()).At(Index()));
  }
};

template <typename ProtoT>
class Options<SubgraphSpecializer, ProtoT> {
 public:
  using Field = OptionsField;
  using Specializer = SubgraphSpecializer;
  using Payload = ProtoT;

  const ProtoT& Get() const { return subgraph_context_->Options<ProtoT>(); }

 protected:
  // Not owned, set by the framework.
  mediapipe::SubgraphContext* subgraph_context_ = nullptr;

  template <typename V, typename SC>
  friend void ::mediapipe::api3::internal_port::SetSubgraphContext(
      V& v, SC& subgraph_context);
};

}  // namespace mediapipe::api3

#endif  // MEDIAPIPE_FRAMEWORK_API3_SUBGRAPH_CONTEXT_H_
