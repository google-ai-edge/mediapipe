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

#ifndef MEDIAPIPE_FRAMEWORK_API3_CALCULATOR_CONTEXT_H_
#define MEDIAPIPE_FRAMEWORK_API3_CALCULATOR_CONTEXT_H_

#include <memory>

#include "absl/log/absl_log.h"
#include "mediapipe/framework/api3/contract.h"
#include "mediapipe/framework/api3/internal/contract_fields.h"
#include "mediapipe/framework/api3/internal/contract_to_tuple.h"
#include "mediapipe/framework/api3/internal/port_base.h"
#include "mediapipe/framework/api3/internal/specializers.h"
#include "mediapipe/framework/api3/packet.h"
#include "mediapipe/framework/calculator_context.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/graph_service.h"
#include "mediapipe/framework/resources.h"
#include "mediapipe/framework/timestamp.h"

namespace mediapipe::api3 {

// Calculator context specialized for a specific node.
//
// (Side)Inputs, (Side)Outputs and Options must be accessed through this
// context. (E.g. cc.input, cc.options etc.)
template <typename NodeT>
class CalculatorContext : public NodeT::template Contract<ContextSpecializer> {
 public:
  explicit CalculatorContext(mediapipe::CalculatorContext& generic_context) {
    holder_ = std::make_unique<internal_port::CalculatorContextHolder>();
    holder_->context = &generic_context;
    typename NodeT::template Contract<ContextSpecializer>* ptr = this;
    auto field_ptrs = ContractToFieldPtrTuple(*ptr);
    std::apply(
        [&](auto&... args) {
          ((internal_port::SetCalculatorContextHolder(*args, *holder_)), ...);
        },
        field_ptrs);
  }

  // Returns the current input timestamp.
  Timestamp InputTimestamp() const {
    return holder_->context->InputTimestamp();
  }

  // Returns a requested service binding.
  //
  // NOTE: you can request service in `UpdateContract` when defining a node or
  //   when implemenating it as a calculator.
  template <typename T>
  ServiceBinding<T> Service(const GraphService<T>& service) {
    return holder_->context->Service(service);
  }

  // Gets interface to access resources (file system, assets, etc.) from
  // calculators.
  //
  // NOTE: this is the preferred way to access resources from subgraphs and
  // calculators as it allows for fine grained per graph configuration.
  //
  // Resources can be configured by setting a custom `kResourcesService` graph
  // service on `CalculatorGraph`. The default resources service can be created
  // and reused through `CreateDefaultResources`.
  const Resources& GetResources() const {
    return holder_->context->GetResources();
  }

  mediapipe::CalculatorContext& GetGenericContext() {
    return *holder_->context;
  }

 private:
  void Reset(mediapipe::CalculatorContext& generic_context) {
    if (holder_->context != nullptr) {
      ABSL_LOG(DFATAL) << "Object must be cleared before resetting.";
    }
    holder_->context = &generic_context;
  }

  void Clear() {
    if (holder_->context == nullptr) {
      ABSL_LOG(DFATAL) << "Object has been already cleared.";
    }
    holder_->context = nullptr;
  }

  std::unique_ptr<internal_port::CalculatorContextHolder> holder_;

  // To Reset/Clear the context.
  template <typename N, typename I>
  friend class Calculator;
};

// +----------------------------------------------------------------------+
// |                                                                      |
// |   Specializations of (Side)Input/Output and Options for CONTEXT.     |
// |                                                                      |
// +----------------------------------------------------------------------+

template <typename PayloadT>
class Input<ContextSpecializer, PayloadT>
    : public internal_port::Port<ContextSpecializer, InputStreamField> {
 public:
  using Payload = PayloadT;
  using internal_port::Port<ContextSpecializer, InputStreamField>::Port;

  explicit operator bool() const {
    auto id = holder_->context->Inputs().GetId(Tag(), Index());
    return id.IsValid() &&
           !holder_->context->Inputs().Get(id).Value().IsEmpty();
  }

  const PayloadT& GetOrDie() const {
    return holder_->context->Inputs()
        .Get(Tag(), Index())
        .Value()
        .template Get<PayloadT>();
  }

  mediapipe::api3::Packet<PayloadT> Packet() const {
    return mediapipe::api3::Packet<PayloadT>(
        holder_->context->Inputs().Get(Tag(), Index()).Value());
  }
};

template <typename PayloadT>
class SideInput<ContextSpecializer, PayloadT>
    : public internal_port::Port<ContextSpecializer, InputSidePacketField> {
 public:
  using Payload = PayloadT;
  using internal_port::Port<ContextSpecializer, InputSidePacketField>::Port;

  virtual explicit operator bool() const {
    auto id = holder_->context->InputSidePackets().GetId(Tag(), Index());
    return id.IsValid() &&
           !holder_->context->InputSidePackets().Get(id).IsEmpty();
  }

  const PayloadT& GetOrDie() const {
    return holder_->context->InputSidePackets()
        .Get(Tag(), Index())
        .template Get<PayloadT>();
  }

  mediapipe::api3::Packet<PayloadT> Packet() const {
    return mediapipe::api3::Packet<PayloadT>(
        holder_->context->InputSidePackets().Get(Tag(), Index()));
  }
};

template <typename PayloadT>
class Output<ContextSpecializer, PayloadT>
    : public internal_port::Port<ContextSpecializer, OutputStreamField> {
 public:
  using Payload = PayloadT;
  using internal_port::Port<ContextSpecializer, OutputStreamField>::Port;

  void Send(const PayloadT& payload) const {
    holder_->context->Outputs()
        .Get(Tag(), Index())
        .AddPacket(mediapipe::MakePacket<PayloadT>(payload).At(
            holder_->context->InputTimestamp()));
  }

  void Send(PayloadT&& payload) const {
    holder_->context->Outputs()
        .Get(Tag(), Index())
        .AddPacket(
            mediapipe::MakePacket<PayloadT>(std::forward<PayloadT>(payload))
                .At(holder_->context->InputTimestamp()));
  }

  void Send(std::unique_ptr<PayloadT> payload) const {
    holder_->context->Outputs()
        .Get(Tag(), Index())
        .AddPacket(mediapipe::Adopt(payload.release())
                       .At(holder_->context->InputTimestamp()));
  }

  void Send(Packet<PayloadT> packet) const {
    holder_->context->Outputs()
        .Get(Tag(), Index())
        .AddPacket(std::move(packet).AsLegacyPacket());
  }

  Timestamp NextTimestampBound() const {
    return holder_->context->Outputs().Get(Tag(), Index()).NextTimestampBound();
  }

  void SetNextTimestampBound(Timestamp timestamp) {
    holder_->context->Outputs()
        .Get(Tag(), Index())
        .SetNextTimestampBound(timestamp);
  }

  bool IsClosed() const {
    return holder_->context->Outputs().Get(Tag(), Index()).IsClosed();
  }

  void Close() const {
    holder_->context->Outputs().Get(Tag(), Index()).Close();
  }
};

template <typename PayloadT>
class SideOutput<ContextSpecializer, PayloadT>
    : public internal_port::Port<ContextSpecializer, OutputSidePacketField> {
 public:
  using Payload = PayloadT;
  using internal_port::Port<ContextSpecializer, OutputSidePacketField>::Port;

  void Set(const PayloadT& payload) const {
    holder_->context->OutputSidePackets()
        .Get(Tag(), Index())
        .Set(mediapipe::MakePacket<PayloadT>(payload));
  }

  void Set(PayloadT&& payload) const {
    holder_->context->OutputSidePackets()
        .Get(Tag(), Index())
        .Set(mediapipe::MakePacket<PayloadT>(std::forward<PayloadT>(payload)));
  }

  void Set(Packet<PayloadT> packet) const {
    holder_->context->OutputSidePackets()
        .Get(Tag(), Index())
        .Set(std::move(packet).AsLegacyPacket());
  }
};

template <typename ProtoT>
class Options<ContextSpecializer, ProtoT> {
 public:
  using Field = OptionsField;
  using Specializer = ContextSpecializer;
  using Payload = ProtoT;

  const ProtoT& Get() const { return holder_->context->Options<ProtoT>(); }

  const ProtoT& operator()() const { return Get(); }

 protected:
  template <typename V, typename H>
  friend void internal_port::SetCalculatorContextHolder(V& v, H& holder);

  // Not owned, set by the framework.
  internal_port::CalculatorContextHolder* holder_ = nullptr;
};

}  // namespace mediapipe::api3

#endif  // MEDIAPIPE_FRAMEWORK_API3_CALCULATOR_CONTEXT_H_
