#ifndef MEDIAPIPE_FRAMEWORK_API3_CALCULATOR_CONTEXT_H_
#define MEDIAPIPE_FRAMEWORK_API3_CALCULATOR_CONTEXT_H_

#include <memory>

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
  explicit CalculatorContext(mediapipe::CalculatorContext& generic_context)
      : generic_context_(generic_context) {
    typename NodeT::template Contract<ContextSpecializer>* ptr = this;
    auto field_ptrs = ContractToFieldPtrTuple(*ptr);
    std::apply(
        [&](auto&... args) {
          ((internal_port::SetCalculatorContext(*args, generic_context)), ...);
        },
        field_ptrs);
  }

  // Returns the current input timestamp.
  Timestamp InputTimestamp() const { return generic_context_.InputTimestamp(); }

  // Returns a requested service binding.
  //
  // NOTE: you can request service in `UpdateContract` when defining a node or
  //   when implemenating it as a calculator.
  template <typename T>
  ServiceBinding<T> Service(const GraphService<T>& service) {
    return generic_context_.Service(service);
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
    return generic_context_.GetResources();
  }

 private:
  mediapipe::CalculatorContext& generic_context_;
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
    auto id = context_->Inputs().GetId(Tag(), Index());
    return id.IsValid() && !context_->Inputs().Get(id).Value().IsEmpty();
  }

  const PayloadT& GetOrDie() const {
    return context_->Inputs()
        .Get(Tag(), Index())
        .Value()
        .template Get<PayloadT>();
  }

  mediapipe::api3::Packet<PayloadT> Packet() const {
    return mediapipe::api3::Packet<PayloadT>(
        context_->Inputs().Get(Tag(), Index()).Value());
  }
};

template <typename PayloadT>
class SideInput<ContextSpecializer, PayloadT>
    : public internal_port::Port<ContextSpecializer, InputSidePacketField> {
 public:
  using Payload = PayloadT;
  using internal_port::Port<ContextSpecializer, InputSidePacketField>::Port;

  virtual explicit operator bool() const {
    auto id = context_->InputSidePackets().GetId(Tag(), Index());
    return id.IsValid() && !context_->InputSidePackets().Get(id).IsEmpty();
  }

  const PayloadT& GetOrDie() const {
    return context_->InputSidePackets()
        .Get(Tag(), Index())
        .template Get<PayloadT>();
  }

  mediapipe::api3::Packet<PayloadT> Packet() const {
    return mediapipe::api3::Packet<PayloadT>(
        context_->InputSidePackets().Get(Tag(), Index()));
  }
};

template <typename PayloadT>
class Output<ContextSpecializer, PayloadT>
    : public internal_port::Port<ContextSpecializer, OutputStreamField> {
 public:
  using Payload = PayloadT;
  using internal_port::Port<ContextSpecializer, OutputStreamField>::Port;

  void Send(const PayloadT& payload) const {
    context_->Outputs()
        .Get(Tag(), Index())
        .AddPacket(mediapipe::MakePacket<PayloadT>(payload).At(
            context_->InputTimestamp()));
  }

  void Send(PayloadT&& payload) const {
    context_->Outputs()
        .Get(Tag(), Index())
        .AddPacket(
            mediapipe::MakePacket<PayloadT>(std::forward<PayloadT>(payload))
                .At(context_->InputTimestamp()));
  }

  void Send(std::unique_ptr<PayloadT> payload) const {
    context_->Outputs()
        .Get(Tag(), Index())
        .AddPacket(
            mediapipe::Adopt(payload.release()).At(context_->InputTimestamp()));
  }

  Timestamp NextTimestampBound() const {
    return context_->Outputs().Get(Tag(), Index()).NextTimestampBound();
  }

  void SetNextTimestampBound(Timestamp timestamp) {
    context_->Outputs().Get(Tag(), Index()).SetNextTimestampBound(timestamp);
  }

  bool IsClosed() const {
    return context_->Outputs().Get(Tag(), Index()).IsClosed();
  }

  void Close() const { context_->Outputs().Get(Tag(), Index()).Close(); }
};

template <typename PayloadT>
class SideOutput<ContextSpecializer, PayloadT>
    : public internal_port::Port<ContextSpecializer, OutputSidePacketField> {
 public:
  using Payload = PayloadT;
  using internal_port::Port<ContextSpecializer, OutputSidePacketField>::Port;

  void Set(const PayloadT& payload) const {
    context_->OutputSidePackets()
        .Get(Tag(), Index())
        .Set(mediapipe::MakePacket<PayloadT>(payload));
  }

  void Set(PayloadT&& payload) const {
    context_->OutputSidePackets()
        .Get(Tag(), Index())
        .Set(mediapipe::MakePacket<PayloadT>(std::forward<PayloadT>(payload)));
  }
};

template <typename ProtoT>
class Options<ContextSpecializer, ProtoT> {
 public:
  using Field = OptionsField;
  using Specializer = ContextSpecializer;
  using Payload = ProtoT;

  const ProtoT& Get() const { return context_->Options<ProtoT>(); }

  const ProtoT& operator()() const { return Get(); }

 protected:
  template <typename V, typename CC>
  friend void internal_port::SetCalculatorContext(V& v, CC& context);

  // Not owned, set by the framework.
  mediapipe::CalculatorContext* context_ = nullptr;
};

}  // namespace mediapipe::api3

#endif  // MEDIAPIPE_FRAMEWORK_API3_CALCULATOR_CONTEXT_H_
