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

#ifndef MEDIAPIPE_FRAMEWORK_API3_CALCULATOR_CONTRACT_H_
#define MEDIAPIPE_FRAMEWORK_API3_CALCULATOR_CONTRACT_H_

#include <string>

#include "absl/functional/any_invocable.h"
#include "absl/status/status.h"
#include "mediapipe/framework/api3/any.h"
#include "mediapipe/framework/api3/contract.h"
#include "mediapipe/framework/api3/internal/contract_fields.h"
#include "mediapipe/framework/api3/internal/contract_to_tuple.h"
#include "mediapipe/framework/api3/internal/has_update_contract.h"
#include "mediapipe/framework/api3/internal/port_base.h"
#include "mediapipe/framework/api3/internal/specializers.h"
#include "mediapipe/framework/calculator_contract.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/graph_service.h"

namespace mediapipe::api3 {

// Calculator contract specialized for a specific node.
//
// (Side)Inputs, (Side)Outputs and Options must be accessed through the
// specialized contract as following:
//
// ```
//   static absl::Status UpdateContract(CalculatorContract<FooNode>& cc) {
//     cc.UseService(kMyService);
//     const FooOptions& options = cc.options.Get();
//     ...
//   }
// ```
template <typename NodeT>
class CalculatorContract
    : public NodeT::template Contract<ContractSpecializer> {
 public:
  explicit CalculatorContract(
      mediapipe::CalculatorContract& generic_contract,
      absl::AnyInvocable<void(absl::Status)> store_status)
      : generic_contract_(generic_contract) {
    using BaseContractT =
        typename NodeT::template Contract<ContractSpecializer>;
    BaseContractT* ptr = this;
    auto field_ptrs = ContractToFieldPtrTuple(*ptr);
    std::apply(
        [&](auto&... args) {
          (([&] {
             internal_port::SetCalculatorContract(*args, generic_contract);
             store_status(
                 internal_port::AddToContract(*args, generic_contract));
           }()),
           ...);
        },
        field_ptrs);

    if constexpr (kHasUpdateContract<BaseContractT,
                                     CalculatorContract<NodeT>>) {
      store_status(BaseContractT::UpdateContract(*this));
    }
  };

  // Returns the name given to this node.
  const std::string& GetNodeName() const {
    return generic_contract_.GetNodeName();
  }

  // Indicates specific `service` is required for graph execution.
  //
  // For services which allow default initialization:
  // - `CalculatorGraph` will try to create corresponding service object by
  //   default even if request is made optional
  //   (`GraphServiceRequest::Optional()`).
  //
  // For services which disallow default initialization:
  // - `CalculatorGraph` requires client to set corresponding service object and
  //   otherwise fails, unless request is made optional
  //   (`GraphServiceRequest::Optional()`).
  mediapipe::CalculatorContract::GraphServiceRequest& UseService(
      const GraphServiceBase& service) {
    return generic_contract_.UseService(service);
  }

  // Specifies the preferred InputStreamHandler for this Node.
  // If there is an InputStreamHandler specified in the graph (.pbtxt) for this
  // Node, then the graph's InputStreamHandler will take priority.
  void SetInputStreamHandler(const std::string& name) {
    generic_contract_.SetInputStreamHandler(name);
  }

  // Returns the name of this Nodes's InputStreamHandler, or empty string if
  // none is set.
  std::string GetInputStreamHandler() const {
    return generic_contract_.GetInputStreamHandler();
  }

  // Sets input stream handler options.
  void SetInputStreamHandlerOptions(const MediaPipeOptions& options) {
    generic_contract_.SetInputStreamHandlerOptions(options);
  }

  // Returns the MediaPipeOptions of this Node's InputStreamHandler, or empty
  // options if none is set.
  MediaPipeOptions GetInputStreamHandlerOptions() const {
    return generic_contract_.GetInputStreamHandlerOptions();
  }

  // The next few methods are concerned with timestamp bound propagation
  // (see scheduling_sync.md#input-policies). Every calculator that processes
  // live inputs should specify either ProcessTimestampBounds or
  // TimestampOffset.  Calculators that produce output at the same timestamp as
  // the input, or with a fixed offset, should declare this fact using
  // SetTimestampOffset.  Calculators that require custom timestamp bound
  // calculations should use SetProcessTimestampBounds.

  // When true, Process is called for every new timestamp bound, with or without
  // new packets. A call to Process with only an input timestamp bound is
  // normally used to compute a new output timestamp bound.
  // NOTE: Also, when true, Process is called when input streams become done,
  // which means, Process needs to handle input streams in "done" state.
  // (Usually, by closing calculators' outputs where and when appropriate.)
  void SetProcessTimestampBounds(bool process_timestamps) {
    generic_contract_.SetProcessTimestampBounds(process_timestamps);
  }
  bool GetProcessTimestampBounds() const {
    return generic_contract_.GetProcessTimestampBounds();
  }

  // Specifies the maximum difference between input and output timestamps.
  // When specified, the mediapipe framework automatically computes output
  // timestamp bounds based on input timestamps.  The special value
  // TimestampDiff::Unset disables the timestamp offset.
  void SetTimestampOffset(TimestampDiff offset) {
    generic_contract_.SetTimestampOffset(offset);
  }
  TimestampDiff GetTimestampOffset() const {
    return generic_contract_.GetTimestampOffset();
  }

  mediapipe::CalculatorContract& GetGenericContract() {
    return generic_contract_;
  }

 private:
  mediapipe::CalculatorContract& generic_contract_;
};

// +----------------------------------------------------------------------+
// |                                                                      |
// |   Specializations of (Side)Input/Output, Options for CONTRACT.       |
// |                                                                      |
// +----------------------------------------------------------------------+

template <typename PayloadT>
class Input<ContractSpecializer, PayloadT>
    : public internal_port::Port<ContractSpecializer, InputStreamField> {
 public:
  using Payload = PayloadT;
  using internal_port::Port<ContractSpecializer, InputStreamField>::Port;
};

template <typename PayloadT>
class SideInput<ContractSpecializer, PayloadT>
    : public internal_port::Port<ContractSpecializer, InputSidePacketField> {
 public:
  using Payload = PayloadT;
  using internal_port::Port<ContractSpecializer, InputSidePacketField>::Port;
};

template <typename PayloadT>
class Output<ContractSpecializer, PayloadT>
    : public internal_port::Port<ContractSpecializer, OutputStreamField> {
 public:
  using Payload = PayloadT;
  using internal_port::Port<ContractSpecializer, OutputStreamField>::Port;
};

template <>
class Output<ContractSpecializer, Any>
    : public internal_port::Port<ContractSpecializer, OutputStreamField> {
 public:
  using Payload = Any;
  using internal_port::Port<ContractSpecializer, OutputStreamField>::Port;

  // SetSameAs should be available only in the cases when node's input is Any,
  // output is Any, but input & output should have the same type.
  void SetSameAs(const Input<ContractSpecializer, Any>& input) {
    contract_->Outputs()
        .Get(this->Tag(), this->Index())
        .SetSameAs(
            contract_->Inputs().Get(input.Tag(), input.Index()).GetSameAs());
  }
};

template <typename PayloadT>
class SideOutput<ContractSpecializer, PayloadT>
    : public internal_port::Port<ContractSpecializer, OutputSidePacketField> {
 public:
  using Payload = PayloadT;
  using internal_port::Port<ContractSpecializer, OutputSidePacketField>::Port;
};

template <>
class SideOutput<ContractSpecializer, Any>
    : public internal_port::Port<ContractSpecializer, OutputSidePacketField> {
 public:
  using Payload = Any;
  using internal_port::Port<ContractSpecializer, OutputSidePacketField>::Port;

  // SetSameAs should be available only in the cases when node's input is Any,
  // output is Any, but input & output should have the same type.
  void SetSameAs(const SideInput<ContractSpecializer, Any>& side_input) {
    contract_->OutputSidePackets()
        .Get(this->Tag(), this->Index())
        .SetSameAs(contract_->InputSidePackets()
                       .Get(side_input.Tag(), side_input.Index())
                       .GetSameAs());
  }
};

template <typename ProtoT>
class Options<ContractSpecializer, ProtoT> {
 public:
  using Field = OptionsField;
  using Specializer = ContractSpecializer;
  using Payload = ProtoT;

  const ProtoT& Get() const { return contract_->Options<ProtoT>(); }

  const ProtoT& operator()() const { return Get(); }

 protected:
  template <typename V, typename CC>
  friend void internal_port::SetCalculatorContract(V& v, CC& contract);

  // Not owned, set by the framework.
  mediapipe::CalculatorContract* contract_ = nullptr;
};

}  // namespace mediapipe::api3

#endif  // MEDIAPIPE_FRAMEWORK_API3_CALCULATOR_CONTRACT_H_
