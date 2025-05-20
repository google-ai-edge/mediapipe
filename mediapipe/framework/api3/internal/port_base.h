#ifndef MEDIAPIPE_FRAMEWORK_API3_INTERNAL_PORT_BASE_H_
#define MEDIAPIPE_FRAMEWORK_API3_INTERNAL_PORT_BASE_H_

#include <memory>
#include <string>
#include <type_traits>
#include <utility>

#include "absl/base/attributes.h"
#include "absl/log/absl_check.h"
#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "mediapipe/framework/api3/internal/contract_fields.h"
#include "mediapipe/framework/api3/internal/dependent_false.h"
#include "mediapipe/framework/api3/internal/specializers.h"
#include "mediapipe/framework/calculator_context.h"
#include "mediapipe/framework/calculator_contract.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/port/status_macros.h"

namespace mediapipe::api3 {

namespace internal_port {

class PortTag {
 public:
  virtual ~PortTag() = default;
  virtual absl::string_view Get() const ABSL_ATTRIBUTE_LIFETIME_BOUND = 0;
};

class StrPortTag : public PortTag {
 public:
  explicit StrPortTag(std::string tag) : tag_(std::move(tag)) {}
  absl::string_view Get() const ABSL_ATTRIBUTE_LIFETIME_BOUND final {
    return tag_;
  }

 private:
  std::string tag_;
};

class /*ABSL_ATTRIBUTE_VIEW*/ StrViewTag : public PortTag {
 public:
  explicit StrViewTag(absl::string_view tag_view ABSL_ATTRIBUTE_LIFETIME_BOUND)
      : tag_view_(tag_view) {}
  absl::string_view Get() const ABSL_ATTRIBUTE_LIFETIME_BOUND final {
    return tag_view_;
  }

 private:
  absl::string_view tag_view_;
};

class TagAndIndex {
 public:
  explicit TagAndIndex(absl::string_view tag)
      : tag_(std::make_unique<StrPortTag>(std::string(tag))) {}
  // Is intended for use only by `Repeated<...>` to provide corresponding port
  // objects by index.
  explicit TagAndIndex(std::unique_ptr<PortTag> tag, int index)
      : tag_(std::move(tag)), index_(index) {}

  virtual ~TagAndIndex() = default;

  // Explicitly disabling copy ctr/assign for any port - they are not designed
  // for copy and reassignment.
  TagAndIndex& operator=(const TagAndIndex& other) = delete;
  TagAndIndex(const TagAndIndex& other) = delete;

  // Explicitly allow move ctr/asgn for ports for sake of easier operation on
  // `Graph` and `Runner` to avoid usage of smart pointers.
  TagAndIndex& operator=(TagAndIndex&& other) = default;
  TagAndIndex(TagAndIndex&& other) = default;

  absl::string_view Tag() const ABSL_ATTRIBUTE_LIFETIME_BOUND {
    return tag_->Get();
  }

  int Index() const { return index_; }

 private:
  std::unique_ptr<PortTag> tag_;
  int index_ = 0;
};

template <typename C>
bool HasTagAndIndex(const C& c, const TagAndIndex& tag_and_index) {
  return c.HasTag(tag_and_index.Tag()) &&
         tag_and_index.Index() < c.NumEntries(tag_and_index.Tag());
}

template <typename V, typename CC>
void SetCalculatorContract(V& v, CC& contract) {
  ABSL_CHECK(v.contract_ == nullptr);
  v.contract_ = &contract;
}

template <typename V, typename G>
void SetGraph(V& v, G& graph) {
  ABSL_CHECK(v.graph_ == nullptr);
  v.graph_ = &graph;
}

template <typename V, typename N>
void SetNode(V& v, N& node) {
  ABSL_CHECK(v.node_ == nullptr);
  v.node_ = &node;
}

template <typename V, typename CC>
void SetCalculatorContext(V& v, CC& context) {
  ABSL_CHECK(v.context_ == nullptr);
  v.context_ = &context;
}

template <typename V>
void SetCalculatorGraphAndName(V& v, CalculatorGraph& calculator_graph,
                               const std::string& name) {
  ABSL_CHECK(v.calculator_graph_ == nullptr);
  v.calculator_graph_ = &calculator_graph;
  v.name_ = name;
}

template <typename SpecializerT, typename FieldT>
class Port;

template <typename FieldT>
class Port<ContractSpecializer, FieldT> : public TagAndIndex {
 public:
  using Field = FieldT;
  using Specializer = ContractSpecializer;
  using TagAndIndex::TagAndIndex;

 protected:
  // IsConnected should be only public for "Optional" ports.
  bool IsConnected() const {
    if constexpr (std::is_same_v<FieldT, InputStreamField>) {
      return HasTagAndIndex(contract_->Inputs(), *this);
    } else if constexpr (std::is_same_v<FieldT, OutputStreamField>) {
      return HasTagAndIndex(contract_->Outputs(), *this);
    } else if constexpr (std::is_same_v<FieldT, InputSidePacketField>) {
      return HasTagAndIndex(contract_->InputSidePackets(), *this);
    } else if constexpr (std::is_same_v<FieldT, OutputSidePacketField>) {
      return HasTagAndIndex(contract_->OutputSidePackets(), *this);
    } else {
      static_assert(dependent_false<FieldT>::value, "Unexpected field type.");
    }
  }

  template <typename V, typename CC>
  friend void SetCalculatorContract(V& v, CC& contract);

  // Not owned, set by the framework.
  mediapipe::CalculatorContract* contract_ = nullptr;
};

template <typename FieldT>
class Port<ContextSpecializer, FieldT> : public TagAndIndex {
 public:
  using Field = FieldT;
  using Specializer = ContextSpecializer;
  using TagAndIndex::TagAndIndex;

 protected:
  // IsConnected should be only public for "Optional" ports.
  bool IsConnected() const {
    if constexpr (std::is_same_v<FieldT, InputStreamField>) {
      return HasTagAndIndex(context_->Inputs(), *this);
    } else if constexpr (std::is_same_v<FieldT, OutputStreamField>) {
      return HasTagAndIndex(context_->Outputs(), *this);
    } else if constexpr (std::is_same_v<FieldT, InputSidePacketField>) {
      return HasTagAndIndex(context_->InputSidePackets(), *this);
    } else if constexpr (std::is_same_v<FieldT, OutputSidePacketField>) {
      return HasTagAndIndex(context_->OutputSidePackets(), *this);
    } else {
      static_assert(dependent_false<FieldT>::value, "Unexpected field type.");
    }
  }

  template <typename V, typename CC>
  friend void SetCalculatorContext(V& v, CC& context);

  // Not owned, set by the framework.
  mediapipe::CalculatorContext* context_ = nullptr;
};

template <typename SpecializerT, typename FieldT>
class RepeatedBase;

template <typename FieldT>
class RepeatedBase<ContractSpecializer, FieldT> : public TagAndIndex {
 public:
  using TagAndIndex::TagAndIndex;

  int Count() const {
    ABSL_CHECK(contract_ != nullptr);
    if constexpr (std::is_same_v<FieldT, InputStreamField>) {
      return contract_->Inputs().NumEntries(Tag());
    } else if constexpr (std::is_same_v<FieldT, OutputStreamField>) {
      return contract_->Outputs().NumEntries(Tag());
    } else if constexpr (std::is_same_v<FieldT, InputSidePacketField>) {
      return contract_->InputSidePackets().NumEntries(Tag());
    } else if constexpr (std::is_same_v<FieldT, OutputSidePacketField>) {
      return contract_->OutputSidePackets().NumEntries(Tag());
    } else {
      static_assert(dependent_false<FieldT>::value, "Unexpected field type.");
    }
  }

 protected:
  void InitPort(Port<ContractSpecializer, FieldT>& p) const {
    SetCalculatorContract(p, *contract_);
  }

  template <typename V, typename CC>
  friend void SetCalculatorContract(V& v, CC& contract);

  // Not owned, set by the framework.
  mediapipe::CalculatorContract* contract_ = nullptr;
};

template <typename FieldT>
class RepeatedBase<ContextSpecializer, FieldT> : public TagAndIndex {
 public:
  using TagAndIndex::TagAndIndex;

  int Count() const {
    if constexpr (std::is_same_v<FieldT, InputStreamField>) {
      return context_->Inputs().NumEntries(Tag());
    } else if constexpr (std::is_same_v<FieldT, OutputStreamField>) {
      return context_->Outputs().NumEntries(Tag());
    } else if constexpr (std::is_same_v<FieldT, InputSidePacketField>) {
      return context_->InputSidePackets().NumEntries(Tag());
    } else if constexpr (std::is_same_v<FieldT, OutputSidePacketField>) {
      return context_->OutputSidePackets().NumEntries(Tag());
    } else {
      static_assert(dependent_false<FieldT>::value, "Unexpected field type.");
    }
  }

 protected:
  void InitPort(Port<ContextSpecializer, FieldT>& p) const {
    SetCalculatorContext(p, *context_);
  }

  template <typename V, typename CC>
  friend void SetCalculatorContext(V& v, CC& context);

  // Not owned, set by the framework.
  mediapipe::CalculatorContext* context_ = nullptr;
};

// Adds `port` to the generic `contract`.
template <typename P, typename CC>
absl::Status AddToContract(P& port, CC& contract, bool optional = false) {
  using Field = typename std::remove_const_t<P>::Field;
  if constexpr (std::is_same_v<Field, InputStreamField>) {
    using Payload = typename std::remove_const_t<P>::Payload;
    auto& v = contract.Inputs().Get(port.Tag(), port.Index());
    v.template Set<Payload>();
    if (optional) {
      v.Optional();
    }
  } else if constexpr (std::is_same_v<Field, OutputStreamField>) {
    using Payload = typename std::remove_const_t<P>::Payload;
    auto& v = contract.Outputs().Get(port.Tag(), port.Index());
    v.template Set<Payload>();
    if (optional) {
      v.Optional();
    }
  } else if constexpr (std::is_same_v<Field, InputSidePacketField>) {
    using Payload = typename std::remove_const_t<P>::Payload;
    auto& v = contract.InputSidePackets().Get(port.Tag(), port.Index());
    v.template Set<Payload>();
    if (optional) {
      v.Optional();
    }
  } else if constexpr (std::is_same_v<Field, OutputSidePacketField>) {
    using Payload = typename std::remove_const_t<P>::Payload;
    auto& v = contract.OutputSidePackets().Get(port.Tag(), port.Index());
    v.template Set<Payload>();
    if (optional) {
      v.Optional();
    }
  } else if constexpr (std::is_same_v<Field, RepeatedField>) {
    for (int i = 0; i < port.Count(); ++i) {
      MP_RETURN_IF_ERROR(AddToContract(port.At(i), contract, optional));
    }
  } else if constexpr (std::is_same_v<Field, OptionalField>) {
    const typename P::Contained& value = port;
    MP_RETURN_IF_ERROR(AddToContract(value, contract, /*optional*/ true));
  } else if constexpr (std::is_same_v<Field, OptionsField>) {
    // Nothing to update in the contract for options.
  } else {
    static_assert(dependent_false_v<Field>, "Unsupported field type.");
  }
  return absl::OkStatus();
}

}  // namespace internal_port

}  // namespace mediapipe::api3

#endif  // MEDIAPIPE_FRAMEWORK_API3_INTERNAL_PORT_BASE_H_
