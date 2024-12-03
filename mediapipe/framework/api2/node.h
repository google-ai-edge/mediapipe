#ifndef MEDIAPIPE_FRAMEWORK_API2_NODE_H_
#define MEDIAPIPE_FRAMEWORK_API2_NODE_H_

#include <memory>
#include <type_traits>

#include "absl/status/status.h"
#include "mediapipe/framework/api2/contract.h"
#include "mediapipe/framework/calculator_base.h"
#include "mediapipe/framework/calculator_context.h"
#include "mediapipe/framework/calculator_contract.h"
#include "mediapipe/framework/deps/registration.h"
#include "mediapipe/framework/subgraph.h"

namespace mediapipe {
namespace api2 {

class NodeIntf {};

class Node : public CalculatorBase {
 public:
  virtual ~Node();
};

}  // namespace api2

namespace internal {

template <class T>
class CalculatorBaseFactoryFor<
    T,
    typename std::enable_if<std::is_base_of<mediapipe::api2::Node, T>{}>::type>
    : public CalculatorBaseFactory {
 public:
  absl::Status GetContract(CalculatorContract* cc) final {
    auto status = T::Contract::GetContract(cc);
    if (status.ok()) {
      status = UpdateContract<T>(cc);
    }
    return status;
  }

  std::unique_ptr<CalculatorBase> CreateCalculator(
      CalculatorContext* calculator_context) final {
    return std::make_unique<T>();
  }

 private:
  template <typename U>
  auto UpdateContract(CalculatorContract* cc)
      -> decltype(U::UpdateContract(cc)) {
    return U::UpdateContract(cc);
  }
  template <typename U>
  absl::Status UpdateContract(...) {
    return {};
  }
};

}  // namespace internal

namespace api2 {
namespace internal {

MEDIAPIPE_STATIC_REGISTRATOR_TEMPLATE(
    NodeRegistrator, mediapipe::CalculatorBaseRegistry, T::kCalculatorName,
    std::make_unique<mediapipe::internal::CalculatorBaseFactoryFor<T>>)

MEDIAPIPE_STATIC_REGISTRATOR_TEMPLATE(SubgraphRegistrator,
                                      mediapipe::SubgraphRegistry,
                                      T::kCalculatorName, std::make_unique<T>)

}  // namespace internal

// By passing the Impl parameter, registration is done automatically. No need
// to use MEDIAPIPE_NODE_IMPLEMENTATION.
// For backward compatibility, Impl can be omitted; use
// MEDIAPIPE_NODE_IMPLEMENTATION with this.
// TODO: migrate and remove.
template <class Impl = void>
class RegisteredNode;

template <class Impl>
class RegisteredNode : public Node, private internal::NodeRegistrator<Impl> {};

// No-op version for backwards compatibility.
template <>
class RegisteredNode<void> : public Node {};

template <class Impl>
struct FunctionNode : public RegisteredNode<Impl> {
  absl::Status Process(CalculatorContext* cc) override {
    return internal::ProcessFnCallers(cc, Impl::kContract.process_items());
  }
};

template <class Intf, class Impl = void>
class NodeImpl : public RegisteredNode<Impl>, public Intf {
 protected:
  // These methods allow accessing a node's ports by tag. This can be useful in
  // a few cases, e.g. if the port is not available as a named constant.
  // They parallel the corresponding methods on builder nodes.
  template <class Tag>
  static constexpr auto Out(Tag t) {
    return Intf::Contract::TaggedOutputs::get(t);
  }

  template <class Tag>
  static constexpr auto In(Tag t) {
    return Intf::Contract::TaggedInputs::get(t);
  }

  template <class Tag>
  static constexpr auto SideOut(Tag t) {
    return Intf::Contract::TaggedSideOutputs::get(t);
  }

  template <class Tag>
  static constexpr auto SideIn(Tag t) {
    return Intf::Contract::TaggedSideInputs::get(t);
  }

  // Convenience.
  template <class Tag, class CC>
  static auto Out(Tag t, CC cc) {
    return Out(t)(cc);
  }
  template <class Tag, class CC>
  static auto In(Tag t, CC cc) {
    return In(t)(cc);
  }
  template <class Tag, class CC>
  static auto SideOut(Tag t, CC cc) {
    return SideOut(t)(cc);
  }
  template <class Tag, class CC>
  static auto SideIn(Tag t, CC cc) {
    return SideIn(t)(cc);
  }
};

// This macro is used to define the contract, without also giving the
// node a type name. It can be used directly in pure interfaces.
#define MEDIAPIPE_NODE_CONTRACT(...)                                          \
  static constexpr auto kContract =                                           \
      mediapipe::api2::internal::MakeContract(__VA_ARGS__);                   \
  using Contract =                                                            \
      typename mediapipe::api2::internal::TaggedContract<decltype(kContract), \
                                                         kContract>;

// This macro is used to define the contract and the type name of a node.
// This saves the name of the calculator, making it available to the
// implementation too, and to the registration macro for it. The reason is
// that the name must be available with the contract (so that it can be used
// to build a graph config, for instance); however, it is the implementation
// that needs to be registered.
// TODO: rename to MEDIAPIPE_NODE_DECLARATION?
// TODO: more detailed explanation.
#define MEDIAPIPE_NODE_INTERFACE(name, ...)        \
  static constexpr char kCalculatorName[] = #name; \
  MEDIAPIPE_NODE_CONTRACT(__VA_ARGS__)

// TODO: verify that the subgraph config fully implements the
// declared interface.
template <class Intf, class Impl>
class SubgraphImpl : public Subgraph,
                     public Intf,
                     private internal::SubgraphRegistrator<Impl> {};

// This macro is used to register a calculator that does not use automatic
// registration. Deprecated.
#define MEDIAPIPE_NODE_IMPLEMENTATION(Impl)                       \
  MEDIAPIPE_REGISTER_FACTORY_FUNCTION_QUALIFIED(                  \
      mediapipe::CalculatorBaseRegistry, calculator_registration, \
      Impl::kCalculatorName,                                      \
      std::make_unique<mediapipe::internal::CalculatorBaseFactoryFor<Impl>>)

// This macro is used to register a non-split-contract calculator. Deprecated.
#define MEDIAPIPE_REGISTER_NODE(name)                                    \
  MEDIAPIPE_REGISTER_FACTORY_FUNCTION_QUALIFIED(                         \
      mediapipe::CalculatorBaseRegistry, calculator_registration, #name, \
      std::make_unique<mediapipe::internal::CalculatorBaseFactoryFor<name>>)

// This macro is used to define a subgraph that does not use automatic
// registration. Deprecated.
#define MEDIAPIPE_SUBGRAPH_IMPLEMENTATION(Impl)           \
  MEDIAPIPE_REGISTER_FACTORY_FUNCTION_QUALIFIED(          \
      mediapipe::SubgraphRegistry, subgraph_registration, \
      Impl::kCalculatorName, std::make_unique<Impl>)

}  // namespace api2
}  // namespace mediapipe

#endif  // MEDIAPIPE_FRAMEWORK_API2_NODE_H_
