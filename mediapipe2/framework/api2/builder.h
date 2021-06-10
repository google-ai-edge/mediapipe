#ifndef MEDIAPIPE_FRAMEWORK_API2_BUILDER_H_
#define MEDIAPIPE_FRAMEWORK_API2_BUILDER_H_

#include <string>
#include <type_traits>

#include "absl/container/flat_hash_map.h"
#include "mediapipe/framework/api2/const_str.h"
#include "mediapipe/framework/api2/contract.h"
#include "mediapipe/framework/api2/node.h"
#include "mediapipe/framework/api2/packet.h"
#include "mediapipe/framework/api2/port.h"
#include "mediapipe/framework/calculator_base.h"
#include "mediapipe/framework/calculator_contract.h"

namespace mediapipe {
namespace api2 {
namespace builder {

template <typename T>
T& GetWithAutoGrow(std::vector<std::unique_ptr<T>>* vecp, int index) {
  auto& vec = *vecp;
  if (vec.size() <= index) {
    vec.resize(index + 1);
  }
  if (vec[index] == nullptr) {
    vec[index] = absl::make_unique<T>();
  }
  return *vec[index];
}

struct TagIndexLocation {
  const std::string& tag;
  std::size_t index;
  std::size_t count;
};

template <typename T>
class TagIndexMap {
 public:
  std::vector<std::unique_ptr<T>>& operator[](const std::string& tag) {
    return map_[tag];
  }

  void Visit(std::function<void(const TagIndexLocation&, const T&)> fun) const {
    for (const auto& tagged : map_) {
      TagIndexLocation loc{tagged.first, 0, tagged.second.size()};
      for (const auto& item : tagged.second) {
        fun(loc, *item);
        ++loc.index;
      }
    }
  }

  void Visit(std::function<void(const TagIndexLocation&, T*)> fun) {
    for (auto& tagged : map_) {
      TagIndexLocation loc{tagged.first, 0, tagged.second.size()};
      for (auto& item : tagged.second) {
        fun(loc, item.get());
        ++loc.index;
      }
    }
  }

  // Note: entries are held by a unique_ptr to ensure pointers remain valid.
  // Should use absl::flat_hash_map but ordering keys for now.
  std::map<std::string, std::vector<std::unique_ptr<T>>> map_;
};

// These structs are used internally to store information about the endpoints
// of a connection.
struct SourceBase;
struct DestinationBase {
  SourceBase* source = nullptr;
};
struct SourceBase {
  std::vector<DestinationBase*> dests_;
  std::string name_;
};

// Following existing GraphConfig usage, we allow using a multiport as a single
// port as well. This is necessary for generic nodes, since we have no
// information about which ports are meant to be multiports or not, but it is
// also convenient with typed nodes.
template <typename Single>
class MultiPort : public Single {
 public:
  using Base = typename Single::Base;

  explicit MultiPort(std::vector<std::unique_ptr<Base>>* vec)
      : Single(vec), vec_(*vec) {}

  Single operator[](int index) {
    CHECK_GE(index, 0);
    return Single{&GetWithAutoGrow(&vec_, index)};
  }

 private:
  std::vector<std::unique_ptr<Base>>& vec_;
};

// These classes wrap references to the underlying source/destination
// endpoints, adding type information and the user-visible API.
template <bool AllowMultiple, bool IsSide, typename T = internal::Generic>
class DestinationImpl {
 public:
  using Base = DestinationBase;

  explicit DestinationImpl(std::vector<std::unique_ptr<Base>>* vec)
      : DestinationImpl(&GetWithAutoGrow(vec, 0)) {}
  explicit DestinationImpl(DestinationBase* base) : base_(*base) {}
  DestinationBase& base_;
};

template <bool IsSide, typename T>
class DestinationImpl<true, IsSide, T>
    : public MultiPort<DestinationImpl<false, IsSide, T>> {
 public:
  using MultiPort<DestinationImpl<false, IsSide, T>>::MultiPort;
};

template <bool AllowMultiple, bool IsSide, typename T = internal::Generic>
class SourceImpl {
 public:
  using Base = SourceBase;

  // Src is used as the return type of fluent methods below. Since these are
  // single-port methods, it is desirable to always decay to a reference to the
  // single-port superclass, even if they are called on a multiport.
  using Src = SourceImpl<false, IsSide, T>;
  template <typename U>
  using Dst = DestinationImpl<false, IsSide, U>;

  // clang-format off
  template <typename U>
  struct AllowConnection : public std::integral_constant<bool,
      std::is_same<T, U>{} || std::is_same<T, internal::Generic>{} ||
      std::is_same<U, internal::Generic>{}> {};
  // clang-format on

  explicit SourceImpl(std::vector<std::unique_ptr<Base>>* vec)
      : SourceImpl(&GetWithAutoGrow(vec, 0)) {}
  explicit SourceImpl(SourceBase* base) : base_(*base) {}

  template <typename U,
            typename std::enable_if<AllowConnection<U>{}, int>::type = 0>
  Src& AddTarget(const Dst<U>& dest) {
    CHECK(dest.base_.source == nullptr);
    dest.base_.source = &base_;
    base_.dests_.emplace_back(&dest.base_);
    return *this;
  }
  Src& SetName(std::string name) {
    base_.name_ = std::move(name);
    return *this;
  }
  template <typename U>
  Src& operator>>(const Dst<U>& dest) {
    return AddTarget(dest);
  }

 private:
  SourceBase& base_;
};

template <bool IsSide, typename T>
class SourceImpl<true, IsSide, T>
    : public MultiPort<SourceImpl<false, IsSide, T>> {
 public:
  using MultiPort<SourceImpl<false, IsSide, T>>::MultiPort;
};

// A source and a destination correspond to an output/input stream on a node,
// and a side source and side destination correspond to an output/input side
// packet.
// For graph inputs/outputs, however, the inputs are sources, and the outputs
// are destinations. This is because graph ports are connected "from inside"
// when building the graph.
template <bool AllowMultiple = false, typename T = internal::Generic>
using Source = SourceImpl<AllowMultiple, false, T>;
template <bool AllowMultiple = false, typename T = internal::Generic>
using SideSource = SourceImpl<AllowMultiple, true, T>;
template <bool AllowMultiple = false, typename T = internal::Generic>
using Destination = DestinationImpl<AllowMultiple, false, T>;
template <bool AllowMultiple = false, typename T = internal::Generic>
using SideDestination = DestinationImpl<AllowMultiple, true, T>;

class NodeBase {
 public:
  // TODO: right now access to an indexed port is made directly by
  // specifying both a tag and an index. It would be better to represent this
  // as a two-step lookup, first getting a multi-port, and then accessing one
  // of its entries by index. However, for nodes without visible contracts we
  // can't know whether a tag is indexable or not, so we would need the
  // multi-port to also be usable as a port directly (representing index 0).
  Source<true> Out(const std::string& tag) {
    return Source<true>(&out_streams_[tag]);
  }

  Destination<true> In(const std::string& tag) {
    return Destination<true>(&in_streams_[tag]);
  }

  SideSource<true> SideOut(const std::string& tag) {
    return SideSource<true>(&out_sides_[tag]);
  }

  SideDestination<true> SideIn(const std::string& tag) {
    return SideDestination<true>(&in_sides_[tag]);
  }

  // Convenience methods for accessing purely index-based ports.
  Source<false> Out(int index) { return Out("")[index]; }

  Destination<false> In(int index) { return In("")[index]; }

  SideSource<false> SideOut(int index) { return SideOut("")[index]; }

  SideDestination<false> SideIn(int index) { return SideIn("")[index]; }

  template <typename T>
  T& GetOptions() {
    options_used_ = true;
    return *options_.MutableExtension(T::ext);
  }

 protected:
  NodeBase(std::string type) : type_(std::move(type)) {}

  std::string type_;
  TagIndexMap<DestinationBase> in_streams_;
  TagIndexMap<SourceBase> out_streams_;
  TagIndexMap<DestinationBase> in_sides_;
  TagIndexMap<SourceBase> out_sides_;
  CalculatorOptions options_;
  // ideally we'd just check if any extensions are set on options_
  bool options_used_ = false;
  friend class Graph;
};

template <class Calc = internal::Generic>
class Node;
#if __cplusplus >= 201703L
// Deduction guide to silence -Wctad-maybe-unsupported.
explicit Node()->Node<internal::Generic>;
#endif  // C++17

template <>
class Node<internal::Generic> : public NodeBase {
 public:
  Node(std::string type) : NodeBase(std::move(type)) {}
};

using GenericNode = Node<internal::Generic>;

template <template <bool, class> class BP, class Port, class TagIndexMapT>
auto MakeBuilderPort(const Port& port, TagIndexMapT& streams) {
  return BP<Port::kMultiple, typename Port::PayloadT>(&streams[port.Tag()]);
}

template <class Calc>
class Node : public NodeBase {
 public:
  Node() : NodeBase(Calc::kCalculatorName) {}
  // Overrides the built-in calculator type std::string with the provided
  // argument. Can be used to create nodes from pure interfaces.
  // TODO: only use this for pure interfaces
  Node(const std::string& type_override) : NodeBase(type_override) {}

  // These methods only allow access to ports declared in the contract.
  // The argument must be a tag object created with the MPP_TAG macro.
  // These objects encode the tag in their type, which allows us to return
  // a result with the appropriate payload type depending on the tag.
  template <class Tag>
  auto Out(Tag tag) {
    constexpr auto& port = Calc::Contract::TaggedOutputs::get(tag);
    return MakeBuilderPort<Source>(port, out_streams_);
  }

  template <class Tag>
  auto In(Tag tag) {
    constexpr auto& port = Calc::Contract::TaggedInputs::get(tag);
    return MakeBuilderPort<Destination>(port, in_streams_);
  }

  template <class Tag>
  auto SideOut(Tag tag) {
    constexpr auto& port = Calc::Contract::TaggedSideOutputs::get(tag);
    return MakeBuilderPort<SideSource>(port, out_sides_);
  }

  template <class Tag>
  auto SideIn(Tag tag) {
    constexpr auto& port = Calc::Contract::TaggedSideInputs::get(tag);
    return MakeBuilderPort<SideDestination>(port, in_sides_);
  }

  // We could allow using the non-checked versions with typed nodes too, but
  // we don't.
  // using NodeBase::Out;
  // using NodeBase::In;
  // using NodeBase::SideOut;
  // using NodeBase::SideIn;
};

// For legacy PacketGenerators.
class PacketGenerator {
 public:
  PacketGenerator(std::string type) : type_(std::move(type)) {}

  SideSource<true> SideOut(const std::string& tag) {
    return SideSource<true>(&out_sides_[tag]);
  }

  SideDestination<true> SideIn(const std::string& tag) {
    return SideDestination<true>(&in_sides_[tag]);
  }

  // Convenience methods for accessing purely index-based ports.
  SideSource<false> SideOut(int index) { return SideOut("")[index]; }
  SideDestination<false> SideIn(int index) { return SideIn("")[index]; }

  template <typename T>
  T& GetOptions() {
    options_used_ = true;
    return *options_.MutableExtension(T::ext);
  }

 private:
  std::string type_;
  TagIndexMap<DestinationBase> in_sides_;
  TagIndexMap<SourceBase> out_sides_;
  mediapipe::PacketGeneratorOptions options_;
  // ideally we'd just check if any extensions are set on options_
  bool options_used_ = false;
  friend class Graph;
};

class Graph {
 public:
  void SetType(std::string type) { type_ = std::move(type); }

  // Creates a node of a specific type. Should be used for calculators whose
  // contract is available.
  template <class Calc>
  Node<Calc>& AddNode() {
    auto node = std::make_unique<Node<Calc>>();
    auto node_p = node.get();
    nodes_.emplace_back(std::move(node));
    return *node_p;
  }

  // Creates a node of a specific type. Should be used for pure interfaces,
  // which do not have a built-in type std::string.
  template <class Calc>
  Node<Calc>& AddNode(const std::string& type) {
    auto node = std::make_unique<Node<Calc>>(type);
    auto node_p = node.get();
    nodes_.emplace_back(std::move(node));
    return *node_p;
  }

  // Creates a generic node, with no compile-time checking of inputs and
  // outputs. This can be used for calculators whose contract is not visible.
  GenericNode& AddNode(const std::string& type) {
    auto node = std::make_unique<GenericNode>(type);
    auto node_p = node.get();
    nodes_.emplace_back(std::move(node));
    return *node_p;
  }

  // For legacy PacketGenerators.
  PacketGenerator& AddPacketGenerator(const std::string& type) {
    auto node = std::make_unique<PacketGenerator>(type);
    auto node_p = node.get();
    packet_gens_.emplace_back(std::move(node));
    return *node_p;
  }

  // Graph ports, non-typed.
  Source<true> In(const std::string& graph_input) {
    return graph_boundary_.Out(graph_input);
  }

  Destination<true> Out(const std::string& graph_output) {
    return graph_boundary_.In(graph_output);
  }

  SideSource<true> SideIn(const std::string& graph_input) {
    return graph_boundary_.SideOut(graph_input);
  }

  SideDestination<true> SideOut(const std::string& graph_output) {
    return graph_boundary_.SideIn(graph_output);
  }

  // Convenience methods for accessing purely index-based ports.
  Source<false> In(int index) { return In("")[0]; }

  Destination<false> Out(int index) { return Out("")[0]; }

  SideSource<false> SideIn(int index) { return SideIn("")[0]; }

  SideDestination<false> SideOut(int index) { return SideOut("")[0]; }

  // Graph ports, typed.
  // TODO: make graph_boundary_ a typed node!
  template <class PortT, class Payload = typename PortT::PayloadT,
            class Src = Source<PortT::kMultiple, Payload>>
  Src In(const PortT& graph_input) {
    return Src(&graph_boundary_.out_streams_[graph_input.Tag()]);
  }

  template <class PortT, class Payload = typename PortT::PayloadT,
            class Dst = Destination<PortT::kMultiple, Payload>>
  Dst Out(const PortT& graph_output) {
    return Dst(&graph_boundary_.in_streams_[graph_output.Tag()]);
  }

  template <class PortT, class Payload = typename PortT::PayloadT,
            class Src = SideSource<PortT::kMultiple, Payload>>
  Src SideIn(const PortT& graph_input) {
    return Src(&graph_boundary_.out_sides_[graph_input.Tag()]);
  }

  template <class PortT, class Payload = typename PortT::PayloadT,
            class Dst = SideDestination<PortT::kMultiple, Payload>>
  Dst SideOut(const PortT& graph_output) {
    return Dst(&graph_boundary_.in_sides_[graph_output.Tag()]);
  }

  // Returns the graph config. This can be used to instantiate and run the
  // graph.
  CalculatorGraphConfig GetConfig() {
    CalculatorGraphConfig config;
    if (!type_.empty()) {
      config.set_type(type_);
    }
    FixUnnamedConnections();
    CHECK_OK(UpdateBoundaryConfig(&config));
    for (const std::unique_ptr<NodeBase>& node : nodes_) {
      auto* out_node = config.add_node();
      CHECK_OK(UpdateNodeConfig(*node, out_node));
    }
    for (const std::unique_ptr<PacketGenerator>& node : packet_gens_) {
      auto* out_node = config.add_packet_generator();
      CHECK_OK(UpdateNodeConfig(*node, out_node));
    }
    return config;
  }

 private:
  void FixUnnamedConnections(NodeBase* node, int* unnamed_count) {
    node->out_streams_.Visit([&](const TagIndexLocation&, SourceBase* source) {
      if (source->name_.empty()) {
        source->name_ = absl::StrCat("__stream_", (*unnamed_count)++);
      }
    });
    node->out_sides_.Visit([&](const TagIndexLocation&, SourceBase* source) {
      if (source->name_.empty()) {
        source->name_ = absl::StrCat("__side_packet_", (*unnamed_count)++);
      }
    });
  }

  void FixUnnamedConnections() {
    int unnamed_count = 0;
    FixUnnamedConnections(&graph_boundary_, &unnamed_count);
    for (std::unique_ptr<NodeBase>& node : nodes_) {
      FixUnnamedConnections(node.get(), &unnamed_count);
    }
    for (std::unique_ptr<PacketGenerator>& node : packet_gens_) {
      node->out_sides_.Visit([&](const TagIndexLocation&, SourceBase* source) {
        if (source->name_.empty()) {
          source->name_ = absl::StrCat("__side_packet_", unnamed_count++);
        }
      });
    }
  }

  std::string TaggedName(const TagIndexLocation& loc, const std::string& name) {
    if (loc.tag.empty()) {
      // ParseTagIndexName does not allow using explicit indices without tags,
      // while ParseTagIndex does.
      // TODO: decide whether we should just allow it.
      return name;
    } else {
      if (loc.count <= 1) {
        return absl::StrCat(loc.tag, ":", name);
      } else {
        return absl::StrCat(loc.tag, ":", loc.index, ":", name);
      }
    }
  }

  absl::Status UpdateNodeConfig(const NodeBase& node,
                                CalculatorGraphConfig::Node* config) {
    config->set_calculator(node.type_);
    node.in_streams_.Visit(
        [&](const TagIndexLocation& loc, const DestinationBase& endpoint) {
          CHECK(endpoint.source != nullptr);
          config->add_input_stream(TaggedName(loc, endpoint.source->name_));
        });
    node.out_streams_.Visit(
        [&](const TagIndexLocation& loc, const SourceBase& endpoint) {
          config->add_output_stream(TaggedName(loc, endpoint.name_));
        });
    node.in_sides_.Visit([&](const TagIndexLocation& loc,
                             const DestinationBase& endpoint) {
      CHECK(endpoint.source != nullptr);
      config->add_input_side_packet(TaggedName(loc, endpoint.source->name_));
    });
    node.out_sides_.Visit(
        [&](const TagIndexLocation& loc, const SourceBase& endpoint) {
          config->add_output_side_packet(TaggedName(loc, endpoint.name_));
        });
    if (node.options_used_) {
      *config->mutable_options() = node.options_;
    }
    return {};
  }

  absl::Status UpdateNodeConfig(const PacketGenerator& node,
                                PacketGeneratorConfig* config) {
    config->set_packet_generator(node.type_);
    node.in_sides_.Visit([&](const TagIndexLocation& loc,
                             const DestinationBase& endpoint) {
      CHECK(endpoint.source != nullptr);
      config->add_input_side_packet(TaggedName(loc, endpoint.source->name_));
    });
    node.out_sides_.Visit(
        [&](const TagIndexLocation& loc, const SourceBase& endpoint) {
          config->add_output_side_packet(TaggedName(loc, endpoint.name_));
        });
    if (node.options_used_) {
      *config->mutable_options() = node.options_;
    }
    return {};
  }

  // For special boundary node.
  absl::Status UpdateBoundaryConfig(CalculatorGraphConfig* config) {
    graph_boundary_.in_streams_.Visit(
        [&](const TagIndexLocation& loc, const DestinationBase& endpoint) {
          CHECK(endpoint.source != nullptr);
          config->add_output_stream(TaggedName(loc, endpoint.source->name_));
        });
    graph_boundary_.out_streams_.Visit(
        [&](const TagIndexLocation& loc, const SourceBase& endpoint) {
          config->add_input_stream(TaggedName(loc, endpoint.name_));
        });
    graph_boundary_.in_sides_.Visit([&](const TagIndexLocation& loc,
                                        const DestinationBase& endpoint) {
      CHECK(endpoint.source != nullptr);
      config->add_output_side_packet(TaggedName(loc, endpoint.source->name_));
    });
    graph_boundary_.out_sides_.Visit(
        [&](const TagIndexLocation& loc, const SourceBase& endpoint) {
          config->add_input_side_packet(TaggedName(loc, endpoint.name_));
        });
    return {};
  }

  std::string type_;
  std::vector<std::unique_ptr<NodeBase>> nodes_;
  std::vector<std::unique_ptr<PacketGenerator>> packet_gens_;
  // Special node representing graph inputs and outputs.
  NodeBase graph_boundary_{"__GRAPH__"};
};

}  // namespace builder
}  // namespace api2
}  // namespace mediapipe

#endif  // MEDIAPIPE_FRAMEWORK_API2_BUILDER_H_
