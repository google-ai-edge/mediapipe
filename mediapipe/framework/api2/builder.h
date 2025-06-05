#ifndef MEDIAPIPE_FRAMEWORK_API2_BUILDER_H_
#define MEDIAPIPE_FRAMEWORK_API2_BUILDER_H_

#include <memory>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "absl/log/absl_check.h"
#include "absl/strings/string_view.h"
#include "google/protobuf/message_lite.h"
#include "mediapipe/framework/api2/packet.h"
#include "mediapipe/framework/api2/port.h"
#include "mediapipe/framework/api3/graph.h"
#include "mediapipe/framework/api3/internal/graph_builder.h"
#include "mediapipe/framework/api3/side_packet.h"
#include "mediapipe/framework/api3/stream.h"
#include "mediapipe/framework/calculator_base.h"
#include "mediapipe/framework/calculator_contract.h"
#include "mediapipe/framework/deps/registration.h"
#include "mediapipe/framework/mediapipe_options.pb.h"
#include "mediapipe/framework/stream_handler.pb.h"

namespace mediapipe {
namespace api2 {
namespace builder {

// Workaround for static_assert(false). Example:
//   dependent_false<T>::value returns false.
// For more information, see:
// https://en.cppreference.com/w/cpp/language/if#Constexpr_If
// TODO: migrate to a common utility when available.
template <class T>
struct dependent_false : std::false_type {};

class Graph;
class NodeBase;
class PacketGenerator;

// Following existing GraphConfig usage, we allow using a multiport as a single
// port as well. This is necessary for generic nodes, since we have no
// information about which ports are meant to be multiports or not, but it is
// also convenient with typed nodes.
//
// NOTE: because MultiPort extends Single, there's a special test case where
// Graph::GetConfig() causes a crash - this happens when repeated input has been
// accessed by incorrect index: e.g. accessing node.In("REPEATED")[1], while not
// accessing [0].
template <typename Single>
class MultiPort : public Single {
 public:
  using Base = typename Single::Base;

  explicit MultiPort(api3::builder::Multi<Base> vec) : Single(vec), vec_(vec) {}

  Single operator[](int index) {
    ABSL_CHECK_GE(index, 0);
    return Single{&vec_.At(index)};
  }

  template <typename U>
  auto Cast() {
    using SingleCastT =
        std::invoke_result_t<decltype(&Single::template Cast<U>), Single*>;
    return MultiPort<SingleCastT>(vec_);
  }

 private:
  api3::builder::Multi<Base> vec_;
};

namespace internal_builder {

template <typename T, typename U>
using AllowCast = std::integral_constant<bool, (std::is_same_v<T, AnyType> ||
                                                std::is_same_v<U, AnyType>) &&
                                                   !std::is_same_v<T, U>>;

}  // namespace internal_builder

template <bool IsSide, typename T = internal::Generic>
class SourceImpl;

template <bool IsSide, typename T = internal::Generic>
class DestinationImpl;

// These classes wrap references to the underlying source/destination
// endpoints, adding type information and the user-visible API.
template <typename T>
class DestinationImpl</*IsSide=*/false, T> {
 public:
  static constexpr bool kIsSide = false;
  using Base = api3::builder::Destination;

  explicit DestinationImpl(api3::builder::Multi<Base> vec)
      : DestinationImpl(&vec.At(0)) {}
  explicit DestinationImpl(Base* base) : base_(*base) {}

  template <typename U,
            std::enable_if_t<internal_builder::AllowCast<T, U>{}, int> = 0>
  DestinationImpl<kIsSide, U> Cast() {
    return DestinationImpl<kIsSide, U>(&base_);
  }

  // Whether the input stream is a back edge.
  //
  // By default, MediaPipe requires graphs to be acyclic and treats cycles in a
  // graph as errors. To allow MediaPipe to accept a cyclic graph, use/make
  // corresponding inputs as back edges. A cyclic graph usually has an obvious
  // forward direction, and a back edge goes in the opposite direction. For a
  // formal definition of a back edge, please see
  // https://en.wikipedia.org/wiki/Depth-first_search.
  //
  // Equivalent of having "input_stream_info" for an input stream in the config:
  //   node {
  //     ...
  //     input_stream: "TAG:0:stream"
  //     input_stream_info {
  //       tag: "TAG:0"
  //       back_edge: true
  //     }
  //   }
  DestinationImpl<kIsSide, T>& AsBackEdge() {
    base_.back_edge = true;
    return *this;
  }

  const std::string& Name() const {
    ABSL_CHECK(base_.source != nullptr)
        << "Destination is not connected to a source.";
    return base_.source->name;
  }

 private:
  Base& base_;

  template <bool Source_IsSide, typename Source_T>
  friend class SourceImpl;
};

template <typename T>
class DestinationImpl</*IsSide=*/true, T> {
 public:
  static constexpr bool kIsSide = true;
  using Base = api3::builder::SideDestination;

  explicit DestinationImpl(api3::builder::Multi<Base> vec)
      : DestinationImpl(&vec.At(0)) {}
  explicit DestinationImpl(Base* base) : base_(*base) {}

  template <typename U,
            std::enable_if_t<internal_builder::AllowCast<T, U>{}, int> = 0>
  DestinationImpl<kIsSide, U> Cast() {
    return DestinationImpl<kIsSide, U>(&base_);
  }

 private:
  Base& base_;

  template <bool Source_IsSide, typename Source_T>
  friend class SourceImpl;
};

template <bool IsSide, typename T>
using SourceImplBase =
    std::conditional_t<IsSide, mediapipe::api3::SidePacket<T>,
                       mediapipe::api3::Stream<T>>;

template <bool IsSide, typename T>
class SourceImpl : public SourceImplBase<IsSide, T> {
 public:
  using Base = std::conditional_t<IsSide, api3::builder::SideSource,
                                  api3::builder::Source>;
  using PayloadT = T;

  // Src is used as the return type of fluent methods below. Since these are
  // single-port methods, it is desirable to always decay to a reference to the
  // single-port superclass, even if they are called on a multiport.
  using Src = SourceImpl<IsSide, T>;
  template <typename U>
  using Dst = DestinationImpl<IsSide, U>;

  // clang-format off
  template <typename U>
  struct AllowConnection : public std::integral_constant<bool,
      std::is_same<T, U>{} || std::is_same<T, internal::Generic>{} ||
      std::is_same<U, internal::Generic>{}> {};
  // clang-format on

  explicit SourceImpl(api3::builder::Multi<Base> src)
      : SourceImpl(&src.At(0)) {}
  explicit SourceImpl(Base* base) : SourceImplBase<IsSide, T>(*base) {}
  explicit SourceImpl(SourceImplBase<IsSide, T> source_base)
      : SourceImplBase<IsSide, T>(source_base) {}

  // Connects MediaPipe stream or side packet to a destination:
  // - node input (input stream) / side input (input side packet)
  // - graph output (output stream) / side output (output side packet).
  //
  // MediaPipe streams and side packets can be connected to multiple
  // destinations. Side packets and packets added to streams are sent to all
  // connected destinations.
  template <typename U,
            typename std::enable_if<AllowConnection<U>{}, int>::type = 0>
  Src& ConnectTo(const Dst<U>& dest) {
    SourceImplBase<IsSide, T>::GetBase()->ConnectTo(dest.base_);
    return *this;
  }

  // Shortcut for `ConnectTo`.
  //
  // Connects MediaPipe stream or side packet to a destination:
  // - node input (input stream) / side input (input side packet)
  // - graph output (output stream) / side output (output side packet).
  //
  // MediaPipe streams and side packets can be connected to multiple
  // destinations. Side packets and packets added to streams are sent to all
  // connected destinations.
  template <typename U>
  Src& operator>>(const Dst<U>& dest) {
    return ConnectTo(dest);
  }

  template <typename U>
  bool operator==(const SourceImpl<IsSide, U>& other) {
    return SourceImplBase<IsSide, T>::GetBase() == other.GetBase();
  }

  template <typename U>
  bool operator!=(const SourceImpl<IsSide, U>& other) {
    return !(*this == other);
  }

  const std::string& Name() const {
    return SourceImplBase<IsSide, T>::GetBase()->name;
  }

  Src& SetName(const char* name) {
    SourceImplBase<IsSide, T>::GetBase()->name = std::string(name);
    return *this;
  }

  Src& SetName(absl::string_view name) {
    SourceImplBase<IsSide, T>::GetBase()->name = std::string(name);
    return *this;
  }

  Src& SetName(std::string name) {
    SourceImplBase<IsSide, T>::GetBase()->name = std::move(name);
    return *this;
  }

  template <typename U,
            std::enable_if_t<internal_builder::AllowCast<T, U>{}, int> = 0>
  SourceImpl<IsSide, U> Cast() {
    return SourceImpl<IsSide, U>(SourceImplBase<IsSide, T>::GetBase());
  }

 private:
  template <bool, typename U>
  friend class SourceImpl;
};

// A source and a destination correspond to an output/input stream on a node,
// and a side source and side destination correspond to an output/input side
// packet.
// For graph inputs/outputs, however, the inputs are sources, and the outputs
// are destinations. This is because graph ports are connected "from inside"
// when building the graph.
template <typename T = internal::Generic>
using Source = SourceImpl<false, T>;

// Represents a stream of packets of a particular type.
//
// The intended use:
// - decouple input/output streams from graph/node during graph construction
// - pass streams around and connect them as needed, extracting reusable parts
//   to utility/convenience functions or classes.
//
// For example:
//   Stream<Image> Resize(Stream<Image> image, const Size& size, Graph& graph) {
//     auto& scaler_node = graph.AddNode("GlScalerCalculator");
//     auto& opts = scaler_node.GetOptions<GlScalerCalculatorOptions>();
//     opts.set_output_width(size.width);
//     opts.set_output_height(size.height);
//     a >> scaler_node.In("IMAGE");
//     return scaler_node.Out("IMAGE").Cast<Image>();
//   }
//
// Where graph can use it as:
//   Graph graph;
//   Stream<Image> input_image = graph.In("INPUT_IMAGE").Cast<Image>();
//   Stream<Image> resized_image = Resize(input_image, {64, 64}, graph);
template <typename T>
using Stream = Source<T>;

template <typename T = internal::Generic>
using MultiSource = MultiPort<Source<T>>;

template <typename T = internal::Generic>
using SideSource = SourceImpl<true, T>;

// Represents a side packet of a particular type.
//
// The intended use:
// - decouple input/output side packets from graph/node during graph
//   construction
// - pass side packets around and connect them as needed, extracting reusable
//   parts utility/convenience functions or classes.
//
// For example:
//   SidePacket<TfLiteModelPtr> GetModel(SidePacket<Resource> model_resource,
//                                       Graph& graph) {
//     auto& model_node = graph.AddNode("TfLiteModelCalculator");
//     model_resource >> model_node.SideIn("MODEL_RESOURCE");
//     return model_node.SideOut("MODEL").Cast<TfLiteModelPtr>();
//   }
//
// Where graph can use it as:
//   Graph graph;
//   SidePacket<Resource> model_resource =
//     graph.SideIn("MODEL_RESOURCE").Cast<Resource>();
//   SidePacket<TfLiteModelPtr> model = GetModel(model_resource, graph);
template <typename T>
using SidePacket = SideSource<T>;

template <typename T = internal::Generic>
using MultiSideSource = MultiPort<SideSource<T>>;

template <typename T = internal::Generic>
using Destination = DestinationImpl<false, T>;
template <typename T = internal::Generic>
using SideDestination = DestinationImpl<true, T>;
template <typename T = internal::Generic>
using MultiDestination = MultiPort<Destination<T>>;
template <typename T = internal::Generic>
using MultiSideDestination = MultiPort<SideDestination<T>>;

using Executor = api3::builder::Executor;
using InputStreamHandler = api3::builder::InputStreamHandler;
using OutputStreamHandler = api3::builder::OutputStreamHandler;

class NodeBase {
 public:
  explicit NodeBase(api3::builder::NodeBuilder& node_builder)
      : node_builder_(node_builder) {};

  ~NodeBase() = default;
  NodeBase(NodeBase&&) = default;
  NodeBase& operator=(NodeBase&&) = default;
  // Explicitly delete copies to improve error messages.
  NodeBase(const NodeBase&) = delete;
  NodeBase& operator=(const NodeBase&) = delete;

  // TODO: right now access to an indexed port is made directly by
  // specifying both a tag and an index. It would be better to represent this
  // as a two-step lookup, first getting a multi-port, and then accessing one
  // of its entries by index. However, for nodes without visible contracts we
  // can't know whether a tag is indexable or not, so we would need the
  // multi-port to also be usable as a port directly (representing index 0).
  MultiSource<> Out(absl::string_view tag) {
    return MultiSource<>(node_builder_.Out(tag));
  }

  MultiDestination<> In(absl::string_view tag) {
    return MultiDestination<>(node_builder_.In(tag));
  }

  MultiSideSource<> SideOut(absl::string_view tag) {
    return MultiSideSource<>(node_builder_.SideOut(tag));
  }

  MultiSideDestination<> SideIn(absl::string_view tag) {
    return MultiSideDestination<>(node_builder_.SideIn(tag));
  }

  template <typename B, typename T, bool kIsOptional, bool kIsMultiple>
  auto operator[](const PortCommon<B, T, kIsOptional, kIsMultiple>& port) {
    using PayloadT =
        typename PortCommon<B, T, kIsOptional, kIsMultiple>::PayloadT;
    if constexpr (std::is_same_v<B, OutputBase>) {
      auto base = node_builder_.Out(port.Tag());
      if constexpr (kIsMultiple) {
        return MultiSource<PayloadT>(base);
      } else {
        return Source<PayloadT>(base);
      }
    } else if constexpr (std::is_same_v<B, InputBase>) {
      auto base = node_builder_.In(port.Tag());
      if constexpr (kIsMultiple) {
        return MultiDestination<PayloadT>(base);
      } else {
        return Destination<PayloadT>(base);
      }
    } else if constexpr (std::is_same_v<B, SideOutputBase>) {
      auto base = node_builder_.SideOut(port.Tag());
      if constexpr (kIsMultiple) {
        return MultiSideSource<PayloadT>(base);
      } else {
        return SideSource<PayloadT>(base);
      }
    } else if constexpr (std::is_same_v<B, SideInputBase>) {
      auto base = node_builder_.SideIn(port.Tag());
      if constexpr (kIsMultiple) {
        return MultiSideDestination<PayloadT>(base);
      } else {
        return SideDestination<PayloadT>(base);
      }
    } else {
      static_assert(dependent_false<B>::value, "Type not supported.");
    }
  }

  // Convenience methods for accessing purely index-based ports.
  Source<> Out(int index) { return Out("")[index]; }

  Destination<> In(int index) { return In("")[index]; }

  SideSource<> SideOut(int index) { return SideOut("")[index]; }

  SideDestination<> SideIn(int index) { return SideIn("")[index]; }

  // Get mutable node options of type Options.
  template <
      typename OptionsT,
      typename std::enable_if<std::is_base_of<
          google::protobuf::MessageLite, OptionsT>::value>::type* = nullptr>
  OptionsT& GetOptions() {
    return node_builder_.GetOptions<OptionsT>();
  }

  // Use this API when the proto extension does not follow the "ext" naming
  // convention.
  template <typename ExtensionT>
  auto& GetOptions(const ExtensionT& ext) {
    return node_builder_.GetOptions(ext);
  }

  void SetExecutor(Executor& executor) { node_builder_.SetExecutor(executor); }

  InputStreamHandler& SetInputStreamHandler(absl::string_view type) {
    return node_builder_.SetInputStreamHandler(type);
  }

  OutputStreamHandler& SetOutputStreamHandler(absl::string_view type) {
    return node_builder_.SetOutputStreamHandler(type);
  }

  void SetSourceLayer(int source_layer) {
    node_builder_.SetSourceLayer(source_layer);
  }

 protected:
  api3::builder::NodeBuilder& node_builder_;

  friend class Graph;
};

template <class Calc = internal::Generic>
class Node;
#if __cplusplus >= 201703L
// Deduction guide to silence -Wctad-maybe-unsupported.
explicit Node() -> Node<internal::Generic>;
#endif  // C++17

template <>
class Node<internal::Generic> : public NodeBase {
 public:
  using NodeBase::NodeBase;
};

using GenericNode = Node<internal::Generic>;

template <class Calc>
class Node : public NodeBase {
 public:
  using NodeBase::NodeBase;

  // These methods only allow access to ports declared in the contract.
  // The argument must be a tag object created with the MPP_TAG macro.
  // These objects encode the tag in their type, which allows us to return
  // a result with the appropriate payload type depending on the tag.
  template <class Tag>
  auto Out(Tag tag) {
    constexpr auto& port = Calc::Contract::TaggedOutputs::get(tag);
    return NodeBase::operator[](port);
  }

  template <class Tag>
  auto In(Tag tag) {
    constexpr auto& port = Calc::Contract::TaggedInputs::get(tag);
    return NodeBase::operator[](port);
  }

  template <class Tag>
  auto SideOut(Tag tag) {
    constexpr auto& port = Calc::Contract::TaggedSideOutputs::get(tag);
    return NodeBase::operator[](port);
  }

  template <class Tag>
  auto SideIn(Tag tag) {
    constexpr auto& port = Calc::Contract::TaggedSideInputs::get(tag);
    return NodeBase::operator[](port);
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
  explicit PacketGenerator(
      api3::builder::PacketGeneratorBuilder& packet_generator_builder)
      : packet_generator_builder_(packet_generator_builder) {}

  MultiSideSource<> SideOut(absl::string_view tag) {
    return MultiSideSource<>(packet_generator_builder_.SideOut(tag));
  }

  MultiSideDestination<> SideIn(absl::string_view tag) {
    return MultiSideDestination<>(packet_generator_builder_.SideIn(tag));
  }

  // Convenience methods for accessing purely index-based ports.
  SideSource<> SideOut(int index) { return SideOut("")[index]; }
  SideDestination<> SideIn(int index) { return SideIn("")[index]; }

  template <typename T>
  T& GetOptions() {
    return GetOptions(T::ext);
  }

  // Use this API when the proto extension does not follow the "ext" naming
  // convention.
  template <typename E>
  auto& GetOptions(const E& extension) {
    return packet_generator_builder_.GetOptions(extension);
  }

  template <typename B, typename T, bool kIsOptional, bool kIsMultiple>
  auto operator[](const PortCommon<B, T, kIsOptional, kIsMultiple>& port) {
    using PayloadT =
        typename PortCommon<B, T, kIsOptional, kIsMultiple>::PayloadT;
    if constexpr (std::is_same_v<B, SideOutputBase>) {
      auto base = packet_generator_builder_.SideOut(port.Tag());
      if constexpr (kIsMultiple) {
        return MultiSideSource<PayloadT>(base);
      } else {
        return SideSource<PayloadT>(base);
      }
    } else if constexpr (std::is_same_v<B, SideInputBase>) {
      auto base = packet_generator_builder_.SideIn(port.Tag());
      if constexpr (kIsMultiple) {
        return MultiSideDestination<PayloadT>(base);
      } else {
        return SideDestination<PayloadT>(base);
      }
    } else {
      static_assert(dependent_false<B>::value, "Type not supported.");
    }
  }

 private:
  api3::builder::PacketGeneratorBuilder& packet_generator_builder_;

  friend class Graph;
};

class Graph : public mediapipe::api3::GenericGraph {
 public:
  Graph() = default;
  ~Graph() = default;
  Graph(Graph&&) = default;
  Graph& operator=(Graph&&) = default;
  // Explicitly delete copies to improve error messages.
  Graph(const Graph&) = delete;
  Graph& operator=(const Graph&) = delete;

  void SetType(std::string type) {
    mediapipe::api3::GenericGraph::graph_.SetType(std::move(type));
  }

  // Creates a node of a specific type. Should be used for calculators whose
  // contract is available.
  template <class Calc>
  Node<Calc>& AddNode() {
    auto node = std::make_unique<Node<Calc>>(
        mediapipe::api3::GenericGraph::graph_.AddNode(
            FunctionRegistry<NodeBase>::GetLookupName(Calc::kCalculatorName)));
    auto node_p = node.get();
    api2_nodes_.emplace_back(std::move(node));
    return *node_p;
  }

  // Creates a node of a specific type. Should be used for pure interfaces,
  // which do not have a built-in type string.
  // `type` is a calculator type-name with dot-separated namespaces.
  template <class Calc>
  Node<Calc>& AddNode(absl::string_view type) {
    auto node = std::make_unique<Node<Calc>>(
        mediapipe::api3::GenericGraph::graph_.AddNode(
            std::string(type.data(), type.size())));
    auto node_p = node.get();
    api2_nodes_.emplace_back(std::move(node));
    return *node_p;
  }

  // Creates a generic node, with no compile-time checking of inputs and
  // outputs. This can be used for calculators whose contract is not visible.
  // `type` is a calculator type-name with dot-separated namespaces.
  GenericNode& AddNode(absl::string_view type) {
    auto node = std::make_unique<GenericNode>(
        mediapipe::api3::GenericGraph::graph_.AddNode(
            std::string(type.data(), type.size())));
    auto node_p = node.get();
    api2_nodes_.emplace_back(std::move(node));
    return *node_p;
  }

  // For legacy PacketGenerators.
  PacketGenerator& AddPacketGenerator(absl::string_view type) {
    auto node = std::make_unique<PacketGenerator>(
        mediapipe::api3::GenericGraph::graph_.AddPacketGenerator(
            std::string(type.data(), type.size())));
    auto node_p = node.get();
    api2_packet_gens_.emplace_back(std::move(node));
    return *node_p;
  }

  Executor& AddExecutor(absl::string_view type) {
    return mediapipe::api3::GenericGraph::graph_.AddExecutor(type);
  }

  // Graph ports, non-typed.
  MultiSource<> In(absl::string_view graph_input) {
    return MultiSource<>(mediapipe::api3::GenericGraph::graph_.In(graph_input));
  }

  MultiDestination<> Out(absl::string_view graph_output) {
    return MultiDestination<>(
        mediapipe::api3::GenericGraph::graph_.Out(graph_output));
  }

  MultiSideSource<> SideIn(absl::string_view graph_input) {
    return MultiSideSource<>(
        mediapipe::api3::GenericGraph::graph_.SideIn(graph_input));
  }

  MultiSideDestination<> SideOut(absl::string_view graph_output) {
    return MultiSideDestination<>(
        mediapipe::api3::GenericGraph::graph_.SideOut(graph_output));
  }

  // Convenience methods for accessing purely index-based ports.
  Source<> In(int index) { return In("")[index]; }

  Destination<> Out(int index) { return Out("")[index]; }

  SideSource<> SideIn(int index) { return SideIn("")[index]; }

  SideDestination<> SideOut(int index) { return SideOut("")[index]; }

  // Graph ports, typed.
  // TODO: make graph_boundary_ a typed node!
  template <class PortT, class Payload = typename PortT::PayloadT>
  auto In(const PortT& graph_input) {
    return (*this)[graph_input];
  }

  template <class PortT, class Payload = typename PortT::PayloadT>
  auto Out(const PortT& graph_output) {
    return (*this)[graph_output];
  }

  template <class PortT, class Payload = typename PortT::PayloadT>
  auto SideIn(const PortT& graph_input) {
    return (*this)[graph_input];
  }

  template <class PortT, class Payload = typename PortT::PayloadT>
  auto SideOut(const PortT& graph_output) {
    return (*this)[graph_output];
  }

  template <typename B, typename T, bool kIsOptional, bool kIsMultiple>
  auto operator[](const PortCommon<B, T, kIsOptional, kIsMultiple>& port) {
    using PayloadT =
        typename PortCommon<B, T, kIsOptional, kIsMultiple>::PayloadT;
    if constexpr (std::is_same_v<B, OutputBase>) {
      auto base = mediapipe::api3::GenericGraph::graph_.Out(port.Tag());
      if constexpr (kIsMultiple) {
        return MultiDestination<PayloadT>(base);
      } else {
        return Destination<PayloadT>(base);
      }
    } else if constexpr (std::is_same_v<B, InputBase>) {
      auto base = mediapipe::api3::GenericGraph::graph_.In(port.Tag());
      if constexpr (kIsMultiple) {
        return MultiSource<PayloadT>(base);
      } else {
        return Source<PayloadT>(base);
      }
    } else if constexpr (std::is_same_v<B, SideOutputBase>) {
      auto base = mediapipe::api3::GenericGraph::graph_.SideOut(port.Tag());
      if constexpr (kIsMultiple) {
        return MultiSideDestination<PayloadT>(base);
      } else {
        return SideDestination<PayloadT>(base);
      }
    } else if constexpr (std::is_same_v<B, SideInputBase>) {
      auto base = mediapipe::api3::GenericGraph::graph_.SideIn(port.Tag());
      if constexpr (kIsMultiple) {
        return MultiSideSource<PayloadT>(base);
      } else {
        return SideSource<PayloadT>(base);
      }
    } else {
      static_assert(dependent_false<B>::value, "Type not supported.");
    }
  }

  // Returns the graph config. This can be used to instantiate and run the
  // graph.
  CalculatorGraphConfig GetConfig() {
    auto config = mediapipe::api3::GenericGraph::graph_.GetConfig();
    ABSL_CHECK_OK(config);
    return std::move(config).value();
  }

 private:
  std::vector<std::unique_ptr<Executor>> api2_executors_;
  std::vector<std::unique_ptr<NodeBase>> api2_nodes_;
  std::vector<std::unique_ptr<PacketGenerator>> api2_packet_gens_;
};

}  // namespace builder
}  // namespace api2
}  // namespace mediapipe

#endif  // MEDIAPIPE_FRAMEWORK_API2_BUILDER_H_
