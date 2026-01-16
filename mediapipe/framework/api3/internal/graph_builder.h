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

#ifndef MEDIAPIPE_FRAMEWORK_API3_INTERNAL_GRAPH_BUILDER_H_
#define MEDIAPIPE_FRAMEWORK_API3_INTERNAL_GRAPH_BUILDER_H_

#include <cstddef>
#include <functional>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "absl/container/btree_map.h"
#include "absl/log/absl_check.h"
#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "google/protobuf/message_lite.h"
#include "mediapipe/framework/calculator_base.h"
#include "mediapipe/framework/calculator_contract.h"
#include "mediapipe/framework/mediapipe_options.pb.h"
#include "mediapipe/framework/port/any_proto.h"
#include "mediapipe/framework/port/ret_check.h"
#include "mediapipe/framework/port/status_builder.h"
#include "mediapipe/framework/port/status_macros.h"
#include "mediapipe/framework/stream_handler.pb.h"
#include "mediapipe/framework/tool/type_util.h"

namespace mediapipe::api3::builder {

template <typename T>
T& GetWithAutoGrow(std::vector<std::unique_ptr<T>>* vecp, size_t index) {
  auto& vec = *vecp;
  if (vec.size() <= index) {
    vec.resize(index + 1);
  }
  if (vec[index] == nullptr) {
    vec[index] = std::make_unique<T>();
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
  std::vector<std::unique_ptr<T>>& operator[](absl::string_view tag) {
    return map_[tag];
  }

  absl::Status Visit(
      std::function<absl::Status(const TagIndexLocation&, T&)> fun) const {
    for (auto& tagged : map_) {
      TagIndexLocation loc{tagged.first, 0, tagged.second.size()};
      for (int i = 0; i < tagged.second.size(); ++i) {
        auto* item = tagged.second[i].get();
        loc.index = i;
        // If the item is nullptr, it means that the connection vector for
        // current tag grew by a GetWithAutoGrow() request but hasn't been
        // populated yet.
        RET_CHECK(item != nullptr) << absl::StrCat(
            "Missing port for tag: \"", loc.tag, "\", index: ", loc.index, ".");
        MP_RETURN_IF_ERROR(fun(loc, *item));
      }
    }
    return absl::OkStatus();
  }

 private:
  // Note: entries are held by a unique_ptr to ensure pointers remain valid.
  // Should use absl::flat_hash_map but ordering keys for now.
  absl::btree_map<std::string, std::vector<std::unique_ptr<T>>> map_;
};

class GraphBuilder;
class NodeBuilder;
class PacketGeneratorBuilder;

// These structs are used internally to store information about the endpoints
// of a connection.
template <typename S, typename D>
struct SourceBase;

template <typename S, typename D>
struct DestinationBase {
  SourceBase<S, D>* source = nullptr;
  bool back_edge = false;

  D& AsBackEdge() {
    back_edge = true;
    return *static_cast<D*>(this);
  }

  DestinationBase() = default;
  DestinationBase(const DestinationBase&) = delete;
  DestinationBase& operator=(const DestinationBase&) = delete;
};

template <typename S, typename D>
struct SourceBase {
  std::vector<DestinationBase<S, D>*> dests;
  std::string name;

  SourceBase() = default;
  SourceBase(const SourceBase&) = delete;
  SourceBase& operator=(const SourceBase&) = delete;

  S& SetName(std::string name) {
    this->name = std::move(name);
    return *static_cast<S*>(this);
  }

  void ConnectTo(DestinationBase<S, D>& dest) {
    ABSL_CHECK(dest.source == nullptr);
    dest.source = this;
    dests.emplace_back(&dest);
  }
};

struct Destination;
struct Source : SourceBase<Source, Destination> {};
struct Destination : DestinationBase<Source, Destination> {};
struct SideDestination;
struct SideSource : SourceBase<SideSource, SideDestination> {};
struct SideDestination : DestinationBase<SideSource, SideDestination> {};

template <typename T>
class /*ABSL_ATTRIBUTE_VIEW*/ Multi {
 public:
  explicit Multi(std::vector<std::unique_ptr<T>>& vec) : vec_(&vec) {}

  T& At(int index) {
    ABSL_CHECK_GE(index, 0);
    return GetWithAutoGrow(vec_, index);
  }

 private:
  // Never null.
  std::vector<std::unique_ptr<T>>* vec_;
};

template <typename OptionsT>
OptionsT& GetOptions(std::optional<mediapipe::MediaPipeOptions>& options) {
  if (!options.has_value()) {
    options = mediapipe::MediaPipeOptions();
  }
  return *options->MutableExtension(OptionsT::ext);
}

class Executor {
 public:
  template <typename OptionsT>
  OptionsT& GetOptions() {
    return ::mediapipe::api3::builder::GetOptions<OptionsT>(options_);
  }

 private:
  explicit Executor(std::string type) : type_(std::move(type)) {}

  std::string type_;
  std::string name_;

  std::optional<mediapipe::MediaPipeOptions> options_;

  friend class GraphBuilder;
};

class NodeBuilder;

class InputStreamHandler {
 public:
  template <typename OptionsT>
  OptionsT& GetOptions() {
    return ::mediapipe::api3::builder::GetOptions<OptionsT>(options_);
  }

 protected:
  explicit InputStreamHandler() = default;

  std::string type_;
  std::optional<mediapipe::MediaPipeOptions> options_;

  friend class NodeBuilder;
  friend class GraphBuilder;
};

class OutputStreamHandler {
 public:
  template <typename OptionsT>
  OptionsT& GetOptions() {
    return ::mediapipe::api3::builder::GetOptions<OptionsT>(options_);
  }

 protected:
  explicit OutputStreamHandler() = default;

  std::string type_;
  std::optional<mediapipe::MediaPipeOptions> options_;

  friend class NodeBuilder;
  friend class GraphBuilder;
};

// Class for building `CalculatorGraphConfig::Node`.
//
// Adding a node:
// ```
//   NodeBuilder& node = graph.AddNode("NodeName");
// ```
// Accessing inputs/outputs:
// ```
//   ... node.In("INPUT_TAG")[0] ...
//   ... node.Out("OUTPUT_TAG")[0] ...
//   ... node.SideIn("SIDE_INPUT_TAG")[0] ...
//   ... node.SideOut("SIDE_OUTPUT_TAG")[0] ...
// ```
class NodeBuilder {
 public:
  explicit NodeBuilder(std::string type) : type_(std::move(type)) {}
  NodeBuilder(NodeBuilder&&) = default;
  NodeBuilder& operator=(NodeBuilder&&) = default;
  // Explicitly delete copies to improve error messages.
  NodeBuilder(const NodeBuilder&) = delete;
  NodeBuilder& operator=(const NodeBuilder&) = delete;

  // Gives access to node output streams:
  // `CalculatorGraphConfig::Node::output_stream`.
  //
  // Usage pattern:
  // ```
  //   node.Out("OUT")[0].ConnectTo(other_node.In("IN")[0]);
  // ```
  Multi<Source> Out(absl::string_view tag) {
    return Multi<Source>(out_streams_[tag]);
  }

  // Gives access to node input streams:
  // `CalculatorGraphConfig::Node::input_stream`.
  //
  // Usage pattern:
  // ```
  //   other_node.Out("OUT")[0].ConnectTo(node.In("IN")[0]);
  // ```
  Multi<Destination> In(absl::string_view tag) {
    return Multi<Destination>(in_streams_[tag]);
  }

  // Gives access to node output side packets:
  // `CalculatorGraphConfig::Node::output_side_packet`.
  //
  // Usage pattern:
  // ```
  //   node.SideOut("SIDE_OUT")[0].ConnectTo(other_node.SideIn("SIDE_IN")[0]);
  // ```
  Multi<SideSource> SideOut(absl::string_view tag) {
    return Multi<SideSource>(out_sides_[tag]);
  }

  // Gives access to node input side packets:
  // `CalculatorGraphConfig::Node::input_side_packet`.
  //
  // Usage pattern:
  // ```
  //   other_node.SideOut("SIDE_OUT")[0].ConnectTo(node.SideIn("SIDE_IN")[0]);
  // ```
  Multi<SideDestination> SideIn(absl::string_view tag) {
    return Multi<SideDestination>(in_sides_[tag]);
  }

  // Get mutable node options of type Options.
  template <
      typename OptionsT,
      typename std::enable_if<std::is_base_of<
          google::protobuf::MessageLite, OptionsT>::value>::type* = nullptr>
  OptionsT& GetOptions() {
    return GetOptionsInternal<OptionsT>(nullptr);
  }

  // Use this API when the proto extension does not follow the "ext" naming
  // convention.
  template <typename ExtensionT>
  auto& GetOptions(const ExtensionT& ext) {
    if (!calculator_option_.has_value()) {
      calculator_option_ = CalculatorOptions();
    }
    return *calculator_option_->MutableExtension(ext);
  }

  // Sets executor corresponding to `CalculatorGraphConfig::Node::executor`.
  void SetExecutor(Executor& executor) { executor_ = &executor; }

  // Sets input stream handler corresponding to
  // `CalculatorGraphConfig::Node::input_stream_handler`.
  InputStreamHandler& SetInputStreamHandler(absl::string_view type) {
    if (!input_stream_handler_) {
      input_stream_handler_ = InputStreamHandler();
    }
    input_stream_handler_->type_ = std::string(type.data(), type.size());
    return *input_stream_handler_;
  }

  // Sets output stream handler corresponding to
  // `CalculatorGraphConfig::Node::output_stream_handler`.
  OutputStreamHandler& SetOutputStreamHandler(absl::string_view type) {
    if (!output_stream_handler_) {
      output_stream_handler_ = OutputStreamHandler();
    }
    output_stream_handler_->type_ = std::string(type.data(), type.size());
    return *output_stream_handler_;
  }

  // Sets source layer corresponding to
  // `CalculatorGraphConfig::Node::source_layer`.
  void SetSourceLayer(int source_layer) { source_layer_ = source_layer; }

 protected:
  // GetOptionsInternal resolutes the overload greedily, which finds the first
  // match then succeed (template specialization tries all matches, thus could
  // be ambiguous)
  template <typename OptionsT>
  OptionsT& GetOptionsInternal(decltype(&OptionsT::ext) /*unused*/) {
    return GetOptions(OptionsT::ext);
  }
  template <typename OptionsT>
  OptionsT& GetOptionsInternal(...) {
    if (node_options_.count(kTypeId<OptionsT>)) {
      return *static_cast<OptionsT*>(
          node_options_[kTypeId<OptionsT>].message.get());
    }
    auto option = std::make_unique<OptionsT>();
    OptionsT* option_ptr = option.get();
    node_options_[kTypeId<OptionsT>] = {
        std::move(option),
        [option_ptr](protobuf::Any& any) { return any.PackFrom(*option_ptr); }};
    return *option_ptr;
  }

  std::string type_;
  TagIndexMap<Destination> in_streams_;
  TagIndexMap<Source> out_streams_;
  TagIndexMap<SideDestination> in_sides_;
  TagIndexMap<SideSource> out_sides_;
  std::optional<CalculatorOptions> calculator_option_;
  // Stores real proto config, and lambda for packing config into Any.
  // We need the lambda because PackFrom() does not work with MessageLite.
  struct MessageAndPacker {
    std::unique_ptr<google::protobuf::MessageLite> message;
    std::function<bool(protobuf::Any&)> packer;
  };
  std::map<TypeId, MessageAndPacker> node_options_;

  Executor* executor_ = nullptr;

  std::optional<InputStreamHandler> input_stream_handler_;
  std::optional<OutputStreamHandler> output_stream_handler_;

  std::optional<int> source_layer_;

  friend class GraphBuilder;
};

// For legacy PacketGenerators.
class PacketGeneratorBuilder {
 public:
  explicit PacketGeneratorBuilder(std::string type) : type_(std::move(type)) {}

  // Accessing generator output side packets
  // `CalculatorGraphConfig::PacketGenerator::output_side_packet`.
  Multi<SideSource> SideOut(absl::string_view tag) {
    return Multi<SideSource>(out_sides_[tag]);
  }

  // Accessing generator input side packets
  // `CalculatorGraphConfig::PacketGenerator::input_side_packet`.
  Multi<SideDestination> SideIn(absl::string_view tag) {
    return Multi<SideDestination>(in_sides_[tag]);
  }

  // Accessing packet generator options
  template <typename T>
  T& GetOptions() {
    return GetOptions(T::ext);
  }

  // Use this API when the proto extension does not follow the "ext" naming
  // convention.
  template <typename E>
  auto& GetOptions(const E& extension) {
    options_used_ = true;
    return *options_.MutableExtension(extension);
  }

 private:
  std::string type_;
  TagIndexMap<SideDestination> in_sides_;
  TagIndexMap<SideSource> out_sides_;
  mediapipe::PacketGeneratorOptions options_;
  // ideally we'd just check if any extensions are set on options_
  bool options_used_ = false;
  friend class GraphBuilder;
};

// Class to build a generic graph - no contract, no typed inputs/outputs, no
// compile time validation.
//
// NOTE: this is an internal builder which is shared between multiple APIs for
//   for more safe and streamlined graph construction.
class GraphBuilder {
 public:
  GraphBuilder() = default;
  ~GraphBuilder() = default;
  GraphBuilder(GraphBuilder&&) = default;
  GraphBuilder& operator=(GraphBuilder&&) = default;
  // Explicitly delete copies to improve error messages.
  GraphBuilder(const GraphBuilder&) = delete;
  GraphBuilder& operator=(const GraphBuilder&) = delete;

  void SetType(std::string type) { type_ = std::move(type); }

  // Creates a node builder, with no compile-time checking of inputs and
  // outputs. This can be used for calculators whose contract is not visible.
  // `type` is a calculator type-name with dot-separated namespaces.
  NodeBuilder& AddNode(absl::string_view type) {
    auto node =
        std::make_unique<NodeBuilder>(std::string(type.data(), type.size()));
    auto node_p = node.get();
    nodes_.emplace_back(std::move(node));
    return *node_p;
  }

  // Adds a legacy PacketGenerators.
  PacketGeneratorBuilder& AddPacketGenerator(absl::string_view type) {
    auto node = std::make_unique<PacketGeneratorBuilder>(
        std::string(type.data(), type.size()));
    auto node_p = node.get();
    packet_gens_.emplace_back(std::move(node));
    return *node_p;
  }

  // Adds an executor to the graph which can be set on a node.
  Executor& AddExecutor(absl::string_view type) {
    auto executor =
        absl::WrapUnique(new Executor(std::string(type.data(), type.size())));
    auto* executor_p = executor.get();
    executors_.emplace_back(std::move(executor));
    return *executor_p;
  }

  // Gives access to graph input streams: `CalculatorGraphConfig::input_stream`.
  // Usage pattern:
  // ```
  //   graph.In("INPUT")[0].ConnectTo(node.In("IN")[0]);
  // ```
  Multi<Source> In(absl::string_view graph_input) {
    return graph_boundary_.Out(graph_input);
  }

  // Gives access to graph output streams:
  // `CalculatorGraphConfig::output_stream`. Usage pattern:
  // ```
  //   node.Out("OUT")[0].ConnectTo(graph.Out("OUTPUT")[0]);
  // ```
  Multi<Destination> Out(absl::string_view graph_output) {
    return graph_boundary_.In(graph_output);
  }

  // Gives access to graph input streams:
  // `CalculatorGraphConfig::input_side_packet`. Usage pattern:
  // ```
  //   graph.SideIn("SIDE_INPUT")[0].ConnectTo(node.SideIn("SIDE_IN")[0]);
  // ```
  Multi<SideSource> SideIn(absl::string_view graph_input) {
    return graph_boundary_.SideOut(graph_input);
  }

  // Gives access to graph output side packets:
  // `CalculatorGraphConfig::output_side_packet`. Usage pattern:
  // ```
  //   node.SideOut("SIDE_OUT")[0].ConnectTo(graph.SideOut("SIDE_OUTPUT")[0]);
  // ```
  Multi<SideDestination> SideOut(absl::string_view graph_output) {
    return graph_boundary_.SideIn(graph_output);
  }

  // Returns the graph config. This can be used to instantiate and run the
  // graph.
  absl::StatusOr<CalculatorGraphConfig> GetConfig() {
    CalculatorGraphConfig config;
    if (!type_.empty()) {
      config.set_type(type_);
    }

    // Name and add executors.
    int executor_index = 0;
    for (std::unique_ptr<Executor>& executor : executors_) {
      // Names starting from "__" are historically reserved for internal
      // executors.
      executor->name_ = absl::StrCat("_b_executor_", executor_index++);

      auto* out_executor = config.add_executor();
      out_executor->set_name(executor->name_);
      out_executor->set_type(executor->type_);
      if (executor->options_) {
        *out_executor->mutable_options() = *executor->options_;
      }
    }

    MP_RETURN_IF_ERROR(FixUnnamedConnections());
    MP_RETURN_IF_ERROR(UpdateBoundaryConfig(&config));
    for (const std::unique_ptr<NodeBuilder>& node : nodes_) {
      auto* out_node = config.add_node();
      MP_RETURN_IF_ERROR(UpdateNodeConfig(*node, out_node));
    }
    for (const std::unique_ptr<PacketGeneratorBuilder>& node : packet_gens_) {
      auto* out_node = config.add_packet_generator();
      MP_RETURN_IF_ERROR(UpdateNodeConfig(*node, out_node));
    }
    return config;
  }

 private:
  absl::Status FixUnnamedConnections(NodeBuilder* node, int* unnamed_count) {
    MP_RETURN_IF_ERROR(node->out_streams_.Visit(
        [&](const TagIndexLocation& loc, Source& source) -> absl::Status {
          if (source.name.empty()) {
            source.name = absl::StrCat("__stream_", (*unnamed_count)++);
          }
          return absl::OkStatus();
        }));

    MP_RETURN_IF_ERROR(node->out_sides_.Visit(
        [&](const TagIndexLocation& loc, SideSource& source) -> absl::Status {
          if (source.name.empty()) {
            source.name = absl::StrCat("__side_packet_", (*unnamed_count)++);
          }
          return absl::OkStatus();
        }));
    return absl::OkStatus();
  }

  absl::Status FixUnnamedConnections() {
    int unnamed_count = 0;
    MP_RETURN_IF_ERROR(FixUnnamedConnections(&graph_boundary_, &unnamed_count));
    for (std::unique_ptr<NodeBuilder>& node : nodes_) {
      MP_RETURN_IF_ERROR(FixUnnamedConnections(node.get(), &unnamed_count));
    }
    for (std::unique_ptr<PacketGeneratorBuilder>& node : packet_gens_) {
      MP_RETURN_IF_ERROR(node->out_sides_.Visit(
          [&](const TagIndexLocation& loc, SideSource& source) -> absl::Status {
            if (source.name.empty()) {
              source.name = absl::StrCat("__side_packet_", unnamed_count++);
            }
            return absl::OkStatus();
          }));
    }
    return absl::OkStatus();
  }

  static std::string TagIndex(const TagIndexLocation& loc) {
    if (loc.count <= 1) {
      return loc.tag;
    }
    return absl::StrCat(loc.tag, ":", loc.index);
  }

  static std::string TaggedName(const TagIndexLocation& loc,
                                absl::string_view name) {
    if (loc.tag.empty()) {
      // ParseTagIndexName does not allow using explicit indices without tags,
      // while ParseTagIndex does.
      // TODO: decide whether we should just allow it.
      return std::string(name);
    } else {
      if (loc.count <= 1) {
        return absl::StrCat(loc.tag, ":", name);
      } else {
        return absl::StrCat(loc.tag, ":", loc.index, ":", name);
      }
    }
  }

  absl::Status UpdateNodeConfig(const NodeBuilder& node,
                                CalculatorGraphConfig::Node* config) {
    config->set_calculator(node.type_);
    MP_RETURN_IF_ERROR(node.in_streams_.Visit(
        [&](const TagIndexLocation& loc,
            const Destination& endpoint) -> absl::Status {
          RET_CHECK(endpoint.source != nullptr)
              << node.type_ << ": Missing source for input stream with tag "
              << (loc.tag.empty() ? "(empty)" : loc.tag) << " at index "
              << loc.index;
          config->add_input_stream(TaggedName(loc, endpoint.source->name));
          if (endpoint.back_edge) {
            auto* info = config->add_input_stream_info();
            info->set_back_edge(true);
            info->set_tag_index(TagIndex(loc));
          }
          return absl::OkStatus();
        }));
    MP_RETURN_IF_ERROR(
        node.out_streams_.Visit([&](const TagIndexLocation& loc,
                                    const Source& endpoint) -> absl::Status {
          config->add_output_stream(TaggedName(loc, endpoint.name));
          return absl::OkStatus();
        }));
    MP_RETURN_IF_ERROR(node.in_sides_.Visit(
        [&](const TagIndexLocation& loc,
            const SideDestination& endpoint) -> absl::Status {
          RET_CHECK(endpoint.source != nullptr)
              << node.type_
              << ": Missing source for input side packet stream with tag "
              << loc.tag << " at index " << loc.index;
          config->add_input_side_packet(TaggedName(loc, endpoint.source->name));
          return absl::OkStatus();
        }));
    MP_RETURN_IF_ERROR(
        node.out_sides_.Visit([&](const TagIndexLocation& loc,
                                  const SideSource& endpoint) -> absl::Status {
          config->add_output_side_packet(TaggedName(loc, endpoint.name));
          return absl::OkStatus();
        }));
    if (node.calculator_option_.has_value()) {
      *config->mutable_options() = *node.calculator_option_;
    }
    for (auto& [type_id, message_and_packer] : node.node_options_) {
      RET_CHECK(message_and_packer.packer(*config->add_node_options()));
    }
    if (node.executor_ != nullptr) {
      config->set_executor(node.executor_->name_);
    }
    if (node.input_stream_handler_) {
      config->mutable_input_stream_handler()->set_input_stream_handler(
          node.input_stream_handler_->type_);
      if (node.input_stream_handler_->options_) {
        *config->mutable_input_stream_handler()->mutable_options() =
            *node.input_stream_handler_->options_;
      }
    }
    if (node.output_stream_handler_) {
      config->mutable_output_stream_handler()->set_output_stream_handler(
          node.output_stream_handler_->type_);
      if (node.output_stream_handler_->options_) {
        *config->mutable_output_stream_handler()->mutable_options() =
            *node.output_stream_handler_->options_;
      }
    }
    if (node.source_layer_.has_value()) {
      config->set_source_layer(node.source_layer_.value());
    }
    return absl::OkStatus();
  }

  absl::Status UpdateNodeConfig(const PacketGeneratorBuilder& node,
                                PacketGeneratorConfig* config) {
    config->set_packet_generator(node.type_);
    MP_RETURN_IF_ERROR(node.in_sides_.Visit(
        [&](const TagIndexLocation& loc,
            const SideDestination& endpoint) -> absl::Status {
          RET_CHECK(endpoint.source != nullptr)
              << node.type_
              << ": Missing source for input side packet stream with tag "
              << (loc.tag.empty() ? "(empty)" : loc.tag) << " at index "
              << loc.index;
          config->add_input_side_packet(TaggedName(loc, endpoint.source->name));
          return absl::OkStatus();
        }));
    MP_RETURN_IF_ERROR(
        node.out_sides_.Visit([&](const TagIndexLocation& loc,
                                  const SideSource& endpoint) -> absl::Status {
          config->add_output_side_packet(TaggedName(loc, endpoint.name));
          return absl::OkStatus();
        }));
    if (node.options_used_) {
      *config->mutable_options() = node.options_;
    }
    return absl::OkStatus();
  }

  // For special boundary node.
  absl::Status UpdateBoundaryConfig(CalculatorGraphConfig* config) {
    MP_RETURN_IF_ERROR(graph_boundary_.in_streams_.Visit(
        [&](const TagIndexLocation& loc,
            const Destination& endpoint) -> absl::Status {
          RET_CHECK(endpoint.source != nullptr)
              << type_ << ": Missing source for graph output stream with tag "
              << (loc.tag.empty() ? "(empty)" : loc.tag) << " at index "
              << loc.index;
          RET_CHECK(!endpoint.back_edge)
              << "Graph output: " << (loc.tag.empty() ? "(empty)" : loc.tag)
              << " at index " << loc.index << " cannot be a back edge";
          config->add_output_stream(TaggedName(loc, endpoint.source->name));

          return absl::OkStatus();
        }));
    MP_RETURN_IF_ERROR(graph_boundary_.out_streams_.Visit(
        [&](const TagIndexLocation& loc,
            const Source& endpoint) -> absl::Status {
          config->add_input_stream(TaggedName(loc, endpoint.name));
          return absl::OkStatus();
        }));
    MP_RETURN_IF_ERROR(graph_boundary_.in_sides_.Visit(
        [&](const TagIndexLocation& loc,
            const SideDestination& endpoint) -> absl::Status {
          RET_CHECK(endpoint.source != nullptr)
              << type_
              << ": Missing source for graph output side packet stream with "
                 "tag "
              << (loc.tag.empty() ? "(empty)" : loc.tag) << " at index "
              << loc.index;
          config->add_output_side_packet(
              TaggedName(loc, endpoint.source->name));
          return absl::OkStatus();
        }));
    MP_RETURN_IF_ERROR(graph_boundary_.out_sides_.Visit(
        [&](const TagIndexLocation& loc,
            const SideSource& endpoint) -> absl::Status {
          config->add_input_side_packet(TaggedName(loc, endpoint.name));

          return absl::OkStatus();
        }));
    return absl::OkStatus();
  }

  std::string type_;
  std::vector<std::unique_ptr<Executor>> executors_;
  std::vector<std::unique_ptr<NodeBuilder>> nodes_;
  std::vector<std::unique_ptr<PacketGeneratorBuilder>> packet_gens_;
  // Special node representing graph inputs and outputs.
  NodeBuilder graph_boundary_{"__GRAPH__"};
};

}  // namespace mediapipe::api3::builder

#endif  // MEDIAPIPE_FRAMEWORK_API3_INTERNAL_GRAPH_BUILDER_H_
