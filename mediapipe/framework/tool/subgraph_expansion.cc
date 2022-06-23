// Copyright 2019 The MediaPipe Authors.
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

#include "mediapipe/framework/tool/subgraph_expansion.h"

#include <algorithm>
#include <functional>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "absl/memory/memory.h"
#include "absl/strings/str_cat.h"
#include "mediapipe/framework/graph_service_manager.h"
#include "mediapipe/framework/packet_generator.pb.h"
#include "mediapipe/framework/port.h"
#include "mediapipe/framework/port/core_proto_inc.h"
#include "mediapipe/framework/port/logging.h"
#include "mediapipe/framework/port/map_util.h"
#include "mediapipe/framework/port/ret_check.h"
#include "mediapipe/framework/port/status_builder.h"
#include "mediapipe/framework/port/status_macros.h"
#include "mediapipe/framework/status_handler.pb.h"
#include "mediapipe/framework/subgraph.h"
#include "mediapipe/framework/tool/name_util.h"
#include "mediapipe/framework/tool/tag_map.h"

namespace mediapipe {

namespace tool {

absl::Status TransformStreamNames(
    proto_ns::RepeatedPtrField<ProtoString>* streams,
    const std::function<std::string(absl::string_view)>& transform) {
  for (auto& stream : *streams) {
    absl::string_view port_and_name(stream);
    auto colon_pos = port_and_name.find_last_of(':');
    auto name_pos = colon_pos == absl::string_view::npos ? 0 : colon_pos + 1;
    stream =
        absl::StrCat(port_and_name.substr(0, name_pos),
                     transform(absl::ClippedSubstr(port_and_name, name_pos)));
  }
  return absl::OkStatus();
}

// Returns subgraph streams not requested by a subgraph-node.
absl::Status FindIgnoredStreams(
    const proto_ns::RepeatedPtrField<ProtoString>& src_streams,
    const proto_ns::RepeatedPtrField<ProtoString>& dst_streams,
    std::set<std::string>* result) {
  ASSIGN_OR_RETURN(auto src_map, tool::TagMap::Create(src_streams));
  ASSIGN_OR_RETURN(auto dst_map, tool::TagMap::Create(dst_streams));
  for (auto id = src_map->BeginId(); id < src_map->EndId(); ++id) {
    std::pair<std::string, int> tag_index = src_map->TagAndIndexFromId(id);
    if (!dst_map->GetId(tag_index.first, tag_index.second).IsValid()) {
      result->insert(src_map->Names()[id.value()]);
    }
  }
  return absl::OkStatus();
}

// Removes subgraph streams not requested by a subgraph-node.
absl::Status RemoveIgnoredStreams(
    proto_ns::RepeatedPtrField<ProtoString>* streams,
    const std::set<std::string>& missing_streams) {
  for (int i = streams->size() - 1; i >= 0; --i) {
    std::string tag, name;
    int index;
    MP_RETURN_IF_ERROR(ParseTagIndexName(streams->Get(i), &tag, &index, &name));
    if (missing_streams.count(name) > 0) {
      streams->DeleteSubrange(i, 1);
    }
  }
  return absl::OkStatus();
}

absl::Status TransformNames(
    CalculatorGraphConfig* config,
    const std::function<std::string(absl::string_view)>& transform) {
  RET_CHECK_EQ(config->packet_factory().size(), 0);
  for (auto* streams :
       {config->mutable_input_stream(), config->mutable_output_stream(),
        config->mutable_input_side_packet(),
        config->mutable_output_side_packet()}) {
    MP_RETURN_IF_ERROR(TransformStreamNames(streams, transform));
  }
  std::vector<std::string> node_names(config->node_size());
  for (int node_id = 0; node_id < config->node_size(); ++node_id) {
    node_names[node_id] = CanonicalNodeName(*config, node_id);
  }
  for (int node_id = 0; node_id < config->node_size(); ++node_id) {
    config->mutable_node(node_id)->set_name(transform(node_names[node_id]));
  }
  for (auto& node : *config->mutable_node()) {
    for (auto* streams :
         {node.mutable_input_stream(), node.mutable_output_stream(),
          node.mutable_input_side_packet(),
          node.mutable_output_side_packet()}) {
      MP_RETURN_IF_ERROR(TransformStreamNames(streams, transform));
    }
  }
  for (auto& generator : *config->mutable_packet_generator()) {
    for (auto* streams : {generator.mutable_input_side_packet(),
                          generator.mutable_output_side_packet()}) {
      MP_RETURN_IF_ERROR(TransformStreamNames(streams, transform));
    }
  }
  for (auto& status_handler : *config->mutable_status_handler()) {
    MP_RETURN_IF_ERROR(TransformStreamNames(
        status_handler.mutable_input_side_packet(), transform));
  }
  return absl::OkStatus();
}

// Adds a prefix to the name of each stream, side packet and node in the
// config. Each call to this method should use a different prefix. For example:
//   1, { foo, bar }  --PrefixNames-> { qsg__foo, qsg__bar }
//   2, { foo, bar }  --PrefixNames-> { rsg__foo, rsg__bar }
// This means that two copies of the same subgraph will not interfere with
// each other.
static absl::Status PrefixNames(std::string prefix,
                                CalculatorGraphConfig* config) {
  std::transform(prefix.begin(), prefix.end(), prefix.begin(), ::tolower);
  std::replace(prefix.begin(), prefix.end(), '.', '_');
  std::replace(prefix.begin(), prefix.end(), ' ', '_');
  std::replace(prefix.begin(), prefix.end(), ':', '_');
  absl::StrAppend(&prefix, "__");
  auto add_prefix = [&prefix](absl::string_view s) {
    return absl::StrCat(prefix, s);
  };
  return TransformNames(config, add_prefix);
}

absl::Status FindCorrespondingStreams(
    std::map<std::string, std::string>* stream_map,
    const proto_ns::RepeatedPtrField<ProtoString>& src_streams,
    const proto_ns::RepeatedPtrField<ProtoString>& dst_streams) {
  ASSIGN_OR_RETURN(auto src_map, tool::TagMap::Create(src_streams));
  ASSIGN_OR_RETURN(auto dst_map, tool::TagMap::Create(dst_streams));
  for (const auto& it : dst_map->Mapping()) {
    const std::string& tag = it.first;
    const TagMap::TagData* src_tag_data =
        mediapipe::FindOrNull(src_map->Mapping(), tag);
    if (!src_tag_data) {
      return mediapipe::InvalidArgumentErrorBuilder(MEDIAPIPE_LOC)
             << "Tag \"" << tag << "\" does not exist in the subgraph config.";
    }
    const TagMap::TagData& dst_tag_data = it.second;
    CollectionItemId src_id = src_tag_data->id;
    CollectionItemId dst_id = dst_tag_data.id;
    if (dst_tag_data.count > src_tag_data->count) {
      return mediapipe::InvalidArgumentErrorBuilder(MEDIAPIPE_LOC)
             << "Tag \"" << tag << "\" has " << dst_tag_data.count
             << " indexes in the subgraph node but has only "
             << src_tag_data->count << " indexes in the subgraph config.";
    }
    CollectionItemId src_end_id =
        src_id + std::min(src_tag_data->count, dst_tag_data.count);
    for (; src_id < src_end_id; ++src_id, ++dst_id) {
      const std::string& src_name = src_map->Names()[src_id.value()];
      const std::string& dst_name = dst_map->Names()[dst_id.value()];
      (*stream_map)[src_name] = dst_name;
    }
  }
  return absl::OkStatus();
}

// The following fields can be used in a Node message for a subgraph:
//   name, calculator, input_stream, output_stream, input_side_packet,
//   output_side_packet, options.
// All other fields are only applicable to calculators.
absl::Status ValidateSubgraphFields(
    const CalculatorGraphConfig::Node& subgraph_node) {
  if (subgraph_node.source_layer() || subgraph_node.buffer_size_hint() ||
      subgraph_node.has_output_stream_handler() ||
      subgraph_node.input_stream_info_size() != 0 ||
      !subgraph_node.executor().empty()) {
    return mediapipe::InvalidArgumentErrorBuilder(MEDIAPIPE_LOC)
           << "Subgraph \"" << subgraph_node.name()
           << "\" has a field that is only applicable to calculators.";
  }
  return absl::OkStatus();
}

absl::Status ConnectSubgraphStreams(
    const CalculatorGraphConfig::Node& subgraph_node,
    CalculatorGraphConfig* subgraph_config) {
  std::map<std::string, std::string> stream_map;
  MP_RETURN_IF_ERROR(FindCorrespondingStreams(&stream_map,
                                              subgraph_config->input_stream(),
                                              subgraph_node.input_stream()))
          .SetPrepend()
      << "while processing the input streams of subgraph node "
      << subgraph_node.calculator() << ": ";
  MP_RETURN_IF_ERROR(FindCorrespondingStreams(&stream_map,
                                              subgraph_config->output_stream(),
                                              subgraph_node.output_stream()))
          .SetPrepend()
      << "while processing the output streams of subgraph node "
      << subgraph_node.calculator() << ": ";
  std::map<std::string, std::string> side_packet_map;
  MP_RETURN_IF_ERROR(FindCorrespondingStreams(
                         &side_packet_map, subgraph_config->input_side_packet(),
                         subgraph_node.input_side_packet()))
          .SetPrepend()
      << "while processing the input side packets of subgraph node "
      << subgraph_node.calculator() << ": ";
  MP_RETURN_IF_ERROR(
      FindCorrespondingStreams(&side_packet_map,
                               subgraph_config->output_side_packet(),
                               subgraph_node.output_side_packet()))
          .SetPrepend()
      << "while processing the output side packets of subgraph node "
      << subgraph_node.calculator() << ": ";
  std::set<std::string> ignored_input_streams;
  MP_RETURN_IF_ERROR(FindIgnoredStreams(subgraph_config->input_stream(),
                                        subgraph_node.input_stream(),
                                        &ignored_input_streams));
  std::set<std::string> ignored_input_side_packets;
  MP_RETURN_IF_ERROR(FindIgnoredStreams(subgraph_config->input_side_packet(),
                                        subgraph_node.input_side_packet(),
                                        &ignored_input_side_packets));
  std::map<std::string, std::string>* name_map;
  auto replace_names = [&name_map](absl::string_view s) {
    std::string original(s);
    std::string* replacement = mediapipe::FindOrNull(*name_map, original);
    return replacement ? *replacement : original;
  };
  for (auto& node : *subgraph_config->mutable_node()) {
    name_map = &stream_map;
    MP_RETURN_IF_ERROR(
        TransformStreamNames(node.mutable_input_stream(), replace_names));
    MP_RETURN_IF_ERROR(
        TransformStreamNames(node.mutable_output_stream(), replace_names));
    name_map = &side_packet_map;
    MP_RETURN_IF_ERROR(
        TransformStreamNames(node.mutable_input_side_packet(), replace_names));
    MP_RETURN_IF_ERROR(
        TransformStreamNames(node.mutable_output_side_packet(), replace_names));

    // Remove input streams and side packets ignored by the subgraph-node.
    MP_RETURN_IF_ERROR(RemoveIgnoredStreams(node.mutable_input_stream(),
                                            ignored_input_streams));
    MP_RETURN_IF_ERROR(RemoveIgnoredStreams(node.mutable_input_side_packet(),
                                            ignored_input_side_packets));
  }
  name_map = &side_packet_map;
  for (auto& generator : *subgraph_config->mutable_packet_generator()) {
    MP_RETURN_IF_ERROR(TransformStreamNames(
        generator.mutable_input_side_packet(), replace_names));
    MP_RETURN_IF_ERROR(TransformStreamNames(
        generator.mutable_output_side_packet(), replace_names));

    // Remove input side packets ignored by the subgraph-node.
    MP_RETURN_IF_ERROR(RemoveIgnoredStreams(
        generator.mutable_input_side_packet(), ignored_input_side_packets));
  }
  return absl::OkStatus();
}

absl::Status ExpandSubgraphs(CalculatorGraphConfig* config,
                             const GraphRegistry* graph_registry,
                             const Subgraph::SubgraphOptions* graph_options,
                             const GraphServiceManager* service_manager) {
  graph_registry =
      graph_registry ? graph_registry : &GraphRegistry::global_graph_registry;
  RET_CHECK(config);

  MP_RETURN_IF_ERROR(mediapipe::tool::DefineGraphOptions(
      graph_options ? *graph_options : CalculatorGraphConfig::Node(), config));
  auto* nodes = config->mutable_node();
  while (1) {
    auto subgraph_nodes_start = std::stable_partition(
        nodes->begin(), nodes->end(),
        [config, graph_registry](CalculatorGraphConfig::Node& node) {
          return !graph_registry->IsRegistered(config->package(),
                                               node.calculator());
        });
    if (subgraph_nodes_start == nodes->end()) break;
    std::vector<CalculatorGraphConfig> subgraphs;
    for (auto it = subgraph_nodes_start; it != nodes->end(); ++it) {
      auto& node = *it;
      int node_id = it - nodes->begin();
      std::string node_name = CanonicalNodeName(*config, node_id);
      MP_RETURN_IF_ERROR(ValidateSubgraphFields(node));
      SubgraphContext subgraph_context(&node, service_manager);
      ASSIGN_OR_RETURN(auto subgraph, graph_registry->CreateByName(
                                          config->package(), node.calculator(),
                                          &subgraph_context));
      MP_RETURN_IF_ERROR(mediapipe::tool::DefineGraphOptions(node, &subgraph));
      MP_RETURN_IF_ERROR(PrefixNames(node_name, &subgraph));
      MP_RETURN_IF_ERROR(ConnectSubgraphStreams(node, &subgraph));
      subgraphs.push_back(subgraph);
    }
    nodes->erase(subgraph_nodes_start, nodes->end());
    for (const auto& subgraph : subgraphs) {
      std::copy(subgraph.node().begin(), subgraph.node().end(),
                proto_ns::RepeatedPtrFieldBackInserter(nodes));
      std::copy(subgraph.packet_generator().begin(),
                subgraph.packet_generator().end(),
                proto_ns::RepeatedPtrFieldBackInserter(
                    config->mutable_packet_generator()));
      std::copy(subgraph.status_handler().begin(),
                subgraph.status_handler().end(),
                proto_ns::RepeatedPtrFieldBackInserter(
                    config->mutable_status_handler()));
    }
  }
  return absl::OkStatus();
}

CalculatorGraphConfig MakeSingleNodeGraph(CalculatorGraphConfig::Node node) {
  using RepeatedStringField = proto_ns::RepeatedPtrField<ProtoString>;
  struct Connections {
    const RepeatedStringField& node_conns;
    RepeatedStringField* graph_conns;
  };
  CalculatorGraphConfig config;
  for (const Connections& item : std::vector<Connections>{
           {node.input_stream(), config.mutable_input_stream()},
           {node.output_stream(), config.mutable_output_stream()},
           {node.input_side_packet(), config.mutable_input_side_packet()},
           {node.output_side_packet(), config.mutable_output_side_packet()}}) {
    for (const auto& conn : item.node_conns) {
      *item.graph_conns->Add() = conn;
    }
  }
  *config.add_node() = std::move(node);
  return config;
}

}  // namespace tool
}  // namespace mediapipe
