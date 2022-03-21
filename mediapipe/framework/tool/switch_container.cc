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

#include <fstream>
#include <iostream>
#include <sstream>

#include "mediapipe/framework/calculator.pb.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/mediapipe_options.pb.h"
#include "mediapipe/framework/port/canonical_errors.h"
#include "mediapipe/framework/port/ret_check.h"
#include "mediapipe/framework/port/status.h"
#include "mediapipe/framework/tool/container_util.h"
#include "mediapipe/framework/tool/name_util.h"
#include "mediapipe/framework/tool/subgraph_expansion.h"
#include "mediapipe/framework/tool/switch_container.pb.h"

namespace mediapipe {
namespace tool {
using mediapipe::SwitchContainerOptions;

// A graph factory producing a CalculatorGraphConfig routing packets to
// one of several contained CalculatorGraphConfigs.
//
// Usage example:
//
//     node {
//       calculator: "SwitchContainer"
//       input_stream: "ENABLE:enable"
//       input_stream: "INPUT_VIDEO:video_frames"
//       output_stream: "OUTPUT_VIDEO:output_frames"
//       options {
//         [mediapipe.SwitchContainerOptions.ext] {
//           contained_node: { calculator: "BasicSubgraph" }
//           contained_node: { calculator: "AdvancedSubgraph" }
//         }
//       }
//     }
//
// Note that the input and output stream tags supplied to the container node
// must match the input and output stream tags required by the contained nodes,
// such as "INPUT_VIDEO" and "OUTPUT_VIDEO" in the example above.
//
// Input stream "ENABLE" specifies routing of packets to either contained_node 0
// or contained_node 1, given "ENABLE:false" or "ENABLE:true" respectively.
// Input-side-packet "ENABLE" and input-stream "SELECT" can also be used
// similarly to specify the active channel.
//
// Note that this container defaults to use ImmediateInputStreamHandler,
// which can be used to accept infrequent "enable" packets asynchronously.
// However, it can be overridden to work with DefaultInputStreamHandler,
// which can be used to accept frequent "enable" packets synchronously.
class SwitchContainer : public Subgraph {
 public:
  SwitchContainer() = default;
  absl::StatusOr<CalculatorGraphConfig> GetConfig(
      const Subgraph::SubgraphOptions& options) override;
};
REGISTER_MEDIAPIPE_GRAPH(SwitchContainer);

using TagIndex = std::pair<std::string, int>;

// Returns the stream name for one of the demux output channels.
// This is the channel number followed by the stream name separated by "__".
// For example, the channel-name for sream "frame" on channel 1 is "c1__frame".
std::string ChannelName(const std::string& name, int channel) {
  return absl::StrCat("c", channel, "__", name);
}

// Returns a SwitchDemuxCalculator node.
CalculatorGraphConfig::Node* BuildDemuxNode(
    const std::map<TagIndex, std::string>& input_tags,
    const CalculatorGraphConfig::Node& container_node,
    CalculatorGraphConfig* config) {
  CalculatorGraphConfig::Node* result = config->add_node();
  *result->mutable_calculator() = "SwitchDemuxCalculator";
  return result;
}

// Returns a SwitchMuxCalculator node.
CalculatorGraphConfig::Node* BuildMuxNode(
    const std::map<TagIndex, std::string>& output_tags,
    CalculatorGraphConfig* config) {
  CalculatorGraphConfig::Node* result = config->add_node();
  *result->mutable_calculator() = "SwitchMuxCalculator";
  return result;
}

// Copies options from one node to another.
void CopyOptions(const CalculatorGraphConfig::Node& source,
                 CalculatorGraphConfig::Node* dest) {
  if (source.has_options()) {
    *dest->mutable_options() = source.options();
  }
  *dest->mutable_node_options() = source.node_options();
}

// Clears options that are consumed by the container and not forwarded.
void ClearContainerOptions(SwitchContainerOptions* result) {
  result->clear_contained_node();
}

// Clears options that are consumed by the container and not forwarded.
void ClearContainerOptions(CalculatorGraphConfig::Node* dest) {
  if (dest->has_options() &&
      dest->mutable_options()->HasExtension(SwitchContainerOptions::ext)) {
    ClearContainerOptions(
        dest->mutable_options()->MutableExtension(SwitchContainerOptions::ext));
  }
  for (google::protobuf::Any& a : *dest->mutable_node_options()) {
    if (a.Is<SwitchContainerOptions>()) {
      SwitchContainerOptions extension;
      a.UnpackTo(&extension);
      ClearContainerOptions(&extension);
      a.PackFrom(extension);
    }
  }
}

// Returns an unused name similar to a specified name.
std::string UniqueName(std::string name, std::set<std::string>* names) {
  CHECK(names != nullptr);
  std::string result = name;
  int suffix = 2;
  while (names->count(result) > 0) {
    result = absl::StrCat(name, "_", suffix++);
  }
  names->insert(result);
  return result;
}

// Parses tag, index, and name from a list of stream identifiers.
void ParseTags(const proto_ns::RepeatedPtrField<std::string>& streams,
               std::map<TagIndex, std::string>* result) {
  CHECK(result != nullptr);
  std::set<std::string> used_names;
  int used_index = -1;
  for (const std::string& stream : streams) {
    std::string name = UniqueName(ParseNameFromStream(stream), &used_names);
    TagIndex tag_index = ParseTagIndexFromStream(stream);
    if (tag_index.second == -1) {
      tag_index.second = ++used_index;
    }
    result->insert({tag_index, name});
  }
}

// Removes the entry for a tag and index from a map.
void EraseTag(const std::string& stream,
              std::map<TagIndex, std::string>* streams) {
  CHECK(streams != nullptr);
  streams->erase(ParseTagIndexFromStream(absl::StrCat(stream, ":u")));
}

// Removes the entry for a tag and index from a list.
void EraseTag(const std::string& stream,
              proto_ns::RepeatedPtrField<std::string>* streams) {
  CHECK(streams != nullptr);
  TagIndex stream_tag = ParseTagIndexFromStream(absl::StrCat(stream, ":u"));
  for (int i = streams->size() - 1; i >= 0; --i) {
    TagIndex tag = ParseTagIndexFromStream(streams->at(i));
    if (tag == stream_tag) {
      streams->erase(streams->begin() + i);
    }
  }
}

// Returns the stream names for the container node.
void GetContainerNodeStreams(const CalculatorGraphConfig::Node& node,
                             CalculatorGraphConfig::Node* result) {
  CHECK(result != nullptr);
  *result->mutable_input_stream() = node.input_stream();
  *result->mutable_output_stream() = node.output_stream();
  *result->mutable_input_side_packet() = node.input_side_packet();
  *result->mutable_output_side_packet() = node.output_side_packet();
  EraseTag("ENABLE", result->mutable_input_stream());
  EraseTag("ENABLE", result->mutable_input_side_packet());
  EraseTag("SELECT", result->mutable_input_stream());
  EraseTag("SELECT", result->mutable_input_side_packet());
}

// Validate all subgraph inputs and outputs.
absl::Status ValidateContract(
    const CalculatorGraphConfig::Node& subgraph_node,
    const Subgraph::SubgraphOptions& subgraph_options) {
  auto options =
      Subgraph::GetOptions<mediapipe::SwitchContainerOptions>(subgraph_options);
  std::map<TagIndex, std::string> input_tags, side_tags;
  ParseTags(subgraph_node.input_stream(), &input_tags);
  ParseTags(subgraph_node.input_side_packet(), &side_tags);
  if (options.has_select() && options.has_enable()) {
    return absl::InvalidArgumentError(
        "Only one of SwitchContainer options 'enable' and 'select' can be "
        "specified");
  }
  if (side_tags.count({"SELECT", 0}) + side_tags.count({"ENABLE", 0}) > 1 ||
      input_tags.count({"SELECT", 0}) + input_tags.count({"ENABLE", 0}) > 1) {
    return absl::InvalidArgumentError(
        "Only one of SwitchContainer inputs 'ENABLE' and 'SELECT' can be "
        "specified");
  }
  return absl::OkStatus();
}

absl::StatusOr<CalculatorGraphConfig> SwitchContainer::GetConfig(
    const Subgraph::SubgraphOptions& options) {
  CalculatorGraphConfig config;
  std::vector<CalculatorGraphConfig::Node*> subnodes;
  std::vector<CalculatorGraphConfig::Node> substreams;

  // Parse all input and output tags from the container node.
  auto container_node = Subgraph::GetNode(options);
  MP_RETURN_IF_ERROR(ValidateContract(container_node, options));
  CalculatorGraphConfig::Node container_streams;
  GetContainerNodeStreams(container_node, &container_streams);
  std::map<TagIndex, std::string> input_tags, output_tags;
  std::map<TagIndex, std::string> side_input_tags, side_output_tags;
  ParseTags(container_streams.input_stream(), &input_tags);
  ParseTags(container_streams.output_stream(), &output_tags);
  ParseTags(container_streams.input_side_packet(), &side_input_tags);
  ParseTags(container_streams.output_side_packet(), &side_output_tags);

  // Add a graph node for the demux, mux.
  auto demux = BuildDemuxNode(input_tags, container_node, &config);
  CopyOptions(container_node, demux);
  ClearContainerOptions(demux);
  demux->add_input_stream("SELECT:gate_select");
  demux->add_input_stream("ENABLE:gate_enable");
  demux->add_input_side_packet("SELECT:gate_select");
  demux->add_input_side_packet("ENABLE:gate_enable");

  auto mux = BuildMuxNode(output_tags, &config);
  CopyOptions(container_node, mux);
  ClearContainerOptions(mux);
  mux->add_input_stream("SELECT:gate_select");
  mux->add_input_stream("ENABLE:gate_enable");
  mux->add_input_side_packet("SELECT:gate_select");
  mux->add_input_side_packet("ENABLE:gate_enable");

  // Add input streams for graph and demux.
  config.add_input_stream("SELECT:gate_select");
  config.add_input_stream("ENABLE:gate_enable");
  config.add_input_side_packet("SELECT:gate_select");
  config.add_input_side_packet("ENABLE:gate_enable");
  for (const auto& p : input_tags) {
    std::string stream = CatStream(p.first, p.second);
    config.add_input_stream(stream);
    demux->add_input_stream(stream);
  }

  // Add output streams for graph and mux.
  for (const auto& p : output_tags) {
    std::string stream = CatStream(p.first, p.second);
    config.add_output_stream(stream);
    mux->add_output_stream(stream);
  }
  for (const auto& p : side_input_tags) {
    std::string side = CatStream(p.first, p.second);
    config.add_input_side_packet(side);
    demux->add_input_side_packet(side);
  }
  for (const auto& p : side_output_tags) {
    std::string side = CatStream(p.first, p.second);
    config.add_output_side_packet(side);
    mux->add_output_side_packet(side);
  }

  // Add a subnode for each contained_node.
  auto nodes = Subgraph::GetOptions<mediapipe::SwitchContainerOptions>(options)
                   .contained_node();
  std::vector<CalculatorGraphConfig::Node> contained_nodes(nodes.begin(),
                                                           nodes.end());
  for (int i = 0; i < contained_nodes.size(); ++i) {
    auto subnode = config.add_node();
    *subnode = contained_nodes[i];
    subnodes.push_back(subnode);
    substreams.push_back(container_streams);
  }

  // Connect each contained graph node to demux and mux.
  for (int channel = 0; channel < subnodes.size(); ++channel) {
    CalculatorGraphConfig::Node& streams = substreams[channel];

    // Connect each contained graph node input to a demux output.
    std::map<TagIndex, std::string> input_stream_tags;
    ParseTags(streams.input_stream(), &input_stream_tags);
    for (auto& it : input_stream_tags) {
      TagIndex tag_index = it.first;
      std::string tag = ChannelTag(tag_index.first, channel);
      std::string name = ChannelName(input_tags[tag_index], channel);
      std::string demux_stream = CatStream({tag, tag_index.second}, name);
      demux->add_output_stream(demux_stream);
      subnodes[channel]->add_input_stream(CatStream(tag_index, name));
    }

    // Connect each contained graph node output to a mux input.
    std::map<TagIndex, std::string> output_stream_tags;
    ParseTags(streams.output_stream(), &output_stream_tags);
    for (auto& it : output_stream_tags) {
      TagIndex tag_index = it.first;
      std::string tag = ChannelTag(tag_index.first, channel);
      std::string name = ChannelName(output_tags[tag_index], channel);
      subnodes[channel]->add_output_stream(CatStream(tag_index, name));
      mux->add_input_stream(CatStream({tag, tag_index.second}, name));
    }

    // Connect each contained graph node side-input to a demux side-output.
    std::map<TagIndex, std::string> input_side_tags;
    ParseTags(streams.input_side_packet(), &input_side_tags);
    for (auto& it : input_side_tags) {
      TagIndex tag_index = it.first;
      std::string tag = ChannelTag(tag_index.first, channel);
      std::string name = ChannelName(side_input_tags[tag_index], channel);
      std::string demux_stream = CatStream({tag, tag_index.second}, name);
      demux->add_output_side_packet(demux_stream);
      subnodes[channel]->add_input_side_packet(CatStream(tag_index, name));
    }

    // Connect each contained graph node side-output to a mux side-input.
    std::map<TagIndex, std::string> output_side_tags;
    ParseTags(streams.output_side_packet(), &output_side_tags);
    for (auto& it : output_side_tags) {
      TagIndex tag_index = it.first;
      std::string tag = ChannelTag(tag_index.first, channel);
      std::string name = ChannelName(side_output_tags[tag_index], channel);
      subnodes[channel]->add_output_side_packet(CatStream(tag_index, name));
      mux->add_input_side_packet(CatStream({tag, tag_index.second}, name));
    }
  }

  return config;
}

}  // namespace tool
}  // namespace mediapipe
