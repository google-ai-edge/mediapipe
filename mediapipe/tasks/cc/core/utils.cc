/* Copyright 2022 The MediaPipe Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "mediapipe/tasks/cc/core/utils.h"

#include <stddef.h>

#include <fstream>
#include <string>

#include "absl/strings/string_view.h"
#include "flatbuffers/flatbuffers.h"
#include "mediapipe/calculators/core/flow_limiter_calculator.pb.h"
#include "mediapipe/tasks/cc/core/external_file_handler.h"
#include "mediapipe/tasks/cc/core/proto/external_file.pb.h"

namespace mediapipe {
namespace tasks {
namespace core {
namespace {
constexpr char kFinishedTag[] = "FINISHED";
constexpr char kFlowLimiterCalculatorName[] = "FlowLimiterCalculator";
constexpr char kPreviousLoopbackCalculatorName[] = "PreviousLoopbackCalculator";

}  // namespace

std::string LoadBinaryContent(const char* filename) {
  proto::ExternalFile external_file;
  external_file.set_file_name(filename);
  auto file_handler =
      ExternalFileHandler::CreateFromExternalFile(&external_file);
  return std::string{(*file_handler)->GetFileContent()};
}

int FindTensorIndexByMetadataName(
    const flatbuffers::Vector<flatbuffers::Offset<tflite::TensorMetadata>>*
        tensor_metadatas,
    absl::string_view name) {
  if (tensor_metadatas == nullptr) {
    return -1;
  }
  for (int i = 0; i < tensor_metadatas->size(); i++) {
    if (name == tensor_metadatas->Get(i)->name()->c_str()) {
      return i;
    }
  }
  // Returns -1 if not found.
  return -1;
}

CalculatorGraphConfig AddFlowLimiterCalculator(
    api2::builder::Graph& graph, api2::builder::GenericNode& task_subgraph,
    std::vector<std::string> input_stream_tags, std::string finished_stream_tag,
    int max_in_flight, int max_in_queue) {
  auto& flow_limiter = graph.AddNode(kFlowLimiterCalculatorName);
  auto& options =
      flow_limiter.GetOptions<mediapipe::FlowLimiterCalculatorOptions>();
  options.set_max_in_flight(max_in_flight);
  options.set_max_in_queue(max_in_queue);
  for (int i = 0; i < input_stream_tags.size(); ++i) {
    graph.In(input_stream_tags[i]) >> flow_limiter.In("")[i];
    flow_limiter.Out("")[i] >> task_subgraph.In(input_stream_tags[i]);
  }
  // Back edge.
  task_subgraph.Out(finished_stream_tag) >> flow_limiter.In(kFinishedTag);

  // As mediapipe GraphBuilder currently doesn't support configuring
  // InputStreamInfo, modifying the CalculatorGraphConfig proto directly.
  CalculatorGraphConfig config = graph.GetConfig();
  for (int i = 0; i < config.node_size(); ++i) {
    if (config.node(i).calculator() == kFlowLimiterCalculatorName) {
      auto* info = config.mutable_node(i)->add_input_stream_info();
      info->set_tag_index(kFinishedTag);
      info->set_back_edge(true);
      break;
    }
  }
  return config;
}

void FixGraphBackEdges(::mediapipe::CalculatorGraphConfig& graph_config) {
  // TODO remove when support is fixed.
  // As mediapipe GraphBuilder currently doesn't support configuring
  // InputStreamInfo, modifying the CalculatorGraphConfig proto directly.
  for (int i = 0; i < graph_config.node_size(); ++i) {
    if (graph_config.node(i).calculator() == kPreviousLoopbackCalculatorName) {
      auto* info = graph_config.mutable_node(i)->add_input_stream_info();
      info->set_tag_index("LOOP");
      info->set_back_edge(true);
    }
  }
}

}  // namespace core
}  // namespace tasks
}  // namespace mediapipe
