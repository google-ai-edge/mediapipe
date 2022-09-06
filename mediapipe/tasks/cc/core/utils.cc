/* Copyright 2022 The MediaPipe Authors. All Rights Reserved.

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

namespace mediapipe {
namespace tasks {
namespace core {
namespace {
constexpr char kFinishedTag[] = "FINISHED";
constexpr char kFlowLimiterCalculatorName[] = "FlowLimiterCalculator";

}  // namespace

std::string LoadBinaryContent(const char* filename) {
  std::ifstream input_file(filename, std::ios::binary | std::ios::ate);
  // Find buffer size from input file, and load the buffer.
  size_t buffer_size = input_file.tellg();
  std::string buffer(buffer_size, '\0');
  input_file.seekg(0, std::ios::beg);
  input_file.read(const_cast<char*>(buffer.c_str()), buffer_size);
  return buffer;
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

}  // namespace core
}  // namespace tasks
}  // namespace mediapipe
