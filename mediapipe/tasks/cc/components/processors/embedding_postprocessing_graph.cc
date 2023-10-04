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

#include "mediapipe/tasks/cc/components/processors/embedding_postprocessing_graph.h"

#include <string>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "mediapipe/framework/api2/builder.h"
#include "mediapipe/framework/api2/port.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/tensor.h"
#include "mediapipe/framework/tool/options_map.h"
#include "mediapipe/tasks/cc/common.h"
#include "mediapipe/tasks/cc/components/calculators/tensors_to_embeddings_calculator.pb.h"
#include "mediapipe/tasks/cc/components/containers/proto/embeddings.pb.h"
#include "mediapipe/tasks/cc/components/processors/proto/embedder_options.pb.h"
#include "mediapipe/tasks/cc/components/processors/proto/embedding_postprocessing_graph_options.pb.h"
#include "mediapipe/tasks/cc/core/model_resources.h"
#include "mediapipe/tasks/cc/metadata/metadata_extractor.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace mediapipe {
namespace tasks {
namespace components {
namespace processors {

namespace {

using ::mediapipe::Tensor;
using ::mediapipe::api2::Input;
using ::mediapipe::api2::Output;
using ::mediapipe::api2::builder::GenericNode;
using ::mediapipe::api2::builder::Graph;
using ::mediapipe::api2::builder::Source;
using ::mediapipe::tasks::components::containers::proto::EmbeddingResult;
using ::mediapipe::tasks::core::ModelResources;

constexpr char kTensorsTag[] = "TENSORS";
constexpr char kEmbeddingsTag[] = "EMBEDDINGS";
constexpr char kTimestampedEmbeddingsTag[] = "TIMESTAMPED_EMBEDDINGS";
constexpr char kTimestampsTag[] = "TIMESTAMPS";

// Struct holding the different output streams produced by the graph.
struct EmbeddingPostprocessingOutputStreams {
  Source<EmbeddingResult> embeddings;
  Source<std::vector<EmbeddingResult>> timestamped_embeddings;
};

// Identifies whether or not the model has quantized outputs, and performs
// sanity checks.
absl::StatusOr<bool> HasQuantizedOutputs(
    const ModelResources& model_resources) {
  const tflite::Model& model = *model_resources.GetTfLiteModel();
  if (model.subgraphs()->size() != 1) {
    return CreateStatusWithPayload(absl::StatusCode::kInvalidArgument,
                                   "Embedding tflite models are "
                                   "assumed to have a single subgraph.",
                                   MediaPipeTasksStatus::kInvalidArgumentError);
  }
  const auto* primary_subgraph = (*model.subgraphs())[0];
  int num_output_tensors = primary_subgraph->outputs()->size();
  // Sanity check tensor types and check if model outputs are quantized or not.
  int num_quantized_tensors = 0;
  for (int i = 0; i < num_output_tensors; ++i) {
    const auto* tensor =
        primary_subgraph->tensors()->Get(primary_subgraph->outputs()->Get(i));
    if (tensor->type() != tflite::TensorType_FLOAT32 &&
        tensor->type() != tflite::TensorType_UINT8) {
      return CreateStatusWithPayload(
          absl::StatusCode::kInvalidArgument,
          absl::StrFormat("Expected output tensor at index %d to have type "
                          "UINT8 or FLOAT32, found %s instead.",
                          i, tflite::EnumNameTensorType(tensor->type())),
          MediaPipeTasksStatus::kInvalidOutputTensorTypeError);
    }
    if (tensor->type() == tflite::TensorType_UINT8) {
      num_quantized_tensors++;
    }
  }
  if (num_quantized_tensors != num_output_tensors &&
      num_quantized_tensors != 0) {
    return CreateStatusWithPayload(
        absl::StatusCode::kInvalidArgument,
        absl::StrFormat(
            "Expected either all or none of the output tensors to be "
            "quantized, but found %d quantized outputs for %d total outputs.",
            num_quantized_tensors, num_output_tensors),
        MediaPipeTasksStatus::kInvalidOutputTensorTypeError);
  }
  // Check if metadata is consistent with model topology.
  const auto* output_tensors_metadata =
      model_resources.GetMetadataExtractor()->GetOutputTensorMetadata();
  if (output_tensors_metadata != nullptr &&
      num_output_tensors != output_tensors_metadata->size()) {
    return CreateStatusWithPayload(
        absl::StatusCode::kInvalidArgument,
        absl::StrFormat("Mismatch between number of output tensors (%d) and "
                        "output tensors metadata (%d).",
                        num_output_tensors, output_tensors_metadata->size()),
        MediaPipeTasksStatus::kMetadataInconsistencyError);
  }
  return num_quantized_tensors > 0;
}

// Extracts head names from model resources. Returns an empty vector if none are
// available. If partially available, the name for heads that don't specify a
// metadata name will be set to the empty string.
absl::StatusOr<std::vector<std::string>> GetHeadNames(
    const ModelResources& model_resources) {
  std::vector<std::string> head_names;
  const auto* output_tensors_metadata =
      model_resources.GetMetadataExtractor()->GetOutputTensorMetadata();
  if (output_tensors_metadata == nullptr) {
    return head_names;
  }
  head_names.reserve(output_tensors_metadata->size());
  bool names_available = false;
  for (const auto& metadata : *output_tensors_metadata) {
    if (metadata->name() != nullptr) {
      names_available = true;
      head_names.push_back(metadata->name()->str());
    } else {
      head_names.push_back("");
    }
  }
  if (!names_available) {
    head_names.clear();
  }
  return head_names;
}

}  // namespace

absl::Status ConfigureEmbeddingPostprocessingGraph(
    const ModelResources& model_resources,
    const proto::EmbedderOptions& embedder_options,
    proto::EmbeddingPostprocessingGraphOptions* options) {
  MP_ASSIGN_OR_RETURN(bool has_quantized_outputs,
                      HasQuantizedOutputs(model_resources));
  options->set_has_quantized_outputs(has_quantized_outputs);
  auto* tensors_to_embeddings_options =
      options->mutable_tensors_to_embeddings_options();
  *tensors_to_embeddings_options->mutable_embedder_options() = embedder_options;
  MP_ASSIGN_OR_RETURN(auto head_names, GetHeadNames(model_resources));
  if (!head_names.empty()) {
    *tensors_to_embeddings_options->mutable_head_names() = {head_names.begin(),
                                                            head_names.end()};
  }
  return absl::OkStatus();
}

// An EmbeddingPostprocessingGraph converts raw tensors into EmbeddingResult
// objects.
// - Accepts CPU input tensors.
//
// Inputs:
//   TENSORS - std::vector<Tensor>
//     The output tensors of an InferenceCalculator, to convert into
//     EmbeddingResult objects. Expected to be of type kFloat32 or kUInt8.
//   TIMESTAMPS - std::vector<Timestamp> @Optional
//     The collection of the timestamps that this calculator should aggregate.
//     This stream is optional: if provided then the TIMESTAMPED_EMBEDDINGS
//     output is used for results. Otherwise as no timestamp aggregation is
//     required the EMBEDDINGS output is used for results.
//
// Outputs:
//   EMBEDDINGS - EmbeddingResult @Optional
//     The embedding results aggregated by head. Must be connected if the
//     TIMESTAMPS input is not connected, as it signals that timestamp
//     aggregation is not required.
//   TIMESTAMPED_EMBEDDINGS - std::vector<EmbeddingResult> @Optional
//     The embedding result aggregated by timestamp, then by head. Must be
//     connected if the TIMESTAMPS input is connected, as it signals that
//     timestamp aggregation is required.
//
// The recommended way of using this graph is through the GraphBuilder API using
// the 'ConfigureEmbeddingPostprocessingGraph()' function. See header file for
// more details.
class EmbeddingPostprocessingGraph : public mediapipe::Subgraph {
 public:
  absl::StatusOr<mediapipe::CalculatorGraphConfig> GetConfig(
      mediapipe::SubgraphContext* sc) override {
    Graph graph;
    MP_ASSIGN_OR_RETURN(
        auto output_streams,
        BuildEmbeddingPostprocessing(
            sc->Options<proto::EmbeddingPostprocessingGraphOptions>(),
            graph[Input<std::vector<Tensor>>(kTensorsTag)],
            graph[Input<std::vector<Timestamp>>(kTimestampsTag)], graph));
    output_streams.embeddings >> graph[Output<EmbeddingResult>(kEmbeddingsTag)];
    output_streams.timestamped_embeddings >>
        graph[Output<std::vector<EmbeddingResult>>(kTimestampedEmbeddingsTag)];
    return graph.GetConfig();
  }

 private:
  // Adds an on-device embedding postprocessing graph into the provided
  // builder::Graph instance. The embedding postprocessing graph takes tensors
  // (std::vector<mediapipe::Tensor>) as input and returns one output stream
  // containing the output embedding results (EmbeddingResult).
  //
  // options: the on-device EmbeddingPostprocessingGraphOptions
  // tensors_in: (std::vector<mediapipe::Tensor>) tensors to postprocess.
  // timestamps_in: (std::vector<mediapipe::Timestamp>) optional collection of
  //   timestamps that should be used to aggregate embedding results.
  // graph: the mediapipe builder::Graph instance to be updated.
  absl::StatusOr<EmbeddingPostprocessingOutputStreams>
  BuildEmbeddingPostprocessing(
      const proto::EmbeddingPostprocessingGraphOptions options,
      Source<std::vector<Tensor>> tensors_in,
      Source<std::vector<Timestamp>> timestamps_in, Graph& graph) {
    // If output tensors are quantized, they must be dequantized first.
    Source<std::vector<Tensor>> dequantized_tensors = tensors_in;
    if (options.has_quantized_outputs()) {
      GenericNode& tensors_dequantization_node =
          graph.AddNode("TensorsDequantizationCalculator");
      tensors_in >> tensors_dequantization_node.In(kTensorsTag);
      dequantized_tensors = tensors_dequantization_node.Out(kTensorsTag)
                                .Cast<std::vector<Tensor>>();
    }

    // Adds TensorsToEmbeddingsCalculator.
    GenericNode& tensors_to_embeddings_node =
        graph.AddNode("TensorsToEmbeddingsCalculator");
    tensors_to_embeddings_node
        .GetOptions<mediapipe::TensorsToEmbeddingsCalculatorOptions>()
        .CopyFrom(options.tensors_to_embeddings_options());
    dequantized_tensors >> tensors_to_embeddings_node.In(kTensorsTag);

    // Adds EmbeddingAggregationCalculator.
    GenericNode& aggregation_node =
        graph.AddNode("EmbeddingAggregationCalculator");
    tensors_to_embeddings_node[Output<EmbeddingResult>(kEmbeddingsTag)] >>
        aggregation_node.In(kEmbeddingsTag);
    timestamps_in >> aggregation_node.In(kTimestampsTag);

    // Connects outputs.
    return EmbeddingPostprocessingOutputStreams{
        /*embeddings=*/aggregation_node[Output<EmbeddingResult>(
            kEmbeddingsTag)],
        /*timestamped_embeddings=*/aggregation_node
            [Output<std::vector<EmbeddingResult>>(kTimestampedEmbeddingsTag)]};
  }
};
REGISTER_MEDIAPIPE_GRAPH(
    ::mediapipe::tasks::components::processors::EmbeddingPostprocessingGraph);

}  // namespace processors
}  // namespace components
}  // namespace tasks
}  // namespace mediapipe
