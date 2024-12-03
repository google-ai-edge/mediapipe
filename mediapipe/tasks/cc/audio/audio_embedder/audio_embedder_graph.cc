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

#include <stdint.h>

#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/optional.h"
#include "mediapipe/calculators/core/constant_side_packet_calculator.pb.h"
#include "mediapipe/calculators/tensor/audio_to_tensor_calculator.pb.h"
#include "mediapipe/framework/api2/builder.h"
#include "mediapipe/framework/api2/port.h"
#include "mediapipe/framework/calculator.pb.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/matrix.h"
#include "mediapipe/tasks/cc/audio/audio_embedder/proto/audio_embedder_graph_options.pb.h"
#include "mediapipe/tasks/cc/audio/utils/audio_tensor_specs.h"
#include "mediapipe/tasks/cc/common.h"
#include "mediapipe/tasks/cc/components/containers/proto/embeddings.pb.h"
#include "mediapipe/tasks/cc/components/processors/embedding_postprocessing_graph.h"
#include "mediapipe/tasks/cc/components/processors/proto/embedding_postprocessing_graph_options.pb.h"
#include "mediapipe/tasks/cc/core/model_resources.h"
#include "mediapipe/tasks/cc/core/model_task_graph.h"
#include "mediapipe/tasks/cc/core/proto/inference_subgraph.pb.h"
#include "mediapipe/tasks/cc/metadata/metadata_extractor.h"
#include "mediapipe/tasks/metadata/metadata_schema_generated.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace mediapipe::tasks::audio::audio_embedder {
namespace {

using ::mediapipe::api2::Input;
using ::mediapipe::api2::Output;
using ::mediapipe::api2::builder::Graph;
using ::mediapipe::api2::builder::Source;
using ::mediapipe::tasks::components::containers::proto::EmbeddingResult;

constexpr char kAudioTag[] = "AUDIO";
constexpr char kEmbeddingsTag[] = "EMBEDDINGS";
constexpr char kTimestampedEmbeddingsTag[] = "TIMESTAMPED_EMBEDDINGS";
constexpr char kSampleRateTag[] = "SAMPLE_RATE";
constexpr char kTensorsTag[] = "TENSORS";
constexpr char kTimestampsTag[] = "TIMESTAMPS";

// Struct holding the different output streams produced by the audio embedder
// graph.
struct AudioEmbedderOutputStreams {
  Source<EmbeddingResult> embeddings;
  Source<std::vector<EmbeddingResult>> timestamped_embeddings;
};

// Builds an AudioTensorSpecs for configuring the preprocessing calculators.
absl::StatusOr<AudioTensorSpecs> BuildPreprocessingSpecs(
    const core::ModelResources& model_resources) {
  const tflite::Model& model = *model_resources.GetTfLiteModel();
  if (model.subgraphs()->size() != 1) {
    return CreateStatusWithPayload(absl::StatusCode::kInvalidArgument,
                                   "Audio embedding tflite models are "
                                   "assumed to have a single subgraph.",
                                   MediaPipeTasksStatus::kInvalidArgumentError);
  }
  const auto* primary_subgraph = (*model.subgraphs())[0];
  if (primary_subgraph->inputs()->size() != 1) {
    return CreateStatusWithPayload(absl::StatusCode::kInvalidArgument,
                                   "Audio embedding tflite models are "
                                   "assumed to have a single input.",
                                   MediaPipeTasksStatus::kInvalidArgumentError);
  }
  const auto* input_tensor =
      (*primary_subgraph->tensors())[(*primary_subgraph->inputs())[0]];
  MP_ASSIGN_OR_RETURN(
      const auto* audio_tensor_metadata,
      GetAudioTensorMetadataIfAny(*model_resources.GetMetadataExtractor(), 0));
  return BuildInputAudioTensorSpecs(*input_tensor, audio_tensor_metadata);
}

// Fills in the AudioToTensorCalculatorOptions based on the AudioTensorSpecs.
void ConfigureAudioToTensorCalculator(
    const AudioTensorSpecs& audio_tensor_specs, bool use_stream_mode,
    AudioToTensorCalculatorOptions* options) {
  options->set_num_channels(audio_tensor_specs.num_channels);
  options->set_num_samples(audio_tensor_specs.num_samples);
  options->set_target_sample_rate(audio_tensor_specs.sample_rate);
  options->set_stream_mode(use_stream_mode);
}
}  // namespace

// An "AudioEmebdderGraph" performs embedding extractions.
// - Accepts CPU audio buffer and outputs embedding results on CPU.
//
// Inputs:
//   AUDIO - Matrix
//     Audio buffer to perform classification on.
//   SAMPLE_RATE - double @Optional
//     The sample rate of the corresponding audio data in the "AUDIO" stream.
//     If sample rate is not provided, the "AUDIO" stream must carry a time
//     series stream header with sample rate info.
//
// Outputs:
//   EMBEDDINGS - EmbeddingResult @Optional
//     The embedding results aggregated by head. Only produces results if
//     the graph if the 'use_stream_mode' option is true.
//   TIMESTAMPED_EMBEDDINGS - std::vector<EmbeddingResult> @Optional
//     The embedding result aggregated by timestamp, then by head. Only
//     produces results if the graph if the 'use_stream_mode' option is false.
//
// Example:
// node {
//   calculator: "mediapipe.tasks.audio.audio_embedder.AudioEmbedderGraph"
//   input_stream: "AUDIO:audio_in"
//   input_stream: "SAMPLE_RATE:sample_rate_in"
//   output_stream: "EMBEDDINGS:embeddings_out"
//   output_stream: "TIMESTAMPED_EMBEDDINGS:timestamped_embeddings_out"
//   options {
//     [mediapipe.tasks.audio.audio_embedder.proto.AudioEmbedderGraphOptions.ext]
//     {
//       base_options {
//         model_asset {
//           file_name: "/path/to/model.tflite"
//         }
//       }
//       embedder_options {
//         l2_normalize: true
//       }
//     }
//   }
// }
class AudioEmbedderGraph : public core::ModelTaskGraph {
 public:
  absl::StatusOr<CalculatorGraphConfig> GetConfig(
      SubgraphContext* sc) override {
    MP_ASSIGN_OR_RETURN(
        const auto* model_resources,
        CreateModelResources<proto::AudioEmbedderGraphOptions>(sc));
    Graph graph;
    MP_ASSIGN_OR_RETURN(
        auto output_streams,
        BuildAudioEmbeddingTask(
            sc->Options<proto::AudioEmbedderGraphOptions>(), *model_resources,
            graph[Input<Matrix>(kAudioTag)],
            absl::make_optional(graph[Input<double>(kSampleRateTag)]), graph));
    output_streams.embeddings >> graph[Output<EmbeddingResult>(kEmbeddingsTag)];
    output_streams.timestamped_embeddings >>
        graph[Output<std::vector<EmbeddingResult>>(kTimestampedEmbeddingsTag)];
    return graph.GetConfig();
  }

 private:
  absl::StatusOr<AudioEmbedderOutputStreams> BuildAudioEmbeddingTask(
      const proto::AudioEmbedderGraphOptions& task_options,
      const core::ModelResources& model_resources, Source<Matrix> audio_in,
      absl::optional<Source<double>> sample_rate_in, Graph& graph) {
    const bool use_stream_mode = task_options.base_options().use_stream_mode();
    const auto* metadata_extractor = model_resources.GetMetadataExtractor();
    // Checks that metadata is available.
    if (metadata_extractor->GetModelMetadata() == nullptr ||
        metadata_extractor->GetModelMetadata()->subgraph_metadata() ==
            nullptr) {
      return CreateStatusWithPayload(
          absl::StatusCode::kInvalidArgument,
          "Audio embedder models require TFLite Model Metadata but none was "
          "found",
          MediaPipeTasksStatus::kMetadataNotFoundError);
    }
    // Adds AudioToTensorCalculator and connects it to the graph input streams.
    MP_ASSIGN_OR_RETURN(auto audio_tensor_specs,
                        BuildPreprocessingSpecs(model_resources));
    auto& audio_to_tensor = graph.AddNode("AudioToTensorCalculator");
    ConfigureAudioToTensorCalculator(
        audio_tensor_specs, use_stream_mode,
        &audio_to_tensor.GetOptions<AudioToTensorCalculatorOptions>());
    audio_in >> audio_to_tensor.In(kAudioTag);
    if (sample_rate_in.has_value()) {
      sample_rate_in.value() >> audio_to_tensor.In(kSampleRateTag);
    }

    // Adds inference subgraph and connects its input stream to the output
    // tensors produced by the AudioToTensorCalculator.
    auto& inference = AddInference(
        model_resources, task_options.base_options().acceleration(), graph);
    audio_to_tensor.Out(kTensorsTag) >> inference.In(kTensorsTag);
    // Adds postprocessing calculators and connects its input stream to the
    // inference results.
    auto& postprocessing = graph.AddNode(
        "mediapipe.tasks.components.processors.EmbeddingPostprocessingGraph");
    MP_RETURN_IF_ERROR(
        components::processors::ConfigureEmbeddingPostprocessingGraph(
            model_resources, task_options.embedder_options(),
            &postprocessing
                 .GetOptions<components::processors::proto::
                                 EmbeddingPostprocessingGraphOptions>()));
    inference.Out(kTensorsTag) >> postprocessing.In(kTensorsTag);
    // Time aggregation is only needed for performing audio embedding on
    // audio files. Disables timestamp aggregation by not connecting the
    // "TIMESTAMPS" streams.
    if (!use_stream_mode) {
      audio_to_tensor.Out(kTimestampsTag) >> postprocessing.In(kTimestampsTag);
    }

    // Outputs both streams as graph output streams/
    return AudioEmbedderOutputStreams{
        /*embeddings=*/postprocessing[Output<EmbeddingResult>(kEmbeddingsTag)],
        /*timestamped_embeddings=*/
        postprocessing[Output<std::vector<EmbeddingResult>>(
            kTimestampedEmbeddingsTag)],
    };
  }
};

REGISTER_MEDIAPIPE_GRAPH(
    ::mediapipe::tasks::audio::audio_embedder::AudioEmbedderGraph);

}  // namespace mediapipe::tasks::audio::audio_embedder
