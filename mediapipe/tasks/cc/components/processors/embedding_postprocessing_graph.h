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

#ifndef MEDIAPIPE_TASKS_CC_COMPONENTS_PROCESSORS_EMBEDDING_POSTPROCESSING_GRAPH_H_
#define MEDIAPIPE_TASKS_CC_COMPONENTS_PROCESSORS_EMBEDDING_POSTPROCESSING_GRAPH_H_

#include "absl/status/status.h"
#include "mediapipe/tasks/cc/components/processors/proto/embedder_options.pb.h"
#include "mediapipe/tasks/cc/components/processors/proto/embedding_postprocessing_graph_options.pb.h"
#include "mediapipe/tasks/cc/core/model_resources.h"

namespace mediapipe {
namespace tasks {
namespace components {
namespace processors {

// Configures an EmbeddingPostprocessingGraph using the provided model resources
// and EmbedderOptions.
// - Accepts CPU input tensors.
//
// Example usage:
//
// auto& postprocessing =
//     graph.AddNode("mediapipe.tasks.components.EmbeddingPostprocessingGraph");
// MP_RETURN_IF_ERROR(ConfigureEmbeddingPostprocessing(
//     model_resources,
//     embedder_options,
//     &postprocessing.GetOptions<EmbeddingPostprocessingGraphOptions>()));
//
// The result EmbeddingPostprocessingGraph has the following I/O:
// Inputs:
//   TENSORS - std::vector<Tensor>
//     The output tensors of an InferenceCalculator, to convert into
//     EmbeddingResult objects. Expected to be of type kFloat32 or kUInt8.
// Outputs:
//   EMBEDDINGS - EmbeddingResult
//     The output EmbeddingResult.
//
// TODO: add support for additional optional "TIMESTAMPS" input for
// embeddings aggregation.
absl::Status ConfigureEmbeddingPostprocessing(
    const tasks::core::ModelResources& model_resources,
    const proto::EmbedderOptions& embedder_options,
    proto::EmbeddingPostprocessingGraphOptions* options);

}  // namespace processors
}  // namespace components
}  // namespace tasks
}  // namespace mediapipe

#endif  // MEDIAPIPE_TASKS_CC_COMPONENTS_PROCESSORS_EMBEDDING_POSTPROCESSING_GRAPH_H_
