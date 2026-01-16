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

#include "absl/status/statusor.h"
#include "mediapipe/framework/api2/builder.h"
#include "mediapipe/framework/api2/port.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/image.h"
#include "mediapipe/framework/formats/rect.pb.h"
#include "mediapipe/tasks/cc/components/containers/proto/embeddings.pb.h"
#include "mediapipe/tasks/cc/components/processors/embedding_postprocessing_graph.h"
#include "mediapipe/tasks/cc/components/processors/image_preprocessing_graph.h"
#include "mediapipe/tasks/cc/components/processors/proto/embedding_postprocessing_graph_options.pb.h"
#include "mediapipe/tasks/cc/components/processors/proto/image_preprocessing_graph_options.pb.h"
#include "mediapipe/tasks/cc/core/model_task_graph.h"
#include "mediapipe/tasks/cc/vision/image_embedder/proto/image_embedder_graph_options.pb.h"

namespace mediapipe {
namespace tasks {
namespace vision {
namespace image_embedder {

namespace {

using ::mediapipe::NormalizedRect;
using ::mediapipe::api2::Input;
using ::mediapipe::api2::Output;
using ::mediapipe::api2::builder::GenericNode;
using ::mediapipe::api2::builder::Graph;
using ::mediapipe::api2::builder::Source;
using ::mediapipe::tasks::components::containers::proto::EmbeddingResult;

constexpr char kEmbeddingsTag[] = "EMBEDDINGS";
constexpr char kImageTag[] = "IMAGE";
constexpr char kNormRectTag[] = "NORM_RECT";
constexpr char kTensorsTag[] = "TENSORS";

// Struct holding the different output streams produced by the image embedder
// graph.
struct ImageEmbedderOutputStreams {
  Source<EmbeddingResult> embedding_result;
  Source<Image> image;
};

}  // namespace

// An ImageEmbedderGraph performs image embedding extraction.
// - Accepts CPU input images and outputs embeddings on CPU.
//
// Inputs:
//   IMAGE - Image
//     Image to perform embedding extraction on.
//   NORM_RECT - NormalizedRect @Optional
//     Describes region of image to perform embedding extraction on.
//     @Optional: rect covering the whole image is used if not specified.
// Outputs:
//   EMBEDDINGS - EmbeddingResult
//     The embedding result.
//   IMAGE - Image
//     The image that embedding extraction runs on.
//
// Example:
// node {
//   calculator: "mediapipe.tasks.vision.image_embedder.ImageEmbedderGraph"
//   input_stream: "IMAGE:image_in"
//   output_stream: "EMBEDDINGS:embedding_result_out"
//   output_stream: "IMAGE:image_out"
//   options {
//     [mediapipe.tasks.vision.image_embedder.proto.ImageEmbedderOptions.ext]
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
class ImageEmbedderGraph : public core::ModelTaskGraph {
 public:
  absl::StatusOr<CalculatorGraphConfig> GetConfig(
      SubgraphContext* sc) override {
    MP_ASSIGN_OR_RETURN(
        const auto* model_resources,
        CreateModelResources<proto::ImageEmbedderGraphOptions>(sc));
    Graph graph;
    MP_ASSIGN_OR_RETURN(
        auto output_streams,
        BuildImageEmbedderTask(
            sc->Options<proto::ImageEmbedderGraphOptions>(), *model_resources,
            graph[Input<Image>(kImageTag)],
            graph[Input<NormalizedRect>::Optional(kNormRectTag)], graph));
    output_streams.embedding_result >>
        graph[Output<EmbeddingResult>(kEmbeddingsTag)];
    output_streams.image >> graph[Output<Image>(kImageTag)];
    return graph.GetConfig();
  }

 private:
  // Adds a mediapipe image embedding teask graph into the provided
  // builder::Graph instance. The image embedding task takes images
  // (mediapipe::Image) and optional region-of-interest
  // (mediapipe::NormalizedRect) as inputs and returns on embedding result per
  // input image.
  //
  // task_options: the mediapipe tasks ImageEmbedderGraphOptions.
  // model_resources: the ModelSources object initialized from an image
  // embedding model file with model optional metadata.
  // image_in: (mediapipe::Image) stream to run embedding extraction on.
  // norm_rect_in: (mediapipe::NormalizedRect) optional region-of-interest to
  // perform embedding extraction on.
  // graph: the mediapipe builder::Graph instance to be updated.
  absl::StatusOr<ImageEmbedderOutputStreams> BuildImageEmbedderTask(
      const proto::ImageEmbedderGraphOptions& task_options,
      const core::ModelResources& model_resources, Source<Image> image_in,
      Source<NormalizedRect> norm_rect_in, Graph& graph) {
    // Adds preprocessing calculators and connects them to the graph input image
    // stream.
    auto& preprocessing = graph.AddNode(
        "mediapipe.tasks.components.processors.ImagePreprocessingGraph");
    bool use_gpu =
        components::processors::DetermineImagePreprocessingGpuBackend(
            task_options.base_options().acceleration());
    MP_RETURN_IF_ERROR(components::processors::ConfigureImagePreprocessingGraph(
        model_resources, use_gpu, task_options.base_options().gpu_origin(),
        &preprocessing.GetOptions<tasks::components::processors::proto::
                                      ImagePreprocessingGraphOptions>()));
    image_in >> preprocessing.In(kImageTag);
    norm_rect_in >> preprocessing.In(kNormRectTag);

    // Adds inference subgraph and connects its input stream to the outoput
    // tensors produced by the ImageToTensorCalculator.
    auto& inference = AddInference(
        model_resources, task_options.base_options().acceleration(), graph);
    preprocessing.Out(kTensorsTag) >> inference.In(kTensorsTag);

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

    // Outputs the embedding results.
    return ImageEmbedderOutputStreams{
        /*embedding_result=*/postprocessing[Output<EmbeddingResult>(
            kEmbeddingsTag)],
        /*image=*/preprocessing[Output<Image>(kImageTag)]};
  }
};
REGISTER_MEDIAPIPE_GRAPH(
    ::mediapipe::tasks::vision::image_embedder::ImageEmbedderGraph);

}  // namespace image_embedder
}  // namespace vision
}  // namespace tasks
}  // namespace mediapipe
