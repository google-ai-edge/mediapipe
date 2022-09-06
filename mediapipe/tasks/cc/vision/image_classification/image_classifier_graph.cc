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

#include <limits>
#include <type_traits>

#include "absl/status/statusor.h"
#include "mediapipe/framework/api2/builder.h"
#include "mediapipe/framework/api2/port.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/image.h"
#include "mediapipe/tasks/cc/components/classification_postprocessing.h"
#include "mediapipe/tasks/cc/components/classification_postprocessing_options.pb.h"
#include "mediapipe/tasks/cc/components/containers/classifications.pb.h"
#include "mediapipe/tasks/cc/components/image_preprocessing.h"
#include "mediapipe/tasks/cc/components/image_preprocessing_options.pb.h"
#include "mediapipe/tasks/cc/core/model_resources.h"
#include "mediapipe/tasks/cc/core/model_task_graph.h"
#include "mediapipe/tasks/cc/vision/image_classification/image_classifier_options.pb.h"

namespace mediapipe {
namespace tasks {
namespace vision {

namespace {

using ::mediapipe::api2::Input;
using ::mediapipe::api2::Output;
using ::mediapipe::api2::builder::GenericNode;
using ::mediapipe::api2::builder::Graph;
using ::mediapipe::api2::builder::Source;

constexpr float kDefaultScoreThreshold = std::numeric_limits<float>::lowest();

constexpr char kClassificationResultTag[] = "CLASSIFICATION_RESULT";
constexpr char kImageTag[] = "IMAGE";
constexpr char kTensorsTag[] = "TENSORS";

}  // namespace

// A "mediapipe.tasks.vision.ImageClassifierGraph" performs image
// classification.
// - Accepts CPU input images and outputs classifications on CPU.
//
// Inputs:
//   IMAGE - Image
//     Image to perform classification on.
//
// Outputs:
//   CLASSIFICATION_RESULT - ClassificationResult
//     The aggregated classification result object has two dimensions:
//     (classification head, classification category)
//
// Example:
// node {
//   calculator: "mediapipe.tasks.vision.ImageClassifierGraph"
//   input_stream: "IMAGE:image_in"
//   output_stream: "CLASSIFICATION_RESULT:classification_result_out"
//   options {
//     [mediapipe.tasks.vision.ImageClassifierOptions.ext] {
//       max_results: 3
//       score_threshold: 0.5
//       category_allowlist: "foo"
//       category_allowlist: "bar"
//     }
//   }
// }

class ImageClassifierGraph : public core::ModelTaskGraph {
 public:
  absl::StatusOr<CalculatorGraphConfig> GetConfig(
      SubgraphContext* sc) override {
    ASSIGN_OR_RETURN(const auto* model_resources,
                     CreateModelResources<ImageClassifierOptions>(sc));
    Graph graph;
    ASSIGN_OR_RETURN(
        auto classification_result_out,
        BuildImageClassificationTask(sc->Options<ImageClassifierOptions>(),
                                     *model_resources,
                                     graph[Input<Image>(kImageTag)], graph));
    classification_result_out >>
        graph[Output<ClassificationResult>(kClassificationResultTag)];
    return graph.GetConfig();
  }

 private:
  // Adds a mediapipe image classification task graph into the provided
  // builder::Graph instance. The image classification task takes images
  // (mediapipe::Image) as input and returns one classification result per input
  // image.
  //
  // task_options: the mediapipe tasks ImageClassifierOptions.
  // model_resources: the ModelSources object initialized from an image
  // classification model file with model metadata.
  // image_in: (mediapipe::Image) stream to run object detection on.
  // graph: the mediapipe builder::Graph instance to be updated.
  absl::StatusOr<Source<ClassificationResult>> BuildImageClassificationTask(
      const ImageClassifierOptions& task_options,
      const core::ModelResources& model_resources, Source<Image> image_in,
      Graph& graph) {
    // Adds preprocessing calculators and connects them to the graph input image
    // stream.
    auto& preprocessing =
        graph.AddNode("mediapipe.tasks.ImagePreprocessingSubgraph");
    MP_RETURN_IF_ERROR(ConfigureImagePreprocessing(
        model_resources,
        &preprocessing.GetOptions<ImagePreprocessingOptions>()));
    image_in >> preprocessing.In(kImageTag);

    // Adds inference subgraph and connects its input stream to the outoput
    // tensors produced by the ImageToTensorCalculator.
    auto& inference = AddInference(model_resources, graph);
    preprocessing.Out(kTensorsTag) >> inference.In(kTensorsTag);

    // Adds postprocessing calculators and connects them to the graph output.
    auto& postprocessing =
        graph.AddNode("mediapipe.tasks.ClassificationPostprocessingSubgraph");
    MP_RETURN_IF_ERROR(ConfigureClassificationPostprocessing(
        model_resources, task_options.classifier_options(),
        &postprocessing.GetOptions<ClassificationPostprocessingOptions>()));
    inference.Out(kTensorsTag) >> postprocessing.In(kTensorsTag);

    // Outputs the aggregated classification result as the subgraph output
    // stream.
    return postprocessing[Output<ClassificationResult>(
        kClassificationResultTag)];
  }
};
REGISTER_MEDIAPIPE_GRAPH(::mediapipe::tasks::vision::ImageClassifierGraph);

}  // namespace vision
}  // namespace tasks
}  // namespace mediapipe
