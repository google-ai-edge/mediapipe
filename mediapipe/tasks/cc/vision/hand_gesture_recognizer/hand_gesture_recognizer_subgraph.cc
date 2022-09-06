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

#include <memory>
#include <type_traits>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "mediapipe/calculators/tensor/tensors_to_classification_calculator.pb.h"
#include "mediapipe/framework/api2/builder.h"
#include "mediapipe/framework/api2/port.h"
#include "mediapipe/framework/formats/classification.pb.h"
#include "mediapipe/framework/formats/landmark.pb.h"
#include "mediapipe/framework/formats/matrix.h"
#include "mediapipe/framework/formats/tensor.h"
#include "mediapipe/tasks/cc/common.h"
#include "mediapipe/tasks/cc/components/classification_postprocessing.h"
#include "mediapipe/tasks/cc/components/classification_postprocessing_options.pb.h"
#include "mediapipe/tasks/cc/components/containers/classifications.pb.h"
#include "mediapipe/tasks/cc/core/model_resources.h"
#include "mediapipe/tasks/cc/core/model_task_graph.h"
#include "mediapipe/tasks/cc/core/proto/inference_subgraph.pb.h"
#include "mediapipe/tasks/cc/core/utils.h"
#include "mediapipe/tasks/cc/vision/hand_gesture_recognizer/proto/hand_gesture_recognizer_subgraph_options.pb.h"
#include "mediapipe/tasks/cc/vision/utils/image_tensor_specs.h"
#include "mediapipe/tasks/metadata/metadata_schema_generated.h"

namespace mediapipe {
namespace tasks {
namespace vision {

namespace {

using ::mediapipe::api2::Input;
using ::mediapipe::api2::Output;
using ::mediapipe::api2::builder::Graph;
using ::mediapipe::api2::builder::Source;
using ::mediapipe::tasks::vision::hand_gesture_recognizer::proto::
    HandGestureRecognizerSubgraphOptions;

absl::Status SanityCheckOptions(
    const HandGestureRecognizerSubgraphOptions& options) {
  if (options.min_tracking_confidence() < 0 ||
      options.min_tracking_confidence() > 1) {
    return CreateStatusWithPayload(absl::StatusCode::kInvalidArgument,
                                   "Invalid `min_tracking_confidence` option: "
                                   "value must be in the range [0.0, 1.0]",
                                   MediaPipeTasksStatus::kInvalidArgumentError);
  }
  return absl::OkStatus();
}

Source<std::vector<Tensor>> ConvertMatrixToTensor(Source<Matrix> matrix,
                                                  Graph& graph) {
  auto& node = graph.AddNode("TensorConverterCalculator");
  matrix >> node.In("MATRIX");
  return node[Output<std::vector<Tensor>>{"TENSORS"}];
}

}  // namespace

// A "mediapipe.tasks.vision.HandGestureRecognizerSubgraph" performs single hand
// gesture recognition. This graph is used as a building block for
// mediapipe.tasks.vision.HandGestureRecognizerGraph.
//
// Inputs:
//   HANDEDNESS - ClassificationList
//     Classification of handedness.
//   LANDMARKS - NormalizedLandmarkList
//     Detected hand landmarks in normalized image coordinates.
//   WORLD_LANDMARKS - LandmarkList
//     Detected hand landmarks in world coordinates.
//   IMAGE_SIZE - std::pair<int, int>
//     The size of image from which the landmarks detected from.
//
// Outputs:
//   HAND_GESTURES - ClassificationResult
//     Recognized hand gestures with sorted order such that the winning label is
//     the first item in the list.
//
//
// Example:
// node {
//   calculator: "mediapipe.tasks.vision.HandGestureRecognizerSubgraph"
//   input_stream: "HANDEDNESS:handedness"
//   input_stream: "LANDMARKS:landmarks"
//   input_stream: "WORLD_LANDMARKS:world_landmarks"
//   input_stream: "IMAGE_SIZE:image_size"
//   output_stream: "HAND_GESTURES:hand_gestures"
//   options {
//     [mediapipe.tasks.vision.hand_gesture_recognizer.proto.HandGestureRecognizerSubgraphOptions.ext]
//     {
//       base_options {
//         model_file: "hand_gesture.tflite"
//       }
//     }
//   }
// }
class HandGestureRecognizerSubgraph : public core::ModelTaskGraph {
 public:
  absl::StatusOr<CalculatorGraphConfig> GetConfig(
      SubgraphContext* sc) override {
    ASSIGN_OR_RETURN(
        const auto* model_resources,
        CreateModelResources<HandGestureRecognizerSubgraphOptions>(sc));
    Graph graph;
    ASSIGN_OR_RETURN(
        auto hand_gestures,
        BuildHandGestureRecognizerGraph(
            sc->Options<HandGestureRecognizerSubgraphOptions>(),
            *model_resources, graph[Input<ClassificationList>("HANDEDNESS")],
            graph[Input<NormalizedLandmarkList>("LANDMARKS")],
            graph[Input<LandmarkList>("WORLD_LANDMARKS")],
            graph[Input<std::pair<int, int>>("IMAGE_SIZE")], graph));
    hand_gestures >> graph[Output<ClassificationResult>("HAND_GESTURES")];
    return graph.GetConfig();
  }

 private:
  absl::StatusOr<Source<ClassificationResult>> BuildHandGestureRecognizerGraph(
      const HandGestureRecognizerSubgraphOptions& graph_options,
      const core::ModelResources& model_resources,
      Source<ClassificationList> handedness,
      Source<NormalizedLandmarkList> hand_landmarks,
      Source<LandmarkList> hand_world_landmarks,
      Source<std::pair<int, int>> image_size, Graph& graph) {
    MP_RETURN_IF_ERROR(SanityCheckOptions(graph_options));

    // Converts the ClassificationList to a matrix.
    auto& handedness_to_matrix = graph.AddNode("HandednessToMatrixCalculator");
    handedness >> handedness_to_matrix.In("HANDEDNESS");
    auto handedness_matrix =
        handedness_to_matrix[Output<Matrix>("HANDEDNESS_MATRIX")];

    // Converts the handedness matrix to a tensor for the inference
    // calculator.
    auto handedness_tensors = ConvertMatrixToTensor(handedness_matrix, graph);

    //  Converts the screen landmarks to a matrix.
    auto& hand_landmarks_to_matrix =
        graph.AddNode("HandLandmarksToMatrixCalculator");
    hand_landmarks >> hand_landmarks_to_matrix.In("HAND_LANDMARKS");
    image_size >> hand_landmarks_to_matrix.In("IMAGE_SIZE");
    auto hand_landmarks_matrix =
        hand_landmarks_to_matrix[Output<Matrix>("LANDMARKS_MATRIX")];

    // Converts the landmarks matrix to a tensor for the inference calculator.
    auto hand_landmarks_tensor =
        ConvertMatrixToTensor(hand_landmarks_matrix, graph);

    // Converts the world landmarks to a matrix.
    auto& hand_world_landmarks_to_matrix =
        graph.AddNode("HandLandmarksToMatrixCalculator");
    hand_world_landmarks >>
        hand_world_landmarks_to_matrix.In("HAND_WORLD_LANDMARKS");
    image_size >> hand_world_landmarks_to_matrix.In("IMAGE_SIZE");
    auto hand_world_landmarks_matrix =
        hand_world_landmarks_to_matrix[Output<Matrix>("LANDMARKS_MATRIX")];

    // Converts the world landmarks matrix to a tensor for the inference
    // calculator.
    auto hand_world_landmarks_tensor =
        ConvertMatrixToTensor(hand_world_landmarks_matrix, graph);

    // Converts a tensor into a vector of tensors for the inference
    // calculator.
    auto& concatenate_tensor_vector =
        graph.AddNode("ConcatenateTensorVectorCalculator");
    hand_landmarks_tensor >> concatenate_tensor_vector.In(0);
    handedness_tensors >> concatenate_tensor_vector.In(1);
    hand_world_landmarks_tensor >> concatenate_tensor_vector.In(2);
    auto concatenated_tensors = concatenate_tensor_vector.Out("");

    // Inference for static hand gesture recognition.
    auto& inference = AddInference(model_resources, graph);
    concatenated_tensors >> inference.In("TENSORS");
    auto inference_output_tensors = inference.Out("TENSORS");

    auto& postprocessing =
        graph.AddNode("mediapipe.tasks.ClassificationPostprocessingSubgraph");
    MP_RETURN_IF_ERROR(ConfigureClassificationPostprocessing(
        model_resources, graph_options.classifier_options(),
        &postprocessing.GetOptions<ClassificationPostprocessingOptions>()));
    inference_output_tensors >> postprocessing.In("TENSORS");
    auto classification_result =
        postprocessing[Output<ClassificationResult>("CLASSIFICATION_RESULT")];

    return {classification_result};
  }
};

REGISTER_MEDIAPIPE_GRAPH(
    ::mediapipe::tasks::vision::HandGestureRecognizerSubgraph);

}  // namespace vision
}  // namespace tasks
}  // namespace mediapipe
