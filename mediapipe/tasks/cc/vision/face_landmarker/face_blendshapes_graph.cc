/* Copyright 2023 The MediaPipe Authors.

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

#include <utility>
#include <vector>

#include "mediapipe/calculators/core/split_vector_calculator.pb.h"
#include "mediapipe/calculators/tensor/landmarks_to_tensor_calculator.h"
#include "mediapipe/calculators/tensor/landmarks_to_tensor_calculator.pb.h"
#include "mediapipe/calculators/tensor/tensors_to_classification_calculator.pb.h"
#include "mediapipe/framework/api2/builder.h"
#include "mediapipe/framework/api2/port.h"
#include "mediapipe/framework/formats/classification.pb.h"
#include "mediapipe/framework/formats/landmark.pb.h"
#include "mediapipe/framework/formats/tensor.h"
#include "mediapipe/framework/port/status_macros.h"
#include "mediapipe/tasks/cc/core/model_task_graph.h"
#include "mediapipe/tasks/cc/vision/face_landmarker/proto/face_blendshapes_graph_options.pb.h"

namespace mediapipe {
namespace tasks {
namespace vision {
namespace face_landmarker {

namespace {

using ::mediapipe::api2::Input;
using ::mediapipe::api2::LandmarksToTensorCalculator;
using ::mediapipe::api2::Output;
using ::mediapipe::api2::builder::Graph;
using ::mediapipe::api2::builder::Stream;
using ::mediapipe::tasks::vision::face_landmarker::proto::
    FaceBlendshapesGraphOptions;

constexpr char kLandmarksTag[] = "LANDMARKS";
constexpr char kImageSizeTag[] = "IMAGE_SIZE";
constexpr char kBlendshapesTag[] = "BLENDSHAPES";

// Subset of 478 landmarks required for the HUND model.
static constexpr std::array<int, 146> kLandmarksSubsetIdxs = {
    0,   1,   4,   5,   6,   7,   8,   10,  13,  14,  17,  21,  33,  37,  39,
    40,  46,  52,  53,  54,  55,  58,  61,  63,  65,  66,  67,  70,  78,  80,
    81,  82,  84,  87,  88,  91,  93,  95,  103, 105, 107, 109, 127, 132, 133,
    136, 144, 145, 146, 148, 149, 150, 152, 153, 154, 155, 157, 158, 159, 160,
    161, 162, 163, 168, 172, 173, 176, 178, 181, 185, 191, 195, 197, 234, 246,
    249, 251, 263, 267, 269, 270, 276, 282, 283, 284, 285, 288, 291, 293, 295,
    296, 297, 300, 308, 310, 311, 312, 314, 317, 318, 321, 323, 324, 332, 334,
    336, 338, 356, 361, 362, 365, 373, 374, 375, 377, 378, 379, 380, 381, 382,
    384, 385, 386, 387, 388, 389, 390, 397, 398, 400, 402, 405, 409, 415, 454,
    466, 468, 469, 470, 471, 472, 473, 474, 475, 476, 477};
static constexpr std::array<absl::string_view, 52> kBlendshapeNames = {
    "_neutral",
    "browDownLeft",
    "browDownRight",
    "browInnerUp",
    "browOuterUpLeft",
    "browOuterUpRight",
    "cheekPuff",
    "cheekSquintLeft",
    "cheekSquintRight",
    "eyeBlinkLeft",
    "eyeBlinkRight",
    "eyeLookDownLeft",
    "eyeLookDownRight",
    "eyeLookInLeft",
    "eyeLookInRight",
    "eyeLookOutLeft",
    "eyeLookOutRight",
    "eyeLookUpLeft",
    "eyeLookUpRight",
    "eyeSquintLeft",
    "eyeSquintRight",
    "eyeWideLeft",
    "eyeWideRight",
    "jawForward",
    "jawLeft",
    "jawOpen",
    "jawRight",
    "mouthClose",
    "mouthDimpleLeft",
    "mouthDimpleRight",
    "mouthFrownLeft",
    "mouthFrownRight",
    "mouthFunnel",
    "mouthLeft",
    "mouthLowerDownLeft",
    "mouthLowerDownRight",
    "mouthPressLeft",
    "mouthPressRight",
    "mouthPucker",
    "mouthRight",
    "mouthRollLower",
    "mouthRollUpper",
    "mouthShrugLower",
    "mouthShrugUpper",
    "mouthSmileLeft",
    "mouthSmileRight",
    "mouthStretchLeft",
    "mouthStretchRight",
    "mouthUpperUpLeft",
    "mouthUpperUpRight",
    "noseSneerLeft",
    "noseSneerRight"};

struct FaceBlendshapesOuts {
  Stream<ClassificationList> blendshapes;
};

Stream<NormalizedLandmarkList> GetLandmarksSubset(
    Stream<NormalizedLandmarkList> landmarks,
    const std::array<int, 146>& subset_idxs, Graph& graph) {
  auto& node = graph.AddNode("SplitNormalizedLandmarkListCalculator");
  auto& opts = node.GetOptions<SplitVectorCalculatorOptions>();
  for (int i = 0; i < subset_idxs.size(); ++i) {
    auto* range = opts.add_ranges();
    range->set_begin(subset_idxs[i]);
    range->set_end(subset_idxs[i] + 1);
  }
  opts.set_combine_outputs(true);
  landmarks >> node.In("");
  return node.Out("").Cast<NormalizedLandmarkList>();
}

Stream<std::vector<Tensor>> ConvertLandmarksToTensor(
    Stream<NormalizedLandmarkList> landmarks,
    Stream<std::pair<int, int>> img_size, Graph& graph) {
  auto& node = graph.AddNode<LandmarksToTensorCalculator>();
  auto& opts = node.GetOptions<LandmarksToTensorCalculatorOptions>();
  opts.add_attributes(LandmarksToTensorCalculatorOptions::X);
  opts.add_attributes(LandmarksToTensorCalculatorOptions::Y);
  opts.set_flatten(false);
  landmarks >> node[LandmarksToTensorCalculator::kInNormLandmarkList];
  img_size >> node[LandmarksToTensorCalculator::kImageSize];
  return node[LandmarksToTensorCalculator::kOutTensors];
}

Stream<std::vector<Tensor>> GetTensorWithBlendshapes(
    Stream<std::vector<Tensor>> tensors, Graph& graph) {
  auto& node = graph.AddNode("SplitTensorVectorCalculator");
  auto& opts = node.GetOptions<SplitVectorCalculatorOptions>();
  auto* range = opts.add_ranges();
  range->set_begin(0);
  range->set_end(1);
  opts.set_combine_outputs(true);
  tensors >> node.In(0);
  return node.Out(0).Cast<std::vector<Tensor>>();
}

Stream<ClassificationList> ConvertTensorToBlendshapes(
    Stream<std::vector<Tensor>> tensors,
    const std::array<absl::string_view, 52>& labels, Graph& graph) {
  auto& node = graph.AddNode("TensorsToClassificationCalculator");
  auto& opts = node.GetOptions<TensorsToClassificationCalculatorOptions>();
  // Disable top_k and min_score_threshold logic to return coefficients as is.
  opts.set_top_k(0);
  opts.set_min_score_threshold(-1.0);
  for (int i = 0; i < labels.size(); ++i) {
    auto* entry = opts.mutable_label_map()->add_entries();
    entry->set_id(i);
    // TODO: Replace with entry->set_label(labels[i])
    entry->mutable_label()->assign(labels[i].data(), labels[i].size());
  }
  tensors >> node.In("TENSORS");
  return node.Out("CLASSIFICATIONS").Cast<ClassificationList>();
}

}  // namespace

// Predicts face blendshapes from landmarks.
//
// Inputs:
//   LANDMARKS - NormalizedLandmarkList
//     478 2.5D face landmarks predicted by an Attention Mesh model.
//   IMAGE_SIZE - std::pair<int, int>
//     Input frame size.
//
// Outputs:
//   FACE_BLENDSHAPES - ClassificationList
//     if EXTRA_FACE_BLENDSHAPES is provided, we have 53 blendshape coeffs
//     output; if not, 52 coeffs output.
//     All 52 blendshape coefficients:
//       0  - _neutral  (ignore it)
//       1  - browDownLeft
//       2  - browDownRight
//       3  - browInnerUp
//       4  - browOuterUpLeft
//       5  - browOuterUpRight
//       6  - cheekPuff
//       7  - cheekSquintLeft
//       8  - cheekSquintRight
//       9  - eyeBlinkLeft
//       10 - eyeBlinkRight
//       11 - eyeLookDownLeft
//       12 - eyeLookDownRight
//       13 - eyeLookInLeft
//       14 - eyeLookInRight
//       15 - eyeLookOutLeft
//       16 - eyeLookOutRight
//       17 - eyeLookUpLeft
//       18 - eyeLookUpRight
//       19 - eyeSquintLeft
//       20 - eyeSquintRight
//       21 - eyeWideLeft
//       22 - eyeWideRight
//       23 - jawForward
//       24 - jawLeft
//       25 - jawOpen
//       26 - jawRight
//       27 - mouthClose
//       28 - mouthDimpleLeft
//       29 - mouthDimpleRight
//       30 - mouthFrownLeft
//       31 - mouthFrownRight
//       32 - mouthFunnel
//       33 - mouthLeft
//       34 - mouthLowerDownLeft
//       35 - mouthLowerDownRight
//       36 - mouthPressLeft
//       37 - mouthPressRight
//       38 - mouthPucker
//       39 - mouthRight
//       40 - mouthRollLower
//       41 - mouthRollUpper
//       42 - mouthShrugLower
//       43 - mouthShrugUpper
//       44 - mouthSmileLeft
//       45 - mouthSmileRight
//       46 - mouthStretchLeft
//       47 - mouthStretchRight
//       48 - mouthUpperUpLeft
//       49 - mouthUpperUpRight
//       50 - noseSneerLeft
//       51 - noseSneerRight
//
// Example:
// node {
//   calculator: "mediapipe.tasks.vision.face_landmarker.FaceBlendshapesGraph"
//   input_stream: "LANDMARKS:face_landmarks"
//   input_stream: "IMAGE_SIZE:image_size"
//   output_stream: "BLENDSHAPES:face_blendshapes"
//   options {
//     [mediapipe.tasks.vision.face_landmarker.proto.FaceBlendshapesGraphOptions.ext]
//     {
//       base_options {
//         model_asset {
//           file_name: "face_blendshapes.tflite"
//         }
//       }
//     }
//   }
// }
class FaceBlendshapesGraph : public core::ModelTaskGraph {
 public:
  absl::StatusOr<CalculatorGraphConfig> GetConfig(SubgraphContext* sc) {
    MP_ASSIGN_OR_RETURN(const auto* model_resources,
                        CreateModelResources<FaceBlendshapesGraphOptions>(sc));
    Graph graph;
    MP_ASSIGN_OR_RETURN(
        auto face_blendshapes_outs,
        BuildFaceBlendshapesSubgraph(
            sc->Options<FaceBlendshapesGraphOptions>(), *model_resources,
            graph[Input<NormalizedLandmarkList>(kLandmarksTag)],
            graph[Input<std::pair<int, int>>(kImageSizeTag)], graph));
    face_blendshapes_outs.blendshapes >>
        graph[Output<ClassificationList>(kBlendshapesTag)];

    return graph.GetConfig();
  }

 private:
  // Updates graph to predict face blendshapes from landmarks. Returns list of
  // blendsahpes.
  //
  // subgraph_options: the mediapipe tasks module FaceBlendshapesGraphOptions.
  // model_resources: the ModelSources object initialized from a face
  // blendshapes model file with model metadata.
  // landmarks: 478 normalized face landmarks
  // img_size: Image size to denormalize landmarks.
  // graph: the mediapipe builder::Graph instance to be updated.
  absl::StatusOr<FaceBlendshapesOuts> BuildFaceBlendshapesSubgraph(
      const FaceBlendshapesGraphOptions& subgraph_options,
      const core::ModelResources& model_resources,
      Stream<NormalizedLandmarkList> landmarks,
      Stream<std::pair<int, int>> img_size, Graph& graph) {
    // Take required subset of landmarks.
    landmarks = GetLandmarksSubset(landmarks, kLandmarksSubsetIdxs, graph);

    // Convert landmarks to input tensor.
    auto tensor_in = ConvertLandmarksToTensor(landmarks, img_size, graph);

    // Run Blendshapes model.
    auto& inference = AddInference(
        model_resources, subgraph_options.base_options().acceleration(), graph);
    tensor_in >> inference.In("TENSORS");
    auto tensors_out = inference.Out("TENSORS").Cast<std::vector<Tensor>>();

    // Take output tensor with blendshapes and wrap it in vector.
    auto blendshapes_tensor = GetTensorWithBlendshapes(tensors_out, graph);

    // Convert tensor to ClassificationList.
    auto face_blendshapes =
        ConvertTensorToBlendshapes(blendshapes_tensor, kBlendshapeNames, graph);

    return FaceBlendshapesOuts{face_blendshapes};
  }
};

REGISTER_MEDIAPIPE_GRAPH(
    ::mediapipe::tasks::vision::face_landmarker::FaceBlendshapesGraph);

}  // namespace face_landmarker
}  // namespace vision
}  // namespace tasks
}  // namespace mediapipe
