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

#include <array>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "mediapipe/calculators/core/split_vector_calculator.pb.h"
#include "mediapipe/calculators/tensor/tensors_to_landmarks_calculator.pb.h"
#include "mediapipe/calculators/util/landmarks_refinement_calculator.pb.h"
#include "mediapipe/framework/api2/builder.h"
#include "mediapipe/framework/calculator.pb.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/landmark.pb.h"
#include "mediapipe/framework/formats/tensor.h"
#include "mediapipe/framework/subgraph.h"
#include "mediapipe/tasks/cc/vision/face_landmarker/proto/tensors_to_face_landmarks_graph_options.pb.h"

namespace mediapipe {
namespace tasks {
namespace vision {
namespace face_landmarker {
namespace {

using ::mediapipe::api2::builder::Graph;
using ::mediapipe::api2::builder::SidePacket;
using ::mediapipe::api2::builder::Stream;

constexpr char kTensorsTag[] = "TENSORS";
constexpr char kNormLandmarksTag[] = "NORM_LANDMARKS";
constexpr char kLandmarksTag[] = "LANDMARKS";
constexpr char kRefinedLandmarksTag[] = "REFINED_LANDMARKS";
constexpr int kMeshLandmarksNum = 468;
constexpr int kMeshWithIrisLandmarksNum = 478;
constexpr int kLipsLandmarksNum = 80;
constexpr int kEyeLandmarksNum = 71;
constexpr int kIrisLandmarksNum = 5;
constexpr int kContoursNumForIrisAvg = 16;

// TODO When model metadata for face detector is ready, move the
// index mapping to metadata.
constexpr std::array<int, kMeshLandmarksNum> kMeshLandmarksIndicesMapping{
    0,   1,   2,   3,   4,   5,   6,   7,   8,   9,   10,  11,  12,  13,  14,
    15,  16,  17,  18,  19,  20,  21,  22,  23,  24,  25,  26,  27,  28,  29,
    30,  31,  32,  33,  34,  35,  36,  37,  38,  39,  40,  41,  42,  43,  44,
    45,  46,  47,  48,  49,  50,  51,  52,  53,  54,  55,  56,  57,  58,  59,
    60,  61,  62,  63,  64,  65,  66,  67,  68,  69,  70,  71,  72,  73,  74,
    75,  76,  77,  78,  79,  80,  81,  82,  83,  84,  85,  86,  87,  88,  89,
    90,  91,  92,  93,  94,  95,  96,  97,  98,  99,  100, 101, 102, 103, 104,
    105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119,
    120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134,
    135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149,
    150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164,
    165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179,
    180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194,
    195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209,
    210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224,
    225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239,
    240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254,
    255, 256, 257, 258, 259, 260, 261, 262, 263, 264, 265, 266, 267, 268, 269,
    270, 271, 272, 273, 274, 275, 276, 277, 278, 279, 280, 281, 282, 283, 284,
    285, 286, 287, 288, 289, 290, 291, 292, 293, 294, 295, 296, 297, 298, 299,
    300, 301, 302, 303, 304, 305, 306, 307, 308, 309, 310, 311, 312, 313, 314,
    315, 316, 317, 318, 319, 320, 321, 322, 323, 324, 325, 326, 327, 328, 329,
    330, 331, 332, 333, 334, 335, 336, 337, 338, 339, 340, 341, 342, 343, 344,
    345, 346, 347, 348, 349, 350, 351, 352, 353, 354, 355, 356, 357, 358, 359,
    360, 361, 362, 363, 364, 365, 366, 367, 368, 369, 370, 371, 372, 373, 374,
    375, 376, 377, 378, 379, 380, 381, 382, 383, 384, 385, 386, 387, 388, 389,
    390, 391, 392, 393, 394, 395, 396, 397, 398, 399, 400, 401, 402, 403, 404,
    405, 406, 407, 408, 409, 410, 411, 412, 413, 414, 415, 416, 417, 418, 419,
    420, 421, 422, 423, 424, 425, 426, 427, 428, 429, 430, 431, 432, 433, 434,
    435, 436, 437, 438, 439, 440, 441, 442, 443, 444, 445, 446, 447, 448, 449,
    450, 451, 452, 453, 454, 455, 456, 457, 458, 459, 460, 461, 462, 463, 464,
    465, 466, 467};

constexpr std::array<int, kLipsLandmarksNum> kLipsLandmarksIndicesMapping{
    // Lower outer.
    61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291,
    // Upper outer (excluding corners).
    185, 40, 39, 37, 0, 267, 269, 270, 409,
    // Lower inner.
    78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308,
    // Upper inner (excluding corners).
    191, 80, 81, 82, 13, 312, 311, 310, 415,
    // Lower semi-outer.
    76, 77, 90, 180, 85, 16, 315, 404, 320, 307, 306,
    // Upper semi-outer (excluding corners).
    184, 74, 73, 72, 11, 302, 303, 304, 408,
    // Lower semi-inner.
    62, 96, 89, 179, 86, 15, 316, 403, 319, 325, 292,
    // Upper semi-inner (excluding corners).
    183, 42, 41, 38, 12, 268, 271, 272, 407};

constexpr std::array<int, kEyeLandmarksNum> kLeftEyeLandmarksIndicesMapping{
    // Lower contour.
    33, 7, 163, 144, 145, 153, 154, 155, 133,
    // upper contour (excluding corners).
    246, 161, 160, 159, 158, 157, 173,
    // Halo x2 lower contour.
    130, 25, 110, 24, 23, 22, 26, 112, 243,
    // Halo x2 upper contour (excluding corners).
    247, 30, 29, 27, 28, 56, 190,
    // Halo x3 lower contour.
    226, 31, 228, 229, 230, 231, 232, 233, 244,
    // Halo x3 upper contour (excluding corners).
    113, 225, 224, 223, 222, 221, 189,
    // Halo x4 upper contour (no lower because of mesh structure) or
    // eyebrow inner contour.
    35, 124, 46, 53, 52, 65,
    // Halo x5 lower contour.
    143, 111, 117, 118, 119, 120, 121, 128, 245,
    // Halo x5 upper contour (excluding corners) or eyebrow outer contour.
    156, 70, 63, 105, 66, 107, 55, 193};

constexpr std::array<int, kEyeLandmarksNum> kRightEyeLandmarksIndicesMapping{
    // Lower contour.
    263, 249, 390, 373, 374, 380, 381, 382, 362,
    // Upper contour (excluding corners).
    466, 388, 387, 386, 385, 384, 398,
    // Halo x2 lower contour.
    359, 255, 339, 254, 253, 252, 256, 341, 463,
    // Halo x2 upper contour (excluding corners).
    467, 260, 259, 257, 258, 286, 414,
    // Halo x3 lower contour.
    446, 261, 448, 449, 450, 451, 452, 453, 464,
    // Halo x3 upper contour (excluding corners).
    342, 445, 444, 443, 442, 441, 413,
    // Halo x4 upper contour (no lower because of mesh structure) or
    // eyebrow inner contour.
    265, 353, 276, 283, 282, 295,
    // Halo x5 lower contour.
    372, 340, 346, 347, 348, 349, 350, 357, 465,
    // Halo x5 upper contour (excluding corners) or eyebrow outer contour.
    383, 300, 293, 334, 296, 336, 285, 417};

constexpr std::array<int, kIrisLandmarksNum> kLeftIrisLandmarksIndicesMapping{
    // Center.
    468,
    // Iris right edge.
    469,
    // Iris top edge.
    470,
    // Iris left edge.
    471,
    // Iris bottom edge.
    472};

constexpr std::array<int, kContoursNumForIrisAvg> kLeftIrisAvgIndices = {
    // Lower contour.
    33, 7, 163, 144, 145, 153, 154, 155, 133,
    // Upper contour (excluding corners).
    246, 161, 160, 159, 158, 157, 173};

constexpr std::array<int, kIrisLandmarksNum> kRightIrisLandmarksIndicesMapping{
    // Center.
    473,
    // Iris right edge.
    474,
    // Iris top edge.
    475,
    // Iris left edge.
    476,
    // Iris bottom edge.
    477};

constexpr std::array<int, kContoursNumForIrisAvg> kRightIrisAvgIndices = {
    // Lower contour.
    263, 249, 390, 373, 374, 380, 381, 382, 362,
    // Upper contour (excluding corners).
    466, 388, 387, 386, 385, 384, 398};

Stream<NormalizedLandmarkList> ConvertTensorsToLandmarks(
    int landmarks_num, int input_image_width, int input_image_height,
    Stream<std::vector<Tensor>> tensors, Graph& graph) {
  auto& tensors_to_landmarks = graph.AddNode("TensorsToLandmarksCalculator");
  auto* options =
      &tensors_to_landmarks
           .GetOptions<mediapipe::TensorsToLandmarksCalculatorOptions>();
  options->set_num_landmarks(landmarks_num);
  options->set_input_image_width(input_image_width);
  options->set_input_image_height(input_image_height);
  tensors >> tensors_to_landmarks.In(kTensorsTag);
  return tensors_to_landmarks.Out(kNormLandmarksTag)
      .Cast<NormalizedLandmarkList>();
}

}  // namespace

// Graph to transform face landmarks model output tensors into landmarks.
//
// Inputs:
//   TENSORS - std::vector<Tensor>
//     Landmarks tensors to be transformed. If regular model, a vector of single
//     Tensor is expected. If a model with attention, a vector of 6 Tensors is
//     expected.
//
// Outputs:
//   NORM_LANDMARKS: - NormalizedLandmarkList
//     Transformed face landmarks.
//
// Example:
// node {
//   calculator:
//   "mediapipe.tasks.vision.face_landmarker.TensorsToFaceLandmarksGraph"
//   input_stream: "TENSORS:tensors"
//   output_stream: "NORM_LANDMARKS:norm_landmarks"
//   options {
//     [mediapipe.tasks.vision.face_landmarker.proto.TensorsToFaceLandmarksGraphOptions.ext]
//     {
//        input_image_width: 192
//        input_image_height: 192
//     }
//   }
// }
class TensorsToFaceLandmarksGraph : public Subgraph {
 public:
  absl::StatusOr<CalculatorGraphConfig> GetConfig(
      SubgraphContext* sc) override {
    Graph graph;
    auto norm_landmarks = BuildTensorsToFaceLandmarksGraph(
        sc->Options<proto::TensorsToFaceLandmarksGraphOptions>(),
        graph.In(kTensorsTag).Cast<std::vector<Tensor>>(), graph);
    norm_landmarks >>
        graph.Out(kNormLandmarksTag).Cast<NormalizedLandmarkList>();
    return graph.GetConfig();
  }

 private:
  Stream<NormalizedLandmarkList> BuildTensorsToFaceLandmarksGraph(
      const proto::TensorsToFaceLandmarksGraphOptions& subgraph_options,
      Stream<std::vector<Tensor>> tensors, Graph& graph) {
    const int input_image_width = subgraph_options.input_image_width();
    const int input_image_height = subgraph_options.input_image_height();
    return ConvertTensorsToLandmarks(kMeshWithIrisLandmarksNum,
                                     input_image_width, input_image_height,
                                     tensors, graph);
  }
};

// clang-format off
REGISTER_MEDIAPIPE_GRAPH(
  ::mediapipe::tasks::vision::face_landmarker::TensorsToFaceLandmarksGraph); // NOLINT
// clang-format on

}  // namespace face_landmarker
}  // namespace vision
}  // namespace tasks
}  // namespace mediapipe
