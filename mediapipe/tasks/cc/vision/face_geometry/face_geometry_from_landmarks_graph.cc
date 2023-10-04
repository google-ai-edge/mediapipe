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

#include <optional>
#include <string>
#include <vector>

#include "absl/status/statusor.h"
#include "mediapipe/calculators/core/split_vector_calculator.pb.h"
#include "mediapipe/framework/api2/builder.h"
#include "mediapipe/framework/calculator.pb.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/landmark.pb.h"
#include "mediapipe/framework/subgraph.h"
#include "mediapipe/tasks/cc/vision/face_geometry/calculators/env_generator_calculator.pb.h"
#include "mediapipe/tasks/cc/vision/face_geometry/calculators/geometry_pipeline_calculator.pb.h"
#include "mediapipe/tasks/cc/vision/face_geometry/proto/environment.pb.h"
#include "mediapipe/tasks/cc/vision/face_geometry/proto/face_geometry.pb.h"
#include "mediapipe/tasks/cc/vision/face_geometry/proto/face_geometry_graph_options.pb.h"
#include "mediapipe/util/graph_builder_utils.h"

namespace mediapipe::tasks::vision::face_geometry {
namespace {

using ::mediapipe::api2::builder::Graph;
using ::mediapipe::api2::builder::SidePacket;
using ::mediapipe::api2::builder::Stream;
using ::mediapipe::tasks::vision::face_geometry::proto::Environment;
using ::mediapipe::tasks::vision::face_geometry::proto::FaceGeometry;

constexpr char kMultiFaceLandmarksTag[] = "MULTI_FACE_LANDMARKS";
constexpr char kMultiFaceGeometryTag[] = "MULTI_FACE_GEOMETRY";
constexpr char kFaceLandmarksTag[] = "FACE_LANDMARKS";
constexpr char kFaceGeometryTag[] = "FACE_GEOMETRY";
constexpr char kImageSizeTag[] = "IMAGE_SIZE";
constexpr char kEnvironmentTag[] = "ENVIRONMENT";
constexpr char kIterableTag[] = "ITERABLE";
constexpr char kBatchEndTag[] = "BATCH_END";
constexpr char kItemTag[] = "ITEM";

struct FaceGeometryOuts {
  Stream<std::vector<FaceGeometry>> multi_face_geometry;
};

void ConfigureSplitNormalizedLandmarkListCalculator(
    mediapipe::SplitVectorCalculatorOptions& options) {
  auto& range = *options.add_ranges();
  // Extract the first 468 face landmarks, excluding iris;
  range.set_begin(0);
  range.set_end(468);
}

void ConfigureFaceGeometryEnvGeneratorCalculator(
    FaceGeometryEnvGeneratorCalculatorOptions& options) {
  options.mutable_environment()->set_origin_point_location(
      proto::OriginPointLocation::TOP_LEFT_CORNER);
  auto& perspective_camera =
      *options.mutable_environment()->mutable_perspective_camera();
  perspective_camera.set_vertical_fov_degrees(63.0 /*degrees*/);
  perspective_camera.set_near(1.0 /* 1cm */);
  perspective_camera.set_far(10000.0 /* 100m */);
}
}  // namespace

// A "mediapipe.tasks.vision.face_landmarker.FaceGeometryFromLandmarksGraph"
// graph to extract 3D transform from the given canonical face to multi face
// landmarks.
//
// It is required that "geometry_pipeline_metadata_from_landmark.binarypb" is
// available at
// "mediapipe/tasks/cc/vision/face_geometry/data/geometry_pipeline_metadata_from_landmarks.binarypb"
// path during execution.
//
//
// Inputs:
//   IMAGE_SIZE - std::pair<int,int>
//     The size of the image that face landmarks are detected on.
//   FACE_LANDMARKS - std::vector<NormalizedLandmarkList>
//     A vector of multiple face landmarks that the given canonical face would
//     transform to.
//
//  SideInputs:
//   ENVIRONMENT - ENVIRONMENT
//     Environment that describes the current virtual scene. If not provided, a
//     default environment will be used which is good enough for most general
//     use cases.
//
//
// Outputs:
//   FACE_GEOMETRY: - std::vector<FaceGeometry>
//    A vector of 3D transform data for each detected face.
//
//
// Example:
// node {
//   calculator:
//   "mediapipe.tasks.vision.face_landmarker.FaceGeometryFromLandmarksGraph"
//   input_stream: "IMAGE_SIZE:image_size"
//   input_stream: "FACE_LANDMARKS:face_landmarks"
//   input_side_packet: "ENVIRONMENT:environment"
//   output_stream: "FACE_GEOMETRY:face_geometry"
// }
class FaceGeometryFromLandmarksGraph : public Subgraph {
 public:
  absl::StatusOr<CalculatorGraphConfig> GetConfig(
      SubgraphContext* sc) override {
    Graph graph;
    std::optional<SidePacket<Environment>> environment;
    if (HasSideInput(sc->OriginalNode(), kEnvironmentTag)) {
      environment = std::make_optional<>(
          graph.SideIn(kEnvironmentTag).Cast<Environment>());
    }
    MP_ASSIGN_OR_RETURN(
        auto outs, BuildFaceGeometryFromLandmarksGraph(
                       *sc->MutableOptions<proto::FaceGeometryGraphOptions>(),
                       graph.In(kFaceLandmarksTag)
                           .Cast<std::vector<NormalizedLandmarkList>>(),
                       graph.In(kImageSizeTag).Cast<std::pair<int, int>>(),
                       environment, graph));
    outs.multi_face_geometry >>
        graph.Out(kFaceGeometryTag).Cast<std::vector<FaceGeometry>>();
    return graph.GetConfig();
  }

 private:
  absl::StatusOr<FaceGeometryOuts> BuildFaceGeometryFromLandmarksGraph(
      proto::FaceGeometryGraphOptions& graph_options,
      Stream<std::vector<NormalizedLandmarkList>> multi_face_landmarks,
      Stream<std::pair<int, int>> image_size,
      std::optional<SidePacket<Environment>> environment, Graph& graph) {
    if (!environment.has_value()) {
      // If there is no provided Environment, use a a default environment which
      // is good enough for most general use cases.
      auto& env_generator = graph.AddNode(
          "mediapipe.tasks.vision.face_geometry."
          "FaceGeometryEnvGeneratorCalculator");
      ConfigureFaceGeometryEnvGeneratorCalculator(
          env_generator
              .GetOptions<FaceGeometryEnvGeneratorCalculatorOptions>());
      environment = std::make_optional<>(
          env_generator.SideOut(kEnvironmentTag).Cast<Environment>());
    }

    // For loop to go over the vector of face landmarks list, and remove the
    // iris landmarks.
    auto& begin_loop_landmark_list_vector =
        graph.AddNode("BeginLoopNormalizedLandmarkListVectorCalculator");
    multi_face_landmarks >> begin_loop_landmark_list_vector.In(kIterableTag);
    auto batch_end = begin_loop_landmark_list_vector.Out(kBatchEndTag);
    auto single_face_landmarks = begin_loop_landmark_list_vector.Out(kItemTag);

    // Take first 468 face landmarks and exclude iris landmarks.
    auto& split_landmark_list =
        graph.AddNode("SplitNormalizedLandmarkListCalculator");
    ConfigureSplitNormalizedLandmarkListCalculator(
        split_landmark_list
            .GetOptions<mediapipe::SplitVectorCalculatorOptions>());
    single_face_landmarks >> split_landmark_list.In("");
    auto single_face_landmarks_no_iris = split_landmark_list.Out("");

    auto& end_loop_landmark_list_vector =
        graph.AddNode("EndLoopNormalizedLandmarkListVectorCalculator");
    batch_end >> end_loop_landmark_list_vector.In(kBatchEndTag);
    single_face_landmarks_no_iris >> end_loop_landmark_list_vector.In(kItemTag);
    auto multi_face_landmarks_no_iris =
        end_loop_landmark_list_vector.Out(kIterableTag)
            .Cast<std::vector<NormalizedLandmarkList>>();

    // Find the transformation from the canonical face to the list of multi face
    // landmarks.
    auto& geometry_pipeline = graph.AddNode(
        "mediapipe.tasks.vision.face_geometry.FaceGeometryPipelineCalculator");
    auto& geometry_pipeline_options =
        geometry_pipeline.GetOptions<FaceGeometryPipelineCalculatorOptions>();
    geometry_pipeline_options.Swap(
        graph_options.mutable_geometry_pipeline_options());
    image_size >> geometry_pipeline.In(kImageSizeTag);
    multi_face_landmarks_no_iris >>
        geometry_pipeline.In(kMultiFaceLandmarksTag);
    environment.value() >> geometry_pipeline.SideIn(kEnvironmentTag);
    auto multi_face_geometry = geometry_pipeline.Out(kMultiFaceGeometryTag)
                                   .Cast<std::vector<FaceGeometry>>();

    return {{/*multi_face_geometry */ multi_face_geometry}};
  }
};

// clang-format off
REGISTER_MEDIAPIPE_GRAPH(
  ::mediapipe::tasks::vision::face_geometry::FaceGeometryFromLandmarksGraph); // NOLINT
// clang-format on

}  // namespace mediapipe::tasks::vision::face_geometry
