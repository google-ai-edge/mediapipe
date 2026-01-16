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

#include <memory>
#include <type_traits>
#include <utility>
#include <vector>

#include "absl/log/absl_log.h"
#include "absl/strings/str_format.h"
#include "mediapipe/calculators/core/clip_vector_size_calculator.pb.h"
#include "mediapipe/calculators/core/concatenate_vector_calculator.h"
#include "mediapipe/calculators/core/gate_calculator.pb.h"
#include "mediapipe/calculators/core/get_vector_item_calculator.h"
#include "mediapipe/calculators/core/get_vector_item_calculator.pb.h"
#include "mediapipe/calculators/util/association_calculator.pb.h"
#include "mediapipe/calculators/util/collection_has_min_size_calculator.pb.h"
#include "mediapipe/framework/api2/builder.h"
#include "mediapipe/framework/api2/port.h"
#include "mediapipe/framework/formats/classification.pb.h"
#include "mediapipe/framework/formats/detection.pb.h"
#include "mediapipe/framework/formats/image.h"
#include "mediapipe/framework/formats/landmark.pb.h"
#include "mediapipe/framework/formats/rect.pb.h"
#include "mediapipe/framework/formats/tensor.h"
#include "mediapipe/framework/port/status_macros.h"
#include "mediapipe/tasks/cc/common.h"
#include "mediapipe/tasks/cc/components/utils/gate.h"
#include "mediapipe/tasks/cc/core/model_asset_bundle_resources.h"
#include "mediapipe/tasks/cc/core/model_resources_cache.h"
#include "mediapipe/tasks/cc/core/model_task_graph.h"
#include "mediapipe/tasks/cc/core/utils.h"
#include "mediapipe/tasks/cc/metadata/utils/zip_utils.h"
#include "mediapipe/tasks/cc/vision/face_detector/proto/face_detector_graph_options.pb.h"
#include "mediapipe/tasks/cc/vision/face_geometry/calculators/geometry_pipeline_calculator.pb.h"
#include "mediapipe/tasks/cc/vision/face_geometry/proto/environment.pb.h"
#include "mediapipe/tasks/cc/vision/face_geometry/proto/face_geometry.pb.h"
#include "mediapipe/tasks/cc/vision/face_geometry/proto/face_geometry_graph_options.pb.h"
#include "mediapipe/tasks/cc/vision/face_landmarker/proto/face_blendshapes_graph_options.pb.h"
#include "mediapipe/tasks/cc/vision/face_landmarker/proto/face_landmarker_graph_options.pb.h"
#include "mediapipe/tasks/cc/vision/face_landmarker/proto/face_landmarks_detector_graph_options.pb.h"
#include "mediapipe/util/graph_builder_utils.h"

namespace mediapipe {
namespace tasks {
namespace vision {
namespace face_landmarker {

namespace {

using ::mediapipe::NormalizedRect;
using ::mediapipe::api2::Input;
using ::mediapipe::api2::Output;
using ::mediapipe::api2::builder::Graph;
using ::mediapipe::api2::builder::SidePacket;
using ::mediapipe::api2::builder::Source;
using ::mediapipe::tasks::components::utils::DisallowIf;
using ::mediapipe::tasks::core::ModelAssetBundleResources;
using ::mediapipe::tasks::metadata::SetExternalFile;
using ::mediapipe::tasks::vision::face_detector::proto::
    FaceDetectorGraphOptions;
using ::mediapipe::tasks::vision::face_geometry::proto::Environment;
using ::mediapipe::tasks::vision::face_geometry::proto::FaceGeometry;
using ::mediapipe::tasks::vision::face_landmarker::proto::
    FaceLandmarkerGraphOptions;
using ::mediapipe::tasks::vision::face_landmarker::proto::
    FaceLandmarksDetectorGraphOptions;

constexpr char kImageTag[] = "IMAGE";
constexpr char kNormRectTag[] = "NORM_RECT";
constexpr char kNormLandmarksTag[] = "NORM_LANDMARKS";
constexpr char kFaceRectsTag[] = "FACE_RECTS";
constexpr char kFaceRectsNextFrameTag[] = "FACE_RECTS_NEXT_FRAME";
constexpr char kExpandedFaceRectsTag[] = "EXPANDED_FACE_RECTS";
constexpr char kDetectionsTag[] = "DETECTIONS";
constexpr char kLoopTag[] = "LOOP";
constexpr char kPrevLoopTag[] = "PREV_LOOP";
constexpr char kMainTag[] = "MAIN";
constexpr char kIterableTag[] = "ITERABLE";
constexpr char kFaceLandmarksTag[] = "FACE_LANDMARKS";
constexpr char kFaceGeometryTag[] = "FACE_GEOMETRY";
constexpr char kEnvironmentTag[] = "ENVIRONMENT";
constexpr char kBlendshapesTag[] = "BLENDSHAPES";
constexpr char kImageSizeTag[] = "IMAGE_SIZE";
constexpr char kSizeTag[] = "SIZE";
constexpr char kVectorTag[] = "VECTOR";
constexpr char kItemTag[] = "ITEM";
constexpr char kNormFilteredLandmarksTag[] = "NORM_FILTERED_LANDMARKS";
constexpr char kFaceDetectorTFLiteName[] = "face_detector.tflite";
constexpr char kFaceLandmarksDetectorTFLiteName[] =
    "face_landmarks_detector.tflite";
constexpr char kFaceBlendshapeTFLiteName[] = "face_blendshapes.tflite";
constexpr char kFaceGeometryPipelineMetadataName[] =
    "geometry_pipeline_metadata_landmarks.binarypb";

struct FaceLandmarkerOutputs {
  Source<std::vector<NormalizedLandmarkList>> landmark_lists;
  Source<std::vector<NormalizedRect>> face_rects_next_frame;
  Source<std::vector<NormalizedRect>> face_rects;
  Source<std::vector<Detection>> detections;
  std::optional<Source<std::vector<ClassificationList>>> face_blendshapes;
  std::optional<Source<std::vector<FaceGeometry>>> face_geometry;
  Source<Image> image;
};

// Sets the base options in the sub tasks.
absl::Status SetSubTaskBaseOptions(const ModelAssetBundleResources& resources,
                                   FaceLandmarkerGraphOptions* options,
                                   bool is_copy) {
  auto* face_detector_graph_options =
      options->mutable_face_detector_graph_options();
  if (!face_detector_graph_options->base_options().has_model_asset()) {
    MP_ASSIGN_OR_RETURN(const auto face_detector_file,
                        resources.GetFile(kFaceDetectorTFLiteName));
    SetExternalFile(face_detector_file,
                    face_detector_graph_options->mutable_base_options()
                        ->mutable_model_asset(),
                    is_copy);
  }
  face_detector_graph_options->mutable_base_options()
      ->mutable_acceleration()
      ->CopyFrom(options->base_options().acceleration());
  face_detector_graph_options->mutable_base_options()->set_use_stream_mode(
      options->base_options().use_stream_mode());
  face_detector_graph_options->mutable_base_options()->set_gpu_origin(
      options->base_options().gpu_origin());

  auto* face_landmarks_detector_graph_options =
      options->mutable_face_landmarks_detector_graph_options();
  if (!face_landmarks_detector_graph_options->base_options()
           .has_model_asset()) {
    MP_ASSIGN_OR_RETURN(const auto face_landmarks_detector_file,
                        resources.GetFile(kFaceLandmarksDetectorTFLiteName));
    SetExternalFile(
        face_landmarks_detector_file,
        face_landmarks_detector_graph_options->mutable_base_options()
            ->mutable_model_asset(),
        is_copy);
  }
  face_landmarks_detector_graph_options->mutable_base_options()
      ->mutable_acceleration()
      ->CopyFrom(options->base_options().acceleration());
  face_landmarks_detector_graph_options->mutable_base_options()
      ->set_use_stream_mode(options->base_options().use_stream_mode());
  face_landmarks_detector_graph_options->mutable_base_options()->set_gpu_origin(
      options->base_options().gpu_origin());

  absl::StatusOr<absl::string_view> face_blendshape_model =
      resources.GetFile(kFaceBlendshapeTFLiteName);
  if (face_blendshape_model.ok()) {
    SetExternalFile(*face_blendshape_model,
                    face_landmarks_detector_graph_options
                        ->mutable_face_blendshapes_graph_options()
                        ->mutable_base_options()
                        ->mutable_model_asset(),
                    is_copy);
    face_landmarks_detector_graph_options
        ->mutable_face_blendshapes_graph_options()
        ->mutable_base_options()
        ->mutable_acceleration()
        ->mutable_xnnpack();
    ABSL_LOG(WARNING) << "Sets FaceBlendshapesGraph acceleration to xnnpack "
                      << "by default.";
  }

  return absl::OkStatus();
}
}  // namespace

// A "mediapipe.tasks.vision.face_landmarker.FaceLandmarkerGraph" performs face
// landmarks detection. The FaceLandmarkerGraph consists of three subgraphs:
// FaceDetectorGraph, MultipleFaceLandmarksDetectorGraph and
// FaceGeometryFromLandmarksGraph.
//
// MultipleFaceLandmarksDetectorGraph detects landmarks from bounding boxes
// produced by FaceDetectorGraph. FaceLandmarkerGraph tracks the landmarks over
// time, and skips the FaceDetectorGraph. If the tracking is lost or the
// detected faces are less than configured max number faces, FaceDetectorGraph
// would be triggered to detect faces.
//
// FaceGeometryFromLandmarksGraph finds the transformation from canonical face
// to the detected faces. This transformation is useful for rendering face
// effects on the detected faces. This subgraph is added if users request a
// FaceGeometry Tag.
//
//
// Inputs:
//   IMAGE - Image
//     Image to perform face landmarks detection on.
//   NORM_RECT - NormalizedRect @Optional
//     Describes image rotation and region of image to perform landmarks
//     detection on. If not provided, whole image is used for face landmarks
//     detection.
//
//  SideInputs:
//   ENVIRONMENT - ENVIRONMENT @optional
//     Environment that describes the current virtual scene. If not provided, a
//     default environment will be used which is good enough for most general
//     use case
//
// Outputs:
//   NORM_LANDMARKS: - std::vector<NormalizedLandmarkList>
//     Vector of detected face landmarks.
//   BLENDSHAPES: - std::vector<ClassificationList> @optional
//     Blendshape classification, available when the given model asset contains
//     blendshapes model.
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
//   FACE_GEOMETRY - std::vector<FaceGeometry> @optional
//     A vector of 3D transform data for each detected face.
//   FACE_RECTS_NEXT_FRAME - std::vector<NormalizedRect>
//     Vector of the expanded rects enclosing the whole face RoI for landmark
//     detection on the next frame.
//   FACE_RECTS - std::vector<NormalizedRect>
//     Detected face bounding boxes in normalized coordinates from face
//     detection.
//   DETECTIONS - std::vector<Detection>
//     Detected faces with maximum `num_faces` specified in options.
//   IMAGE - Image
//     The input image that the face landmarker runs on and has the pixel data
//     stored on the target storage (CPU vs GPU).
// All returned coordinates are in the unrotated and uncropped input image
// coordinates system.
//
// Example:
// node {
//   calculator: "mediapipe.tasks.vision.face_landmarker.FaceLandmarkerGraph"
//   input_stream: "IMAGE:image_in"
//   input_stream: "NORM_RECT:norm_rect"
//   output_stream: "NORM_LANDMARKS:face_landmarks"
//   output_stream: "BLENDSHAPES:face_blendshapes"
//   output_stream: "FACE_GEOMETRY:face_geometry"
//   output_stream: "FACE_RECTS_NEXT_FRAME:face_rects_next_frame"
//   output_stream: "FACE_RECTS:face_rects"
//   output_stream: "DETECTIONS:detections"
//   output_stream: "IMAGE:image_out"
//   options {
//     [mediapipe.tasks.vision.face_landmarker.proto.FaceLandmarkerGraphOptions.ext]
//     {
//       base_options {
//          model_asset {
//            file_name: "face_landmarker.task"
//          }
//       }
//       face_detector_graph_options {
//         min_detection_confidence: 0.5
//         num_faces: 2
//       }
//       face_landmarks_detector_graph_options {
//         min_detection_confidence: 0.5
//       }
//     }
//   }
// }
class FaceLandmarkerGraph : public core::ModelTaskGraph {
 public:
  absl::StatusOr<CalculatorGraphConfig> GetConfig(
      SubgraphContext* sc) override {
    Graph graph;
    bool output_geometry = HasOutput(sc->OriginalNode(), kFaceGeometryTag);
    if (sc->Options<FaceLandmarkerGraphOptions>()
            .base_options()
            .has_model_asset()) {
      MP_ASSIGN_OR_RETURN(
          const auto* model_asset_bundle_resources,
          CreateModelAssetBundleResources<FaceLandmarkerGraphOptions>(sc));
      // Copies the file content instead of passing the pointer of file in
      // memory if the subgraph model resource service is not available.
      MP_RETURN_IF_ERROR(SetSubTaskBaseOptions(
          *model_asset_bundle_resources,
          sc->MutableOptions<FaceLandmarkerGraphOptions>(),
          !sc->Service(::mediapipe::tasks::core::kModelResourcesCacheService)
               .IsAvailable()));
      if (output_geometry) {
        // Set the face geometry metadata file for
        // FaceGeometryFromLandmarksGraph.
        MP_ASSIGN_OR_RETURN(auto face_geometry_pipeline_metadata_file,
                            model_asset_bundle_resources->GetFile(
                                kFaceGeometryPipelineMetadataName));
        SetExternalFile(face_geometry_pipeline_metadata_file,
                        sc->MutableOptions<FaceLandmarkerGraphOptions>()
                            ->mutable_face_geometry_graph_options()
                            ->mutable_geometry_pipeline_options()
                            ->mutable_metadata_file());
      }
    }
    std::optional<SidePacket<Environment>> environment;
    if (HasSideInput(sc->OriginalNode(), kEnvironmentTag)) {
      environment = std::make_optional<>(
          graph.SideIn(kEnvironmentTag).Cast<Environment>());
    }
    bool output_blendshapes = HasOutput(sc->OriginalNode(), kBlendshapesTag);
    if (output_blendshapes && !sc->Options<FaceLandmarkerGraphOptions>()
                                   .face_landmarks_detector_graph_options()
                                   .has_face_blendshapes_graph_options()) {
      return absl::InvalidArgumentError(absl::StrFormat(
          "BLENDSHAPES Tag and blendshapes model must be both set. Get "
          "BLENDSHAPES is set: %v, blendshapes "
          "model "
          "is set: %v",
          output_blendshapes,
          sc->Options<FaceLandmarkerGraphOptions>()
              .face_landmarks_detector_graph_options()
              .has_face_blendshapes_graph_options()));
    }
    std::optional<Source<NormalizedRect>> norm_rect_in;
    if (HasInput(sc->OriginalNode(), kNormRectTag)) {
      norm_rect_in = graph.In(kNormRectTag).Cast<NormalizedRect>();
    }
    MP_ASSIGN_OR_RETURN(
        auto outs,
        BuildFaceLandmarkerGraph(
            *sc->MutableOptions<FaceLandmarkerGraphOptions>(),
            graph[Input<Image>(kImageTag)], norm_rect_in, environment,
            output_blendshapes, output_geometry, graph));
    outs.landmark_lists >>
        graph[Output<std::vector<NormalizedLandmarkList>>(kNormLandmarksTag)];
    outs.face_rects_next_frame >>
        graph[Output<std::vector<NormalizedRect>>(kFaceRectsNextFrameTag)];
    outs.face_rects >>
        graph[Output<std::vector<NormalizedRect>>(kFaceRectsTag)];
    outs.detections >> graph[Output<std::vector<Detection>>(kDetectionsTag)];
    outs.image >> graph[Output<Image>(kImageTag)];
    if (outs.face_blendshapes) {
      *outs.face_blendshapes >>
          graph[Output<std::vector<ClassificationList>>(kBlendshapesTag)];
    }
    if (outs.face_geometry) {
      *outs.face_geometry >>
          graph[Output<std::vector<FaceGeometry>>(kFaceGeometryTag)];
    }

    CalculatorGraphConfig config = graph.GetConfig();
    core::FixGraphBackEdges(config);
    return config;
  }

 private:
  // Adds a mediapipe face landmarker graph into the provided builder::Graph
  // instance.
  //
  // tasks_options: the mediapipe tasks module FaceLandmarkerGraphOptions.
  // image_in: (mediapipe::Image) stream to run face landmark detection on.
  // graph: the mediapipe graph instance to be updated.
  absl::StatusOr<FaceLandmarkerOutputs> BuildFaceLandmarkerGraph(
      FaceLandmarkerGraphOptions& tasks_options, Source<Image> image_in,
      std::optional<Source<NormalizedRect>> norm_rect_in,
      std::optional<SidePacket<Environment>> environment,
      bool output_blendshapes, bool output_geometry, Graph& graph) {
    const int max_num_faces =
        tasks_options.face_detector_graph_options().num_faces();

    auto& face_detector =
        graph.AddNode("mediapipe.tasks.vision.face_detector.FaceDetectorGraph");
    face_detector.GetOptions<FaceDetectorGraphOptions>().Swap(
        tasks_options.mutable_face_detector_graph_options());
    const auto& face_detector_options =
        face_detector.GetOptions<FaceDetectorGraphOptions>();
    auto& clip_face_rects =
        graph.AddNode("ClipNormalizedRectVectorSizeCalculator");
    clip_face_rects.GetOptions<ClipVectorSizeCalculatorOptions>()
        .set_max_vec_size(max_num_faces);
    auto clipped_face_rects = clip_face_rects.Out("");

    auto& face_landmarks_detector_graph = graph.AddNode(
        "mediapipe.tasks.vision.face_landmarker."
        "MultiFaceLandmarksDetectorGraph");
    face_landmarks_detector_graph
        .GetOptions<FaceLandmarksDetectorGraphOptions>()
        .Swap(tasks_options.mutable_face_landmarks_detector_graph_options());
    image_in >> face_landmarks_detector_graph.In(kImageTag);
    clipped_face_rects >> face_landmarks_detector_graph.In(kNormRectTag);

    Source<std::vector<NormalizedLandmarkList>> face_landmarks =
        face_landmarks_detector_graph.Out(kNormLandmarksTag)
            .Cast<std::vector<NormalizedLandmarkList>>();
    auto face_rects_for_next_frame =
        face_landmarks_detector_graph.Out(kFaceRectsNextFrameTag)
            .Cast<std::vector<NormalizedRect>>();

    auto& image_properties = graph.AddNode("ImagePropertiesCalculator");
    image_in >> image_properties.In(kImageTag);
    auto image_size = image_properties.Out(kSizeTag);

    // Apply smoothing filter only on the single face landmarks, because
    // landmarks smoothing calculator doesn't support multiple landmarks yet.
    if (face_detector_options.num_faces() == 1) {
      face_landmarks_detector_graph
          .GetOptions<FaceLandmarksDetectorGraphOptions>()
          .set_smooth_landmarks(tasks_options.base_options().use_stream_mode());
    } else if (face_detector_options.num_faces() > 1 &&
               face_landmarks_detector_graph
                   .GetOptions<FaceLandmarksDetectorGraphOptions>()
                   .smooth_landmarks()) {
      return absl::InvalidArgumentError(
          "Currently face landmarks smoothing only support a single face.");
    }

    if (tasks_options.base_options().use_stream_mode()) {
      auto& previous_loopback = graph.AddNode("PreviousLoopbackCalculator");
      image_in >> previous_loopback.In(kMainTag);
      auto prev_face_rects_from_landmarks =
          previous_loopback[Output<std::vector<NormalizedRect>>(kPrevLoopTag)];

      auto& min_size_node =
          graph.AddNode("NormalizedRectVectorHasMinSizeCalculator");
      prev_face_rects_from_landmarks >> min_size_node.In(kIterableTag);
      min_size_node.GetOptions<CollectionHasMinSizeCalculatorOptions>()
          .set_min_size(max_num_faces);
      auto has_enough_faces = min_size_node.Out("").Cast<bool>();

      // While in stream mode, skip face detector graph when we successfully
      // track the faces from the last frame.
      auto image_for_face_detector =
          DisallowIf(image_in, has_enough_faces, graph);
      image_for_face_detector >> face_detector.In(kImageTag);
      std::optional<Source<NormalizedRect>> norm_rect_in_for_face_detector;
      if (norm_rect_in) {
        norm_rect_in_for_face_detector =
            DisallowIf(norm_rect_in.value(), has_enough_faces, graph);
      }
      if (norm_rect_in_for_face_detector) {
        *norm_rect_in_for_face_detector >> face_detector.In("NORM_RECT");
      }
      auto expanded_face_rects_from_face_detector =
          face_detector.Out(kExpandedFaceRectsTag);
      auto& face_association = graph.AddNode("AssociationNormRectCalculator");
      face_association.GetOptions<mediapipe::AssociationCalculatorOptions>()
          .set_min_similarity_threshold(
              tasks_options.min_tracking_confidence());
      prev_face_rects_from_landmarks >>
          face_association[Input<std::vector<NormalizedRect>>::Multiple("")][0];
      expanded_face_rects_from_face_detector >>
          face_association[Input<std::vector<NormalizedRect>>::Multiple("")][1];
      auto face_rects = face_association.Out("");
      face_rects >> clip_face_rects.In("");
      // Back edge.
      face_rects_for_next_frame >> previous_loopback.In(kLoopTag);
    } else {
      // While not in stream mode, the input images are not guaranteed to be
      // in series, and we don't want to enable the tracking and rect
      // associations between input images. Always use the face detector
      // graph.
      image_in >> face_detector.In(kImageTag);
      if (norm_rect_in) {
        *norm_rect_in >> face_detector.In(kNormRectTag);
      }
      auto face_rects = face_detector.Out(kExpandedFaceRectsTag);
      face_rects >> clip_face_rects.In("");
    }

    // Optional blendshape output.
    std::optional<Source<std::vector<ClassificationList>>> blendshapes;
    if (output_blendshapes) {
      blendshapes = std::make_optional<>(
          face_landmarks_detector_graph.Out(kBlendshapesTag)
              .Cast<std::vector<ClassificationList>>());
    }

    // Optional face geometry output.
    std::optional<Source<std::vector<FaceGeometry>>> face_geometry;
    if (output_geometry) {
      auto& face_geometry_from_landmarks = graph.AddNode(
          "mediapipe.tasks.vision.face_geometry."
          "FaceGeometryFromLandmarksGraph");
      face_geometry_from_landmarks
          .GetOptions<face_geometry::proto::FaceGeometryGraphOptions>()
          .Swap(tasks_options.mutable_face_geometry_graph_options());
      if (environment.has_value()) {
        *environment >> face_geometry_from_landmarks.SideIn(kEnvironmentTag);
      }
      face_landmarks >> face_geometry_from_landmarks.In(kFaceLandmarksTag);
      image_size >> face_geometry_from_landmarks.In(kImageSizeTag);
      face_geometry = face_geometry_from_landmarks.Out(kFaceGeometryTag)
                          .Cast<std::vector<FaceGeometry>>();
    }

    // TODO: Replace PassThroughCalculator with a calculator that
    // converts the pixel data to be stored on the target storage (CPU vs
    // GPU).
    auto& pass_through = graph.AddNode("PassThroughCalculator");
    image_in >> pass_through.In("");

    return {{
        /* landmark_lists= */ face_landmarks,
        /* face_rects_next_frame= */
        face_rects_for_next_frame,
        /* face_rects= */
        face_detector.Out(kFaceRectsTag).Cast<std::vector<NormalizedRect>>(),
        /* face_detections */
        face_detector.Out(kDetectionsTag).Cast<std::vector<Detection>>(),
        /* face_blendshapes= */ blendshapes,
        /* face_geometry= */ face_geometry,
        /* image= */
        pass_through[Output<Image>("")],
    }};
  }
};

REGISTER_MEDIAPIPE_GRAPH(
    ::mediapipe::tasks::vision::face_landmarker::FaceLandmarkerGraph);

}  // namespace face_landmarker
}  // namespace vision
}  // namespace tasks
}  // namespace mediapipe
