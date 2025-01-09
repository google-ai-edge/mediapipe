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

#include <memory>
#include <type_traits>
#include <vector>

#include "absl/log/absl_log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "mediapipe/framework/api2/builder.h"
#include "mediapipe/framework/api2/port.h"
#include "mediapipe/framework/formats/classification.pb.h"
#include "mediapipe/framework/formats/detection.pb.h"
#include "mediapipe/framework/formats/image.h"
#include "mediapipe/framework/formats/landmark.pb.h"
#include "mediapipe/framework/formats/rect.pb.h"
#include "mediapipe/framework/port/status_macros.h"
#include "mediapipe/tasks/cc/common.h"
#include "mediapipe/tasks/cc/core/model_asset_bundle_resources.h"
#include "mediapipe/tasks/cc/core/model_resources_cache.h"
#include "mediapipe/tasks/cc/core/model_task_graph.h"
#include "mediapipe/tasks/cc/core/utils.h"
#include "mediapipe/tasks/cc/metadata/utils/zip_utils.h"
#include "mediapipe/tasks/cc/vision/gesture_recognizer/proto/gesture_recognizer_graph_options.pb.h"
#include "mediapipe/tasks/cc/vision/gesture_recognizer/proto/hand_gesture_recognizer_graph_options.pb.h"
#include "mediapipe/tasks/cc/vision/hand_detector/proto/hand_detector_graph_options.pb.h"
#include "mediapipe/tasks/cc/vision/hand_landmarker/proto/hand_landmarker_graph_options.pb.h"
#include "mediapipe/tasks/cc/vision/hand_landmarker/proto/hand_landmarks_detector_graph_options.pb.h"
#include "mediapipe/tasks/metadata/metadata_schema_generated.h"

namespace mediapipe {
namespace tasks {
namespace vision {
namespace gesture_recognizer {

namespace {

using ::mediapipe::NormalizedRect;
using ::mediapipe::api2::Input;
using ::mediapipe::api2::Output;
using ::mediapipe::api2::builder::Graph;
using ::mediapipe::api2::builder::Source;
using ::mediapipe::tasks::core::ModelAssetBundleResources;
using ::mediapipe::tasks::metadata::SetExternalFile;
using ::mediapipe::tasks::vision::gesture_recognizer::proto::
    GestureRecognizerGraphOptions;
using ::mediapipe::tasks::vision::gesture_recognizer::proto::
    HandGestureRecognizerGraphOptions;
using ::mediapipe::tasks::vision::hand_landmarker::proto::
    HandLandmarkerGraphOptions;

constexpr char kImageTag[] = "IMAGE";
constexpr char kNormRectTag[] = "NORM_RECT";
constexpr char kLandmarksTag[] = "LANDMARKS";
constexpr char kWorldLandmarksTag[] = "WORLD_LANDMARKS";
constexpr char kHandednessTag[] = "HANDEDNESS";
constexpr char kImageSizeTag[] = "IMAGE_SIZE";
constexpr char kHandGesturesTag[] = "HAND_GESTURES";
constexpr char kHandTrackingIdsTag[] = "HAND_TRACKING_IDS";
constexpr char kRectNextFrameTag[] = "HAND_RECT_NEXT_FRAME";
constexpr char kPalmRectsTag[] = "PALM_RECTS";
constexpr char kPalmDetectionsTag[] = "PALM_DETECTIONS";
constexpr char kHandLandmarkerBundleAssetName[] = "hand_landmarker.task";
constexpr char kHandGestureRecognizerBundleAssetName[] =
    "hand_gesture_recognizer.task";

struct GestureRecognizerOutputs {
  Source<std::vector<ClassificationList>> gesture;
  Source<std::vector<ClassificationList>> handedness;
  Source<std::vector<NormalizedLandmarkList>> hand_landmarks;
  Source<std::vector<LandmarkList>> hand_world_landmarks;
  Source<std::vector<NormalizedRect>> hand_rects_next_frame;
  Source<std::vector<NormalizedRect>> palm_rects;
  Source<std::vector<Detection>> palm_detections;
  Source<Image> image;
};

// Sets the base options in the sub tasks.
absl::Status SetSubTaskBaseOptions(const ModelAssetBundleResources& resources,
                                   GestureRecognizerGraphOptions* options,
                                   bool is_copy) {
  MP_ASSIGN_OR_RETURN(const auto hand_landmarker_file,
                      resources.GetFile(kHandLandmarkerBundleAssetName));
  auto* hand_landmarker_graph_options =
      options->mutable_hand_landmarker_graph_options();
  SetExternalFile(hand_landmarker_file,
                  hand_landmarker_graph_options->mutable_base_options()
                      ->mutable_model_asset(),
                  is_copy);
  hand_landmarker_graph_options->mutable_base_options()
      ->mutable_acceleration()
      ->CopyFrom(options->base_options().acceleration());
  hand_landmarker_graph_options->mutable_base_options()->set_use_stream_mode(
      options->base_options().use_stream_mode());

  MP_ASSIGN_OR_RETURN(const auto hand_gesture_recognizer_file,
                      resources.GetFile(kHandGestureRecognizerBundleAssetName));
  auto* hand_gesture_recognizer_graph_options =
      options->mutable_hand_gesture_recognizer_graph_options();
  SetExternalFile(hand_gesture_recognizer_file,
                  hand_gesture_recognizer_graph_options->mutable_base_options()
                      ->mutable_model_asset(),
                  is_copy);
  hand_gesture_recognizer_graph_options->mutable_base_options()
      ->mutable_acceleration()
      ->CopyFrom(options->base_options().acceleration());
  if (!hand_gesture_recognizer_graph_options->base_options()
           .acceleration()
           .has_xnnpack() &&
      !hand_gesture_recognizer_graph_options->base_options()
           .acceleration()
           .has_tflite()) {
    hand_gesture_recognizer_graph_options->mutable_base_options()
        ->mutable_acceleration()
        ->mutable_xnnpack();
    ABSL_LOG(WARNING) << "Hand Gesture Recognizer contains CPU only ops. Sets "
                      << "HandGestureRecognizerGraph acceleration to Xnnpack.";
  }
  hand_gesture_recognizer_graph_options->mutable_base_options()
      ->set_use_stream_mode(options->base_options().use_stream_mode());

  hand_landmarker_graph_options->mutable_base_options()->set_gpu_origin(
      options->base_options().gpu_origin());
  hand_gesture_recognizer_graph_options->mutable_base_options()->set_gpu_origin(
      options->base_options().gpu_origin());
  return absl::OkStatus();
}

}  // namespace

// A "mediapipe.tasks.vision.gesture_recognizer.GestureRecognizerGraph" performs
// hand gesture recognition.
//
// Inputs:
//   IMAGE - Image
//     Image to perform hand gesture recognition on.
//   NORM_RECT - NormalizedRect @Optional
//     Describes image rotation and region of image to perform landmarks
//     detection on. If not provided, whole image is used for gesture
//     recognition.
//
// Outputs:
//   HAND_GESTURES - std::vector<ClassificationList>
//     Recognized hand gestures with sorted order such that the winning label is
//     the first item in the list.
//   LANDMARKS: - std::vector<NormalizedLandmarkList>
//     Detected hand landmarks.
//   WORLD_LANDMARKS - std::vector<LandmarkList>
//     Detected hand landmarks in world coordinates.
//   HAND_RECT_NEXT_FRAME - std::vector<NormalizedRect>
//     The predicted Rect enclosing the hand RoI for landmark detection on the
//     next frame.
//   HANDEDNESS - std::vector<ClassificationList>
//     Classification of handedness.
//   IMAGE - mediapipe::Image
//     The image that gesture recognizer runs on and has the pixel data stored
//     on the target storage (CPU vs GPU).
// All returned coordinates are in the unrotated and uncropped input image
// coordinates system.
//
// Example:
// node {
//   calculator:
//   "mediapipe.tasks.vision.gesture_recognizer.GestureRecognizerGraph"
//   input_stream: "IMAGE:image_in"
//   input_stream: "NORM_RECT:norm_rect"
//   output_stream: "HAND_GESTURES:hand_gestures"
//   output_stream: "LANDMARKS:hand_landmarks"
//   output_stream: "WORLD_LANDMARKS:world_hand_landmarks"
//   output_stream: "HAND_RECT_NEXT_FRAME:hand_rect_next_frame"
//   output_stream: "HANDEDNESS:handedness"
//   output_stream: "IMAGE:image_out"
//   options {
//     [mediapipe.tasks.vision.gesture_recognizer.proto.GestureRecognizerGraphOptions.ext]
//     {
//       base_options {
//         model_asset {
//           file_name: "hand_gesture.tflite"
//         }
//       }
//       hand_landmark_detector_options {
//         base_options {
//           model_asset {
//             file_name: "hand_landmark.tflite"
//           }
//         }
//       }
//     }
//   }
// }
class GestureRecognizerGraph : public core::ModelTaskGraph {
 public:
  absl::StatusOr<CalculatorGraphConfig> GetConfig(
      SubgraphContext* sc) override {
    Graph graph;
    if (sc->Options<GestureRecognizerGraphOptions>()
            .base_options()
            .has_model_asset()) {
      MP_ASSIGN_OR_RETURN(
          const auto* model_asset_bundle_resources,
          CreateModelAssetBundleResources<GestureRecognizerGraphOptions>(sc));
      // When the model resources cache service is available, filling in
      // the file pointer meta in the subtasks' base options. Otherwise,
      // providing the file contents instead.
      MP_RETURN_IF_ERROR(SetSubTaskBaseOptions(
          *model_asset_bundle_resources,
          sc->MutableOptions<GestureRecognizerGraphOptions>(),
          !sc->Service(::mediapipe::tasks::core::kModelResourcesCacheService)
               .IsAvailable()));
    }
    MP_ASSIGN_OR_RETURN(
        auto hand_gesture_recognition_output,
        BuildGestureRecognizerGraph(
            *sc->MutableOptions<GestureRecognizerGraphOptions>(),
            graph[Input<Image>(kImageTag)],
            graph[Input<NormalizedRect>::Optional(kNormRectTag)], graph));
    hand_gesture_recognition_output.gesture >>
        graph[Output<std::vector<ClassificationList>>(kHandGesturesTag)];
    hand_gesture_recognition_output.handedness >>
        graph[Output<std::vector<ClassificationList>>(kHandednessTag)];
    hand_gesture_recognition_output.hand_landmarks >>
        graph[Output<std::vector<NormalizedLandmarkList>>(kLandmarksTag)];
    hand_gesture_recognition_output.hand_world_landmarks >>
        graph[Output<std::vector<LandmarkList>>(kWorldLandmarksTag)];
    hand_gesture_recognition_output.image >> graph[Output<Image>(kImageTag)];
    hand_gesture_recognition_output.hand_rects_next_frame >>
        graph[Output<std::vector<NormalizedRect>>(kRectNextFrameTag)];
    hand_gesture_recognition_output.palm_rects >>
        graph[Output<std::vector<NormalizedRect>>(kPalmRectsTag)];
    hand_gesture_recognition_output.palm_detections >>
        graph[Output<std::vector<Detection>>(kPalmDetectionsTag)];
    return graph.GetConfig();
  }

 private:
  absl::StatusOr<GestureRecognizerOutputs> BuildGestureRecognizerGraph(
      GestureRecognizerGraphOptions& graph_options, Source<Image> image_in,
      Source<NormalizedRect> norm_rect_in, Graph& graph) {
    auto& image_property = graph.AddNode("ImagePropertiesCalculator");
    image_in >> image_property.In("IMAGE");
    auto image_size = image_property.Out("SIZE");

    // Hand landmarker graph.
    auto& hand_landmarker_graph = graph.AddNode(
        "mediapipe.tasks.vision.hand_landmarker.HandLandmarkerGraph");
    auto& hand_landmarker_graph_options =
        hand_landmarker_graph.GetOptions<HandLandmarkerGraphOptions>();
    hand_landmarker_graph_options.Swap(
        graph_options.mutable_hand_landmarker_graph_options());

    image_in >> hand_landmarker_graph.In(kImageTag);
    norm_rect_in >> hand_landmarker_graph.In(kNormRectTag);
    auto hand_landmarks =
        hand_landmarker_graph[Output<std::vector<NormalizedLandmarkList>>(
            kLandmarksTag)];
    auto hand_world_landmarks =
        hand_landmarker_graph[Output<std::vector<LandmarkList>>(
            kWorldLandmarksTag)];
    auto handedness =
        hand_landmarker_graph[Output<std::vector<ClassificationList>>(
            kHandednessTag)];

    auto& vector_indices =
        graph.AddNode("NormalizedLandmarkListVectorIndicesCalculator");
    hand_landmarks >> vector_indices.In("VECTOR");
    auto hand_landmarks_id = vector_indices.Out("INDICES");

    // Hand gesture recognizer subgraph.
    auto& hand_gesture_subgraph = graph.AddNode(
        "mediapipe.tasks.vision.gesture_recognizer."
        "MultipleHandGestureRecognizerGraph");
    hand_gesture_subgraph.GetOptions<HandGestureRecognizerGraphOptions>().Swap(
        graph_options.mutable_hand_gesture_recognizer_graph_options());
    hand_landmarks >> hand_gesture_subgraph.In(kLandmarksTag);
    hand_world_landmarks >> hand_gesture_subgraph.In(kWorldLandmarksTag);
    handedness >> hand_gesture_subgraph.In(kHandednessTag);
    image_size >> hand_gesture_subgraph.In(kImageSizeTag);
    norm_rect_in >> hand_gesture_subgraph.In(kNormRectTag);
    hand_landmarks_id >> hand_gesture_subgraph.In(kHandTrackingIdsTag);
    auto hand_gestures =
        hand_gesture_subgraph[Output<std::vector<ClassificationList>>(
            kHandGesturesTag)];

    return GestureRecognizerOutputs{
        /*gesture=*/hand_gestures,
        /*handedness=*/handedness,
        /*hand_landmarks=*/hand_landmarks,
        /*hand_world_landmarks=*/hand_world_landmarks,
        /*hand_rects_next_frame =*/
        hand_landmarker_graph[Output<std::vector<NormalizedRect>>(
            kRectNextFrameTag)],
        /*palm_rects =*/
        hand_landmarker_graph[Output<std::vector<NormalizedRect>>(
            kPalmRectsTag)],
        /*palm_detections =*/
        hand_landmarker_graph[Output<std::vector<Detection>>(
            kPalmDetectionsTag)],
        /*image=*/hand_landmarker_graph[Output<Image>(kImageTag)],
    };
  }
};

// clang-format off
REGISTER_MEDIAPIPE_GRAPH(
  ::mediapipe::tasks::vision::gesture_recognizer::GestureRecognizerGraph);  // NOLINT
// clang-format on

}  // namespace gesture_recognizer
}  // namespace vision
}  // namespace tasks
}  // namespace mediapipe
