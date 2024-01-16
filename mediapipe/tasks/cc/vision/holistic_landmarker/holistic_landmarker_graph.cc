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

#include <string>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "mediapipe/framework/api2/builder.h"
#include "mediapipe/framework/api2/stream/split.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/classification.pb.h"
#include "mediapipe/framework/formats/image.h"
#include "mediapipe/framework/formats/landmark.pb.h"
#include "mediapipe/framework/formats/rect.pb.h"
#include "mediapipe/framework/port/ret_check.h"
#include "mediapipe/framework/port/status_macros.h"
#include "mediapipe/tasks/cc/core/model_asset_bundle_resources.h"
#include "mediapipe/tasks/cc/core/model_resources_cache.h"
#include "mediapipe/tasks/cc/core/model_task_graph.h"
#include "mediapipe/tasks/cc/core/utils.h"
#include "mediapipe/tasks/cc/metadata/utils/zip_utils.h"
#include "mediapipe/tasks/cc/vision/face_detector/proto/face_detector_graph_options.pb.h"
#include "mediapipe/tasks/cc/vision/face_landmarker/proto/face_blendshapes_graph_options.pb.h"
#include "mediapipe/tasks/cc/vision/face_landmarker/proto/face_landmarks_detector_graph_options.pb.h"
#include "mediapipe/tasks/cc/vision/hand_landmarker/proto/hand_landmarks_detector_graph_options.pb.h"
#include "mediapipe/tasks/cc/vision/hand_landmarker/proto/hand_roi_refinement_graph_options.pb.h"
#include "mediapipe/tasks/cc/vision/holistic_landmarker/holistic_face_tracking.h"
#include "mediapipe/tasks/cc/vision/holistic_landmarker/holistic_hand_tracking.h"
#include "mediapipe/tasks/cc/vision/holistic_landmarker/holistic_pose_tracking.h"
#include "mediapipe/tasks/cc/vision/holistic_landmarker/proto/holistic_landmarker_graph_options.pb.h"
#include "mediapipe/tasks/cc/vision/pose_detector/proto/pose_detector_graph_options.pb.h"
#include "mediapipe/tasks/cc/vision/pose_landmarker/pose_topology.h"
#include "mediapipe/tasks/cc/vision/pose_landmarker/proto/pose_landmarks_detector_graph_options.pb.h"
#include "mediapipe/util/graph_builder_utils.h"

namespace mediapipe {
namespace tasks {
namespace vision {
namespace holistic_landmarker {
namespace {

using ::mediapipe::api2::Output;
using ::mediapipe::api2::builder::Graph;
using ::mediapipe::api2::builder::Stream;
using ::mediapipe::tasks::metadata::SetExternalFile;

constexpr absl::string_view kHandLandmarksDetectorModelName =
    "hand_landmarks_detector.tflite";
constexpr absl::string_view kHandRoiRefinementModelName =
    "hand_roi_refinement.tflite";
constexpr absl::string_view kFaceDetectorModelName = "face_detector.tflite";
constexpr absl::string_view kFaceLandmarksDetectorModelName =
    "face_landmarks_detector.tflite";
constexpr absl::string_view kFaceBlendshapesModelName =
    "face_blendshapes.tflite";
constexpr absl::string_view kPoseDetectorModelName = "pose_detector.tflite";
constexpr absl::string_view kPoseLandmarksDetectorModelName =
    "pose_landmarks_detector.tflite";

absl::Status SetGraphPoseOutputs(
    const HolisticPoseTrackingRequest& pose_request,
    const CalculatorGraphConfig::Node& node,
    HolisticPoseTrackingOutput& pose_output, Graph& graph) {
  // Main outputs.
  if (pose_request.landmarks) {
    RET_CHECK(pose_output.landmarks.has_value())
        << "POSE_LANDMARKS output is not supported.";
    pose_output.landmarks->ConnectTo(graph.Out("POSE_LANDMARKS"));
  }
  if (pose_request.world_landmarks) {
    RET_CHECK(pose_output.world_landmarks.has_value())
        << "POSE_WORLD_LANDMARKS output is not supported.";
    pose_output.world_landmarks->ConnectTo(graph.Out("POSE_WORLD_LANDMARKS"));
  }
  if (pose_request.segmentation_mask) {
    RET_CHECK(pose_output.segmentation_mask.has_value())
        << "POSE_SEGMENTATION_MASK output is not supported.";
    pose_output.segmentation_mask->ConnectTo(
        graph.Out("POSE_SEGMENTATION_MASK"));
  }

  // Debug outputs.
  if (HasOutput(node, "POSE_AUXILIARY_LANDMARKS")) {
    pose_output.debug_output.auxiliary_landmarks.ConnectTo(
        graph.Out("POSE_AUXILIARY_LANDMARKS"));
  }
  if (HasOutput(node, "POSE_LANDMARKS_ROI")) {
    pose_output.debug_output.roi_from_landmarks.ConnectTo(
        graph.Out("POSE_LANDMARKS_ROI"));
  }

  return absl::OkStatus();
}

// Sets the base options in the sub tasks.
template <typename T>
absl::Status SetSubTaskBaseOptions(
    const core::ModelAssetBundleResources* resources,
    proto::HolisticLandmarkerGraphOptions* options, T* sub_task_options,
    absl::string_view model_name, bool is_copy) {
  if (!sub_task_options->base_options().has_model_asset()) {
    MP_ASSIGN_OR_RETURN(const auto model_file_content,
                        resources->GetFile(std::string(model_name)));
    SetExternalFile(
        model_file_content,
        sub_task_options->mutable_base_options()->mutable_model_asset(),
        is_copy);
  }
  sub_task_options->mutable_base_options()->mutable_acceleration()->CopyFrom(
      options->base_options().acceleration());
  sub_task_options->mutable_base_options()->set_use_stream_mode(
      options->base_options().use_stream_mode());
  sub_task_options->mutable_base_options()->set_gpu_origin(
      options->base_options().gpu_origin());
  return absl::OkStatus();
}

void SetGraphHandOutputs(bool is_left, const CalculatorGraphConfig::Node& node,
                         HolisticHandTrackingOutput& hand_output,
                         Graph& graph) {
  const std::string hand_side = is_left ? "LEFT" : "RIGHT";

  if (hand_output.landmarks) {
    hand_output.landmarks->ConnectTo(graph.Out(hand_side + "_HAND_LANDMARKS"));
  }
  if (hand_output.world_landmarks) {
    hand_output.world_landmarks->ConnectTo(
        graph.Out(hand_side + "_HAND_WORLD_LANDMARKS"));
  }

  // Debug outputs.
  if (HasOutput(node, hand_side + "_HAND_ROI_FROM_POSE")) {
    hand_output.debug_output.roi_from_pose.ConnectTo(
        graph.Out(hand_side + "_HAND_ROI_FROM_POSE"));
  }
  if (HasOutput(node, hand_side + "_HAND_ROI_FROM_RECROP")) {
    hand_output.debug_output.roi_from_recrop.ConnectTo(
        graph.Out(hand_side + "_HAND_ROI_FROM_RECROP"));
  }
  if (HasOutput(node, hand_side + "_HAND_TRACKING_ROI")) {
    hand_output.debug_output.tracking_roi.ConnectTo(
        graph.Out(hand_side + "_HAND_TRACKING_ROI"));
  }
}

void SetGraphFaceOutputs(const CalculatorGraphConfig::Node& node,
                         HolisticFaceTrackingOutput& face_output,
                         Graph& graph) {
  if (face_output.landmarks) {
    face_output.landmarks->ConnectTo(graph.Out("FACE_LANDMARKS"));
  }
  if (face_output.classifications) {
    face_output.classifications->ConnectTo(graph.Out("FACE_BLENDSHAPES"));
  }

  // Face detection debug outputs
  if (HasOutput(node, "FACE_ROI_FROM_POSE")) {
    face_output.debug_output.roi_from_pose.ConnectTo(
        graph.Out("FACE_ROI_FROM_POSE"));
  }
  if (HasOutput(node, "FACE_ROI_FROM_DETECTION")) {
    face_output.debug_output.roi_from_detection.ConnectTo(
        graph.Out("FACE_ROI_FROM_DETECTION"));
  }
  if (HasOutput(node, "FACE_TRACKING_ROI")) {
    face_output.debug_output.tracking_roi.ConnectTo(
        graph.Out("FACE_TRACKING_ROI"));
  }
}

}  // namespace

// Tracks pose and detects hands and face.
//
// NOTE: for GPU works only with image having GpuOrigin::TOP_LEFT
//
// Inputs:
//   IMAGE - Image
//     Image to perform detection on.
//
// Outputs:
//   POSE_LANDMARKS - NormalizedLandmarkList
//     33 landmarks (see pose_landmarker/pose_topology.h)
//     0 - nose
//     1 - left eye (inner)
//     2 - left eye
//     3 - left eye (outer)
//     4 - right eye (inner)
//     5 - right eye
//     6 - right eye (outer)
//     7 - left ear
//     8 - right ear
//     9 - mouth (left)
//     10 - mouth (right)
//     11 - left shoulder
//     12 - right shoulder
//     13 - left elbow
//     14 - right elbow
//     15 - left wrist
//     16 - right wrist
//     17 - left pinky
//     18 - right pinky
//     19 - left index
//     20 - right index
//     21 - left thumb
//     22 - right thumb
//     23 - left hip
//     24 - right hip
//     25 - left knee
//     26 - right knee
//     27 - left ankle
//     28 - right ankle
//     29 - left heel
//     30 - right heel
//     31 - left foot index
//     32 - right foot index
//   POSE_WORLD_LANDMARKS - LandmarkList
//     World landmarks are real world 3D coordinates with origin in hips center
//     and coordinates in meters. To understand the difference: POSE_LANDMARKS
//     stream provides coordinates (in pixels) of 3D object projected on a 2D
//     surface of the image (check on how perspective projection works), while
//     POSE_WORLD_LANDMARKS stream provides coordinates (in meters) of the 3D
//     object itself. POSE_WORLD_LANDMARKS has the same landmarks topology,
//     visibility and presence as POSE_LANDMARKS.
//   POSE_SEGMENTATION_MASK - Image
//     Separates person from background. Mask is stored as gray float32 image
//     with [0.0, 1.0] range for pixels (1 for person and 0 for background) on
//     CPU and, on GPU - RGBA texture with R channel indicating person vs.
//     background probability.
//   LEFT_HAND_LANDMARKS - NormalizedLandmarkList
//     21 left hand landmarks.
//   RIGHT_HAND_LANDMARKS - NormalizedLandmarkList
//     21 right hand landmarks.
//   FACE_LANDMARKS - NormalizedLandmarkList
//     468 face landmarks.
//   FACE_BLENDSHAPES - ClassificationList
//     Supplementary blendshape coefficients that are predicted directly from
//     the input image.
//   LEFT_HAND_WORLD_LANDMARKS - LandmarkList
//     21 left hand world 3D landmarks.
//     Hand landmarks are aligned with pose landmarks: translated so that wrist
//     from # hand matches wrist from pose in pose coordinates system.
//   RIGHT_HAND_WORLD_LANDMARKS - LandmarkList
//     21 right hand world 3D landmarks.
//     Hand landmarks are aligned with pose landmarks: translated so that wrist
//     from # hand matches wrist from pose in pose coordinates system.
//   IMAGE - Image
//     The input image that the hiolistic landmarker runs on and has the pixel
//     data stored on the target storage (CPU vs GPU).
//
// Debug outputs:
//   POSE_AUXILIARY_LANDMARKS - NormalizedLandmarkList
//     TODO: Return ROI rather than auxiliary landmarks
//     Auxiliary landmarks for deriving the ROI in the subsequent image.
//     0 - hidden center point
//     1 - hidden scale point
//   POSE_LANDMARKS_ROI - NormalizedRect
//     Region of interest calculated based on landmarks.
//   LEFT_HAND_ROI_FROM_POSE - NormalizedLandmarkList
//   LEFT_HAND_ROI_FROM_RECROP - NormalizedLandmarkList
//   LEFT_HAND_TRACKING_ROI - NormalizedLandmarkList
//   RIGHT_HAND_ROI_FROM_POSE - NormalizedLandmarkList
//   RIGHT_HAND_ROI_FROM_RECROP - NormalizedLandmarkList
//   RIGHT_HAND_TRACKING_ROI - NormalizedLandmarkList
//   FACE_ROI_FROM_POSE - NormalizedLandmarkList
//   FACE_ROI_FROM_DETECTION - NormalizedLandmarkList
//   FACE_TRACKING_ROI - NormalizedLandmarkList
//
//   NOTE: failure is reported if some output has been requested, but specified
//     model doesn't support it.
//
//   NOTE: there will not be an output packet in an output stream for a
//     particular timestamp if nothing is detected. However, the MediaPipe
//     framework will internally inform the downstream calculators of the
//     absence of this packet so that they don't wait for it unnecessarily.
//
// Example:
// node {
//   calculator:
//   "mediapipe.tasks.vision.holistic_landmarker.HolisticLandmarkerGraph"
//   input_stream: "IMAGE:input_frames_image"
//   output_stream: "POSE_LANDMARKS:pose_landmarks"
//   output_stream: "POSE_WORLD_LANDMARKS:pose_world_landmarks"
//   output_stream: "FACE_LANDMARKS:face_landmarks"
//   output_stream: "FACE_BLENDSHAPES:extra_blendshapes"
//   output_stream: "LEFT_HAND_LANDMARKS:left_hand_landmarks"
//   output_stream: "LEFT_HAND_WORLD_LANDMARKS:left_hand_world_landmarks"
//   output_stream: "RIGHT_HAND_LANDMARKS:right_hand_landmarks"
//   output_stream: "RIGHT_HAND_WORLD_LANDMARKS:right_hand_world_landmarks"
//   node_options {
//     [type.googleapis.com/mediapipe.tasks.vision.holistic_landmarker.proto.HolisticLandmarkerGraphOptions]
//     {
//       base_options {
//         model_asset {
//           file_name:
//           "mediapipe/tasks/testdata/vision/holistic_landmarker.task"
//         }
//       }
//       face_detector_graph_options: {
//         num_faces: 1
//       }
//       pose_detector_graph_options: {
//         num_poses: 1
//       }
//     }
//   }
// }
class HolisticLandmarkerGraph : public core::ModelTaskGraph {
 public:
  absl::StatusOr<CalculatorGraphConfig> GetConfig(
      SubgraphContext* sc) override {
    Graph graph;
    const auto& holistic_node = sc->OriginalNode();
    proto::HolisticLandmarkerGraphOptions* holistic_options =
        sc->MutableOptions<proto::HolisticLandmarkerGraphOptions>();
    const core::ModelAssetBundleResources* model_asset_bundle_resources;
    if (holistic_options->base_options().has_model_asset()) {
      MP_ASSIGN_OR_RETURN(model_asset_bundle_resources,
                          CreateModelAssetBundleResources<
                              proto::HolisticLandmarkerGraphOptions>(sc));
    }
    // Copies the file content instead of passing the pointer of file in
    // memory if the subgraph model resource service is not available.
    bool create_copy =
        !sc->Service(::mediapipe::tasks::core::kModelResourcesCacheService)
             .IsAvailable();

    Stream<Image> image = graph.In("IMAGE").Cast<Image>();

    // Check whether Hand requested
    const bool is_left_hand_requested =
        HasOutput(holistic_node, "LEFT_HAND_LANDMARKS");
    const bool is_right_hand_requested =
        HasOutput(holistic_node, "RIGHT_HAND_LANDMARKS");
    const bool is_left_hand_world_requested =
        HasOutput(holistic_node, "LEFT_HAND_WORLD_LANDMARKS");
    const bool is_right_hand_world_requested =
        HasOutput(holistic_node, "RIGHT_HAND_WORLD_LANDMARKS");
    const bool hands_requested =
        is_left_hand_requested || is_right_hand_requested ||
        is_left_hand_world_requested || is_right_hand_world_requested;
    if (hands_requested) {
      MP_RETURN_IF_ERROR(SetSubTaskBaseOptions(
          model_asset_bundle_resources, holistic_options,
          holistic_options->mutable_hand_landmarks_detector_graph_options(),
          kHandLandmarksDetectorModelName, create_copy));
      MP_RETURN_IF_ERROR(SetSubTaskBaseOptions(
          model_asset_bundle_resources, holistic_options,
          holistic_options->mutable_hand_roi_refinement_graph_options(),
          kHandRoiRefinementModelName, create_copy));
    }

    // Check whether Face requested
    const bool is_face_requested = HasOutput(holistic_node, "FACE_LANDMARKS");
    const bool is_face_blendshapes_requested =
        HasOutput(holistic_node, "FACE_BLENDSHAPES");
    const bool face_requested =
        is_face_requested || is_face_blendshapes_requested;
    if (face_requested) {
      MP_RETURN_IF_ERROR(SetSubTaskBaseOptions(
          model_asset_bundle_resources, holistic_options,
          holistic_options->mutable_face_detector_graph_options(),
          kFaceDetectorModelName, create_copy));
      // Forcely set num_faces to 1, because holistic landmarker only supports a
      // single subject for now.
      holistic_options->mutable_face_detector_graph_options()->set_num_faces(1);
      MP_RETURN_IF_ERROR(SetSubTaskBaseOptions(
          model_asset_bundle_resources, holistic_options,
          holistic_options->mutable_face_landmarks_detector_graph_options(),
          kFaceLandmarksDetectorModelName, create_copy));
      if (is_face_blendshapes_requested) {
        MP_RETURN_IF_ERROR(SetSubTaskBaseOptions(
            model_asset_bundle_resources, holistic_options,
            holistic_options->mutable_face_landmarks_detector_graph_options()
                ->mutable_face_blendshapes_graph_options(),
            kFaceBlendshapesModelName, create_copy));
      }
    }

    MP_RETURN_IF_ERROR(SetSubTaskBaseOptions(
        model_asset_bundle_resources, holistic_options,
        holistic_options->mutable_pose_detector_graph_options(),
        kPoseDetectorModelName, create_copy));
    // Forcely set num_poses to 1, because holistic landmarker sonly supports a
    // single subject for now.
    holistic_options->mutable_pose_detector_graph_options()->set_num_poses(1);
    MP_RETURN_IF_ERROR(SetSubTaskBaseOptions(
        model_asset_bundle_resources, holistic_options,
        holistic_options->mutable_pose_landmarks_detector_graph_options(),
        kPoseLandmarksDetectorModelName, create_copy));

    HolisticPoseTrackingRequest pose_request = {
        /*.landmarks=*/HasOutput(holistic_node, "POSE_LANDMARKS") ||
            hands_requested || face_requested,
        /*.world_landmarks=*/HasOutput(holistic_node, "POSE_WORLD_LANDMARKS") ||
            hands_requested,
        /*.segmentation_mask=*/
        HasOutput(holistic_node, "POSE_SEGMENTATION_MASK")};

    // Detect and track pose.
    MP_ASSIGN_OR_RETURN(
        HolisticPoseTrackingOutput pose_output,
        TrackHolisticPose(
            image, holistic_options->pose_detector_graph_options(),
            holistic_options->pose_landmarks_detector_graph_options(),
            pose_request, graph));
    MP_RETURN_IF_ERROR(
        SetGraphPoseOutputs(pose_request, holistic_node, pose_output, graph));

    // Detect and track hand.
    if (hands_requested) {
      if (is_left_hand_requested || is_left_hand_world_requested) {
        RET_CHECK(pose_output.landmarks.has_value());
        RET_CHECK(pose_output.world_landmarks.has_value());

        PoseIndices pose_indices = {
            /*.wrist_idx =*/
            static_cast<int>(pose_landmarker::PoseLandmarkName::kLeftWrist),
            /*.pinky_idx =*/
            static_cast<int>(pose_landmarker::PoseLandmarkName::kLeftPinky1),
            /*.index_idx = */
            static_cast<int>(pose_landmarker::PoseLandmarkName::kLeftIndex1),
        };
        HolisticHandTrackingRequest hand_request = {
            /*.landmarks = */ is_left_hand_requested,
            /*.world_landmarks = */ is_left_hand_world_requested,
        };
        MP_ASSIGN_OR_RETURN(
            HolisticHandTrackingOutput hand_output,
            TrackHolisticHand(
                image, *pose_output.landmarks, *pose_output.world_landmarks,
                holistic_options->hand_landmarks_detector_graph_options(),
                holistic_options->hand_roi_refinement_graph_options(),
                pose_indices, hand_request, graph

                ));
        SetGraphHandOutputs(/*is_left=*/true, holistic_node, hand_output,
                            graph);
      }

      if (is_right_hand_requested || is_right_hand_world_requested) {
        RET_CHECK(pose_output.landmarks.has_value());
        RET_CHECK(pose_output.world_landmarks.has_value());

        PoseIndices pose_indices = {
            /*.wrist_idx = */ static_cast<int>(
                pose_landmarker::PoseLandmarkName::kRightWrist),
            /*.pinky_idx = */
            static_cast<int>(pose_landmarker::PoseLandmarkName::kRightPinky1),
            /*.index_idx = */
            static_cast<int>(pose_landmarker::PoseLandmarkName::kRightIndex1),
        };
        HolisticHandTrackingRequest hand_request = {
            /*.landmarks = */ is_right_hand_requested,
            /*.world_landmarks = */ is_right_hand_world_requested,
        };
        MP_ASSIGN_OR_RETURN(
            HolisticHandTrackingOutput hand_output,
            TrackHolisticHand(
                image, *pose_output.landmarks, *pose_output.world_landmarks,
                holistic_options->hand_landmarks_detector_graph_options(),
                holistic_options->hand_roi_refinement_graph_options(),
                pose_indices, hand_request, graph

                ));
        SetGraphHandOutputs(/*is_left=*/false, holistic_node, hand_output,
                            graph);
      }
    }

    // Detect and track face.
    if (face_requested) {
      RET_CHECK(pose_output.landmarks.has_value());

      Stream<mediapipe::NormalizedLandmarkList> face_landmarks_from_pose =
          api2::builder::SplitToRanges(*pose_output.landmarks, {{0, 11}},
                                       graph)[0];

      HolisticFaceTrackingRequest face_request = {
          /*.classifications = */ is_face_blendshapes_requested,
      };
      MP_ASSIGN_OR_RETURN(
          HolisticFaceTrackingOutput face_output,
          TrackHolisticFace(
              image, face_landmarks_from_pose,
              holistic_options->face_detector_graph_options(),
              holistic_options->face_landmarks_detector_graph_options(),
              face_request, graph));
      SetGraphFaceOutputs(holistic_node, face_output, graph);
    }

    auto& pass_through = graph.AddNode("PassThroughCalculator");
    image >> pass_through.In("");
    pass_through.Out("") >> graph.Out("IMAGE");

    auto config = graph.GetConfig();
    core::FixGraphBackEdges(config);
    return config;
  }
};

REGISTER_MEDIAPIPE_GRAPH(
    ::mediapipe::tasks::vision::holistic_landmarker::HolisticLandmarkerGraph);

}  // namespace holistic_landmarker
}  // namespace vision
}  // namespace tasks
}  // namespace mediapipe
