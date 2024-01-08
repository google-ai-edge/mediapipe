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

#include "mediapipe/tasks/cc/vision/holistic_landmarker/holistic_hand_tracking.h"

#include <functional>
#include <optional>
#include <utility>

#include "absl/status/statusor.h"
#include "mediapipe/calculators/util/align_hand_to_pose_in_world_calculator.h"
#include "mediapipe/calculators/util/align_hand_to_pose_in_world_calculator.pb.h"
#include "mediapipe/framework/api2/builder.h"
#include "mediapipe/framework/api2/stream/image_size.h"
#include "mediapipe/framework/api2/stream/landmarks_to_detection.h"
#include "mediapipe/framework/api2/stream/loopback.h"
#include "mediapipe/framework/api2/stream/rect_transformation.h"
#include "mediapipe/framework/api2/stream/split.h"
#include "mediapipe/framework/api2/stream/threshold.h"
#include "mediapipe/framework/formats/image.h"
#include "mediapipe/framework/formats/landmark.pb.h"
#include "mediapipe/framework/formats/rect.pb.h"
#include "mediapipe/framework/port/status_macros.h"
#include "mediapipe/modules/holistic_landmark/calculators/roi_tracking_calculator.pb.h"
#include "mediapipe/tasks/cc/components/utils/gate.h"
#include "mediapipe/tasks/cc/vision/hand_landmarker/proto/hand_landmarks_detector_graph_options.pb.h"
#include "mediapipe/tasks/cc/vision/hand_landmarker/proto/hand_roi_refinement_graph_options.pb.h"

namespace mediapipe {
namespace tasks {
namespace vision {
namespace holistic_landmarker {

namespace {

using ::mediapipe::NormalizedRect;
using ::mediapipe::api2::AlignHandToPoseInWorldCalculator;
using ::mediapipe::api2::builder::ConvertLandmarksToDetection;
using ::mediapipe::api2::builder::GetImageSize;
using ::mediapipe::api2::builder::GetLoopbackData;
using ::mediapipe::api2::builder::Graph;
using ::mediapipe::api2::builder::IsOverThreshold;
using ::mediapipe::api2::builder::ScaleAndShiftAndMakeSquareLong;
using ::mediapipe::api2::builder::SplitAndCombine;
using ::mediapipe::api2::builder::Stream;
using ::mediapipe::tasks::components::utils::AllowIf;

struct HandLandmarksResult {
  std::optional<Stream<NormalizedLandmarkList>> landmarks;
  std::optional<Stream<LandmarkList>> world_landmarks;
};

Stream<LandmarkList> AlignHandToPoseInWorldCalculator(
    Stream<LandmarkList> hand_world_landmarks,
    Stream<LandmarkList> pose_world_landmarks, int pose_wrist_idx,
    Graph& graph) {
  auto& node = graph.AddNode("AlignHandToPoseInWorldCalculator");
  auto& opts = node.GetOptions<AlignHandToPoseInWorldCalculatorOptions>();
  opts.set_hand_wrist_idx(0);
  opts.set_pose_wrist_idx(pose_wrist_idx);
  hand_world_landmarks.ConnectTo(
      node[AlignHandToPoseInWorldCalculator::kInHandLandmarks]);
  pose_world_landmarks.ConnectTo(
      node[AlignHandToPoseInWorldCalculator::kInPoseLandmarks]);
  return node[AlignHandToPoseInWorldCalculator::kOutHandLandmarks];
}

Stream<bool> GetPosePalmVisibility(
    Stream<NormalizedLandmarkList> pose_palm_landmarks, Graph& graph) {
  // Get wrist landmark.
  auto pose_wrist = SplitAndCombine(pose_palm_landmarks, {0}, graph);

  // Get visibility score.
  auto& score_node = graph.AddNode("LandmarkVisibilityCalculator");
  pose_wrist.ConnectTo(score_node.In("NORM_LANDMARKS"));
  Stream<float> score = score_node.Out("VISIBILITY").Cast<float>();

  // Convert score into flag.
  return IsOverThreshold(score, /*threshold=*/0.1, graph);
}

Stream<NormalizedRect> GetHandRoiFromPosePalmLandmarks(
    Stream<NormalizedLandmarkList> pose_palm_landmarks,
    Stream<std::pair<int, int>> image_size, Graph& graph) {
  // Convert pose palm landmarks to detection.
  auto detection = ConvertLandmarksToDetection(pose_palm_landmarks, graph);

  // Convert detection to rect.
  auto& rect_node = graph.AddNode("HandDetectionsFromPoseToRectsCalculator");
  detection.ConnectTo(rect_node.In("DETECTION"));
  image_size.ConnectTo(rect_node.In("IMAGE_SIZE"));
  Stream<NormalizedRect> rect =
      rect_node.Out("NORM_RECT").Cast<NormalizedRect>();

  return ScaleAndShiftAndMakeSquareLong(rect, image_size,
                                        /*scale_x_factor=*/2.7,
                                        /*scale_y_factor=*/2.7, /*shift_x=*/0,
                                        /*shift_y=*/-0.1, graph);
}

absl::StatusOr<Stream<NormalizedRect>> RefineHandRoi(
    Stream<Image> image, Stream<NormalizedRect> roi,
    const hand_landmarker::proto::HandRoiRefinementGraphOptions&
        hand_roi_refinenement_graph_options,
    Graph& graph) {
  auto& hand_roi_refinement = graph.AddNode(
      "mediapipe.tasks.vision.hand_landmarker.HandRoiRefinementGraph");
  hand_roi_refinement
      .GetOptions<hand_landmarker::proto::HandRoiRefinementGraphOptions>() =
      hand_roi_refinenement_graph_options;
  image >> hand_roi_refinement.In("IMAGE");
  roi >> hand_roi_refinement.In("NORM_RECT");
  return hand_roi_refinement.Out("NORM_RECT").Cast<NormalizedRect>();
}

Stream<NormalizedRect> TrackHandRoi(
    Stream<NormalizedLandmarkList> prev_landmarks, Stream<NormalizedRect> roi,
    Stream<std::pair<int, int>> image_size, Graph& graph) {
  // Convert hand landmarks to tight rect.
  auto& prev_rect_node = graph.AddNode("HandLandmarksToRectCalculator");
  prev_landmarks.ConnectTo(prev_rect_node.In("NORM_LANDMARKS"));
  image_size.ConnectTo(prev_rect_node.In("IMAGE_SIZE"));
  Stream<NormalizedRect> prev_rect =
      prev_rect_node.Out("NORM_RECT").Cast<NormalizedRect>();

  // Convert tight hand rect to hand roi.
  Stream<NormalizedRect> prev_roi =
      ScaleAndShiftAndMakeSquareLong(prev_rect, image_size,
                                     /*scale_x_factor=*/2.0,
                                     /*scale_y_factor=*/2.0, /*shift_x=*/0,
                                     /*shift_y=*/-0.1, graph);

  auto& tracking_node = graph.AddNode("RoiTrackingCalculator");
  auto& tracking_node_opts =
      tracking_node.GetOptions<RoiTrackingCalculatorOptions>();
  auto* rect_requirements = tracking_node_opts.mutable_rect_requirements();
  rect_requirements->set_rotation_degrees(40.0);
  rect_requirements->set_translation(0.2);
  rect_requirements->set_scale(0.4);
  auto* landmarks_requirements =
      tracking_node_opts.mutable_landmarks_requirements();
  landmarks_requirements->set_recrop_rect_margin(-0.1);
  prev_landmarks.ConnectTo(tracking_node.In("PREV_LANDMARKS"));
  prev_roi.ConnectTo(tracking_node.In("PREV_LANDMARKS_RECT"));
  roi.ConnectTo(tracking_node.In("RECROP_RECT"));
  image_size.ConnectTo(tracking_node.In("IMAGE_SIZE"));
  return tracking_node.Out("TRACKING_RECT").Cast<NormalizedRect>();
}

HandLandmarksResult GetHandLandmarksDetection(
    Stream<Image> image, Stream<NormalizedRect> roi,
    const hand_landmarker::proto::HandLandmarksDetectorGraphOptions&
        hand_landmarks_detector_graph_options,
    const HolisticHandTrackingRequest& request, Graph& graph) {
  HandLandmarksResult result;
  auto& hand_landmarks_detector_graph = graph.AddNode(
      "mediapipe.tasks.vision.hand_landmarker."
      "SingleHandLandmarksDetectorGraph");
  hand_landmarks_detector_graph
      .GetOptions<hand_landmarker::proto::HandLandmarksDetectorGraphOptions>() =
      hand_landmarks_detector_graph_options;

  image >> hand_landmarks_detector_graph.In("IMAGE");
  roi >> hand_landmarks_detector_graph.In("HAND_RECT");

  if (request.landmarks) {
    result.landmarks = hand_landmarks_detector_graph.Out("LANDMARKS")
                           .Cast<NormalizedLandmarkList>();
  }
  if (request.world_landmarks) {
    result.world_landmarks =
        hand_landmarks_detector_graph.Out("WORLD_LANDMARKS")
            .Cast<LandmarkList>();
  }
  return result;
}

}  // namespace

absl::StatusOr<HolisticHandTrackingOutput> TrackHolisticHand(
    Stream<Image> image, Stream<NormalizedLandmarkList> pose_landmarks,
    Stream<LandmarkList> pose_world_landmarks,
    const hand_landmarker::proto::HandLandmarksDetectorGraphOptions&
        hand_landmarks_detector_graph_options,
    const hand_landmarker::proto::HandRoiRefinementGraphOptions&
        hand_roi_refinement_graph_options,
    const PoseIndices& pose_indices, const HolisticHandTrackingRequest& request,
    Graph& graph) {
  // Extracts pose palm landmarks.
  Stream<NormalizedLandmarkList> pose_palm_landmarks = SplitAndCombine(
      pose_landmarks,
      {pose_indices.wrist_idx, pose_indices.pinky_idx, pose_indices.index_idx},
      graph);

  // Get pose palm visibility.
  Stream<bool> is_pose_palm_visible =
      GetPosePalmVisibility(pose_palm_landmarks, graph);

  // Drop pose palm landmarks if pose palm is invisible.
  pose_palm_landmarks =
      AllowIf(pose_palm_landmarks, is_pose_palm_visible, graph);

  // Extracts image size from the input images.
  Stream<std::pair<int, int>> image_size = GetImageSize(image, graph);

  // Get hand ROI from pose palm landmarks.
  Stream<NormalizedRect> roi_from_pose =
      GetHandRoiFromPosePalmLandmarks(pose_palm_landmarks, image_size, graph);

  // Refine hand ROI with re-crop model.
  MP_ASSIGN_OR_RETURN(Stream<NormalizedRect> roi_from_recrop,
                      RefineHandRoi(image, roi_from_pose,
                                    hand_roi_refinement_graph_options, graph));

  // Loop for previous frame landmarks.
  auto [prev_landmarks, set_prev_landmarks_fn] =
      GetLoopbackData<NormalizedLandmarkList>(/*tick=*/image_size, graph);

  // Track hand ROI.
  auto tracking_roi =
      TrackHandRoi(prev_landmarks, roi_from_recrop, image_size, graph);

  // Predict hand landmarks.
  auto landmarks_detection_result = GetHandLandmarksDetection(
      image, tracking_roi, hand_landmarks_detector_graph_options, request,
      graph);

  // Set previous landmarks for ROI tracking.
  set_prev_landmarks_fn(landmarks_detection_result.landmarks.value());

  // Output landmarks.
  std::optional<Stream<NormalizedLandmarkList>> hand_landmarks;
  if (request.landmarks) {
    hand_landmarks = landmarks_detection_result.landmarks;
  }

  // Output world landmarks.
  std::optional<Stream<LandmarkList>> hand_world_landmarks;
  if (request.world_landmarks) {
    hand_world_landmarks = landmarks_detection_result.world_landmarks;

    // Align hand world landmarks with pose world landmarks.
    hand_world_landmarks = AlignHandToPoseInWorldCalculator(
        hand_world_landmarks.value(), pose_world_landmarks,
        pose_indices.wrist_idx, graph);
  }

  return {{/*landmarks=*/hand_landmarks,
           /*world_landmarks=*/hand_world_landmarks,
           /*debug_output=*/
           {
               /*roi_from_pose=*/roi_from_pose,
               /*roi_from_recrop=*/roi_from_recrop,
               /*tracking_roi=*/tracking_roi,
           }}};
}

}  // namespace holistic_landmarker
}  // namespace vision
}  // namespace tasks
}  // namespace mediapipe
