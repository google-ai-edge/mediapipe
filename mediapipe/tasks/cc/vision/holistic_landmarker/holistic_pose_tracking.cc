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

#include "mediapipe/tasks/cc/vision/holistic_landmarker/holistic_pose_tracking.h"

#include <optional>
#include <utility>
#include <vector>

#include "absl/status/statusor.h"
#include "mediapipe/framework/api2/builder.h"
#include "mediapipe/framework/api2/stream/detections_to_rects.h"
#include "mediapipe/framework/api2/stream/image_size.h"
#include "mediapipe/framework/api2/stream/landmarks_to_detection.h"
#include "mediapipe/framework/api2/stream/loopback.h"
#include "mediapipe/framework/api2/stream/merge.h"
#include "mediapipe/framework/api2/stream/presence.h"
#include "mediapipe/framework/api2/stream/rect_transformation.h"
#include "mediapipe/framework/api2/stream/segmentation_smoothing.h"
#include "mediapipe/framework/api2/stream/smoothing.h"
#include "mediapipe/framework/api2/stream/split.h"
#include "mediapipe/framework/formats/detection.pb.h"
#include "mediapipe/framework/formats/image.h"
#include "mediapipe/framework/formats/landmark.pb.h"
#include "mediapipe/framework/formats/rect.pb.h"
#include "mediapipe/framework/port/ret_check.h"
#include "mediapipe/framework/port/status_macros.h"
#include "mediapipe/tasks/cc/components/utils/gate.h"
#include "mediapipe/tasks/cc/vision/pose_detector/proto/pose_detector_graph_options.pb.h"
#include "mediapipe/tasks/cc/vision/pose_landmarker/proto/pose_landmarks_detector_graph_options.pb.h"

namespace mediapipe {
namespace tasks {
namespace vision {
namespace holistic_landmarker {

namespace {

using ::mediapipe::NormalizedRect;
using ::mediapipe::api2::builder::ConvertAlignmentPointsDetectionsToRect;
using ::mediapipe::api2::builder::ConvertAlignmentPointsDetectionToRect;
using ::mediapipe::api2::builder::ConvertLandmarksToDetection;
using ::mediapipe::api2::builder::GenericNode;
using ::mediapipe::api2::builder::GetImageSize;
using ::mediapipe::api2::builder::GetLoopbackData;
using ::mediapipe::api2::builder::Graph;
using ::mediapipe::api2::builder::IsPresent;
using ::mediapipe::api2::builder::Merge;
using ::mediapipe::api2::builder::ScaleAndMakeSquare;
using ::mediapipe::api2::builder::SmoothLandmarks;
using ::mediapipe::api2::builder::SmoothLandmarksVisibility;
using ::mediapipe::api2::builder::SmoothSegmentationMask;
using ::mediapipe::api2::builder::SplitToRanges;
using ::mediapipe::api2::builder::Stream;
using ::mediapipe::tasks::components::utils::DisallowIf;
using Size = std::pair<int, int>;

constexpr int kAuxLandmarksStartKeypointIndex = 0;
constexpr int kAuxLandmarksEndKeypointIndex = 1;
constexpr float kAuxLandmarksTargetAngle = 90;
constexpr float kRoiFromDetectionScaleFactor = 1.25f;
constexpr float kRoiFromLandmarksScaleFactor = 1.25f;

Stream<NormalizedRect> CalculateRoiFromDetections(
    Stream<std::vector<Detection>> detections, Stream<Size> image_size,
    Graph& graph) {
  auto roi = ConvertAlignmentPointsDetectionsToRect(detections, image_size,
                                                    /*start_keypoint_index=*/0,
                                                    /*end_keypoint_index=*/1,
                                                    /*target_angle=*/90, graph);
  return ScaleAndMakeSquare(
      roi, image_size, /*scale_x_factor=*/kRoiFromDetectionScaleFactor,
      /*scale_y_factor=*/kRoiFromDetectionScaleFactor, graph);
}

Stream<NormalizedRect> CalculateScaleRoiFromAuxiliaryLandmarks(
    Stream<NormalizedLandmarkList> landmarks, Stream<Size> image_size,
    Graph& graph) {
  // TODO: consider calculating ROI directly from landmarks.
  auto detection = ConvertLandmarksToDetection(landmarks, graph);
  return ConvertAlignmentPointsDetectionToRect(
      detection, image_size, kAuxLandmarksStartKeypointIndex,
      kAuxLandmarksEndKeypointIndex, kAuxLandmarksTargetAngle, graph);
}

Stream<NormalizedRect> CalculateRoiFromAuxiliaryLandmarks(
    Stream<NormalizedLandmarkList> landmarks, Stream<Size> image_size,
    Graph& graph) {
  // TODO: consider calculating ROI directly from landmarks.
  auto detection = ConvertLandmarksToDetection(landmarks, graph);
  auto roi = ConvertAlignmentPointsDetectionToRect(
      detection, image_size, kAuxLandmarksStartKeypointIndex,
      kAuxLandmarksEndKeypointIndex, kAuxLandmarksTargetAngle, graph);
  return ScaleAndMakeSquare(
      roi, image_size, /*scale_x_factor=*/kRoiFromLandmarksScaleFactor,
      /*scale_y_factor=*/kRoiFromLandmarksScaleFactor, graph);
}

struct PoseLandmarksResult {
  std::optional<Stream<NormalizedLandmarkList>> landmarks;
  std::optional<Stream<LandmarkList>> world_landmarks;
  std::optional<Stream<NormalizedLandmarkList>> auxiliary_landmarks;
  std::optional<Stream<Image>> segmentation_mask;
};

PoseLandmarksResult RunLandmarksDetection(
    Stream<Image> image, Stream<NormalizedRect> roi,
    const pose_landmarker::proto::PoseLandmarksDetectorGraphOptions&
        pose_landmarks_detector_graph_options,
    const HolisticPoseTrackingRequest& request, Graph& graph) {
  GenericNode& landmarks_graph = graph.AddNode(
      "mediapipe.tasks.vision.pose_landmarker."
      "SinglePoseLandmarksDetectorGraph");
  landmarks_graph
      .GetOptions<pose_landmarker::proto::PoseLandmarksDetectorGraphOptions>() =
      pose_landmarks_detector_graph_options;
  image >> landmarks_graph.In("IMAGE");
  roi >> landmarks_graph.In("NORM_RECT");

  PoseLandmarksResult result;
  if (request.landmarks) {
    result.landmarks =
        landmarks_graph.Out("LANDMARKS").Cast<NormalizedLandmarkList>();
    result.auxiliary_landmarks = landmarks_graph.Out("AUXILIARY_LANDMARKS")
                                     .Cast<NormalizedLandmarkList>();
  }
  if (request.world_landmarks) {
    result.world_landmarks =
        landmarks_graph.Out("WORLD_LANDMARKS").Cast<LandmarkList>();
  }
  if (request.segmentation_mask) {
    result.segmentation_mask =
        landmarks_graph.Out("SEGMENTATION_MASK").Cast<Image>();
  }
  return result;
}

}  // namespace

absl::StatusOr<HolisticPoseTrackingOutput>
TrackHolisticPoseUsingCustomPoseDetection(
    Stream<Image> image, PoseDetectionFn pose_detection_fn,
    const pose_landmarker::proto::PoseLandmarksDetectorGraphOptions&
        pose_landmarks_detector_graph_options,
    const HolisticPoseTrackingRequest& request, Graph& graph) {
  // Calculate ROI from scratch (pose detection) or reuse one from the
  // previous run if available.
  auto [previous_roi, set_previous_roi_fn] =
      GetLoopbackData<NormalizedRect>(/*tick=*/image, graph);
  auto is_previous_roi_available = IsPresent(previous_roi, graph);
  auto image_for_detection =
      DisallowIf(image, is_previous_roi_available, graph);
  MP_ASSIGN_OR_RETURN(auto pose_detections,
                      pose_detection_fn(image_for_detection, graph));
  auto roi_from_detections = CalculateRoiFromDetections(
      pose_detections, GetImageSize(image_for_detection, graph), graph);
  // Take first non-empty.
  auto roi = Merge(roi_from_detections, previous_roi, graph);

  // Calculate landmarks and other outputs (if requested) in the specified ROI.
  auto landmarks_detection_result = RunLandmarksDetection(
      image, roi, pose_landmarks_detector_graph_options,
      {
          // Landmarks are required for tracking, hence force-requesting them.
          /*.landmarks = */ true,
          /*.world_landmarks = */ request.world_landmarks,
          /*.segmentation_mask = */ request.segmentation_mask,
      },
      graph);
  RET_CHECK(landmarks_detection_result.landmarks.has_value() &&
            landmarks_detection_result.auxiliary_landmarks.has_value())
      << "Failed to calculate landmarks required for tracking.";

  // Split landmarks to pose landmarks and auxiliary landmarks.
  auto pose_landmarks_raw = *landmarks_detection_result.landmarks;
  auto auxiliary_landmarks = *landmarks_detection_result.auxiliary_landmarks;

  auto image_size = GetImageSize(image, graph);

  // TODO: b/305750053 - Apply adaptive crop by adding AdaptiveCropCalculator.

  // Calculate ROI from smoothed auxiliary landmarks.
  auto scale_roi = CalculateScaleRoiFromAuxiliaryLandmarks(auxiliary_landmarks,
                                                           image_size, graph);
  auto auxiliary_landmarks_smoothed = SmoothLandmarks(
      auxiliary_landmarks, image_size, scale_roi,
      {// Min cutoff 0.01 results into ~0.002 alpha in landmark EMA filter when
       // landmark is static.
       /*.min_cutoff = */ 0.01,
       // Beta 10.0 in combintation with min_cutoff 0.01 results into ~0.68
       // alpha in landmark EMA filter when landmark is moving fast.
       /*.beta = */ 10.0,
       // Derivative cutoff 1.0 results into ~0.17 alpha in landmark velocity
       // EMA filter.
       /*.derivate_cutoff = */ 1.0},
      graph);
  auto roi_from_auxiliary_landmarks = CalculateRoiFromAuxiliaryLandmarks(
      auxiliary_landmarks_smoothed, image_size, graph);

  // Make ROI from auxiliary landmarks to be used as "previous" ROI for a
  // subsequent run.
  set_previous_roi_fn(roi_from_auxiliary_landmarks);

  // Populate and smooth pose landmarks if corresponding output has been
  // requested.
  std::optional<Stream<NormalizedLandmarkList>> pose_landmarks;
  if (request.landmarks) {
    pose_landmarks = SmoothLandmarksVisibility(
        pose_landmarks_raw, /*low_pass_filter_alpha=*/0.1f, graph);
    pose_landmarks = SmoothLandmarks(
        *pose_landmarks, image_size, scale_roi,
        {// Min cutoff 0.05 results into ~0.01 alpha in landmark EMA filter when
         // landmark is static.
         /*.min_cutoff = */ 0.05f,
         // Beta 80.0 in combination with min_cutoff 0.05 results into ~0.94
         // alpha in landmark EMA filter when landmark is moving fast.
         /*.beta = */ 80.0f,
         // Derivative cutoff 1.0 results into ~0.17 alpha in landmark velocity
         // EMA filter.
         /*.derivate_cutoff = */ 1.0f},
        graph);
  }

  // Populate and smooth world landmarks if available.
  std::optional<Stream<LandmarkList>> world_landmarks;
  if (landmarks_detection_result.world_landmarks) {
    world_landmarks = SplitToRanges(*landmarks_detection_result.world_landmarks,
                                    /*ranges*/ {{0, 33}}, graph)[0];
    world_landmarks = SmoothLandmarksVisibility(
        *world_landmarks, /*low_pass_filter_alpha=*/0.1f, graph);
    world_landmarks = SmoothLandmarks(
        *world_landmarks,
        /*scale_roi=*/std::nullopt,
        {// Min cutoff 0.1 results into ~ 0.02 alpha in landmark EMA filter when
         // landmark is static.
         /*.min_cutoff = */ 0.1f,
         // Beta 40.0 in combination with min_cutoff 0.1 results into ~0.8
         // alpha in landmark EMA filter when landmark is moving fast.
         /*.beta = */ 40.0f,
         // Derivative cutoff 1.0 results into ~0.17 alpha in landmark velocity
         // EMA filter.
         /*.derivate_cutoff = */ 1.0f},
        graph);
  }

  // Populate and smooth segmentation mask if available.
  std::optional<Stream<Image>> segmentation_mask;
  if (landmarks_detection_result.segmentation_mask) {
    auto mask = *landmarks_detection_result.segmentation_mask;
    auto [prev_mask_as_img, set_prev_mask_as_img_fn] =
        GetLoopbackData<mediapipe::Image>(
            /*tick=*/*landmarks_detection_result.segmentation_mask, graph);
    auto mask_smoothed =
        SmoothSegmentationMask(mask, prev_mask_as_img,
                               /*combine_with_previous_ratio=*/0.7f, graph);
    set_prev_mask_as_img_fn(mask_smoothed);
    segmentation_mask = mask_smoothed;
  }

  return {{/*landmarks=*/pose_landmarks,
           /*world_landmarks=*/world_landmarks,
           /*segmentation_mask=*/segmentation_mask,
           /*debug_output=*/
           {/*auxiliary_landmarks=*/auxiliary_landmarks_smoothed,
            /*roi_from_landmarks=*/roi_from_auxiliary_landmarks,
            /*detections*/ pose_detections}}};
}

absl::StatusOr<HolisticPoseTrackingOutput> TrackHolisticPose(
    Stream<Image> image,
    const pose_detector::proto::PoseDetectorGraphOptions&
        pose_detector_graph_options,
    const pose_landmarker::proto::PoseLandmarksDetectorGraphOptions&
        pose_landmarks_detector_graph_options,
    const HolisticPoseTrackingRequest& request, Graph& graph) {
  PoseDetectionFn pose_detection_fn = [&pose_detector_graph_options](
                                          Stream<Image> image, Graph& graph)
      -> absl::StatusOr<Stream<std::vector<mediapipe::Detection>>> {
    GenericNode& pose_detector =
        graph.AddNode("mediapipe.tasks.vision.pose_detector.PoseDetectorGraph");
    pose_detector.GetOptions<pose_detector::proto::PoseDetectorGraphOptions>() =
        pose_detector_graph_options;
    image >> pose_detector.In("IMAGE");
    return pose_detector.Out("DETECTIONS")
        .Cast<std::vector<mediapipe::Detection>>();
  };
  return TrackHolisticPoseUsingCustomPoseDetection(
      image, pose_detection_fn, pose_landmarks_detector_graph_options, request,
      graph);
}

}  // namespace holistic_landmarker
}  // namespace vision
}  // namespace tasks
}  // namespace mediapipe
