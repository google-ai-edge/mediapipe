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

#ifndef MEDIAPIPE_TASKS_CC_VISION_HOLISTIC_LANDMARKER_HOLISTIC_POSE_TRACKING_H_
#define MEDIAPIPE_TASKS_CC_VISION_HOLISTIC_LANDMARKER_HOLISTIC_POSE_TRACKING_H_

#include <functional>
#include <optional>
#include <vector>

#include "absl/status/statusor.h"
#include "mediapipe/framework/api2/builder.h"
#include "mediapipe/framework/formats/detection.pb.h"
#include "mediapipe/framework/formats/image.h"
#include "mediapipe/framework/formats/landmark.pb.h"
#include "mediapipe/framework/formats/rect.pb.h"
#include "mediapipe/tasks/cc/vision/pose_detector/proto/pose_detector_graph_options.pb.h"
#include "mediapipe/tasks/cc/vision/pose_landmarker/proto/pose_landmarks_detector_graph_options.pb.h"

namespace mediapipe {
namespace tasks {
namespace vision {
namespace holistic_landmarker {

// Type of pose detection function that can be used to customize pose tracking,
// by supplying the function into a corresponding `TrackPose` function overload.
//
// Function should update provided graph with node/nodes that accept image
// stream and produce stream of detections.
using PoseDetectionFn = std::function<
    absl::StatusOr<api2::builder::Stream<std::vector<mediapipe::Detection>>>(
        api2::builder::Stream<Image>, api2::builder::Graph&)>;

struct HolisticPoseTrackingRequest {
  bool landmarks = false;
  bool world_landmarks = false;
  bool segmentation_mask = false;
};

struct HolisticPoseTrackingOutput {
  std::optional<api2::builder::Stream<mediapipe::NormalizedLandmarkList>>
      landmarks;
  std::optional<api2::builder::Stream<mediapipe::LandmarkList>> world_landmarks;
  std::optional<api2::builder::Stream<Image>> segmentation_mask;

  struct DebugOutput {
    api2::builder::Stream<mediapipe::NormalizedLandmarkList>
        auxiliary_landmarks;
    api2::builder::Stream<NormalizedRect> roi_from_landmarks;
    api2::builder::Stream<std::vector<mediapipe::Detection>> detections;
  };

  DebugOutput debug_output;
};

// Updates @graph to track pose in @image.
//
// @image - ImageFrame/GpuBuffer to track pose in.
// @pose_detection_fn - pose detection function that takes @image as input and
//   produces stream of pose detections.
// @pose_landmarks_detector_graph_options - options of the
//   PoseLandmarksDetectorGraph used to detect the pose landmarks.
// @request - object to request specific pose tracking outputs.
//   NOTE: Outputs that were not requested won't be returned and corresponding
//   parts of the graph won't be genertaed all.
// @graph - graph to update.
absl::StatusOr<HolisticPoseTrackingOutput>
TrackHolisticPoseUsingCustomPoseDetection(
    api2::builder::Stream<Image> image, PoseDetectionFn pose_detection_fn,
    const pose_landmarker::proto::PoseLandmarksDetectorGraphOptions&
        pose_landmarks_detector_graph_options,
    const HolisticPoseTrackingRequest& request, api2::builder::Graph& graph);

// Updates @graph to track pose in @image.
//
// @image - ImageFrame/GpuBuffer to track pose in.
// @pose_detector_graph_options - options of the PoseDetectorGraph used to
// detect the pose.
// @pose_landmarks_detector_graph_options - options of the
//   PoseLandmarksDetectorGraph used to detect the pose landmarks.
// @request - object to request specific pose tracking outputs.
//   NOTE: Outputs that were not requested won't be returned and corresponding
//   parts of the graph won't be genertaed all.
// @graph - graph to update.
absl::StatusOr<HolisticPoseTrackingOutput> TrackHolisticPose(
    api2::builder::Stream<Image> image,
    const pose_detector::proto::PoseDetectorGraphOptions&
        pose_detector_graph_options,
    const pose_landmarker::proto::PoseLandmarksDetectorGraphOptions&
        pose_landmarks_detector_graph_options,
    const HolisticPoseTrackingRequest& request, api2::builder::Graph& graph);

}  // namespace holistic_landmarker
}  // namespace vision
}  // namespace tasks
}  // namespace mediapipe

#endif  // MEDIAPIPE_TASKS_CC_VISION_HOLISTIC_LANDMARKER_HOLISTIC_POSE_TRACKING_H_
