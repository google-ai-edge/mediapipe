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

#ifndef MEDIAPIPE_TASKS_CC_VISION_HOLISTIC_LANDMARKER_HOLISTIC_HAND_TRACKING_H_
#define MEDIAPIPE_TASKS_CC_VISION_HOLISTIC_LANDMARKER_HOLISTIC_HAND_TRACKING_H_

#include <optional>

#include "absl/status/statusor.h"
#include "mediapipe/framework/api2/builder.h"
#include "mediapipe/framework/formats/image.h"
#include "mediapipe/framework/formats/landmark.pb.h"
#include "mediapipe/framework/formats/rect.pb.h"
#include "mediapipe/tasks/cc/vision/hand_landmarker/proto/hand_landmarks_detector_graph_options.pb.h"
#include "mediapipe/tasks/cc/vision/hand_landmarker/proto/hand_roi_refinement_graph_options.pb.h"

namespace mediapipe {
namespace tasks {
namespace vision {
namespace holistic_landmarker {

struct PoseIndices {
  int wrist_idx;
  int pinky_idx;
  int index_idx;
};

struct HolisticHandTrackingRequest {
  bool landmarks = false;
  bool world_landmarks = false;
};

struct HolisticHandTrackingOutput {
  std::optional<api2::builder::Stream<mediapipe::NormalizedLandmarkList>>
      landmarks;
  std::optional<api2::builder::Stream<mediapipe::LandmarkList>> world_landmarks;

  struct DebugOutput {
    api2::builder::Stream<mediapipe::NormalizedRect> roi_from_pose;
    api2::builder::Stream<mediapipe::NormalizedRect> roi_from_recrop;
    api2::builder::Stream<mediapipe::NormalizedRect> tracking_roi;
  };

  DebugOutput debug_output;
};

// Updates @graph to track a single hand in @image based on pose landmarks.
//
// To track single hand this subgraph uses pose palm landmarks to obtain
// approximate hand location, refines it with re-crop model and then runs hand
// landmarks model. It can also reuse hand ROI from the previous frame if hand
// hasn't moved too much.
//
// @image - ImageFrame/GpuBuffer to track a single hand in.
// @pose_landmarks - Pose landmarks to derive initial hand location from.
// @pose_world_landmarks - Pose world landmarks to align hand world landmarks
//   wrist with.
// @ hand_landmarks_detector_graph_options - Options of the
// HandLandmarksDetectorGraph used to detect the hand landmarks.
// @ hand_roi_refinement_graph_options - Options of HandRoiRefinementGraph used
// to refine the hand RoIs got from Pose landmarks.
// @request - object to request specific hand tracking outputs.
//   NOTE: Outputs that were not requested won't be returned and corresponding
//   parts of the graph won't be genertaed.
// @graph - graph to update.
absl::StatusOr<HolisticHandTrackingOutput> TrackHolisticHand(
    api2::builder::Stream<Image> image,
    api2::builder::Stream<mediapipe::NormalizedLandmarkList> pose_landmarks,
    api2::builder::Stream<mediapipe::LandmarkList> pose_world_landmarks,
    const hand_landmarker::proto::HandLandmarksDetectorGraphOptions&
        hand_landmarks_detector_graph_options,
    const hand_landmarker::proto::HandRoiRefinementGraphOptions&
        hand_roi_refinement_graph_options,
    const PoseIndices& pose_indices, const HolisticHandTrackingRequest& request,
    mediapipe::api2::builder::Graph& graph);

}  // namespace holistic_landmarker
}  // namespace vision
}  // namespace tasks
}  // namespace mediapipe

#endif  // MEDIAPIPE_TASKS_CC_VISION_HOLISTIC_LANDMARKER_HOLISTIC_HAND_TRACKING_H_
