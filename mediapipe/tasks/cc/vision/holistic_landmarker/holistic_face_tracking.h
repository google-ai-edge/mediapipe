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

#ifndef MEDIAPIPE_TASKS_CC_VISION_HOLISTIC_LANDMARKER_HOLISTIC_FACE_TRACKING_H_
#define MEDIAPIPE_TASKS_CC_VISION_HOLISTIC_LANDMARKER_HOLISTIC_FACE_TRACKING_H_

#include <optional>

#include "absl/status/statusor.h"
#include "mediapipe/framework/api2/builder.h"
#include "mediapipe/framework/formats/classification.pb.h"
#include "mediapipe/framework/formats/image.h"
#include "mediapipe/framework/formats/landmark.pb.h"
#include "mediapipe/framework/formats/rect.pb.h"
#include "mediapipe/tasks/cc/vision/face_detector/proto/face_detector_graph_options.pb.h"
#include "mediapipe/tasks/cc/vision/face_landmarker/proto/face_landmarks_detector_graph_options.pb.h"

namespace mediapipe {
namespace tasks {
namespace vision {
namespace holistic_landmarker {

struct HolisticFaceTrackingRequest {
  bool classifications = false;
};

struct HolisticFaceTrackingOutput {
  std::optional<api2::builder::Stream<mediapipe::NormalizedLandmarkList>>
      landmarks;
  std::optional<api2::builder::Stream<mediapipe::ClassificationList>>
      classifications;

  struct DebugOutput {
    api2::builder::Stream<mediapipe::NormalizedRect> roi_from_pose;
    api2::builder::Stream<mediapipe::NormalizedRect> roi_from_detection;
    api2::builder::Stream<mediapipe::NormalizedRect> tracking_roi;
  };

  DebugOutput debug_output;
};

// Updates @graph to track a single face in @image based on pose landmarks.
//
// To track single face this subgraph uses pose face landmarks to obtain
// approximate face location, refines it with face detector model and then runs
// face landmarks model. It can also reuse face ROI from the previous frame if
// face hasn't moved too much.
//
// @image - Image to track a single face in.
// @pose_face_landmarks - Pose face landmarks to derive initial face location
//  from.
// @face_detector_graph_options - face detector graph options used to detect the
//   face within the RoI constructed from the pose face landmarks.
// @face_landmarks_detector_graph_options - face landmarks detector graph
//   options used to detect face landmarks within the RoI given be the face
//   detector graph.
// @request - object to request specific face tracking outputs.
//   NOTE: Outputs that were not requested won't be returned and corresponding
//   parts of the graph won't be genertaed.
// @graph - graph to update.
absl::StatusOr<HolisticFaceTrackingOutput> TrackHolisticFace(
    api2::builder::Stream<Image> image,
    api2::builder::Stream<mediapipe::NormalizedLandmarkList>
        pose_face_landmarks,
    const face_detector::proto::FaceDetectorGraphOptions&
        face_detector_graph_options,
    const face_landmarker::proto::FaceLandmarksDetectorGraphOptions&
        face_landmarks_detector_graph_options,
    const HolisticFaceTrackingRequest& request,
    mediapipe::api2::builder::Graph& graph);

}  // namespace holistic_landmarker
}  // namespace vision
}  // namespace tasks
}  // namespace mediapipe

#endif  // MEDIAPIPE_TASKS_CC_VISION_HOLISTIC_LANDMARKER_HOLISTIC_FACE_TRACKING_H_
