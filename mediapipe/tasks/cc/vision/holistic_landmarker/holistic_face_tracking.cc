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

#include "mediapipe/tasks/cc/vision/holistic_landmarker/holistic_face_tracking.h"

#include <functional>
#include <optional>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "mediapipe/framework/api2/builder.h"
#include "mediapipe/framework/api2/stream/detections_to_rects.h"
#include "mediapipe/framework/api2/stream/image_size.h"
#include "mediapipe/framework/api2/stream/landmarks_to_detection.h"
#include "mediapipe/framework/api2/stream/loopback.h"
#include "mediapipe/framework/api2/stream/rect_transformation.h"
#include "mediapipe/framework/formats/classification.pb.h"
#include "mediapipe/framework/formats/detection.pb.h"
#include "mediapipe/framework/formats/image.h"
#include "mediapipe/framework/formats/landmark.pb.h"
#include "mediapipe/framework/formats/rect.pb.h"
#include "mediapipe/framework/port/status_macros.h"
#include "mediapipe/modules/holistic_landmark/calculators/roi_tracking_calculator.pb.h"
#include "mediapipe/tasks/cc/vision/face_detector/proto/face_detector_graph_options.pb.h"
#include "mediapipe/tasks/cc/vision/face_landmarker/proto/face_blendshapes_graph_options.pb.h"
#include "mediapipe/tasks/cc/vision/face_landmarker/proto/face_landmarks_detector_graph_options.pb.h"

namespace mediapipe {
namespace tasks {
namespace vision {
namespace holistic_landmarker {

namespace {

using ::mediapipe::NormalizedRect;
using ::mediapipe::api2::builder::ConvertDetectionsToRectUsingKeypoints;
using ::mediapipe::api2::builder::ConvertDetectionToRect;
using ::mediapipe::api2::builder::ConvertLandmarksToDetection;
using ::mediapipe::api2::builder::GetImageSize;
using ::mediapipe::api2::builder::GetLoopbackData;
using ::mediapipe::api2::builder::Graph;
using ::mediapipe::api2::builder::Scale;
using ::mediapipe::api2::builder::ScaleAndMakeSquare;
using ::mediapipe::api2::builder::Stream;

struct FaceLandmarksResult {
  std::optional<Stream<NormalizedLandmarkList>> landmarks;
  std::optional<Stream<ClassificationList>> classifications;
};

absl::Status ValidateGraphOptions(
    const face_detector::proto::FaceDetectorGraphOptions&
        face_detector_graph_options,
    const face_landmarker::proto::FaceLandmarksDetectorGraphOptions&
        face_landmarks_detector_graph_options,
    const HolisticFaceTrackingRequest& request) {
  if (face_detector_graph_options.num_faces() != 1) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "Only support num_faces to be 1, but got num_faces = %d.",
        face_detector_graph_options.num_faces()));
  }
  if (request.classifications && !face_landmarks_detector_graph_options
                                      .has_face_blendshapes_graph_options()) {
    return absl::InvalidArgumentError(
        "Blendshapes detection is requested, but "
        "face_blendshapes_graph_options is not configured.");
  }
  return absl::OkStatus();
}

Stream<NormalizedRect> GetFaceRoiFromPoseFaceLandmarks(
    Stream<NormalizedLandmarkList> pose_face_landmarks,
    Stream<std::pair<int, int>> image_size, Graph& graph) {
  Stream<mediapipe::Detection> detection =
      ConvertLandmarksToDetection(pose_face_landmarks, graph);

  // Refer the pose face landmarks indices here:
  // https://developers.google.com/mediapipe/solutions/vision/pose_landmarker#pose_landmarker_model
  Stream<NormalizedRect> rect = ConvertDetectionToRect(
      detection, image_size, /*start_keypoint_index=*/5,
      /*end_keypoint_index=*/2, /*target_angle=*/0, graph);

  // Scale the face RoI from a tight rect enclosing the pose face landmarks, to
  // a larger square so that the whole face is within the RoI.
  return ScaleAndMakeSquare(rect, image_size,
                            /*scale_x_factor=*/3.0,
                            /*scale_y_factor=*/3.0, graph);
}

Stream<NormalizedRect> GetFaceRoiFromFaceLandmarks(
    Stream<NormalizedLandmarkList> face_landmarks,
    Stream<std::pair<int, int>> image_size, Graph& graph) {
  Stream<mediapipe::Detection> detection =
      ConvertLandmarksToDetection(face_landmarks, graph);

  Stream<NormalizedRect> rect = ConvertDetectionToRect(
      detection, image_size, /*start_keypoint_index=*/33,
      /*end_keypoint_index=*/263, /*target_angle=*/0, graph);

  return Scale(rect, image_size,
               /*scale_x_factor=*/1.5,
               /*scale_y_factor=*/1.5, graph);
}

Stream<std::vector<Detection>> GetFaceDetections(
    Stream<Image> image, Stream<NormalizedRect> roi,
    const face_detector::proto::FaceDetectorGraphOptions&
        face_detector_graph_options,
    Graph& graph) {
  auto& face_detector_graph =
      graph.AddNode("mediapipe.tasks.vision.face_detector.FaceDetectorGraph");
  face_detector_graph
      .GetOptions<face_detector::proto::FaceDetectorGraphOptions>() =
      face_detector_graph_options;
  image >> face_detector_graph.In("IMAGE");
  roi >> face_detector_graph.In("NORM_RECT");
  return face_detector_graph.Out("DETECTIONS").Cast<std::vector<Detection>>();
}

Stream<NormalizedRect> GetFaceRoiFromFaceDetections(
    Stream<std::vector<Detection>> face_detections,
    Stream<std::pair<int, int>> image_size, Graph& graph) {
  // Convert detection to rect.
  Stream<NormalizedRect> rect = ConvertDetectionsToRectUsingKeypoints(
      face_detections, image_size, /*start_keypoint_index=*/0,
      /*end_keypoint_index=*/1, /*target_angle=*/0, graph);

  return ScaleAndMakeSquare(rect, image_size,
                            /*scale_x_factor=*/2.0,
                            /*scale_y_factor=*/2.0, graph);
}

Stream<NormalizedRect> TrackFaceRoi(
    Stream<NormalizedLandmarkList> prev_landmarks, Stream<NormalizedRect> roi,
    Stream<std::pair<int, int>> image_size, Graph& graph) {
  // Gets face ROI from previous frame face landmarks.
  Stream<NormalizedRect> prev_roi =
      GetFaceRoiFromFaceLandmarks(prev_landmarks, image_size, graph);

  auto& tracking_node = graph.AddNode("RoiTrackingCalculator");
  auto& tracking_node_opts =
      tracking_node.GetOptions<RoiTrackingCalculatorOptions>();
  auto* rect_requirements = tracking_node_opts.mutable_rect_requirements();
  rect_requirements->set_rotation_degrees(15.0);
  rect_requirements->set_translation(0.1);
  rect_requirements->set_scale(0.3);
  auto* landmarks_requirements =
      tracking_node_opts.mutable_landmarks_requirements();
  landmarks_requirements->set_recrop_rect_margin(-0.2);
  prev_landmarks.ConnectTo(tracking_node.In("PREV_LANDMARKS"));
  prev_roi.ConnectTo(tracking_node.In("PREV_LANDMARKS_RECT"));
  roi.ConnectTo(tracking_node.In("RECROP_RECT"));
  image_size.ConnectTo(tracking_node.In("IMAGE_SIZE"));
  return tracking_node.Out("TRACKING_RECT").Cast<NormalizedRect>();
}

FaceLandmarksResult GetFaceLandmarksDetection(
    Stream<Image> image, Stream<NormalizedRect> roi,
    Stream<std::pair<int, int>> image_size,
    const face_landmarker::proto::FaceLandmarksDetectorGraphOptions&
        face_landmarks_detector_graph_options,
    const HolisticFaceTrackingRequest& request, Graph& graph) {
  FaceLandmarksResult result;
  auto& face_landmarks_detector_graph = graph.AddNode(
      "mediapipe.tasks.vision.face_landmarker."
      "SingleFaceLandmarksDetectorGraph");
  face_landmarks_detector_graph
      .GetOptions<face_landmarker::proto::FaceLandmarksDetectorGraphOptions>() =
      face_landmarks_detector_graph_options;
  image >> face_landmarks_detector_graph.In("IMAGE");
  roi >> face_landmarks_detector_graph.In("NORM_RECT");
  auto landmarks = face_landmarks_detector_graph.Out("NORM_LANDMARKS")
                       .Cast<NormalizedLandmarkList>();
  result.landmarks = landmarks;
  if (request.classifications) {
    auto& blendshapes_graph = graph.AddNode(
        "mediapipe.tasks.vision.face_landmarker.FaceBlendshapesGraph");
    blendshapes_graph
        .GetOptions<face_landmarker::proto::FaceBlendshapesGraphOptions>() =
        face_landmarks_detector_graph_options.face_blendshapes_graph_options();
    landmarks >> blendshapes_graph.In("LANDMARKS");
    image_size >> blendshapes_graph.In("IMAGE_SIZE");
    result.classifications =
        blendshapes_graph.Out("BLENDSHAPES").Cast<ClassificationList>();
  }
  return result;
}

}  // namespace

absl::StatusOr<HolisticFaceTrackingOutput> TrackHolisticFace(
    Stream<Image> image, Stream<NormalizedLandmarkList> pose_face_landmarks,
    const face_detector::proto::FaceDetectorGraphOptions&
        face_detector_graph_options,
    const face_landmarker::proto::FaceLandmarksDetectorGraphOptions&
        face_landmarks_detector_graph_options,
    const HolisticFaceTrackingRequest& request, Graph& graph) {
  MP_RETURN_IF_ERROR(ValidateGraphOptions(face_detector_graph_options,
                                          face_landmarks_detector_graph_options,
                                          request));

  // Extracts image size from the input images.
  Stream<std::pair<int, int>> image_size = GetImageSize(image, graph);

  // Gets face ROI from pose face landmarks.
  Stream<NormalizedRect> roi_from_pose =
      GetFaceRoiFromPoseFaceLandmarks(pose_face_landmarks, image_size, graph);

  // Detects faces within ROI of pose face.
  Stream<std::vector<Detection>> face_detections = GetFaceDetections(
      image, roi_from_pose, face_detector_graph_options, graph);

  // Gets face ROI from face detector.
  Stream<NormalizedRect> roi_from_detection =
      GetFaceRoiFromFaceDetections(face_detections, image_size, graph);

  // Loop for previous frame landmarks.
  auto [prev_landmarks, set_prev_landmarks_fn] =
      GetLoopbackData<NormalizedLandmarkList>(/*tick=*/image_size, graph);

  // Tracks face ROI.
  auto tracking_roi =
      TrackFaceRoi(prev_landmarks, roi_from_detection, image_size, graph);

  // Predicts face landmarks.
  auto landmarks_detection_result = GetFaceLandmarksDetection(
      image, tracking_roi, image_size, face_landmarks_detector_graph_options,
      request, graph);

  // Sets previous landmarks for ROI tracking.
  set_prev_landmarks_fn(landmarks_detection_result.landmarks.value());

  return {{/*landmarks=*/landmarks_detection_result.landmarks,
           /*classifications=*/landmarks_detection_result.classifications,
           /*debug_output=*/
           {
               /*roi_from_pose=*/roi_from_pose,
               /*roi_from_detection=*/roi_from_detection,
               /*tracking_roi=*/tracking_roi,
           }}};
}

}  // namespace holistic_landmarker
}  // namespace vision
}  // namespace tasks
}  // namespace mediapipe
