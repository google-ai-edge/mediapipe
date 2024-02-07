// Copyright 2024 The MediaPipe Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#import "mediapipe/tasks/ios/vision/holistic_landmarker/utils/sources/MPPHolisticLandmarkerOptions+Helpers.h"

#import "mediapipe/tasks/ios/common/utils/sources/NSString+Helpers.h"
#import "mediapipe/tasks/ios/core/utils/sources/MPPBaseOptions+Helpers.h"

#include "mediapipe/tasks/cc/vision/face_detector/proto/face_detector_graph_options.pb.h"
#include "mediapipe/tasks/cc/vision/face_landmarker/proto/face_landmarks_detector_graph_options.pb.h"
#include "mediapipe/tasks/cc/vision/hand_landmarker/proto/hand_landmarks_detector_graph_options.pb.h"
#include "mediapipe/tasks/cc/vision/holistic_landmarker/proto/holistic_landmarker_graph_options.pb.h"
#include "mediapipe/tasks/cc/vision/pose_detector/proto/pose_detector_graph_options.pb.h"
#include "mediapipe/tasks/cc/vision/pose_landmarker/proto/pose_landmarks_detector_graph_options.pb.h"

namespace {
using ::google::protobuf::Any;
using FaceDetectorGraphOptionsProto =
    ::mediapipe::tasks::vision::face_detector::proto::FaceDetectorGraphOptions;
using FaceLandmarksDetectorGraphOptionsProto =
    ::mediapipe::tasks::vision::face_landmarker::proto::FaceLandmarksDetectorGraphOptions;
using HandLandmarksDetectorGraphOptionsProto =
    ::mediapipe::tasks::vision::hand_landmarker::proto::HandLandmarksDetectorGraphOptions;
using HolisticLandmarkerGraphOptionsProto =
    ::mediapipe::tasks::vision::holistic_landmarker::proto::HolisticLandmarkerGraphOptions;
using PoseDetectorGraphOptionsProto =
    ::mediapipe::tasks::vision::pose_detector::proto::PoseDetectorGraphOptions;
using PoseLandmarksDetectorGraphOptionsProto =
    ::mediapipe::tasks::vision::pose_landmarker::proto::PoseLandmarksDetectorGraphOptions;
}  // namespace

@implementation MPPHolisticLandmarkerOptions (Helpers)

- (void)copyToAnyProto:(Any *)optionsProto {
  HolisticLandmarkerGraphOptionsProto holisticLandmarkerGraphOptions;

  [self.baseOptions copyToProto:holisticLandmarkerGraphOptions.mutable_base_options()
              withUseStreamMode:self.runningMode != MPPRunningModeImage];

  FaceLandmarksDetectorGraphOptionsProto *faceLandmarksDetectorGraphOptions =
      holisticLandmarkerGraphOptions.mutable_face_landmarks_detector_graph_options();
  faceLandmarksDetectorGraphOptions->set_min_detection_confidence(self.minFacePresenceConfidence);

  FaceDetectorGraphOptionsProto *faceDetectorGraphOptions =
      holisticLandmarkerGraphOptions.mutable_face_detector_graph_options();
  faceDetectorGraphOptions->set_min_detection_confidence(self.minFaceDetectionConfidence);
  faceDetectorGraphOptions->set_min_suppression_threshold(self.minFaceSuppressionThreshold);

  PoseLandmarksDetectorGraphOptionsProto *poseLandmarksDetectorGraphOptions =
      holisticLandmarkerGraphOptions.mutable_pose_landmarks_detector_graph_options();
  poseLandmarksDetectorGraphOptions->set_min_detection_confidence(self.minPosePresenceConfidence);

  PoseDetectorGraphOptionsProto *poseDetectorGraphOptions =
      holisticLandmarkerGraphOptions.mutable_pose_detector_graph_options();
  poseDetectorGraphOptions->set_min_detection_confidence(self.minPoseDetectionConfidence);
  poseDetectorGraphOptions->set_min_suppression_threshold(self.minPoseSuppressionThreshold);

  HandLandmarksDetectorGraphOptionsProto *handLandmarsDetectorGraphOptions =
      holisticLandmarkerGraphOptions.mutable_hand_landmarks_detector_graph_options();
  handLandmarsDetectorGraphOptions->set_min_detection_confidence(self.minHandLandmarksConfidence);

  optionsProto->PackFrom(holisticLandmarkerGraphOptions);
}

@end
