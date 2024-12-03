// Copyright 2023 The MediaPipe Authors.
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

#import "mediapipe/tasks/ios/vision/pose_landmarker/utils/sources/MPPPoseLandmarkerOptions+Helpers.h"

#import "mediapipe/tasks/ios/common/utils/sources/NSString+Helpers.h"
#import "mediapipe/tasks/ios/core/utils/sources/MPPBaseOptions+Helpers.h"

#include "mediapipe/tasks/cc/vision/pose_landmarker/proto/pose_landmarker_graph_options.pb.h"
#include "mediapipe/tasks/cc/vision/pose_landmarker/proto/pose_landmarks_detector_graph_options.pb.h"
#include "mediapipe/tasks/cc/vision/pose_detector/proto/pose_detector_graph_options.pb.h"

using CalculatorOptionsProto = ::mediapipe::CalculatorOptions;
using PoseDetectorGraphOptionsProto =
    ::mediapipe::tasks::vision::pose_detector::proto::PoseDetectorGraphOptions;
using PoseLandmarksDetectorGraphOptionsProto =
    ::mediapipe::tasks::vision::pose_landmarker::proto::PoseLandmarksDetectorGraphOptions;
using PoseLandmarkerGraphOptionsProto =
    ::mediapipe::tasks::vision::pose_landmarker::proto::PoseLandmarkerGraphOptions;

@implementation MPPPoseLandmarkerOptions (Helpers)

- (void)copyToProto:(CalculatorOptionsProto *)optionsProto {
  PoseLandmarkerGraphOptionsProto *poseLandmarkerGraphOptions =
      optionsProto->MutableExtension(PoseLandmarkerGraphOptionsProto::ext);
  poseLandmarkerGraphOptions->Clear();

  [self.baseOptions copyToProto:poseLandmarkerGraphOptions->mutable_base_options()
              withUseStreamMode:self.runningMode != MPPRunningModeImage];
  poseLandmarkerGraphOptions->set_min_tracking_confidence(self.minTrackingConfidence);

  PoseLandmarksDetectorGraphOptionsProto *poseLandmarksDetectorGraphOptions =
      poseLandmarkerGraphOptions->mutable_pose_landmarks_detector_graph_options();
  poseLandmarksDetectorGraphOptions->set_min_detection_confidence(self.minPosePresenceConfidence);

  PoseDetectorGraphOptionsProto *poseDetectorGraphOptions =
      poseLandmarkerGraphOptions->mutable_pose_detector_graph_options();
  poseDetectorGraphOptions->set_num_poses(self.numPoses);
  poseDetectorGraphOptions->set_min_detection_confidence(self.minPoseDetectionConfidence);
}

@end
