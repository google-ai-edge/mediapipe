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

#import "mediapipe/tasks/ios/core/utils/sources/MPPBaseOptions+Helpers.h"

#include "mediapipe/tasks/c/vision/holistic_landmarker/holistic_landmarker.h"

@implementation MPPHolisticLandmarkerOptions (Helpers)

- (void)copyToCOptions:(MpHolisticLandmarkerOptions *)cOptions {
  if (!cOptions) {
    return;
  }
  *cOptions = {};
  [self.baseOptions copyToCOptions:&cOptions->base_options];

  switch (self.runningMode) {
    case MPPRunningModeImage:
      cOptions->running_mode = MP_RUNNING_MODE_IMAGE;
      break;
    case MPPRunningModeVideo:
      cOptions->running_mode = MP_RUNNING_MODE_VIDEO;
      break;
    case MPPRunningModeLiveStream:
      cOptions->running_mode = MP_RUNNING_MODE_LIVE_STREAM;
      break;
  }

  cOptions->min_face_detection_confidence = self.minFaceDetectionConfidence;
  cOptions->min_face_suppression_threshold = self.minFaceSuppressionThreshold;
  cOptions->min_face_presence_confidence = self.minFacePresenceConfidence;
  cOptions->min_hand_landmarks_confidence = self.minHandLandmarksConfidence;
  cOptions->min_pose_detection_confidence = self.minPoseDetectionConfidence;
  cOptions->min_pose_suppression_threshold = self.minPoseSuppressionThreshold;
  cOptions->min_pose_presence_confidence = self.minPosePresenceConfidence;
  cOptions->output_face_blendshapes = self.outputFaceBlendshapes;
  cOptions->output_pose_segmentation_masks = self.outputPoseSegmentationMasks;
  cOptions->result_callback = nullptr;
}

+ (void)freeCOptions:(MpHolisticLandmarkerOptions *)cOptions {
  if (!cOptions) {
    return;
  }
  [MPPBaseOptions freeCOptions:&cOptions->base_options];
}

@end
