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

#import "mediapipe/tasks/ios/vision/hand_landmarker/utils/sources/MPPHandLandmarkerOptions+Helpers.h"

#import "mediapipe/tasks/ios/core/utils/sources/MPPBaseOptions+Helpers.h"

#include "mediapipe/tasks/c/vision/hand_landmarker/hand_landmarker.h"

@implementation MPPHandLandmarkerOptions (Helpers)

- (void)copyToCOptions:(MpHandLandmarkerOptions *)cOptions {
  if (!cOptions) {
    return;
  }
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

  cOptions->num_hands = static_cast<int>(self.numHands);
  cOptions->min_hand_detection_confidence = self.minHandDetectionConfidence;
  cOptions->min_hand_presence_confidence = self.minHandPresenceConfidence;
  cOptions->min_tracking_confidence = self.minTrackingConfidence;
  cOptions->result_callback = nullptr;
}

+ (void)freeCOptions:(MpHandLandmarkerOptions *)cOptions {
  if (!cOptions) {
    return;
  }
  [MPPBaseOptions freeCOptions:&cOptions->base_options];
}

@end
