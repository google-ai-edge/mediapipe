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

#import "mediapipe/tasks/ios/vision/pose_landmarker/sources/MPPPoseLandmarkerOptions.h"

@implementation MPPPoseLandmarkerOptions

- (instancetype)init {
  self = [super init];
  if (self) {
    _numPoses = 1;
    _minPoseDetectionConfidence = 0.5f;
    _minPosePresenceConfidence = 0.5f;
    _minTrackingConfidence = 0.5f;
    _shouldOutputSegmentationMasks = NO;
  }
  return self;
}

- (id)copyWithZone:(NSZone *)zone {
  MPPPoseLandmarkerOptions *poseLandmarkerOptions = [super copyWithZone:zone];

  poseLandmarkerOptions.runningMode = self.runningMode;
  poseLandmarkerOptions.numPoses = self.numPoses;
  poseLandmarkerOptions.minPoseDetectionConfidence = self.minPoseDetectionConfidence;
  poseLandmarkerOptions.minPosePresenceConfidence = self.minPosePresenceConfidence;
  poseLandmarkerOptions.minTrackingConfidence = self.minTrackingConfidence;
  poseLandmarkerOptions.shouldOutputSegmentationMasks = self.shouldOutputSegmentationMasks;
  poseLandmarkerOptions.poseLandmarkerLiveStreamDelegate = self.poseLandmarkerLiveStreamDelegate;

  return poseLandmarkerOptions;
}

@end
