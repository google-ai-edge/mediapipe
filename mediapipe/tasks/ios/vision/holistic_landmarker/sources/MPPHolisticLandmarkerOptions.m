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

#import "mediapipe/tasks/ios/vision/holistic_landmarker/sources/MPPHolisticLandmarkerOptions.h"

@implementation MPPHolisticLandmarkerOptions

- (instancetype)init {
  self = [super init];
  if (self) {
    _minFaceDetectionConfidence = 0.5f;
    _minFaceSuppressionThreshold = 0.3f;
    _minFacePresenceConfidence = 0.5f;
    _minPoseDetectionConfidence = 0.5f;
    _minPoseSuppressionThreshold = 0.3f;
    _minPosePresenceConfidence = 0.5f;
    _outputFaceBlendshapes = NO;
    _outputPoseSegmentationMasks = NO;
    _minHandLandmarksConfidence = 0.5f;
  }
  return self;
}

- (id)copyWithZone:(NSZone *)zone {
  MPPHolisticLandmarkerOptions *holisticLandmarkerOptions = [super copyWithZone:zone];

  holisticLandmarkerOptions.runningMode = self.runningMode;
  holisticLandmarkerOptions.minFaceDetectionConfidence = self.minFaceDetectionConfidence;
  holisticLandmarkerOptions.minFaceSuppressionThreshold = self.minFaceSuppressionThreshold;
  holisticLandmarkerOptions.minFacePresenceConfidence = self.minFacePresenceConfidence;
  holisticLandmarkerOptions.minPoseDetectionConfidence = self.minPoseDetectionConfidence;
  holisticLandmarkerOptions.minPoseSuppressionThreshold = self.minPoseSuppressionThreshold;
  holisticLandmarkerOptions.minPosePresenceConfidence = self.minPosePresenceConfidence;
  holisticLandmarkerOptions.outputFaceBlendshapes = self.outputFaceBlendshapes;
  holisticLandmarkerOptions.outputPoseSegmentationMasks = self.outputPoseSegmentationMasks;
  holisticLandmarkerOptions.minHandLandmarksConfidence = self.minHandLandmarksConfidence;

  return holisticLandmarkerOptions;
}

@end
