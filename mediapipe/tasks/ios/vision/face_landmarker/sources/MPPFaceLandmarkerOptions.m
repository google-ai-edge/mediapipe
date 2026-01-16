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

#import "mediapipe/tasks/ios/vision/face_landmarker/sources/MPPFaceLandmarkerOptions.h"

@implementation MPPFaceLandmarkerOptions

- (instancetype)init {
  self = [super init];
  if (self) {
    _numFaces = 1;
    _minFaceDetectionConfidence = 0.5f;
    _minFacePresenceConfidence = 0.5f;
    _minTrackingConfidence = 0.5f;
    _outputFaceBlendshapes = NO;
    _outputFacialTransformationMatrixes = NO;
    _outputFacialTransformationMatrixes = NO;
  }
  return self;
}

- (id)copyWithZone:(NSZone *)zone {
  MPPFaceLandmarkerOptions *faceLandmarkerOptions = [super copyWithZone:zone];

  faceLandmarkerOptions.runningMode = self.runningMode;
  faceLandmarkerOptions.numFaces = self.numFaces;
  faceLandmarkerOptions.minFaceDetectionConfidence = self.minFaceDetectionConfidence;
  faceLandmarkerOptions.minFacePresenceConfidence = self.minFacePresenceConfidence;
  faceLandmarkerOptions.minTrackingConfidence = self.minTrackingConfidence;
  faceLandmarkerOptions.outputFaceBlendshapes = self.outputFaceBlendshapes;
  faceLandmarkerOptions.outputFacialTransformationMatrixes =
      self.outputFacialTransformationMatrixes;
  faceLandmarkerOptions.faceLandmarkerLiveStreamDelegate = self.faceLandmarkerLiveStreamDelegate;

  return faceLandmarkerOptions;
}

@end
