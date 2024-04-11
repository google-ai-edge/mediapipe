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

#import <Foundation/Foundation.h>

#import "mediapipe/tasks/ios/vision/holistic_landmarker/sources/MPPHolisticLandmarkerResult.h"

@implementation MPPHolisticLandmarkerResult

- (instancetype)initWithFaceLandmarks:(NSArray<MPPNormalizedLandmark *> *)faceLandmarks
                      faceBlendshapes:(nullable MPPClassifications *)faceBlendshapes
                        poseLandmarks:(NSArray<MPPNormalizedLandmark *> *)poseLandmarks
                   poseWorldLandmarks:(NSArray<MPPLandmark *> *)poseWorldLandmarks
                 poseSegmentationMask:(nullable MPPMask *)poseSegmentationMask
                    leftHandLandmarks:(NSArray<MPPNormalizedLandmark *> *)leftHandLandmarks
               leftHandWorldLandmarks:(NSArray<MPPLandmark *> *)leftHandWorldLandmarks
                   rightHandLandmarks:(NSArray<MPPNormalizedLandmark *> *)rightHandLandmarks
              rightHandWorldLandmarks:(NSArray<MPPLandmark *> *)rightHandWorldLandmarks
              timestampInMilliseconds:(NSInteger)timestampInMilliseconds {
  self = [super initWithTimestampInMilliseconds:timestampInMilliseconds];
  if (self) {
    _faceLandmarks = faceLandmarks;
    _faceBlendshapes = faceBlendshapes;
    _poseLandmarks = poseLandmarks;
    _poseWorldLandmarks = poseWorldLandmarks;
    _poseSegmentationMask = poseSegmentationMask;
    _leftHandLandmarks = leftHandLandmarks;
    _leftHandWorldLandmarks = leftHandWorldLandmarks;
    _rightHandLandmarks = rightHandLandmarks;
    _rightHandWorldLandmarks = rightHandWorldLandmarks;
  }
  return self;
}

@end
