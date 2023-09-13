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

#import "mediapipe/tasks/ios/vision/gesture_recognizer/sources/MPPGestureRecognizerOptions.h"

@implementation MPPGestureRecognizerOptions

- (instancetype)init {
  self = [super init];
  if (self) {
    _numHands = 1;
    _minHandDetectionConfidence = 0.5f;
    _minHandPresenceConfidence = 0.5f;
    _minTrackingConfidence = 0.5f;
  }
  return self;
}

- (id)copyWithZone:(NSZone *)zone {
  MPPGestureRecognizerOptions *gestureRecognizerOptions = [super copyWithZone:zone];

  gestureRecognizerOptions.runningMode = self.runningMode;
  gestureRecognizerOptions.gestureRecognizerLiveStreamDelegate =
      self.gestureRecognizerLiveStreamDelegate;
  gestureRecognizerOptions.numHands = self.numHands;
  gestureRecognizerOptions.minHandDetectionConfidence = self.minHandDetectionConfidence;
  gestureRecognizerOptions.minHandPresenceConfidence = self.minHandPresenceConfidence;
  gestureRecognizerOptions.minTrackingConfidence = self.minTrackingConfidence;
  gestureRecognizerOptions.cannedGesturesClassifierOptions = self.cannedGesturesClassifierOptions;
  gestureRecognizerOptions.customGesturesClassifierOptions = self.customGesturesClassifierOptions;

  return gestureRecognizerOptions;
}

@end
