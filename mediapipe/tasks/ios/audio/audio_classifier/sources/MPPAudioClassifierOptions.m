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

#import "mediapipe/tasks/ios/audio/audio_classifier/sources/MPPAudioClassifierOptions.h"

@implementation MPPAudioClassifierOptions

- (instancetype)init {
  self = [super init];
  if (self) {
    _maxResults = -1;
    _scoreThreshold = 0.0f;
  }
  return self;
}

- (id)copyWithZone:(NSZone *)zone {
  MPPAudioClassifierOptions *audioClassifierOptions = [super copyWithZone:zone];

  audioClassifierOptions.runningMode = self.runningMode;
  audioClassifierOptions.scoreThreshold = self.scoreThreshold;
  audioClassifierOptions.maxResults = self.maxResults;
  audioClassifierOptions.categoryDenylist = self.categoryDenylist;
  audioClassifierOptions.categoryAllowlist = self.categoryAllowlist;
  audioClassifierOptions.displayNamesLocale = self.displayNamesLocale;
  audioClassifierOptions.audioClassifierStreamDelegate = self.audioClassifierStreamDelegate;

  return audioClassifierOptions;
}

@end
