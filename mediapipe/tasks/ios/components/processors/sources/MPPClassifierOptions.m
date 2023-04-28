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

#import "mediapipe/tasks/ios/components/processors/sources/MPPClassifierOptions.h"

@implementation MPPClassifierOptions

- (instancetype)init {
  self = [super init];
  if (self) {
    _maxResults = -1;
    _scoreThreshold = 0;
  }
  return self;
}

- (id)copyWithZone:(NSZone *)zone {
  MPPClassifierOptions *classifierOptions = [[MPPClassifierOptions alloc] init];

  classifierOptions.displayNamesLocale = self.displayNamesLocale;
  classifierOptions.maxResults = self.maxResults;
  classifierOptions.scoreThreshold = self.scoreThreshold;
  classifierOptions.categoryAllowlist = self.categoryAllowlist;
  classifierOptions.categoryDenylist = self.categoryDenylist;

  return classifierOptions;
}

@end
