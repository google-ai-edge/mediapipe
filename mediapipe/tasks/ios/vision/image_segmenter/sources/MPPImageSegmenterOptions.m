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

#import "mediapipe/tasks/ios/vision/image_segmenter/sources/MPPImageSegmenterOptions.h"

@implementation MPPImageSegmenterOptions

- (instancetype)init {
  self = [super init];
  if (self) {
    _displayNamesLocale = @"en";
    _shouldOutputConfidenceMasks = YES;
  }
  return self;
}

- (id)copyWithZone:(NSZone *)zone {
  MPPImageSegmenterOptions *imageSegmenterOptions = [super copyWithZone:zone];

  imageSegmenterOptions.runningMode = self.runningMode;
  imageSegmenterOptions.shouldOutputConfidenceMasks = self.shouldOutputConfidenceMasks;
  imageSegmenterOptions.shouldOutputCategoryMask = self.shouldOutputCategoryMask;
  imageSegmenterOptions.displayNamesLocale = self.displayNamesLocale;
  imageSegmenterOptions.imageSegmenterLiveStreamDelegate = self.imageSegmenterLiveStreamDelegate;

  return imageSegmenterOptions;
}

@end
