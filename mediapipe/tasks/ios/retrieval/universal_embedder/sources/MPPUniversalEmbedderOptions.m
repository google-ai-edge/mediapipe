// Copyright 2026 The MediaPipe Authors. All Rights Reserved.
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

#import "mediapipe/tasks/ios/retrieval/universal_embedder/sources/MPPUniversalEmbedderOptions.h"

@implementation MPPUniversalEmbedderOptions

- (instancetype)init {
  self = [super init];
  if (self) {
    _l2Normalize = YES;
    _maxInputLength = 0;
    _visionTokensPerImage = 0;
    _activationDataType = MPPActivationDataTypeDefault;
  }
  return self;
}

- (id)copyWithZone:(NSZone *)zone {
  MPPUniversalEmbedderOptions *universalEmbedderOptions = [super copyWithZone:zone];
  universalEmbedderOptions.l2Normalize = self.l2Normalize;
  universalEmbedderOptions.maxInputLength = self.maxInputLength;
  universalEmbedderOptions.visionTokensPerImage = self.visionTokensPerImage;
  universalEmbedderOptions.activationDataType = self.activationDataType;
  universalEmbedderOptions.cacheDir = self.cacheDir;
  return universalEmbedderOptions;
}

@end
