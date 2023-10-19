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

#import "mediapipe/tasks/ios/text/language_detector/sources/MPPLanguageDetectorResult.h"

@implementation MPPLanguagePrediction

- (instancetype)initWithLanguageCode:(NSString *)languageCode probability:(float)probability {
  self = [super init];
  if (self) {
    _languageCode = languageCode;
    _probability = probability;
  }
  return self;
}

@end

@implementation MPPLanguageDetectorResult

- (instancetype)initWithLanguagePredictions:(NSArray<MPPLanguagePrediction *> *)languagePredictions
                    timestampInMilliseconds:(NSInteger)timestampInMilliseconds {
  self = [super initWithTimestampInMilliseconds:timestampInMilliseconds];
  if (self) {
    _languagePredictions = languagePredictions;
  }
  return self;
}

@end
