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

#import "mediapipe/tasks/ios/text/text_classifier/utils/sources/MPPTextClassifierOptions+Helpers.h"

#import "mediapipe/tasks/ios/components/processors/sources/MPPClassifierOptions.h"
#import "mediapipe/tasks/ios/components/processors/utils/sources/MPPClassifierOptions+Helpers.h"
#import "mediapipe/tasks/ios/core/utils/sources/MPPBaseOptions+Helpers.h"

#include "mediapipe/tasks/c/text/text_classifier/text_classifier.h"

@implementation MPPTextClassifierOptions (Helpers)

- (void)copyToCOptions:(MpTextClassifierOptions *)cOptions {
  [self.baseOptions copyToCOptions:&(cOptions->base_options)];

  MPPClassifierOptions *classifierOptions = [[MPPClassifierOptions alloc] init];
  classifierOptions.displayNamesLocale = self.displayNamesLocale;
  classifierOptions.maxResults = self.maxResults;
  classifierOptions.scoreThreshold = self.scoreThreshold;
  classifierOptions.categoryAllowlist = self.categoryAllowlist;
  classifierOptions.categoryDenylist = self.categoryDenylist;
  [classifierOptions copyToCOptions:&(cOptions->classifier_options)];
}

+ (void)freeCOptions:(MpTextClassifierOptions *)cOptions {
  if (cOptions) {
    [MPPBaseOptions freeCOptions:&(cOptions->base_options)];
    [MPPClassifierOptions freeCOptions:&(cOptions->classifier_options)];
  }
}

@end
