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

#import "mediapipe/tasks/ios/vision/image_classifier/utils/sources/MPPImageClassifierResult+Helpers.h"

#import "mediapipe/tasks/ios/components/containers/utils/sources/MPPClassificationResult+Helpers.h"

#include "mediapipe/tasks/c/components/containers/classification_result.h"
#include "mediapipe/tasks/c/vision/image_classifier/image_classifier.h"

@implementation MPPImageClassifierResult (Helpers)

+ (nullable MPPImageClassifierResult *)
    imageClassifierResultWithCResult:(const MpImageClassifierResult &)cResult
             timestampInMilliseconds:(NSInteger)timestampInMilliseconds {
  MPPClassificationResult *classificationResult =
      [MPPClassificationResult classificationResultWithCResult:cResult];
  if (!classificationResult) {
    return nil;
  }
  return [[MPPImageClassifierResult alloc] initWithClassificationResult:classificationResult
                                                timestampInMilliseconds:timestampInMilliseconds];
}

@end
