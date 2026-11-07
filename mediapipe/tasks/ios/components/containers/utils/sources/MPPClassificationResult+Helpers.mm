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

#import "mediapipe/tasks/ios/components/containers/utils/sources/MPPClassificationResult+Helpers.h"

#import "mediapipe/tasks/ios/components/containers/utils/sources/MPPCategory+Helpers.h"

#include "mediapipe/tasks/c/components/containers/category.h"
#include "mediapipe/tasks/c/components/containers/classification_result.h"

@implementation MPPClassifications (Helpers)

+ (MPPClassifications *)classificationsWithCClassifications:
    (const MpClassifications &)cClassifications {
  NSMutableArray<MPPCategory *> *categories =
      [NSMutableArray arrayWithCapacity:cClassifications.categories_count];
  for (uint32_t i = 0; i < cClassifications.categories_count; ++i) {
    [categories addObject:[MPPCategory categoryWithCCategory:cClassifications.categories[i]]];
  }

  NSString *headName =
      cClassifications.head_name ? [NSString stringWithUTF8String:cClassifications.head_name] : @"";

  return [[MPPClassifications alloc] initWithHeadIndex:cClassifications.head_index
                                              headName:headName
                                            categories:categories];
}

@end

@implementation MPPClassificationResult (Helpers)

+ (MPPClassificationResult *)classificationResultWithCResult:
    (const MpClassificationResult &)cResult {
  NSMutableArray<MPPClassifications *> *classifications =
      [NSMutableArray arrayWithCapacity:cResult.classifications_count];
  for (uint32_t i = 0; i < cResult.classifications_count; ++i) {
    [classifications
        addObject:[MPPClassifications
                      classificationsWithCClassifications:cResult.classifications[i]]];
  }
  NSInteger timestampInMilliseconds = 0;
  if (cResult.has_timestamp_ms) {
    timestampInMilliseconds = (NSInteger)cResult.timestamp_ms;
  }
  return [[MPPClassificationResult alloc] initWithClassifications:classifications
                                          timestampInMilliseconds:timestampInMilliseconds];
}

@end
