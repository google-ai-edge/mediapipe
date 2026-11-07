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

#import "mediapipe/tasks/ios/components/containers/utils/sources/MPPDetection+Helpers.h"

#import "mediapipe/tasks/ios/components/containers/sources/MPPCategory.h"

#include "mediapipe/tasks/c/components/containers/category.h"
#include "mediapipe/tasks/c/components/containers/detection_result.h"
#include "mediapipe/tasks/c/components/containers/keypoint.h"
#include "mediapipe/tasks/c/components/containers/rect.h"

@implementation MPPDetection (Helpers)

+ (MPPNormalizedKeypoint *)normalizedKeypointWithCKeypoint:
    (const MpNormalizedKeypoint &)cKeypoint {
  NSString *label = cKeypoint.label ? [NSString stringWithUTF8String:cKeypoint.label] : nil;
  float score = cKeypoint.has_score ? cKeypoint.score : 0.0f;
  return [[MPPNormalizedKeypoint alloc] initWithLocation:CGPointMake(cKeypoint.x, cKeypoint.y)
                                                   label:label
                                                   score:score];
}

+ (MPPDetection *)detectionWithCDetection:(const MpDetection &)cDetection {
  CGRect boundingBox = CGRectMake(cDetection.bounding_box.left, cDetection.bounding_box.top,
                                  cDetection.bounding_box.right - cDetection.bounding_box.left,
                                  cDetection.bounding_box.bottom - cDetection.bounding_box.top);
  NSMutableArray<MPPCategory *> *categories =
      [NSMutableArray arrayWithCapacity:cDetection.categories_count];
  for (uint32_t i = 0; i < cDetection.categories_count; ++i) {
    const MpCategory &cCategory = cDetection.categories[i];
    NSString *categoryName =
        (cCategory.category_name && cCategory.category_name[0] != '\0')
            ? [NSString stringWithUTF8String:cCategory.category_name]
            : nil;
    NSString *displayName =
        (cCategory.display_name && cCategory.display_name[0] != '\0')
            ? [NSString stringWithUTF8String:cCategory.display_name]
            : nil;
    [categories addObject:[[MPPCategory alloc] initWithIndex:cCategory.index
                                                       score:cCategory.score
                                                categoryName:categoryName
                                                 displayName:displayName]];
  }
  NSMutableArray<MPPNormalizedKeypoint *> *keypoints =
      cDetection.keypoints != nullptr
          ? [NSMutableArray arrayWithCapacity:cDetection.keypoints_count]
          : nil;
  if (cDetection.keypoints != nullptr) {
    for (uint32_t i = 0; i < cDetection.keypoints_count; ++i) {
      [keypoints addObject:[MPPDetection normalizedKeypointWithCKeypoint:cDetection.keypoints[i]]];
    }
  }
  return [[MPPDetection alloc] initWithCategories:categories
                                      boundingBox:boundingBox
                                        keypoints:keypoints];
}

+ (NSArray<MPPDetection *> *)detectionsWithCDetectionResult:
    (const MpDetectionResult &)cResult {
  NSMutableArray<MPPDetection *> *detections =
      [NSMutableArray arrayWithCapacity:cResult.detections_count];
  for (uint32_t i = 0; i < cResult.detections_count; ++i) {
    [detections addObject:[MPPDetection detectionWithCDetection:cResult.detections[i]]];
  }
  return detections;
}

@end
