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

#import "mediapipe/tasks/ios/vision/gesture_recognizer/utils/sources/MPPGestureRecognizerResult+Helpers.h"

#include "mediapipe/tasks/c/vision/gesture_recognizer/gesture_recognizer_result.h"
#import "mediapipe/tasks/ios/components/containers/utils/sources/MPPCategory+Helpers.h"
#import "mediapipe/tasks/ios/components/containers/utils/sources/MPPLandmark+Helpers.h"

static const NSInteger kDefaultGestureIndex = -1;

@implementation MPPGestureRecognizerResult (Helpers)

+ (MPPGestureRecognizerResult *)
    gestureRecognizerResultWithCResult:(const MpGestureRecognizerResult &)cResult
               timestampInMilliseconds:(NSInteger)timestampInMilliseconds {
  NSMutableArray<NSMutableArray<MPPCategory *> *> *multiHandGestures =
      [NSMutableArray arrayWithCapacity:cResult.gestures_count];
  for (uint32_t i = 0; i < cResult.gestures_count; ++i) {
    const auto &cCategories = cResult.gestures[i];
    NSMutableArray<MPPCategory *> *gestures =
        [NSMutableArray arrayWithCapacity:cCategories.categories_count];
    for (uint32_t j = 0; j < cCategories.categories_count; ++j) {
      MPPCategory *c = [MPPCategory categoryWithCCategory:cCategories.categories[j]];
      if (c) {
        MPPCategory *category = [[MPPCategory alloc] initWithIndex:kDefaultGestureIndex
                                                             score:c.score
                                                      categoryName:c.categoryName
                                                       displayName:c.displayName];
        [gestures addObject:category];
      }
    }
    [multiHandGestures addObject:gestures];
  }

  NSMutableArray<NSMutableArray<MPPCategory *> *> *multiHandHandedness =
      [NSMutableArray arrayWithCapacity:cResult.handedness_count];
  for (uint32_t i = 0; i < cResult.handedness_count; ++i) {
    const auto &cCategories = cResult.handedness[i];
    NSMutableArray<MPPCategory *> *handedness =
        [NSMutableArray arrayWithCapacity:cCategories.categories_count];
    for (uint32_t j = 0; j < cCategories.categories_count; ++j) {
      MPPCategory *category = [MPPCategory categoryWithCCategory:cCategories.categories[j]];
      if (category) {
        [handedness addObject:category];
      }
    }
    [multiHandHandedness addObject:handedness];
  }

  NSMutableArray<NSArray<MPPNormalizedLandmark *> *> *multiHandLandmarks =
      [NSMutableArray arrayWithCapacity:cResult.hand_landmarks_count];
  for (uint32_t i = 0; i < cResult.hand_landmarks_count; ++i) {
    NSArray<MPPNormalizedLandmark *> *landmarks = [MPPNormalizedLandmark
        normalizedLandmarksWithCNormalizedLandmarks:cResult.hand_landmarks[i]];
    [multiHandLandmarks addObject:landmarks ?: @[]];
  }

  NSMutableArray<NSArray<MPPLandmark *> *> *multiHandWorldLandmarks =
      [NSMutableArray arrayWithCapacity:cResult.hand_world_landmarks_count];
  for (uint32_t i = 0; i < cResult.hand_world_landmarks_count; ++i) {
    NSArray<MPPLandmark *> *worldLandmarks =
        [MPPLandmark landmarksWithCLandmarks:cResult.hand_world_landmarks[i]];
    [multiHandWorldLandmarks addObject:worldLandmarks ?: @[]];
  }

  return [[MPPGestureRecognizerResult alloc] initWithGestures:multiHandGestures
                                                   handedness:multiHandHandedness
                                                    landmarks:multiHandLandmarks
                                               worldLandmarks:multiHandWorldLandmarks
                                      timestampInMilliseconds:timestampInMilliseconds];
}

@end
