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

#import "mediapipe/tasks/ios/components/containers/utils/sources/MPPRegionOfInterest+Helpers.h"

#import "mediapipe/tasks/ios/common/sources/MPPCommon.h"
#import "mediapipe/tasks/ios/common/utils/sources/MPPCommonUtils.h"
#import "mediapipe/tasks/ios/components/containers/sources/MPPDetection.h"

#include <cstdlib>
#include <cstring>

#include "mediapipe/tasks/c/components/containers/keypoint.h"
#include "mediapipe/tasks/c/vision/interactive_segmenter_legacy/interactive_segmenter_legacy.h"

@implementation MPPRegionOfInterest (Helpers)

- (BOOL)copyToCRegionOfInterest:(MpRegionOfInterest *)cRoi error:(NSError **)error {
  std::memset(cRoi, 0, sizeof(MpRegionOfInterest));
  if (self.keypoint) {
    cRoi->format = MP_REGION_OF_INTEREST_FORMAT_KEYPOINT;
    cRoi->keypoint.x = self.keypoint.location.x;
    cRoi->keypoint.y = self.keypoint.location.y;
    cRoi->keypoint.label = nullptr;
    cRoi->keypoint.score = self.keypoint.score;
    cRoi->keypoint.has_score = true;
    return YES;
  } else if (self.scribbles && self.scribbles.count > 0) {
    cRoi->format = MP_REGION_OF_INTEREST_FORMAT_SCRIBBLE;
    cRoi->scribble_count = (uint32_t)self.scribbles.count;
    cRoi->scribble =
        (MpNormalizedKeypoint *)std::malloc(sizeof(MpNormalizedKeypoint) * cRoi->scribble_count);
    for (uint32_t i = 0; i < cRoi->scribble_count; ++i) {
      MPPNormalizedKeypoint *kp = self.scribbles[i];
      cRoi->scribble[i].x = kp.location.x;
      cRoi->scribble[i].y = kp.location.y;
      cRoi->scribble[i].label = nullptr;
      cRoi->scribble[i].score = kp.score;
      cRoi->scribble[i].has_score = true;
    }
    return YES;
  }

  [MPPCommonUtils createCustomError:error
                           withCode:MPPTasksErrorCodeInvalidArgumentError
                        description:@"RegionOfInterest does not include a valid user interaction."];
  return NO;
}

+ (void)freeCRegionOfInterest:(nullable MpRegionOfInterest *)cRoi {
  if (cRoi) {
    if (cRoi->scribble) {
      std::free(cRoi->scribble);
      cRoi->scribble = nullptr;
    }
    cRoi->scribble_count = 0;
  }
}

@end
