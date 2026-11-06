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

#import "mediapipe/tasks/ios/vision/pose_landmarker/utils/sources/MPPPoseLandmarkerResult+Helpers.h"

#include "mediapipe/tasks/c/vision/core/image.h"
#include "mediapipe/tasks/c/vision/pose_landmarker/pose_landmarker_result.h"
#import "mediapipe/tasks/ios/components/containers/utils/sources/MPPLandmark+Helpers.h"

@implementation MPPPoseLandmarkerResult (Helpers)

+ (MPPPoseLandmarkerResult *)poseLandmarkerResultWithCResult:(const MpPoseLandmarkerResult &)cResult
                                     timestampInMilliseconds:(NSInteger)timestampInMilliseconds {
  NSMutableArray<NSArray<MPPNormalizedLandmark *> *> *multiPoseLandmarks =
      [NSMutableArray arrayWithCapacity:cResult.pose_landmarks_count];
  for (uint32_t i = 0; i < cResult.pose_landmarks_count; ++i) {
    NSArray<MPPNormalizedLandmark *> *landmarks = [MPPNormalizedLandmark
        normalizedLandmarksWithCNormalizedLandmarks:cResult.pose_landmarks[i]];
    [multiPoseLandmarks addObject:landmarks ?: @[]];
  }

  NSMutableArray<NSArray<MPPLandmark *> *> *multiPoseWorldLandmarks =
      [NSMutableArray arrayWithCapacity:cResult.pose_world_landmarks_count];
  for (uint32_t i = 0; i < cResult.pose_world_landmarks_count; ++i) {
    NSArray<MPPLandmark *> *worldLandmarks =
        [MPPLandmark landmarksWithCLandmarks:cResult.pose_world_landmarks[i]];
    [multiPoseWorldLandmarks addObject:worldLandmarks ?: @[]];
  }

  NSMutableArray<MPPMask *> *confidenceMasks =
      [NSMutableArray arrayWithCapacity:cResult.segmentation_masks_count];
  if (cResult.segmentation_masks) {
    for (uint32_t i = 0; i < cResult.segmentation_masks_count; ++i) {
      MpImagePtr maskImage = cResult.segmentation_masks[i];
      if (maskImage) {
        const float *pixelData = nullptr;
        if (MpImageDataFloat32(maskImage, &pixelData, nullptr) == kMpOk && pixelData) {
          int width = MpImageGetWidth(maskImage);
          int height = MpImageGetHeight(maskImage);
          MPPMask *mask = [[MPPMask alloc] initWithFloat32Data:pixelData
                                                         width:width
                                                        height:height
                                                    shouldCopy:YES];
          if (mask) {
            [confidenceMasks addObject:mask];
          }
        }
      }
    }
  }

  return [[MPPPoseLandmarkerResult alloc] initWithLandmarks:multiPoseLandmarks
                                             worldLandmarks:multiPoseWorldLandmarks
                                          segmentationMasks:confidenceMasks
                                    timestampInMilliseconds:timestampInMilliseconds];
}

@end
