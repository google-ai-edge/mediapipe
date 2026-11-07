// Copyright 2024 The MediaPipe Authors.
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

#import "mediapipe/tasks/ios/vision/holistic_landmarker/utils/sources/MPPHolisticLandmarkerResult+Helpers.h"

#include "mediapipe/tasks/c/core/mp_status.h"
#include "mediapipe/tasks/c/vision/core/image.h"
#include "mediapipe/tasks/c/vision/holistic_landmarker/holistic_landmarker_result.h"
#import "mediapipe/tasks/ios/components/containers/utils/sources/MPPCategory+Helpers.h"
#import "mediapipe/tasks/ios/components/containers/utils/sources/MPPLandmark+Helpers.h"

@implementation MPPHolisticLandmarkerResult (Helpers)

+ (MPPHolisticLandmarkerResult *)
    holisticLandmarkerResultWithCResult:(const MpHolisticLandmarkerResult &)cResult
                timestampInMilliseconds:(NSInteger)timestampInMilliseconds {
  NSArray<MPPNormalizedLandmark *> *faceLandmarks =
      [MPPNormalizedLandmark normalizedLandmarksWithCNormalizedLandmarks:cResult.face_landmarks]
          ?: @[];
  NSArray<MPPNormalizedLandmark *> *poseLandmarks =
      [MPPNormalizedLandmark normalizedLandmarksWithCNormalizedLandmarks:cResult.pose_landmarks]
          ?: @[];
  NSArray<MPPLandmark *> *poseWorldLandmarks =
      [MPPLandmark landmarksWithCLandmarks:cResult.pose_world_landmarks] ?: @[];
  NSArray<MPPNormalizedLandmark *> *leftHandLandmarks =
      [MPPNormalizedLandmark
          normalizedLandmarksWithCNormalizedLandmarks:cResult.left_hand_landmarks]
          ?: @[];
  NSArray<MPPLandmark *> *leftHandWorldLandmarks =
      [MPPLandmark landmarksWithCLandmarks:cResult.left_hand_world_landmarks] ?: @[];
  NSArray<MPPNormalizedLandmark *> *rightHandLandmarks =
      [MPPNormalizedLandmark
          normalizedLandmarksWithCNormalizedLandmarks:cResult.right_hand_landmarks]
          ?: @[];
  NSArray<MPPLandmark *> *rightHandWorldLandmarks =
      [MPPLandmark landmarksWithCLandmarks:cResult.right_hand_world_landmarks] ?: @[];

  MPPClassifications *faceBlendshapes = nil;
  if (cResult.face_blendshapes.categories && cResult.face_blendshapes.categories_count > 0) {
    NSMutableArray<MPPCategory *> *categories =
        [NSMutableArray arrayWithCapacity:cResult.face_blendshapes.categories_count];
    for (uint32_t j = 0; j < cResult.face_blendshapes.categories_count; ++j) {
      MPPCategory *category =
          [MPPCategory categoryWithCCategory:cResult.face_blendshapes.categories[j]];
      if (category) {
        [categories addObject:category];
      }
    }
    faceBlendshapes = [[MPPClassifications alloc] initWithHeadIndex:0
                                                           headName:@""
                                                         categories:categories];
  }

  MPPMask *poseSegmentationMask = nil;
  if (cResult.pose_segmentation_mask) {
    const float *pixelData = nullptr;
    if (MpImageDataFloat32(cResult.pose_segmentation_mask, &pixelData, nullptr) == kMpOk &&
        pixelData) {
      int width = MpImageGetWidth(cResult.pose_segmentation_mask);
      int height = MpImageGetHeight(cResult.pose_segmentation_mask);
      poseSegmentationMask = [[MPPMask alloc] initWithFloat32Data:pixelData
                                                            width:width
                                                           height:height
                                                       shouldCopy:YES];
    }
  }

  return [[MPPHolisticLandmarkerResult alloc] initWithFaceLandmarks:faceLandmarks
                                                    faceBlendshapes:faceBlendshapes
                                                      poseLandmarks:poseLandmarks
                                                 poseWorldLandmarks:poseWorldLandmarks
                                               poseSegmentationMask:poseSegmentationMask
                                                  leftHandLandmarks:leftHandLandmarks
                                             leftHandWorldLandmarks:leftHandWorldLandmarks
                                                 rightHandLandmarks:rightHandLandmarks
                                            rightHandWorldLandmarks:rightHandWorldLandmarks
                                            timestampInMilliseconds:timestampInMilliseconds];
}

@end
