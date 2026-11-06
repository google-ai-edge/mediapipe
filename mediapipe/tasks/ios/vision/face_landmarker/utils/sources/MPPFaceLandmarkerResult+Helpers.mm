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

#import "mediapipe/tasks/ios/vision/face_landmarker/utils/sources/MPPFaceLandmarkerResult+Helpers.h"

#include "mediapipe/tasks/c/vision/face_landmarker/face_landmarker_result.h"
#import "mediapipe/tasks/ios/components/containers/utils/sources/MPPCategory+Helpers.h"
#import "mediapipe/tasks/ios/components/containers/utils/sources/MPPLandmark+Helpers.h"

@implementation MPPFaceLandmarkerResult (Helpers)

+ (MPPFaceLandmarkerResult *)faceLandmarkerResultWithCResult:(const MpFaceLandmarkerResult &)cResult
                                     timestampInMilliseconds:(NSInteger)timestampInMilliseconds {
  NSMutableArray<NSArray<MPPNormalizedLandmark *> *> *faceLandmarks =
      [NSMutableArray arrayWithCapacity:cResult.face_landmarks_count];
  for (uint32_t i = 0; i < cResult.face_landmarks_count; ++i) {
    NSArray<MPPNormalizedLandmark *> *landmarks = [MPPNormalizedLandmark
        normalizedLandmarksWithCNormalizedLandmarks:cResult.face_landmarks[i]];
    [faceLandmarks addObject:landmarks ?: @[]];
  }

  NSMutableArray<MPPClassifications *> *faceBlendshapes =
      [NSMutableArray arrayWithCapacity:cResult.face_blendshapes_count];
  for (uint32_t i = 0; i < cResult.face_blendshapes_count; ++i) {
    const auto &cCategories = cResult.face_blendshapes[i];
    NSMutableArray<MPPCategory *> *categories =
        [NSMutableArray arrayWithCapacity:cCategories.categories_count];
    for (uint32_t j = 0; j < cCategories.categories_count; ++j) {
      MPPCategory *category = [MPPCategory categoryWithCCategory:cCategories.categories[j]];
      if (category) {
        [categories addObject:category];
      }
    }
    MPPClassifications *classifications = [[MPPClassifications alloc] initWithHeadIndex:0
                                                                               headName:@""
                                                                             categories:categories];
    [faceBlendshapes addObject:classifications];
  }

  NSMutableArray<MPPTransformMatrix *> *facialTransformationMatrixes =
      [NSMutableArray arrayWithCapacity:cResult.facial_transformation_matrixes_count];
  for (uint32_t i = 0; i < cResult.facial_transformation_matrixes_count; ++i) {
    const auto &matrix = cResult.facial_transformation_matrixes[i];
    MPPTransformMatrix *transformMatrix = [[MPPTransformMatrix alloc] initWithData:matrix.data
                                                                              rows:matrix.rows
                                                                           columns:matrix.cols];
    [facialTransformationMatrixes addObject:transformMatrix];
  }

  return [[MPPFaceLandmarkerResult alloc] initWithFaceLandmarks:faceLandmarks
                                                faceBlendshapes:faceBlendshapes
                                   facialTransformationMatrixes:facialTransformationMatrixes
                                        timestampInMilliseconds:timestampInMilliseconds];
}

@end
