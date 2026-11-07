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

#import "mediapipe/tasks/ios/vision/image_segmenter/utils/sources/MPPImageSegmenterResult+Helpers.h"

#include "mediapipe/tasks/c/vision/core/image.h"
#include "mediapipe/tasks/c/vision/image_segmenter/image_segmenter_result.h"

@implementation MPPImageSegmenterResult (Helpers)

+ (nullable MPPImageSegmenterResult *)
    imageSegmenterResultWithCResult:(const struct MpImageSegmenterResult &)cResult
            timestampInMilliseconds:(NSInteger)timestampInMilliseconds
           shouldCopyMaskPacketData:(BOOL)shouldCopyMaskPacketData {
  NSMutableArray<MPPMask *> *confidenceMasks = nil;
  if (cResult.confidence_masks && cResult.confidence_masks_count > 0) {
    confidenceMasks = [NSMutableArray arrayWithCapacity:cResult.confidence_masks_count];
    for (uint32_t i = 0; i < cResult.confidence_masks_count; ++i) {
      MpImagePtr maskImage = cResult.confidence_masks[i];
      if (maskImage) {
        const float *pixelData = nullptr;
        if (MpImageDataFloat32(maskImage, &pixelData, nullptr) == kMpOk && pixelData) {
          int width = MpImageGetWidth(maskImage);
          int height = MpImageGetHeight(maskImage);
          MPPMask *mask = [[MPPMask alloc] initWithFloat32Data:pixelData
                                                         width:width
                                                        height:height
                                                    shouldCopy:shouldCopyMaskPacketData];
          if (mask) {
            [confidenceMasks addObject:mask];
          }
        }
      }
    }
  }

  MPPMask *categoryMask = nil;
  if (cResult.has_category_mask && cResult.category_mask) {
    const uint8_t *pixelData = nullptr;
    if (MpImageDataUint8(cResult.category_mask, &pixelData, nullptr) == kMpOk && pixelData) {
      int width = MpImageGetWidth(cResult.category_mask);
      int height = MpImageGetHeight(cResult.category_mask);
      categoryMask = [[MPPMask alloc] initWithUInt8Data:pixelData
                                                  width:width
                                                 height:height
                                             shouldCopy:shouldCopyMaskPacketData];
    }
  }

  NSMutableArray<NSNumber *> *qualityScores = nil;
  if (cResult.quality_scores && cResult.quality_scores_count > 0) {
    qualityScores = [NSMutableArray arrayWithCapacity:cResult.quality_scores_count];
    for (uint32_t i = 0; i < cResult.quality_scores_count; ++i) {
      [qualityScores addObject:@(cResult.quality_scores[i])];
    }
  }

  return [[MPPImageSegmenterResult alloc] initWithConfidenceMasks:confidenceMasks
                                                     categoryMask:categoryMask
                                                    qualityScores:qualityScores
                                          timestampInMilliseconds:timestampInMilliseconds];
}

@end
