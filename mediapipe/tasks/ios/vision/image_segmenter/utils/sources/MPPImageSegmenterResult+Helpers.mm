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

#include "mediapipe/framework/formats/image.h"
#include "mediapipe/framework/packet.h"

namespace {
using ::mediapipe::Image;
using ::mediapipe::ImageFrameSharedPtr;
using ::mediapipe::Packet;
}  // namespace

@implementation MPPImageSegmenterResult (Helpers)

+ (MPPImageSegmenterResult *)
    imageSegmenterResultWithConfidenceMasksPacket:(const Packet &)confidenceMasksPacket
                               categoryMaskPacket:(const Packet &)categoryMaskPacket
                              qualityScoresPacket:(const Packet &)qualityScoresPacket
                          timestampInMilliseconds:(NSInteger)timestampInMilliseconds
                         shouldCopyMaskPacketData:(BOOL)shouldCopyMaskPacketData {
  NSMutableArray<MPPMask *> *confidenceMasks;
  MPPMask *categoryMask;
  NSMutableArray<NSNumber *> *qualityScores;

  if (confidenceMasksPacket.ValidateAsType<std::vector<Image>>().ok()) {
    std::vector<Image> cppConfidenceMasks = confidenceMasksPacket.Get<std::vector<Image>>();
    confidenceMasks = [NSMutableArray arrayWithCapacity:(NSUInteger)cppConfidenceMasks.size()];

    for (const auto &confidenceMask : cppConfidenceMasks) {
      [confidenceMasks
          addObject:[[MPPMask alloc]
                        initWithFloat32Data:(float *)confidenceMask.GetImageFrameSharedPtr()
                                                .get()
                                                ->PixelData()
                                      width:confidenceMask.width()
                                     height:confidenceMask.height()
                                 shouldCopy:shouldCopyMaskPacketData]];
    }
  }

  if (categoryMaskPacket.ValidateAsType<Image>().ok()) {
    const Image &cppCategoryMask = categoryMaskPacket.Get<Image>();
    categoryMask = [[MPPMask alloc]
        initWithUInt8Data:(UInt8 *)cppCategoryMask.GetImageFrameSharedPtr().get()->PixelData()
                    width:cppCategoryMask.width()
                   height:cppCategoryMask.height()
               shouldCopy:shouldCopyMaskPacketData];
  }

  if (qualityScoresPacket.ValidateAsType<std::vector<float>>().ok()) {
    std::vector<float> cppQualityScores = qualityScoresPacket.Get<std::vector<float>>();
    qualityScores = [NSMutableArray arrayWithCapacity:(NSUInteger)cppQualityScores.size()];

    for (const auto &qualityScore : cppQualityScores) {
      [qualityScores addObject:[NSNumber numberWithFloat:qualityScore]];
    }
  }

  return [[MPPImageSegmenterResult alloc] initWithConfidenceMasks:confidenceMasks
                                                     categoryMask:categoryMask
                                                    qualityScores:qualityScores
                                          timestampInMilliseconds:timestampInMilliseconds];
}

@end
