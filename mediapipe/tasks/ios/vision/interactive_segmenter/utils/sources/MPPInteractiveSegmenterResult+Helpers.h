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

#import "mediapipe/tasks/ios/vision/interactive_segmenter/sources/MPPInteractiveSegmenterResult.h"

#include "mediapipe/framework/packet.h"

NS_ASSUME_NONNULL_BEGIN

@interface MPPInteractiveSegmenterResult (Helpers)

/**
 * Creates an `MPPInteractiveSegmenterResult` from confidence masks, category mask and quality
 * scores packets.
 *
 * If `shouldCopyMaskPacketData` is set to `YES`, the confidence and catergory masks of the newly
 * created `MPPInteractiveSegmenterResult` holds references to deep copied pixel data of the output
 * respective masks.
 *
 * @param confidenceMasksPacket A MediaPipe packet wrapping a `std::vector<mediapipe::Image>`.
 * @param categoryMaskPacket A MediaPipe packet wrapping a `<mediapipe::Image>`.
 * @param qualityScoresPacket A MediaPipe packet wrapping a `std::vector<float>`.
 * @param shouldCopyMaskPacketData A `BOOL` which indicates if the pixel data of the output masks
 * must be deep copied to the newly created `MPPInteractiveSegmenterResult`.
 *
 * @return  An `MPPInteractiveSegmenterResult` object that contains the image segmentation results.
 */
+ (MPPInteractiveSegmenterResult *)
    interactiveSegmenterResultWithConfidenceMasksPacket:
        (const mediapipe::Packet &)confidenceMasksPacket
                                     categoryMaskPacket:
                                         (const mediapipe::Packet &)categoryMaskPacket
                                    qualityScoresPacket:
                                        (const mediapipe::Packet &)qualityScoresPacket
                                timestampInMilliseconds:(NSInteger)timestampInMilliseconds
                               shouldCopyMaskPacketData:(BOOL)shouldCopyMaskPacketData;

@end

NS_ASSUME_NONNULL_END
