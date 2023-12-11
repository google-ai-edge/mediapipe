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

#import "mediapipe/tasks/ios/components/containers/utils/sources/MPPLandmark+Helpers.h"

namespace {
using LandmarkListProto = ::mediapipe::LandmarkList;
using NormalizedLandmarkListProto = ::mediapipe::NormalizedLandmarkList;
using ::mediapipe::Image;
using ::mediapipe::Packet;
}  // namespace

@implementation MPPPoseLandmarkerResult (Helpers)

+ (MPPPoseLandmarkerResult *)emptyPoseLandmarkerResultWithTimestampInMilliseconds:
    (NSInteger)timestampInMilliseconds {
  return [[MPPPoseLandmarkerResult alloc] initWithLandmarks:@[]
                                             worldLandmarks:@[]
                                          segmentationMasks:@[]
                                    timestampInMilliseconds:timestampInMilliseconds];
}

+ (MPPPoseLandmarkerResult *)
    poseLandmarkerResultWithLandmarksProto:
        (const std::vector<NormalizedLandmarkListProto> &)landmarksProto
                       worldLandmarksProto:
                           (const std::vector<LandmarkListProto> &)worldLandmarksProto
                         segmentationMasks:(nullable const std::vector<Image> *)segmentationMasks
                   timestampInMilliseconds:(NSInteger)timestampInMilliseconds {
  NSMutableArray<NSMutableArray<MPPNormalizedLandmark *> *> *multiplePoseLandmarks =
      [NSMutableArray arrayWithCapacity:(NSUInteger)landmarksProto.size()];

  for (const auto &landmarkListProto : landmarksProto) {
    NSMutableArray<MPPNormalizedLandmark *> *landmarks =
        [NSMutableArray arrayWithCapacity:(NSUInteger)landmarkListProto.landmark().size()];
    for (const auto &normalizedLandmarkProto : landmarkListProto.landmark()) {
      MPPNormalizedLandmark *normalizedLandmark =
          [MPPNormalizedLandmark normalizedLandmarkWithProto:normalizedLandmarkProto];
      [landmarks addObject:normalizedLandmark];
    }
    [multiplePoseLandmarks addObject:landmarks];
  }

  NSMutableArray<NSMutableArray<MPPLandmark *> *> *multiplePoseWorldLandmarks =
      [NSMutableArray arrayWithCapacity:(NSUInteger)worldLandmarksProto.size()];

  for (const auto &worldLandmarkListProto : worldLandmarksProto) {
    NSMutableArray<MPPLandmark *> *worldLandmarks =
        [NSMutableArray arrayWithCapacity:(NSUInteger)worldLandmarkListProto.landmark().size()];
    for (const auto &landmarkProto : worldLandmarkListProto.landmark()) {
      MPPLandmark *landmark = [MPPLandmark landmarkWithProto:landmarkProto];
      [worldLandmarks addObject:landmark];
    }
    [multiplePoseWorldLandmarks addObject:worldLandmarks];
  }

  if (!segmentationMasks) {
    return [[MPPPoseLandmarkerResult alloc] initWithLandmarks:multiplePoseLandmarks
                                               worldLandmarks:multiplePoseWorldLandmarks
                                            segmentationMasks:nil
                                      timestampInMilliseconds:timestampInMilliseconds];
  }
  NSMutableArray<MPPMask *> *confidenceMasks =
      [NSMutableArray arrayWithCapacity:(NSUInteger)segmentationMasks->size()];

  for (const auto &segmentationMask : *segmentationMasks) {
    [confidenceMasks addObject:[[MPPMask alloc] initWithFloat32Data:(float *)segmentationMask
                                                                        .GetImageFrameSharedPtr()
                                                                        .get()
                                                                        ->PixelData()
                                                              width:segmentationMask.width()
                                                             height:segmentationMask.height()
                                                         /** Always deep copy */
                                                         shouldCopy:YES]];
  }

  return [[MPPPoseLandmarkerResult alloc] initWithLandmarks:multiplePoseLandmarks
                                             worldLandmarks:multiplePoseWorldLandmarks
                                          segmentationMasks:confidenceMasks
                                    timestampInMilliseconds:timestampInMilliseconds];
  ;
}

+ (MPPPoseLandmarkerResult *)
    poseLandmarkerResultWithLandmarksPacket:(const Packet &)landmarksPacket
                       worldLandmarksPacket:(const Packet &)worldLandmarksPacket
                    segmentationMasksPacket:(const Packet *)segmentationMasksPacket {
  NSInteger timestampInMilliseconds =
      (NSInteger)(landmarksPacket.Timestamp().Value() / kMicrosecondsPerMillisecond);

  if (landmarksPacket.IsEmpty()) {
    return [MPPPoseLandmarkerResult
        emptyPoseLandmarkerResultWithTimestampInMilliseconds:timestampInMilliseconds];
  }

  if (!landmarksPacket.ValidateAsType<std::vector<NormalizedLandmarkListProto>>().ok() ||
      !worldLandmarksPacket.ValidateAsType<std::vector<LandmarkListProto>>().ok()) {
    return [MPPPoseLandmarkerResult
        emptyPoseLandmarkerResultWithTimestampInMilliseconds:timestampInMilliseconds];
  }

  const std::vector<Image> *segmentationMasks =
      segmentationMasksPacket ? &(segmentationMasksPacket->Get<std::vector<Image>>()) : nullptr;

  return [MPPPoseLandmarkerResult
      poseLandmarkerResultWithLandmarksProto:landmarksPacket
                                                 .Get<std::vector<NormalizedLandmarkListProto>>()
                         worldLandmarksProto:worldLandmarksPacket
                                                 .Get<std::vector<LandmarkListProto>>()
                           segmentationMasks:segmentationMasks
                     timestampInMilliseconds:timestampInMilliseconds];
}

@end
