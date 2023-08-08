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
    using ImageProto = ::mediapipe::Image;
    using LandmarkListProto = ::mediapipe::LandmarkList;
    using NormalizedLandmarkListProto = ::mediapipe::NormalizedLandmarkList;
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

+ (MPPPoseLandmarkerResult *)poseLandmarkerResultWithLandmarksPacket:(const mediapipe::Packet &)landmarksPacket
                                                worldLandmarksPacket:(const mediapipe::Packet &)worldLandmarksPacket
                                             segmentationMasksPacket:(const mediapipe::Packet &)segmentationMasksPacket
                                            shouldCopyMaskPacketData:(BOOL)shouldCopyMaskPacketData {
    NSInteger timestampInMilliseconds =
        (NSInteger)(landmarksPacket.Timestamp().Value() / kMicroSecondsPerMilliSecond);

    if (landmarksPacket.IsEmpty()) {
        return [MPPPoseLandmarkerResult
                emptyPoseLandmarkerResultWithTimestampInMilliseconds:timestampInMilliseconds];
    }
    if (!landmarksPacket.ValidateAsType<std::vector<NormalizedLandmarkListProto> >().ok() ||
        !worldLandmarksPacket.ValidateAsType<std::vector<LandmarkListProto> >().ok()) {
        return [MPPPoseLandmarkerResult
                emptyPoseLandmarkerResultWithTimestampInMilliseconds:timestampInMilliseconds];
    }

    const std::vector<NormalizedLandmarkListProto> &landmarksProto =
            landmarksPacket.Get<std::vector<NormalizedLandmarkListProto>>();
    NSMutableArray<NSMutableArray<MPPNormalizedLandmark *> *> *multiPoseLandmarks =
        [NSMutableArray arrayWithCapacity:(NSUInteger)landmarksProto.size()];

    for (const auto &landmarkListProto : landmarksProto) {
        NSMutableArray<MPPNormalizedLandmark *> *landmarks =
            [NSMutableArray arrayWithCapacity:(NSUInteger)landmarkListProto.landmark().size()];

        for (const auto &normalizedLandmarkProto : landmarkListProto.landmark()) {
            MPPNormalizedLandmark *normalizedLandmark =
                [MPPNormalizedLandmark normalizedLandmarkWithProto:normalizedLandmarkProto];
            [landmarks addObject:normalizedLandmark];
        }

        [multiPoseLandmarks addObject:landmarks];
    }

    const std::vector<LandmarkListProto> &worldLandmarksProto =
            worldLandmarksPacket.Get<std::vector<LandmarkListProto>>();
    NSMutableArray<NSMutableArray<MPPLandmark *> *> *multiPoseWorldLandmarks =
        [NSMutableArray arrayWithCapacity:(NSUInteger)worldLandmarksProto.size()];

    for (const auto &worldLandmarkListProto : worldLandmarksProto) {
        NSMutableArray<MPPLandmark *> *worldLandmarks =
            [NSMutableArray arrayWithCapacity:(NSUInteger)worldLandmarkListProto.landmark().size()];

        for (const auto &landmarkProto : worldLandmarkListProto.landmark()) {
            MPPLandmark *landmark = [MPPLandmark landmarkWithProto:landmarkProto];
            [worldLandmarks addObject:landmark];
        }

        [multiPoseWorldLandmarks addObject:worldLandmarks];
    }

    NSMutableArray<MPPMask *> *multiPoseSegmentationMasksProto = [[NSMutableArray alloc] init];
    if (segmentationMasksPacket.ValidateAsType<std::vector<ImageProto> >().ok()) {
        const std::vector<ImageProto> &segmentationMasksProto = segmentationMasksPacket.Get<std::vector<ImageProto> >();
        for (const auto &imageProto : segmentationMasksProto) {
            MPPMask *segmentationMasks = [[MPPMask alloc] initWithFloat32Data:(float *)imageProto.GetImageFrameSharedPtr().get()->PixelData() width:imageProto.width() height:imageProto.height() shouldCopy:shouldCopyMaskPacketData ? YES : NO];
            [multiPoseSegmentationMasksProto addObject:segmentationMasks];
        }
    }
    MPPPoseLandmarkerResult *poseLandmarkerResult =
        [[MPPPoseLandmarkerResult alloc] initWithLandmarks:multiPoseLandmarks
                                            worldLandmarks:multiPoseWorldLandmarks
                                         segmentationMasks:multiPoseSegmentationMasksProto
                                   timestampInMilliseconds:timestampInMilliseconds];
    return poseLandmarkerResult;
}

@end
