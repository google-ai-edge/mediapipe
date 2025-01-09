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

#import "mediapipe/tasks/ios/vision/pose_landmarker/sources/MPPPoseLandmarkerResult.h"

#include "mediapipe/framework/formats/image.h"
#include "mediapipe/framework/formats/landmark.pb.h"
#include "mediapipe/framework/packet.h"

NS_ASSUME_NONNULL_BEGIN

static const int kMicrosecondsPerMillisecond = 1000;

@interface MPPPoseLandmarkerResult (Helpers)

/**
 * Creates an `MPPPoseLandmarkerResult` from landmarks, world landmarks and segmentation mask
 * packets.
 *
 * @param landmarksPacket A MediaPipe packet wrapping a `std::vector<NormalizedlandmarkListProto>`.
 * @param worldLandmarksPacket A MediaPipe packet wrapping a `std::vector<LandmarkListProto>`.
 * @param segmentationMasksPacket a MediaPipe packet wrapping a `std::vector<Image>`.
 *
 * @return  An `MPPPoseLandmarkerResult` object that contains the hand landmark detection
 * results.
 */
+ (MPPPoseLandmarkerResult *)
    poseLandmarkerResultWithLandmarksPacket:(const mediapipe::Packet &)landmarksPacket
                       worldLandmarksPacket:(const mediapipe::Packet &)worldLandmarksPacket
                    segmentationMasksPacket:(const mediapipe::Packet *)segmentationMasksPacket;

/**
 * Creates an `MPPPoseLandmarkerResult` from landmarks, world landmarks and segmentation mask
 * images.
 *
 * @param landmarksProto A vector of protos of type `std::vector<NormalizedlandmarkListProto>`.
 * @param worldLandmarksProto A vector of protos of type `std::vector<LandmarkListProto>`.
 * @param segmentationMasks A vector of type `std::vector<Image>`.
 * @param timestampInMilliSeconds The timestamp of the Packet that contained the result.
 *
 * @return  An `MPPPoseLandmarkerResult` object that contains the pose landmark detection
 * results.
 */
+ (MPPPoseLandmarkerResult *)
    poseLandmarkerResultWithLandmarksProto:
        (const std::vector<::mediapipe::NormalizedLandmarkList> &)landmarksProto
                       worldLandmarksProto:
                           (const std::vector<::mediapipe::LandmarkList> &)worldLandmarksProto
                         segmentationMasks:
                             (nullable const std::vector<mediapipe::Image> *)segmentationMasks
                   timestampInMilliseconds:(NSInteger)timestampInMilliseconds;

@end

NS_ASSUME_NONNULL_END
