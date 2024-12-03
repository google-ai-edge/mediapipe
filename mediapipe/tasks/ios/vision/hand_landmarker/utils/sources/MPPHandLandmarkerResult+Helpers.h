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

#import "mediapipe/tasks/ios/vision/hand_landmarker/sources/MPPHandLandmarkerResult.h"

#include "mediapipe/framework/formats/classification.pb.h"
#include "mediapipe/framework/formats/landmark.pb.h"
#include "mediapipe/framework/packet.h"

NS_ASSUME_NONNULL_BEGIN

static const int kMicrosecondsPerMillisecond = 1000;

@interface MPPHandLandmarkerResult (Helpers)

/**
 * Creates an `MPPHandLandmarkerResult` from landmarks, world landmarks and handedness packets.
 *
 * @param landmarksPacket A MediaPipe packet wrapping a `std::vector<NormalizedlandmarkListProto>`.
 * @param worldLandmarksPacket A MediaPipe packet wrapping a `std::vector<LandmarkListProto>`.
 * @param handednessPacket a MediaPipe packet wrapping a `std::vector<ClassificationListProto>`.
 *
 * @return  An `MPPHandLandmarkerResult` object that contains the hand landmark detection
 * results.
 */
+ (MPPHandLandmarkerResult *)
    handLandmarkerResultWithLandmarksPacket:(const mediapipe::Packet &)handLandmarksPacket
                       worldLandmarksPacket:(const mediapipe::Packet &)worldLandmarksPacket
                           handednessPacket:(const mediapipe::Packet &)handednessPacket;

/**
 * Creates an `MPPHandLandmarkerResult` from landmarks, world landmarks and handedness proto
 * vectors.
 *
 * @param landmarksProto A vector of protos of type `std::vector<NormalizedlandmarkListProto>`.
 * @param worldLandmarksPacket A vector of protos of type `std::vector<LandmarkListProto>`.
 * @param handednessPacket A vector of protos of type `std::vector<ClassificationListProto>`.
 * @param timestampInMilliSeconds The timestamp of the Packet that contained the result.
 *
 * @return  An `MPPHandLandmarkerResult` object that contains the hand landmark detection
 * results.
 */
+ (MPPHandLandmarkerResult *)
    handLandmarkerResultWithLandmarksProto:
        (const std::vector<::mediapipe::NormalizedLandmarkList> &)landmarksProto
                       worldLandmarksProto:
                           (const std::vector<::mediapipe::LandmarkList> &)worldLandmarksProto
                           handednessProto:
                               (const std::vector<::mediapipe::ClassificationList> &)handednessProto
                   timestampInMilliSeconds:(NSInteger)timestampInMilliseconds;

@end

NS_ASSUME_NONNULL_END
