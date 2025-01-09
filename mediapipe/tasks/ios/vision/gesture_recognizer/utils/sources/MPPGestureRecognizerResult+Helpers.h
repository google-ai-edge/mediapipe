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

#import "mediapipe/tasks/ios/vision/gesture_recognizer/sources/MPPGestureRecognizerResult.h"

#include "mediapipe/framework/formats/classification.pb.h"
#include "mediapipe/framework/formats/landmark.pb.h"
#include "mediapipe/framework/packet.h"

NS_ASSUME_NONNULL_BEGIN

static const int kMicrosecondsPerMillisecond = 1000;

@interface MPPGestureRecognizerResult (Helpers)

/**
 * Creates an `MPPGestureRecognizerResult` from  hand gestures, handedness, hand landmarks and world
 * landmarks packets.
 *
 * @param handGesturesPacket A MediaPipe packet wrapping a`std::vector<ClassificationListProto>`.
 * @param handednessPacket A MediaPipe packet wrapping a`std::vector<ClassificationListProto>`.
 * @param handLandmarksPacket A MediaPipe packet wrapping
 * a`std::vector<NormalizedlandmarkListProto>`.
 * @param worldLandmarksPacket A MediaPipe packet wrapping a`std::vector<LandmarkListProto>`.
 *
 * @return  An `MPPGestureRecognizerResult` object that contains the hand gesture recognition
 * results.
 */
+ (MPPGestureRecognizerResult *)
    gestureRecognizerResultWithHandGesturesPacket:(const mediapipe::Packet &)handGesturesPacket
                                 handednessPacket:(const mediapipe::Packet &)handednessPacket
                              handLandmarksPacket:(const mediapipe::Packet &)handLandmarksPacket
                             worldLandmarksPacket:(const mediapipe::Packet &)worldLandmarksPacket;

/**
 * Creates an `MPPGestureRecognizerResult` from hand gestures, handedness, hand landmarks and world
 * landmarks proto vectors.
 *
 * @param handGesturesProto A vector of protos of type `std::vector<ClassificationListProto>`.
 * @param handednessPacket A vector of protos of type  `std::vector<ClassificationListProto>`.
 * @param handLandmarksPacket A vector of protos of type `std::vector<NormalizedlandmarkListProto>`.
 * @param handLandmarksPacket A vector of protos of type `std::vector<LandmarkListProto>`.
 * @param timestampInMilliSeconds The timestamp of the result.
 *
 * @return  An `MPPGestureRecognizerResult` object that contains the hand gesture recognition
 * results.
 */
+ (MPPGestureRecognizerResult *)
    gestureRecognizerResultWithHandGesturesProto:
        (const std::vector<::mediapipe::ClassificationList> &)handGesturesProto
                                 handednessProto:
                                     (const std::vector<::mediapipe::ClassificationList> &)
                                         handednessProto
                              handLandmarksProto:
                                  (const std::vector<::mediapipe::NormalizedLandmarkList> &)
                                      handLandmarksProto
                             worldLandmarksProto:
                                 (const std::vector<::mediapipe::LandmarkList> &)worldLandmarksProto
                         timestampInMilliseconds:(NSInteger)timestampInMilliseconds;
@end

NS_ASSUME_NONNULL_END
