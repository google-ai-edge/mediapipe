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

#import "mediapipe/tasks/ios/vision/gesture_recognizer/utils/sources/MPPGestureRecognizerResult+Helpers.h"

#import "mediapipe/tasks/ios/components/containers/utils/sources/MPPCategory+Helpers.h"
#import "mediapipe/tasks/ios/components/containers/utils/sources/MPPLandmark+Helpers.h"

#include "mediapipe/framework/formats/classification.pb.h"
#include "mediapipe/framework/formats/landmark.pb.h"
#include "mediapipe/framework/packet.h"

namespace {
using ClassificationListProto = ::mediapipe::ClassificationList;
using LandmarkListProto = ::mediapipe::LandmarkList;
using NormalizedLandmarkListProto = ::mediapipe::NormalizedLandmarkList;
using ::mediapipe::Packet;
}  // namespace

@implementation MPPGestureRecognizerResult (Helpers)

+ (MPPGestureRecognizerResult *)
    gestureRecognizerResultWithHandGesturesPacket:(const Packet &)handGesturesPacket
                                 handednessPacket:(const Packet &)handednessPacket
                              handLandmarksPacket:(const Packet &)handLandmarksPacket
                             worldLandmarksPacket:(const Packet &)worldLandmarksPacket {
  NSInteger timestampInMilliseconds =
      (NSInteger)(handGesturesPacket.Timestamp().Value() / kMicroSecondsPerMilliSecond);
  
  if (handGesturesPacket.IsEmpty()) {
    return [[MPPGestureRecognizerResult alloc] initWithGestures:@[]
                                                  handedness:@[] 
                                                  landmarks:@[] 
                                                    worldLandmarks:@[]
                                         timestampInMilliseconds:timestampInMilliseconds];
  }

  if (!handGesturesPacket.ValidateAsType<std::vector<ClassificationListProto>>().ok() ||
      !handednessPacket.ValidateAsType<std::vector<ClassificationListProto>>().ok() ||
      !handLandmarksPacket.ValidateAsType<std::vector<NormalizedLandmarkListProto>>().ok() ||
      !worldLandmarksPacket.ValidateAsType<std::vector<LandmarkListProto>>().ok()) {
    return nil;
  }

  const std::vector<ClassificationListProto> &handGesturesClassificationListProtos =
      handGesturesPacket.Get<std::vector<ClassificationListProto>>();
  NSMutableArray<NSMutableArray<MPPCategory *> *> *multiHandGestures =
      [NSMutableArray arrayWithCapacity:(NSUInteger)handGesturesClassificationListProtos.size()];

  for (const auto &classificationListProto : handGesturesClassificationListProtos) {
    NSMutableArray<MPPCategory *> *gestures = [NSMutableArray
        arrayWithCapacity:(NSUInteger)classificationListProto.classification().size()];
    for (const auto &classificationProto : classificationListProto.classification()) {
      MPPCategory *category = [MPPCategory categoryWithProto:classificationProto];
      [gestures addObject:category];
    }
    [multiHandGestures addObject:[gestures copy]];
  }

  const std::vector<ClassificationListProto> &handednessClassificationListProtos =
      handednessPacket.Get<std::vector<ClassificationListProto>>();
  NSMutableArray<NSMutableArray<MPPCategory *> *> *multiHandHandedness =
      [NSMutableArray arrayWithCapacity:(NSUInteger)handednessClassificationListProtos.size()];

  for (const auto &classificationListProto : handednessClassificationListProtos) {
    NSMutableArray<MPPCategory *> *handedness = [NSMutableArray
        arrayWithCapacity:(NSUInteger)classificationListProto.classification().size()];
    for (const auto &classificationProto : classificationListProto.classification()) {
      MPPCategory *category = [MPPCategory categoryWithProto:classificationProto];
      [handedness addObject:category];
    }
    [multiHandHandedness addObject:[handedness copy]];
  }

  const std::vector<NormalizedLandmarkListProto> &handLandmarkListProtos =
      handLandmarksPacket.Get<std::vector<NormalizedLandmarkListProto>>();
  NSMutableArray<NSMutableArray<MPPNormalizedLandmark *> *> *multiHandLandmarks =
      [NSMutableArray arrayWithCapacity:(NSUInteger)handLandmarkListProtos.size()];

  for (const auto &handLandmarkListProto : handLandmarkListProtos) {
    NSMutableArray<MPPNormalizedLandmark *> *handLandmarks =
        [NSMutableArray arrayWithCapacity:(NSUInteger)handLandmarkListProto.landmark().size()];
    for (const auto &normalizedLandmarkProto : handLandmarkListProto.landmark()) {
      MPPNormalizedLandmark *normalizedLandmark =
          [MPPNormalizedLandmark normalizedLandmarkWithProto:normalizedLandmarkProto];   
      [handLandmarks addObject:normalizedLandmark];
    }
    [multiHandLandmarks addObject:[handLandmarks copy]];
  }

  const std::vector<LandmarkListProto> &worldLandmarkListProtos =
      worldLandmarksPacket.Get<std::vector<LandmarkListProto>>();
  NSMutableArray<NSMutableArray<MPPLandmark *> *> *multiHandWorldLandmarks =
      [NSMutableArray arrayWithCapacity:(NSUInteger)worldLandmarkListProtos.size()];

  for (const auto &worldLandmarkListProto : worldLandmarkListProtos) {
    NSMutableArray<MPPLandmark *> *worldLandmarks =
        [NSMutableArray arrayWithCapacity:(NSUInteger)worldLandmarkListProto.landmark().size()];
    for (const auto &landmarkProto : worldLandmarkListProto.landmark()) {
      MPPLandmark *landmark =
          [MPPLandmark landmarkWithProto:landmarkProto];
      [worldLandmarks addObject:landmark];
    }
   [multiHandWorldLandmarks addObject:[worldLandmarks copy]];
  }

  MPPGestureRecognizerResult *gestureRecognizerResult =
      [[MPPGestureRecognizerResult alloc] initWithGestures:[multiHandGestures copy]
                                                handedness:[multiHandHandedness copy]
                                                 landmarks:[multiHandLandmarks copy]
                                            worldLandmarks:[multiHandWorldLandmarks copy]
                                   timestampInMilliseconds:timestampInMilliseconds];

  return gestureRecognizerResult;
}

@end
