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

namespace {
using ClassificationListProto = ::mediapipe::ClassificationList;
using LandmarkListProto = ::mediapipe::LandmarkList;
using NormalizedLandmarkListProto = ::mediapipe::NormalizedLandmarkList;
using ::mediapipe::Packet;
}  // namespace

static const NSInteger kDefaultGestureIndex = -1;

@implementation MPPGestureRecognizerResult (Helpers)

+ (MPPGestureRecognizerResult *)emptyGestureRecognizerResultWithTimestampInMilliseconds:
    (NSInteger)timestampInMilliseconds {
  return [[MPPGestureRecognizerResult alloc] initWithGestures:@[]
                                                   handedness:@[]
                                                    landmarks:@[]
                                               worldLandmarks:@[]
                                      timestampInMilliseconds:timestampInMilliseconds];
}

+ (MPPGestureRecognizerResult *)
    gestureRecognizerResultWithHandGesturesProto:
        (const std::vector<ClassificationListProto> &)handGesturesProto
                                 handednessProto:
                                     (const std::vector<ClassificationListProto> &)handednessProto
                              handLandmarksProto:(const std::vector<NormalizedLandmarkListProto> &)
                                                     handLandmarksProto
                             worldLandmarksProto:
                                 (const std::vector<LandmarkListProto> &)worldLandmarksProto
                         timestampInMilliseconds:(NSInteger)timestampInMilliseconds {
  NSMutableArray<NSMutableArray<MPPCategory *> *> *multiHandGestures =
      [NSMutableArray arrayWithCapacity:(NSUInteger)handGesturesProto.size()];

  for (const auto &classificationListProto : handGesturesProto) {
    NSMutableArray<MPPCategory *> *gestures = [NSMutableArray
        arrayWithCapacity:(NSUInteger)classificationListProto.classification().size()];
    for (const auto &classificationProto : classificationListProto.classification()) {
      MPPCategory *category = [MPPCategory categoryWithProto:classificationProto
                                                       index:kDefaultGestureIndex];
      [gestures addObject:category];
    }
    [multiHandGestures addObject:gestures];
  }

  NSMutableArray<NSMutableArray<MPPCategory *> *> *multiHandHandedness =
      [NSMutableArray arrayWithCapacity:(NSUInteger)handednessProto.size()];

  for (const auto &classificationListProto : handednessProto) {
    NSMutableArray<MPPCategory *> *handedness = [NSMutableArray
        arrayWithCapacity:(NSUInteger)classificationListProto.classification().size()];
    for (const auto &classificationProto : classificationListProto.classification()) {
      MPPCategory *category = [MPPCategory categoryWithProto:classificationProto];
      [handedness addObject:category];
    }
    [multiHandHandedness addObject:handedness];
  }

  NSMutableArray<NSMutableArray<MPPNormalizedLandmark *> *> *multiHandLandmarks =
      [NSMutableArray arrayWithCapacity:(NSUInteger)handLandmarksProto.size()];

  for (const auto &handLandmarkListProto : handLandmarksProto) {
    NSMutableArray<MPPNormalizedLandmark *> *handLandmarks =
        [NSMutableArray arrayWithCapacity:(NSUInteger)handLandmarkListProto.landmark().size()];
    for (const auto &normalizedLandmarkProto : handLandmarkListProto.landmark()) {
      MPPNormalizedLandmark *normalizedLandmark =
          [MPPNormalizedLandmark normalizedLandmarkWithProto:normalizedLandmarkProto];
      [handLandmarks addObject:normalizedLandmark];
    }
    [multiHandLandmarks addObject:handLandmarks];
  }

  NSMutableArray<NSMutableArray<MPPLandmark *> *> *multiHandWorldLandmarks =
      [NSMutableArray arrayWithCapacity:(NSUInteger)worldLandmarksProto.size()];

  for (const auto &worldLandmarkListProto : worldLandmarksProto) {
    NSMutableArray<MPPLandmark *> *worldLandmarks =
        [NSMutableArray arrayWithCapacity:(NSUInteger)worldLandmarkListProto.landmark().size()];
    for (const auto &landmarkProto : worldLandmarkListProto.landmark()) {
      MPPLandmark *landmark = [MPPLandmark landmarkWithProto:landmarkProto];
      [worldLandmarks addObject:landmark];
    }
    [multiHandWorldLandmarks addObject:worldLandmarks];
  }

  MPPGestureRecognizerResult *gestureRecognizerResult =
      [[MPPGestureRecognizerResult alloc] initWithGestures:multiHandGestures
                                                handedness:multiHandHandedness
                                                 landmarks:multiHandLandmarks
                                            worldLandmarks:multiHandWorldLandmarks
                                   timestampInMilliseconds:timestampInMilliseconds];

  return gestureRecognizerResult;
}

+ (MPPGestureRecognizerResult *)
    gestureRecognizerResultWithHandGesturesPacket:(const Packet &)handGesturesPacket
                                 handednessPacket:(const Packet &)handednessPacket
                              handLandmarksPacket:(const Packet &)handLandmarksPacket
                             worldLandmarksPacket:(const Packet &)worldLandmarksPacket {
  NSInteger timestampInMilliseconds =
      (NSInteger)(handGesturesPacket.Timestamp().Value() / kMicrosecondsPerMillisecond);

  if (handGesturesPacket.IsEmpty()) {
    return [MPPGestureRecognizerResult
        emptyGestureRecognizerResultWithTimestampInMilliseconds:timestampInMilliseconds];
  }

  if (!handGesturesPacket.ValidateAsType<std::vector<ClassificationListProto>>().ok() ||
      !handednessPacket.ValidateAsType<std::vector<ClassificationListProto>>().ok() ||
      !handLandmarksPacket.ValidateAsType<std::vector<NormalizedLandmarkListProto>>().ok() ||
      !worldLandmarksPacket.ValidateAsType<std::vector<LandmarkListProto>>().ok()) {
    return [MPPGestureRecognizerResult
        emptyGestureRecognizerResultWithTimestampInMilliseconds:timestampInMilliseconds];
  }

  return [MPPGestureRecognizerResult
      gestureRecognizerResultWithHandGesturesProto:handGesturesPacket
                                                       .Get<std::vector<ClassificationListProto>>()
                                   handednessProto:handednessPacket
                                                       .Get<std::vector<ClassificationListProto>>()
                                handLandmarksProto:handLandmarksPacket.Get<
                                                       std::vector<NormalizedLandmarkListProto>>()
                               worldLandmarksProto:worldLandmarksPacket
                                                       .Get<std::vector<LandmarkListProto>>()
                           timestampInMilliseconds:timestampInMilliseconds];
}

@end
