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

#import "mediapipe/tasks/ios/vision/hand_landmarker/utils/sources/MPPHandLandmarkerResult+Helpers.h"

#import "mediapipe/tasks/ios/components/containers/utils/sources/MPPCategory+Helpers.h"
#import "mediapipe/tasks/ios/components/containers/utils/sources/MPPLandmark+Helpers.h"

namespace {
using ClassificationListProto = ::mediapipe::ClassificationList;
using LandmarkListProto = ::mediapipe::LandmarkList;
using NormalizedLandmarkListProto = ::mediapipe::NormalizedLandmarkList;
using ::mediapipe::Packet;
}  // namespace

@implementation MPPHandLandmarkerResult (Helpers)

+ (MPPHandLandmarkerResult *)emptyHandLandmarkerResultWithTimestampInMilliseconds:
    (NSInteger)timestampInMilliseconds {
  return [[MPPHandLandmarkerResult alloc] initWithLandmarks:@[]
                                             worldLandmarks:@[]
                                                 handedness:@[]
                                    timestampInMilliseconds:timestampInMilliseconds];
}

+ (MPPHandLandmarkerResult *)
    handLandmarkerResultWithLandmarksProto:
        (const std::vector<NormalizedLandmarkListProto> &)landmarksProto
                       worldLandmarksProto:
                           (const std::vector<LandmarkListProto> &)worldLandmarksProto
                           handednessProto:
                               (const std::vector<ClassificationListProto> &)handednessProto
                   timestampInMilliSeconds:(NSInteger)timestampInMilliseconds {
  NSMutableArray<NSMutableArray<MPPNormalizedLandmark *> *> *multiHandLandmarks =
      [NSMutableArray arrayWithCapacity:(NSUInteger)landmarksProto.size()];

  for (const auto &landmarkListProto : landmarksProto) {
    NSMutableArray<MPPNormalizedLandmark *> *landmarks =
        [NSMutableArray arrayWithCapacity:(NSUInteger)landmarkListProto.landmark().size()];
    for (const auto &normalizedLandmarkProto : landmarkListProto.landmark()) {
      MPPNormalizedLandmark *normalizedLandmark =
          [MPPNormalizedLandmark normalizedLandmarkWithProto:normalizedLandmarkProto];
      [landmarks addObject:normalizedLandmark];
    }
    [multiHandLandmarks addObject:landmarks];
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

  MPPHandLandmarkerResult *handLandmarkerResult =
      [[MPPHandLandmarkerResult alloc] initWithLandmarks:multiHandLandmarks
                                          worldLandmarks:multiHandWorldLandmarks
                                              handedness:multiHandHandedness
                                 timestampInMilliseconds:timestampInMilliseconds];
  return handLandmarkerResult;
}

+ (MPPHandLandmarkerResult *)
    handLandmarkerResultWithLandmarksPacket:(const Packet &)landmarksPacket
                       worldLandmarksPacket:(const Packet &)worldLandmarksPacket
                           handednessPacket:(const Packet &)handednessPacket {
  NSInteger timestampInMilliseconds =
      (NSInteger)(landmarksPacket.Timestamp().Value() / kMicrosecondsPerMillisecond);

  if (landmarksPacket.IsEmpty()) {
    return [MPPHandLandmarkerResult
        emptyHandLandmarkerResultWithTimestampInMilliseconds:timestampInMilliseconds];
  }

  if (!handednessPacket.ValidateAsType<std::vector<ClassificationListProto>>().ok() ||
      !landmarksPacket.ValidateAsType<std::vector<NormalizedLandmarkListProto>>().ok() ||
      !worldLandmarksPacket.ValidateAsType<std::vector<LandmarkListProto>>().ok()) {
    return [MPPHandLandmarkerResult
        emptyHandLandmarkerResultWithTimestampInMilliseconds:timestampInMilliseconds];
  }

  return [MPPHandLandmarkerResult
      handLandmarkerResultWithLandmarksProto:landmarksPacket
                                                 .Get<std::vector<NormalizedLandmarkListProto>>()
                         worldLandmarksProto:worldLandmarksPacket
                                                 .Get<std::vector<LandmarkListProto>>()
                             handednessProto:handednessPacket
                                                 .Get<std::vector<ClassificationListProto>>()
                     timestampInMilliSeconds:timestampInMilliseconds];
}

@end
