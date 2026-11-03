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

#import "mediapipe/tasks/ios/components/containers/utils/sources/MPPLandmark+Helpers.h"

#include "mediapipe/tasks/c/components/containers/landmark.h"

@implementation MPPLandmark (Helpers)

+ (MPPLandmark *)landmarkWithProto:(const ::mediapipe::Landmark &)landmarkProto {
  return [[MPPLandmark alloc]
       initWithX:landmarkProto.x()
               y:landmarkProto.y()
               z:landmarkProto.z()
      visibility:landmarkProto.has_visibility() ? @(landmarkProto.visibility()) : nil
        presence:landmarkProto.has_presence() ? @(landmarkProto.presence()) : nil];
}

+ (MPPLandmark *)landmarkWithCLandmark:(const MpLandmark &)cLandmark {
  NSNumber *visibility = cLandmark.has_visibility ? @(cLandmark.visibility) : nil;
  NSNumber *presence = cLandmark.has_presence ? @(cLandmark.presence) : nil;
  return [[MPPLandmark alloc] initWithX:cLandmark.x
                                      y:cLandmark.y
                                      z:cLandmark.z
                             visibility:visibility
                               presence:presence];
}

+ (NSArray<MPPLandmark *> *)landmarksWithCLandmarks:(const MpLandmarks &)cLandmarks {
  NSMutableArray<MPPLandmark *> *landmarks =
      [NSMutableArray arrayWithCapacity:cLandmarks.landmarks_count];
  for (uint32_t i = 0; i < cLandmarks.landmarks_count; ++i) {
    [landmarks addObject:[MPPLandmark landmarkWithCLandmark:cLandmarks.landmarks[i]]];
  }
  return landmarks;
}

@end

@implementation MPPNormalizedLandmark (Helpers)

+ (MPPNormalizedLandmark *)normalizedLandmarkWithProto:
    (const ::mediapipe::NormalizedLandmark &)normalizedLandmarkProto {
  return [[MPPNormalizedLandmark alloc]
       initWithX:normalizedLandmarkProto.x()
               y:normalizedLandmarkProto.y()
               z:normalizedLandmarkProto.z()
      visibility:normalizedLandmarkProto.has_visibility() ? @(normalizedLandmarkProto.visibility())
                                                          : nil
        presence:normalizedLandmarkProto.has_presence() ? @(normalizedLandmarkProto.presence())
                                                        : nil];
}

+ (MPPNormalizedLandmark *)normalizedLandmarkWithCNormalizedLandmark:
    (const MpNormalizedLandmark &)cNormalizedLandmark {
  NSNumber *visibility =
      cNormalizedLandmark.has_visibility ? @(cNormalizedLandmark.visibility) : nil;
  NSNumber *presence = cNormalizedLandmark.has_presence ? @(cNormalizedLandmark.presence) : nil;
  return [[MPPNormalizedLandmark alloc] initWithX:cNormalizedLandmark.x
                                                y:cNormalizedLandmark.y
                                                z:cNormalizedLandmark.z
                                       visibility:visibility
                                         presence:presence];
}

+ (NSArray<MPPNormalizedLandmark *> *)normalizedLandmarksWithCNormalizedLandmarks:
    (const MpNormalizedLandmarks &)cNormalizedLandmarks {
  NSMutableArray<MPPNormalizedLandmark *> *landmarks =
      [NSMutableArray arrayWithCapacity:cNormalizedLandmarks.landmarks_count];
  for (uint32_t i = 0; i < cNormalizedLandmarks.landmarks_count; ++i) {
    [landmarks
        addObject:[MPPNormalizedLandmark
                      normalizedLandmarkWithCNormalizedLandmark:cNormalizedLandmarks.landmarks[i]]];
  }
  return landmarks;
}

@end
