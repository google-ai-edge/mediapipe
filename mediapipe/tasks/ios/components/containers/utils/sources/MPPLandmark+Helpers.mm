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

#import "mediapipe/tasks/ios/common/utils/sources/NSString+Helpers.h"

namespace {
using LandmarkProto = ::mediapipe::Landmark;
using NormalizedLandmarkProto = ::mediapipe::NormalizedLandmark;
}  // namespace

@implementation MPPLandmark (Helpers)

+ (MPPLandmark *)landmarkWithProto:(const ::mediapipe::Landmark &)landmarkProto {
  return [[MPPLandmark alloc]
       initWithX:landmarkProto.x()
               y:landmarkProto.y()
               z:landmarkProto.z()
      visibility:landmarkProto.has_visibility() ? @(landmarkProto.visibility()) : nil
        presence:landmarkProto.has_presence() ? @(landmarkProto.presence()) : nil];
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

@end
