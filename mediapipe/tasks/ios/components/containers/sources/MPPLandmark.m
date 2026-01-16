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

#import "mediapipe/tasks/ios/components/containers/sources/MPPLandmark.h"

static const float kFloatDifferenceTolerance = 1e-6f;

@implementation MPPLandmark

- (instancetype)initWithX:(float)x
                        y:(float)y
                        z:(float)z
               visibility:(NSNumber *)visibility
                 presence:(NSNumber *)presence {
  self = [super init];
  if (self) {
    _x = x;
    _y = y;
    _z = z;
    _visibility = visibility;
    _presence = presence;
  }
  return self;
}

- (NSUInteger)hash {
  return @(self.x).hash ^ @(self.y).hash ^ @(self.z).hash;
}

- (BOOL)isEqual:(nullable id)object {
  if (!object) {
    return NO;
  }

  if (self == object) {
    return YES;
  }

  if (![object isKindOfClass:[MPPLandmark class]]) {
    return NO;
  }

  MPPLandmark *otherLandmark = (MPPLandmark *)object;

  return fabsf(otherLandmark.x - self.x) < kFloatDifferenceTolerance &&
         fabsf(otherLandmark.y - self.y) < kFloatDifferenceTolerance &&
         fabsf(otherLandmark.z - self.z) < kFloatDifferenceTolerance;
}

@end

@implementation MPPNormalizedLandmark

- (instancetype)initWithX:(float)x
                        y:(float)y
                        z:(float)z
               visibility:(NSNumber *)visibility
                 presence:(NSNumber *)presence {
  self = [super init];
  if (self) {
    _x = x;
    _y = y;
    _z = z;
    _visibility = visibility;
    _presence = presence;
  }
  return self;
}

- (NSUInteger)hash {
  return @(self.x).hash ^ @(self.y).hash ^ @(self.z).hash;
}

- (BOOL)isEqual:(nullable id)object {
  if (!object) {
    return NO;
  }

  if (self == object) {
    return YES;
  }

  if (![object isKindOfClass:[MPPNormalizedLandmark class]]) {
    return NO;
  }

  MPPNormalizedLandmark *otherLandmark = (MPPNormalizedLandmark *)object;

  return fabsf(otherLandmark.x - self.x) < kFloatDifferenceTolerance &&
         fabsf(otherLandmark.y - self.y) < kFloatDifferenceTolerance &&
         fabsf(otherLandmark.z - self.z) < kFloatDifferenceTolerance;
}

@end
