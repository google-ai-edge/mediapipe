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

#import "mediapipe/tasks/ios/components/containers/sources/MPPDetection.h"

@implementation MPPNormalizedKeypoint

- (instancetype)initWithLocation:(CGPoint)location
                           label:(nullable NSString *)label
                           score:(float)score {
  self = [super init];
  if (self) {
    _location = location;
    _label = label;
    _score = score;
  }
  return self;
}

- (NSUInteger)hash {
  NSUInteger nonNullPropertiesHash =
      @(self.location.x).hash ^ @(self.location.y).hash ^ @(self.score).hash;

  return self.label ? nonNullPropertiesHash ^ self.label.hash : nonNullPropertiesHash;
}

- (BOOL)isEqual:(nullable id)object {
  if (!object) {
    return NO;
  }

  if (self == object) {
    return YES;
  }

  if (![object isKindOfClass:[MPPNormalizedKeypoint class]]) {
    return NO;
  }

  MPPNormalizedKeypoint *otherKeypoint = (MPPNormalizedKeypoint *)object;

  return CGPointEqualToPoint(self.location, otherKeypoint.location) &&
         (self.label == otherKeypoint.label) && (self.score == otherKeypoint.score);
}

@end

@implementation MPPDetection

- (instancetype)initWithCategories:(NSArray<MPPCategory *> *)categories
                       boundingBox:(CGRect)boundingBox
                         keypoints:(nullable NSArray<MPPNormalizedKeypoint *> *)keypoints {
  self = [super init];
  if (self) {
    _categories = categories;
    _boundingBox = boundingBox;
    _keypoints = keypoints;
  }
  return self;
}

@end
