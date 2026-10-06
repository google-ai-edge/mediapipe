// Copyright 2026 The MediaPipe Authors.
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

#import <Foundation/Foundation.h>
#import "mediapipe/tasks/ios/components/containers/sources/MPPDetection.h"

NS_ASSUME_NONNULL_BEGIN

/**
 * @brief Specifies the brush mode used for a stroke.
 */
typedef NS_ENUM(NSUInteger, MPPBrushMode) {
  MPPBrushModeUnspecified = 0,
  MPPBrushModePositive = 1,
  MPPBrushModeNegative = 2,
  MPPBrushModeLasso = 3,
} NS_SWIFT_NAME(BrushMode);

/**
 * @brief Represents a single user-drawn stroke for interactive segmentation.
 */
NS_SWIFT_NAME(Stroke)
@interface MPPStroke : NSObject

/**
 * The sequence of normalized key points forming the stroke.
 */
@property(nonatomic, readonly) NSArray<MPPNormalizedKeypoint *> *points;

/**
 * The brush mode used for this stroke.
 */
@property(nonatomic, readonly) MPPBrushMode brushMode;

/**
 * Whether the stroke is complete.
 */
@property(nonatomic, readonly) BOOL isCompleted;

/**
 * Initializes a new `Stroke` with the given points, brush mode, and completion status.
 *
 * @param points The sequence of normalized key points forming the stroke.
 * @param brushMode The brush mode used for this stroke.
 * @param isCompleted Whether the stroke is complete.
 *
 * @return An instance of `Stroke` initialized with the given parameters.
 */
- (instancetype)initWithPoints:(NSArray<MPPNormalizedKeypoint *> *)points
                     brushMode:(MPPBrushMode)brushMode
                   isCompleted:(BOOL)isCompleted NS_DESIGNATED_INITIALIZER;

- (instancetype)init NS_UNAVAILABLE;

+ (instancetype)new NS_UNAVAILABLE;

@end

NS_ASSUME_NONNULL_END
