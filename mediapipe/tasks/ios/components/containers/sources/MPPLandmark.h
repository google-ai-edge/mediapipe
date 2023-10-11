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

#import <Foundation/Foundation.h>

NS_ASSUME_NONNULL_BEGIN

/**
 * Landmark represents a point in 3D space with x, y, z coordinates. The landmark coordinates are in
 * meters. z represents the landmark depth, and the smaller the value the closer the world landmark
 * is to the camera.
 */
NS_SWIFT_NAME(Landmark)
@interface MPPLandmark : NSObject

/** The x coordinates of the landmark. */
@property(nonatomic, readonly) float x;

/** The y coordinates of the landmark. */
@property(nonatomic, readonly) float y;

/** The z coordinates of the landmark. */
@property(nonatomic, readonly) float z;

/**
 * Landmark visibility. Should be `nil` if not supported. Float score of whether landmark is visible
 * or occluded by other objects. Landmark considered as invisible also if it is not present on the
 * screen (out of scene bounds). Depending on the model, visibility value is either a sigmoid or an
 * argument of sigmoid.
 */
@property(nonatomic, readonly, nullable) NSNumber *visibility;

/**
 * Landmark presence. Should stay unset if not supported. Float score of whether landmark is present
 * on the scene (located within scene bounds). Depending on the model, presence value is either a
 * result of sigmoid or an argument of sigmoid function to get landmark presence probability.
 */
@property(nonatomic, readonly, nullable) NSNumber *presence;

/**
 * Initializes a new `MPPLandmark` object with the given x, y and z coordinates.
 *
 * @param x The x coordinates of the landmark.
 * @param y The y coordinates of the landmark.
 * @param z The z coordinates of the landmark.
 *
 * @return An instance of `MPPLandmark` initialized with the given x, y and z coordinates.
 */
- (instancetype)initWithX:(float)x
                        y:(float)y
                        z:(float)z
               visibility:(nullable NSNumber *)visibility
                 presence:(nullable NSNumber *)presence NS_DESIGNATED_INITIALIZER;

- (instancetype)init NS_UNAVAILABLE;

+ (instancetype)new NS_UNAVAILABLE;

@end

/**
 * Normalized Landmark represents a point in 3D space with x, y, z coordinates. x and y are
 * normalized to [0.0, 1.0] by the image width and height respectively. z represents the landmark
 * depth, and the smaller the value the closer the landmark is to the camera. The magnitude of z
 * uses roughly the same scale as x.
 */
NS_SWIFT_NAME(NormalizedLandmark)
@interface MPPNormalizedLandmark : NSObject

/** The x coordinates of the landmark. */
@property(nonatomic, readonly) float x;

/** The y coordinates of the landmark. */
@property(nonatomic, readonly) float y;

/** The z coordinates of the landmark. */
@property(nonatomic, readonly) float z;

/**
 * Landmark visibility. Should be `nil` if not supported. Float score of whether landmark is visible
 * or occluded by other objects. Landmark considered as invisible also if it is not present on the
 * screen (out of scene bounds). Depending on the model, visibility value is either a sigmoid or an
 * argument of sigmoid.
 */
@property(nonatomic, readonly, nullable) NSNumber *visibility;

/**
 * Landmark presence. Should stay unset if not supported. Float score of whether landmark is present
 * on the scene (located within scene bounds). Depending on the model, presence value is either a
 * result of sigmoid or an argument of sigmoid function to get landmark presence probability.
 */
@property(nonatomic, readonly, nullable) NSNumber *presence;

/**
 * Initializes a new `MPPNormalizedLandmark` object with the given x, y and z coordinates.
 *
 * @param x The x coordinates of the landmark.
 * @param y The y coordinates of the landmark.
 * @param z The z coordinates of the landmark.
 *
 * @return An instance of `MPPNormalizedLandmark` initialized with the given x, y and z coordinates.
 */
- (instancetype)initWithX:(float)x
                        y:(float)y
                        z:(float)z
               visibility:(nullable NSNumber *)visibility
                 presence:(nullable NSNumber *)presence NS_DESIGNATED_INITIALIZER;

- (instancetype)init NS_UNAVAILABLE;

+ (instancetype)new NS_UNAVAILABLE;

@end

NS_ASSUME_NONNULL_END
