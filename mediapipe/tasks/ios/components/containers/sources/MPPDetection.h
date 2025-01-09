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
#import <UIKit/UIKit.h>
#import "mediapipe/tasks/ios/components/containers/sources/MPPCategory.h"

NS_ASSUME_NONNULL_BEGIN

/**
 * Normalized keypoint represents a point in 2D space with x, y coordinates. x and y are normalized
 * to [0.0, 1.0] by the image width and height respectively.
 */
NS_SWIFT_NAME(NormalizedKeypoint)
@interface MPPNormalizedKeypoint : NSObject

/** The (x,y) coordinates location of the normalized keypoint. */
@property(nonatomic, readonly) CGPoint location;

/** The optional label of the normalized keypoint. */
@property(nonatomic, readonly, nullable) NSString *label;

/** The optional score of the normalized keypoint. If score is absent, it will be equal to 0.0. */
@property(nonatomic, readonly) float score;

/**
 * Initializes a new `NormalizedKeypoint` object with the given location, label and score.
 * You must pass 0.0 for `score` if it is not present.
 *
 * @param location The (x,y) coordinates location of the normalized keypoint.
 * @param label  The optional label of the normalized keypoint.
 * @param score The optional score of the normalized keypoint. You must pass 0.0 for score if it
 * is not present.
 *
 * @return An instance of `NormalizedKeypoint` initialized with the given given location, label
 * and score.
 */
- (instancetype)initWithLocation:(CGPoint)location
                           label:(nullable NSString *)label
                           score:(float)score NS_DESIGNATED_INITIALIZER;

- (instancetype)init NS_UNAVAILABLE;

+ (instancetype)new NS_UNAVAILABLE;

@end

/** Represents one detected object in the results of `ObjectDetector`. */
NS_SWIFT_NAME(Detection)
@interface MPPDetection : NSObject

/** An array of `Category` objects containing the predicted categories. */
@property(nonatomic, readonly) NSArray<MPPCategory *> *categories;

/** The bounding box of the detected object. */
@property(nonatomic, readonly) CGRect boundingBox;

/**
 * An optional array of `NormalizedKeypoint` objects associated with the detection. Keypoints
 * represent interesting points related to the detection. For example, the keypoints represent the
 * eyes, ear and mouth from the from detection model. In template matching detection, e.g. KNIFT,
 * they can instead represent the feature points for template matching.
 */
@property(nonatomic, readonly, nullable) NSArray<MPPNormalizedKeypoint *> *keypoints;

/**
 * Initializes a new `Detection` object with the given array of categories, bounding box and
 * optional array of keypoints;
 *
 * @param categories A list of `Category` objects that contain category name, display name,
 * score, and the label index.
 * @param boundingBox  A `CGRect` that represents the bounding box.
 * @param keypoints: An optional array of `NormalizedKeypoint` objects associated with the
 * detection. Keypoints represent interesting points related to the detection. For example, the
 * keypoints represent the eyes, ear and mouth from the face detection model. In template matching
 * detection, e.g. KNIFT, they can instead represent the feature points for template matching.
 *
 * @return An instance of `Detection` initialized with the given array of categories, bounding
 * box and `nil` keypoints.
 */
- (instancetype)initWithCategories:(NSArray<MPPCategory *> *)categories
                       boundingBox:(CGRect)boundingBox
                         keypoints:(nullable NSArray<MPPNormalizedKeypoint *> *)keypoints
    NS_DESIGNATED_INITIALIZER;

- (instancetype)init NS_UNAVAILABLE;

+ (instancetype)new NS_UNAVAILABLE;

@end

NS_ASSUME_NONNULL_END
