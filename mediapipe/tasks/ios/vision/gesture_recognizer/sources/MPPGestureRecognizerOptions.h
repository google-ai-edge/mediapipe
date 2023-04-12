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

#import "mediapipe/tasks/ios/components/processors/sources/MPPClassifierOptions.h"
#import "mediapipe/tasks/ios/core/sources/MPPTaskOptions.h"
#import "mediapipe/tasks/ios/vision/core/sources/MPPRunningMode.h"
#import "mediapipe/tasks/ios/vision/gesture_recognizer/sources/MPPGestureRecognizerResult.h"

NS_ASSUME_NONNULL_BEGIN

/** Options for setting up a `MPPGestureRecognizer`. */
NS_SWIFT_NAME(GestureRecognizerOptions)
@interface MPPGestureRecognizerOptions : MPPTaskOptions <NSCopying>

@property(nonatomic) MPPRunningMode runningMode;

/**
 * The user-defined result callback for processing live stream data. The result callback should only
 * be specified when the running mode is set to the live stream mode.
 * TODO: Add parameter `MPPImage` in the callback.
 */
@property(nonatomic, copy) void (^completion)
    (MPPGestureRecognizerResult *result, NSInteger timestampMs, NSError *error);

/** Sets the maximum number of hands can be detected by the GestureRecognizer. */
@property(nonatomic) NSInteger numHands;

/** Sets minimum confidence score for the hand detection to be considered successful */
@property(nonatomic) float minHandDetectionConfidence;

/** Sets minimum confidence score of hand presence score in the hand landmark detection. */
@property(nonatomic) float minHandPresenceConfidence;

/** Sets the minimum confidence score for the hand tracking to be considered successful. */
@property(nonatomic) float minTrackingConfidence;

/**
 * Sets the optional `MPPClassifierOptions` controlling the canned gestures classifier, such as
 * score threshold, allow list and deny list of gestures. The categories for canned gesture
 * classifiers are: ["None", "Closed_Fist", "Open_Palm", "Pointing_Up", "Thumb_Down", "Thumb_Up",
 * "Victory", "ILoveYou"].
 *
 * TODO:  Note this option is subject to change, after scoring merging calculator is implemented.
 */
@property(nonatomic, copy, nullable) MPPClassifierOptions *cannedGesturesClassifierOptions;

/**
 * Sets the optional {@link ClassifierOptions} controlling the custom gestures classifier, such as
 * score threshold, allow list and deny list of gestures.
 *
 * TODO:  Note this option is subject to change, after scoring merging calculator is implemented.
 */
@property(nonatomic, copy, nullable) MPPClassifierOptions *customGesturesClassifierOptions;

@end

NS_ASSUME_NONNULL_END
