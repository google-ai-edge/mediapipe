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

@class MPPGestureRecognizer;

/**
 * This protocol defines an interface for the delegates of `GestureRecognizer` object to receive
 * results of performing asynchronous gesture recognition on images (i.e, when `runningMode` =
 * `.liveStream`).
 *
 * The delegate of `GestureRecognizer` must adopt `GestureRecognizerLiveStreamDelegate` protocol.
 * The methods in this protocol are optional.
 */
NS_SWIFT_NAME(GestureRecognizerLiveStreamDelegate)
@protocol MPPGestureRecognizerLiveStreamDelegate <NSObject>

@optional

/**
 * This method notifies a delegate that the results of asynchronous gesture recognition of an image
 * submitted to the `GestureRecognizer` is available.
 *
 * This method is called on a private serial dispatch queue created by the `GestureRecognizer` for
 * performing the asynchronous delegates calls.
 *
 * @param gestureRecognizer The gesture recognizer which performed the gesture recognition. This is
 * useful to test equality when there are multiple instances of `GestureRecognizer`.
 * @param result The `GestureRecognizerResult` object that contains a list of detections, each
 * detection has a bounding box that is expressed in the unrotated input frame of reference
 * coordinates system, i.e. in `[0,image_width) x [0,image_height)`, which are the dimensions of the
 * underlying image data.
 * @param timestampInMilliseconds The timestamp (in milliseconds) which indicates when the input
 * image was sent to the gesture recognizer.
 * @param error An optional error parameter populated when there is an error in performing gesture
 * recognition on the input live stream image data.
 *
 */
- (void)gestureRecognizer:(MPPGestureRecognizer *)gestureRecognizer
    didFinishRecognitionWithResult:(nullable MPPGestureRecognizerResult *)result
           timestampInMilliseconds:(NSInteger)timestampInMilliseconds
                             error:(nullable NSError *)error
    NS_SWIFT_NAME(gestureRecognizer(_:didFinishGestureRecognition:timestampInMilliseconds:error:));
@end

/** Options for setting up a `GestureRecognizer`. */
NS_SWIFT_NAME(GestureRecognizerOptions)
@interface MPPGestureRecognizerOptions : MPPTaskOptions <NSCopying>

/**
 * Running mode of the gesture recognizer task. Defaults to `.video`.
 * `GestureRecognizer` can be created with one of the following running modes:
 *  1. `image`: The mode for performing gesture recognition on single image inputs.
 *  2. `video`: The mode for performing gesture recognition on the decoded frames of a video.
 *  3. `liveStream`: The mode for performing gesture recognition on a live stream of input data,
 * such as from the camera.
 */
@property(nonatomic) MPPRunningMode runningMode;

/**
 * An object that confirms to `GestureRecognizerLiveStreamDelegate` protocol. This object must
 * implement `gestureRecognizer(_:didFinishRecognitionWithResult:timestampInMilliseconds:error:)` to
 * receive the results of performing asynchronous gesture recognition on images (i.e, when
 * `runningMode` = `.liveStream`).
 */
@property(nonatomic, weak, nullable) id<MPPGestureRecognizerLiveStreamDelegate>
    gestureRecognizerLiveStreamDelegate;

/** Sets the maximum number of hands can be detected by the GestureRecognizer. */
@property(nonatomic) NSInteger numHands;

/** Sets minimum confidence score for the hand detection to be considered successful */
@property(nonatomic) float minHandDetectionConfidence;

/** Sets minimum confidence score of hand presence score in the hand landmark detection. */
@property(nonatomic) float minHandPresenceConfidence;

/** Sets the minimum confidence score for the hand tracking to be considered successful. */
@property(nonatomic) float minTrackingConfidence;

/**
 * Sets the optional `ClassifierOptions` controlling the canned gestures classifier, such as score
 * threshold, allow list and deny list of gestures. The categories for canned gesture classifiers
 * are: ["None", "Closed_Fist", "Open_Palm", "Pointing_Up", "Thumb_Down", "Thumb_Up", "Victory",
 * "ILoveYou"].
 *
 * TODO:  Note this option is subject to change, after scoring merging calculator is implemented.
 */
@property(nonatomic, copy, nullable) MPPClassifierOptions *cannedGesturesClassifierOptions;

/**
 * Sets the optional `ClassifierOptions` controlling the custom gestures classifier, such as score
 * threshold, allow list and deny list of gestures.
 *
 * TODO:  Note this option is subject to change, after scoring merging calculator is implemented.
 */
@property(nonatomic, copy, nullable) MPPClassifierOptions *customGesturesClassifierOptions;

@end

NS_ASSUME_NONNULL_END
