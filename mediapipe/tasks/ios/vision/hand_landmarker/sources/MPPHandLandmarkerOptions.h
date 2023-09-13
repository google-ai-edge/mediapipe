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

#import "mediapipe/tasks/ios/core/sources/MPPTaskOptions.h"
#import "mediapipe/tasks/ios/vision/core/sources/MPPRunningMode.h"
#import "mediapipe/tasks/ios/vision/hand_landmarker/sources/MPPHandLandmarkerResult.h"

NS_ASSUME_NONNULL_BEGIN

@class MPPHandLandmarker;

/**
 * This protocol defines an interface for the delegates of `HandLandmarker` object to receive
 * results of performing asynchronous hand landmark detection on images (i.e, when
 * `runningMode` = `.liveStream`).
 *
 * The delegate of `HandLandmarker` must adopt `HandLandmarkerLiveStreamDelegate` protocol.
 * The methods in this protocol are optional.
 */
NS_SWIFT_NAME(HandLandmarkerLiveStreamDelegate)
@protocol MPPHandLandmarkerLiveStreamDelegate <NSObject>

@optional

/**
 * This method notifies a delegate that the results of asynchronous hand landmark detection of an
 * image submitted to the `HandLandmarker` is available.
 *
 * This method is called on a private serial dispatch queue created by the `HandLandmarker` for
 * performing the asynchronous delegates calls.
 *
 * @param handLandmarker The hand landmarker which performed the hand landmarking.
 * This is useful to test equality when there are multiple instances of `HandLandmarker`.
 * @param result The `HandLandmarkerResult` object that contains a list of detections, each
 * detection has a bounding box that is expressed in the unrotated input frame of reference
 * coordinates system, i.e. in `[0,image_width) x [0,image_height)`, which are the dimensions of the
 * underlying image data.
 * @param timestampInMilliseconds The timestamp (in milliseconds) which indicates when the input
 * image was sent to the hand landmarker.
 * @param error An optional error parameter populated when there is an error in performing hand
 * landmark detection on the input live stream image data.
 */
- (void)handLandmarker:(MPPHandLandmarker *)handLandmarker
    didFinishDetectionWithResult:(nullable MPPHandLandmarkerResult *)result
         timestampInMilliseconds:(NSInteger)timestampInMilliseconds
                           error:(nullable NSError *)error
    NS_SWIFT_NAME(handLandmarker(_:didFinishDetection:timestampInMilliseconds:error:));
@end

/** Options for setting up a `HandLandmarker`. */
NS_SWIFT_NAME(HandLandmarkerOptions)
@interface MPPHandLandmarkerOptions : MPPTaskOptions <NSCopying>

/**
 * Running mode of the hand landmarker task. Defaults to `.image`.
 * `HandLandmarker` can be created with one of the following running modes:
 *  1. `image`: The mode for performing hand landmark detection on single image inputs.
 *  2. `video`: The mode for performing hand landmark detection on the decoded frames of a video.
 *  3. `liveStream`: The mode for performing hand landmark detection on a live stream of input data,
 * such as from the camera.
 */
@property(nonatomic) MPPRunningMode runningMode;

/**
 * An object that confirms to `HandLandmarkerLiveStreamDelegate` protocol. This object must
 * implement `handLandmarker:didFinishDetectionWithResult:timestampInMilliseconds:error:` to
 * receive the results of performing asynchronous hand landmark detection on images (i.e, when
 * `runningMode` = `.liveStream`).
 */
@property(nonatomic, weak, nullable) id<MPPHandLandmarkerLiveStreamDelegate>
    handLandmarkerLiveStreamDelegate;

/** The maximum number of hands that can be detected by the `HandLandmarker`. */
@property(nonatomic) NSInteger numHands;

/** The minimum confidence score for the hand detection to be considered successful. */
@property(nonatomic) float minHandDetectionConfidence;

/** The minimum confidence score of hand presence score in the hand landmark detection. */
@property(nonatomic) float minHandPresenceConfidence;

/** The minimum confidence score for the hand tracking to be considered successful. */
@property(nonatomic) float minTrackingConfidence;

@end

NS_ASSUME_NONNULL_END
