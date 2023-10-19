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
#import "mediapipe/tasks/ios/vision/pose_landmarker/sources/MPPPoseLandmarkerResult.h"

NS_ASSUME_NONNULL_BEGIN

@class MPPPoseLandmarker;

/**
 * This protocol defines an interface for the delegates of `PoseLandmarker` to receive
 * results of performing asynchronous pose landmark detection on images (i.e, when `runningMode` =
 * `.liveStream`).
 *
 * The delegate of `PoseLandmarker` must adopt `PoseLandmarkerLiveStreamDelegate` protocol.
 * The methods in this protocol are optional.
 */
NS_SWIFT_NAME(PoseLandmarkerLiveStreamDelegate)
@protocol MPPPoseLandmarkerLiveStreamDelegate <NSObject>

/**
 * This method notifies a delegate that the results of asynchronous pose landmark detection of an
 * image submitted to the `PoseLandmarker` is available.
 *
 * This method is called on a private serial dispatch queue created by the `PoseLandmarker`
 * for performing the asynchronous delegates calls.
 *
 * @param poseLandmarker The pose landmarker which performed the pose landmark detection.
 * This is useful to test equality when there are multiple instances of `PoseLandmarker`.
 * @param result The `PoseLandmarkerResult` object that contains a list of landmark.
 * @param timestampInMilliseconds The timestamp (in milliseconds) which indicates when the input
 * image was sent to the pose landmarker.
 * @param error An optional error parameter populated when there is an error in performing pose
 * landmark detection on the input live stream image data.
 */
- (void)poseLandmarker:(MPPPoseLandmarker *)poseLandmarker
    didFinishDetectionWithResult:(nullable MPPPoseLandmarkerResult *)result
         timestampInMilliseconds:(NSInteger)timestampInMilliseconds
                           error:(nullable NSError *)error
    NS_SWIFT_NAME(poseLandmarker(_:didFinishDetection:timestampInMilliseconds:error:));
@end

/** Options for setting up a `PoseLandmarker`. */
NS_SWIFT_NAME(PoseLandmarkerOptions)
@interface MPPPoseLandmarkerOptions : MPPTaskOptions <NSCopying>

/**
 * Running mode of the pose landmark dection task. Defaults to `.image`. `PoseLandmarker` can be
 * created with one of the following running modes:
 *  1. `.image`: The mode for performing pose landmark detection on single image inputs.
 *  2. `.video`: The mode for performing pose landmark detection on the decoded frames of a video.
 *  3. `.liveStream`: The mode for performing pose landmark detection on a live stream of input
 * data, such as from the camera.
 */
@property(nonatomic) MPPRunningMode runningMode;

/**
 * An object that confirms to `PoseLandmarkerLiveStreamDelegate` protocol. This object must
 * implement `poseLandmarker(_:didFinishDetectionWithResult:timestampInMilliseconds:error:)` to
 * receive the results of performing asynchronous pose landmark detection on images (i.e, when
 * `runningMode` = `.liveStream`).
 */
@property(nonatomic, weak, nullable) id<MPPPoseLandmarkerLiveStreamDelegate>
    poseLandmarkerLiveStreamDelegate;

/** The maximum number of poses that can be detected by the `PoseLandmarker`. Defaults to 1. */
@property(nonatomic) NSInteger numPoses;

/**
 * The minimum confidence score for pose detection to be considered successful. Defaults to 0.5.
 */
@property(nonatomic) float minPoseDetectionConfidence;

/**
 * The minimum confidence score of pose presence score in the pose landmark detection. Defaults to
 * 0.5.
 */
@property(nonatomic) float minPosePresenceConfidence;

/**
 * The minimum confidence score for pose tracking to be considered successful. Defaults to 0.5.
 */
@property(nonatomic) float minTrackingConfidence;

/**
 * Whether to output segmentation masks. Defaults to `false`.
 */
@property(nonatomic) BOOL shouldOutputSegmentationMasks;

@end

NS_ASSUME_NONNULL_END
