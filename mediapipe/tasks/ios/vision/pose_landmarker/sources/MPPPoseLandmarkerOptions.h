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
 * This protocol defines an interface for the delegates of `MPPPoseLandmarker` object to receive
 * results of performing asynchronous pose landmark detection on images (i.e, when `runningMode` =
 * `MPPRunningModeLiveStream`).
 *
 * The delegate of `MPPPoseLandmarker` must adopt `MPPPoseLandmarkerLiveStreamDelegate` protocol.
 * The methods in this protocol are optional.
 */
NS_SWIFT_NAME(PoseLandmarkerLiveStreamDelegate)
@protocol MPPPoseLandmarkerLiveStreamDelegate <NSObject>

@optional

/**
 * This method notifies a delegate that the results of asynchronous pose landmark detection of an
 * image submitted to the `MPPPoseLandmarker` is available.
 *
 * This method is called on a private serial dispatch queue created by the `MPPPoseLandmarker`
 * for performing the asynchronous delegates calls.
 *
 * @param poseLandmarker The pose landmarker which performed the pose landmarking.
 * This is useful to test equality when there are multiple instances of `MPPPoseLandmarker`.
 * @param result The `MPPPoseLandmarkerResult` object that contains a list of detections, each
 * detection has a bounding box that is expressed in the unrotated input frame of reference
 * coordinates system, i.e. in `[0,image_width) x [0,image_height)`, which are the dimensions of the
 * underlying image data.
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

/** Options for setting up a `MPPPoseLandmarker`. */
NS_SWIFT_NAME(PoseLandmarkerOptions)
@interface MPPPoseLandmarkerOptions : MPPTaskOptions <NSCopying>

/**
 * Running mode of the pose landmarker task. Defaults to `MPPRunningModeImage`.
 * `MPPPoseLandmarker` can be created with one of the following running modes:
 *  1. `MPPRunningModeImage`: The mode for performing pose landmark detection on single image
 * inputs.
 *  2. `MPPRunningModeVideo`: The mode for performing pose landmark detection on the decoded frames
 * of a video.
 *  3. `MPPRunningModeLiveStream`: The mode for performing pose landmark detection on a live stream
 * of input data, such as from the camera.
 */
@property(nonatomic) MPPRunningMode runningMode;

/**
 * An object that confirms to `MPPPoseLandmarkerLiveStreamDelegate` protocol. This object must
 * implement `poseLandmarker:didFinishDetectionWithResult:timestampInMilliseconds:error:` to
 * receive the results of performing asynchronous pose landmark detection on images (i.e, when
 * `runningMode` = `MPPRunningModeLiveStream`).
 */
@property(nonatomic, weak, nullable) id<MPPPoseLandmarkerLiveStreamDelegate>
    poseLandmarkerLiveStreamDelegate;

/** The maximum number of poses that can be detected by the `MPPPoseLandmarker`. */
@property(nonatomic) NSInteger numPoses;

/** The minimum confidence score for the pose detection to be considered successful. */
@property(nonatomic) float minPoseDetectionConfidence;

/** The minimum confidence score of pose presence score in the pose landmark detection. */
@property(nonatomic) float minPosePresenceConfidence;

/** The minimum confidence score for the pose tracking to be considered successful. */
@property(nonatomic) float minTrackingConfidence;

// Whether to output segmentation masks.
@property(nonatomic) BOOL outputSegmentationMasks;

@end

NS_ASSUME_NONNULL_END
