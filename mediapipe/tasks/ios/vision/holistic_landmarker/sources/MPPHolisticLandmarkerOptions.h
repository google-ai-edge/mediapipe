// Copyright 2024 The MediaPipe Authors.
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
#import "mediapipe/tasks/ios/vision/holistic_landmarker/sources/MPPHolisticLandmarkerResult.h"

NS_ASSUME_NONNULL_BEGIN

@class MPPHolisticLandmarker;

/**
 * This protocol defines an interface for the delegates of `HolisticLandmarker` to receive
 * results of performing asynchronous holistic landmarks detection on images (i.e, when
 * `runningMode` = `.liveStream`).
 *
 * The delegate of `FaceLandmarker` must adopt `FaceLandmarkerLiveStreamDelegate` protocol.
 * The methods in this protocol are optional.
 */
NS_SWIFT_NAME(HolisticLandmarkerLiveStreamDelegate)
@protocol MPPHolisticLandmarkerLiveStreamDelegate <NSObject>

/**
 * This method notifies a delegate that the results of asynchronous holistic landmarks detection of
 * an image submitted to the `HolisticsLandmarker` is available.
 *
 * This method is called on a private serial dispatch queue created by the `HolisticLandmarker` for
 * performing the asynchronous delegates calls.
 *
 * @param holisticLandmarker The holistic landmarker which performed the holistic landmarks
 * detection. This is useful to test equality when there are multiple instances of
 * `HolisticLandmarker`.
 * @param result The `HolisticLandmarkerResult` object that contains a list of landmarks.
 * @param timestampInMilliseconds The timestamp (in milliseconds) which indicates when the input
 * image was sent to the holistic landmarker.
 * @param error An optional error parameter populated when there is an error in performing holistic
 * landmarks detection on the input live stream image data.
 */
- (void)holisticLandmarker:(MPPHolisticLandmarker *)holisticLandmarker
    didFinishDetectionWithResult:(nullable MPPHolisticLandmarkerResult *)result
         timestampInMilliseconds:(NSInteger)timestampInMilliseconds
                           error:(nullable NSError *)error
    NS_SWIFT_NAME(holisticLandmarker(_:didFinishDetection:timestampInMilliseconds:error:));
@end

/** Options for setting up a `HolisticLandmarker`. */
NS_SWIFT_NAME(HolisticLandmarkerOptions)
@interface MPPHolisticLandmarkerOptions : MPPTaskOptions <NSCopying>

/**
 * Running mode of the holistic landmarks dection task. Defaults to `.image`. `HolisticLandmarker`
 * can be created with one of the following running modes:
 *  1. `.image`: The mode for performing holistic landmarks detection on single image inputs.
 *  2. `.video`: The mode for performing holistic landmarks detection on the decoded frames of a
 * video.
 *  3. `.liveStream`: The mode for performing holistic landmarks detection on a live stream of input
 * data, such as from the camera.
 */
@property(nonatomic) MPPRunningMode runningMode;

/**
 * An object that confirms to `HolisticLandmarkerLiveStreamDelegate` protocol. This object must
 * implement `holisticLandmarker(_:didFinishDetectionWithResult:timestampInMilliseconds:error:)` to
 * receive the results of performing asynchronous holistic landmarks detection on images (i.e, when
 * `runningMode` = `.liveStream`).
 */
@property(nonatomic, weak, nullable) id<MPPHolisticLandmarkerLiveStreamDelegate>
    holisticLandmarkerLiveStreamDelegate;

/**
 * The minimum confidence score for the face detection to be considered successful. Defaults to 0.5.
 */
@property(nonatomic) float minFaceDetectionConfidence;

/**
 * The minimum threshold for the face suppression score in the face detection. Defaults to 0.3.
 */
@property(nonatomic) float minFaceSuppressionThreshold;

/**
 * The minimum confidence score of face presence score in the face landmark detection. Defaults to
 * 0.5.
 */
@property(nonatomic) float minFacePresenceConfidence;

/**
 * The minimum confidence score for pose detection to be considered successful. Defaults to 0.5.
 */
@property(nonatomic) float minPoseDetectionConfidence;

/**
 * The minimum non-maximum-suppression threshold for pose detection to be considered overlapped.
 * Defaults to 0.3.
 */
@property(nonatomic) float minPoseSuppressionThreshold;

/**
 * The minimum confidence score of pose presence score in the pose landmark detection. Defaults to
 * 0.5.
 */
@property(nonatomic) float minPosePresenceConfidence;

/**
 * Whether FaceLandmarker outputs face blendshapes classification. Face blendshapes are used for
 * rendering the 3D face model.
 */
@property(nonatomic) BOOL outputFaceBlendshapes;

/** Whether to output segmentation masks. Defaults to false. */
@property(nonatomic) BOOL outputPoseSegmentationMasks;

/** The minimum confidence score of hand presence score in the hand landmarks detection. Defaults to
 * 0.5. */
@property(nonatomic) float minHandLandmarksConfidence;

@end

NS_ASSUME_NONNULL_END
