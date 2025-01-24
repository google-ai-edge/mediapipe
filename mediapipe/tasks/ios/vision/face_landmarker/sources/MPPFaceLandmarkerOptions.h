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
#import "mediapipe/tasks/ios/vision/face_landmarker/sources/MPPFaceLandmarkerResult.h"

NS_ASSUME_NONNULL_BEGIN

@class MPPFaceLandmarker;

/**
 * This protocol defines an interface for the delegates of `FaceLandmarker` face to receive
 * results of performing asynchronous face detection on images (i.e, when `runningMode` =
 * `.liveStream`).
 *
 * The delegate of `FaceLandmarker` must adopt `FaceLandmarkerLiveStreamDelegate` protocol.
 * The methods in this protocol are optional.
 */
NS_SWIFT_NAME(FaceLandmarkerLiveStreamDelegate)
@protocol MPPFaceLandmarkerLiveStreamDelegate <NSObject>

/**
 * This method notifies a delegate that the results of asynchronous face detection of
 * an image submitted to the `FaceLandmarker` is available.
 *
 * This method is called on a private serial dispatch queue created by the `FaceLandmarker`
 * for performing the asynchronous delegates calls.
 *
 * @param faceLandmarker The face landmarker which performed the face landmark detections.
 * This is useful to test equality when there are multiple instances of `FaceLandmarker`.
 * @param result The `FaceLandmarkerResult` object that contains a list of landmarks.
 * @param timestampInMilliseconds The timestamp (in milliseconds) which indicates when the input
 * image was sent to the face detector.
 * @param error An optional error parameter populated when there is an error in performing face
 * detection on the input live stream image data.
 */
- (void)faceLandmarker:(MPPFaceLandmarker *)faceLandmarker
    didFinishDetectionWithResult:(nullable MPPFaceLandmarkerResult *)result
         timestampInMilliseconds:(NSInteger)timestampInMilliseconds
                           error:(nullable NSError *)error
    NS_SWIFT_NAME(faceLandmarker(_:didFinishDetection:timestampInMilliseconds:error:));
@end

/** Options for setting up a `FaceLandmarker`. */
NS_SWIFT_NAME(FaceLandmarkerOptions)
@interface MPPFaceLandmarkerOptions : MPPTaskOptions <NSCopying>

/**
 * Running mode of the face landmark detection task. Defaults to `.image`. `FaceLandmarker` can be
 * created with one of the following running modes:
 *  1. `.image`: The mode for performing face detection on single image inputs.
 *  2. `.video`: The mode for performing face detection on the decoded frames of a video.
 *  3. `.liveStream`: The mode for performing face detection on a live stream of input data, such as
 * from the camera.
 */
@property(nonatomic) MPPRunningMode runningMode;

/**
 * An object that confirms to `FaceLandmarkerLiveStreamDelegate` protocol. This object must
 * implement `faceLandmarker(_:didFinishDetectionWithResult:timestampInMilliseconds:error:)` to
 * receive the results of performing asynchronous face landmark detection on images (i.e, when
 * `runningMode` = `.liveStream`).
 */
@property(nonatomic, weak, nullable) id<MPPFaceLandmarkerLiveStreamDelegate>
    faceLandmarkerLiveStreamDelegate;

/** The maximum number of faces can be detected by the FaceLandmarker. Defaults to 1. */
@property(nonatomic) NSInteger numFaces;

/**
 * The minimum confidence score for the face detection to be considered successful. Defaults to 0.5.
 */
@property(nonatomic) float minFaceDetectionConfidence;

/**
 * The minimum confidence score of face presence score in the face landmark detection. Defaults to
 * 0.5.
 */
@property(nonatomic) float minFacePresenceConfidence;

/**
 * The minimum confidence score for the face tracking to be considered successful. Defaults to 0.5.
 */
@property(nonatomic) float minTrackingConfidence;

/**
 * Whether FaceLandmarker outputs face blendshapes classification. Face blendshapes are used for
 * rendering the 3D face model.
 */
@property(nonatomic) BOOL outputFaceBlendshapes;

/**
 * Whether FaceLandmarker outputs facial transformation_matrix. Facial transformation matrix is used
 * to transform the face landmarks in canonical face to the detected face, so that users can apply
 * face effects on the detected landmarks.
 */
@property(nonatomic) BOOL outputFacialTransformationMatrixes;

@end

NS_ASSUME_NONNULL_END
