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

NS_ASSUME_NONNULL_BEGIN

/** Options for setting up a `MPPFaceLandmarker`. */
NS_SWIFT_NAME(FaceLandmarkerOptions)
@interface MPPFaceLandmarkerOptions : MPPTaskOptions <NSCopying>

/**
 * Running mode of the face landmark dection task. Defaults to `MPPRunningModeImage`.
 * `MPPFaceLandmarker` can be created with one of the following running modes:
 *  1. `MPPRunningModeImage`: The mode for performing face detection on single image inputs.
 *  2. `MPPRunningModeVideo`: The mode for performing face detection on the decoded frames of a
 *      video.
 *  3. `MPPRunningModeLiveStream`: The mode for performing face detection on a live stream of
 *      input data, such as from the camera.
 */
@property(nonatomic) MPPRunningMode runningMode;

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

@end

NS_ASSUME_NONNULL_END
