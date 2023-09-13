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
#import "mediapipe/tasks/ios/vision/object_detector/sources/MPPObjectDetectorResult.h"

NS_ASSUME_NONNULL_BEGIN

@class MPPObjectDetector;

/**
 * This protocol defines an interface for the delegates of `ObjectDetector` object to receive
 * results of performing asynchronous object detection on images (i.e, when `runningMode` =
 * `.liveStream`).
 *
 * The delegate of `ObjectDetector` must adopt `ObjectDetectorLiveStreamDelegate` protocol.
 * The methods in this protocol are optional.
 */
NS_SWIFT_NAME(ObjectDetectorLiveStreamDelegate)
@protocol MPPObjectDetectorLiveStreamDelegate <NSObject>

@optional

/**
 * This method notifies a delegate that the results of asynchronous object detection of
 * an image submitted to the `ObjectDetector` is available.
 *
 * This method is called on a private serial dispatch queue created by the `ObjectDetector`
 * for performing the asynchronous delegates calls.
 *
 * @param objectDetector The object detector which performed the object detection.
 * This is useful to test equality when there are multiple instances of `ObjectDetector`.
 * @param result The `ObjectDetectorResult` object that contains a list of detections, each
 * detection has a bounding box that is expressed in the unrotated input frame of reference
 * coordinates system, i.e. in `[0,image_width) x [0,image_height)`, which are the dimensions of the
 * underlying image data.
 * @param timestampInMilliseconds The timestamp (in milliseconds) which indicates when the input
 * image was sent to the object detector.
 * @param error An optional error parameter populated when there is an error in performing object
 * detection on the input live stream image data.
 */
- (void)objectDetector:(MPPObjectDetector *)objectDetector
    didFinishDetectionWithResult:(nullable MPPObjectDetectorResult *)result
         timestampInMilliseconds:(NSInteger)timestampInMilliseconds
                           error:(nullable NSError *)error
    NS_SWIFT_NAME(objectDetector(_:didFinishDetection:timestampInMilliseconds:error:));
@end

/** Options for setting up a `ObjectDetector`. */
NS_SWIFT_NAME(ObjectDetectorOptions)
@interface MPPObjectDetectorOptions : MPPTaskOptions <NSCopying>

/**
 * Running mode of the object detector task. Defaults to `.image`.
 * `ObjectDetector` can be created with one of the following running modes:
 *  1. `.image`: The mode for performing object detection on single image inputs.
 *  2. `.video`: The mode for performing object detection on the decoded frames of a
 *      video.
 *  3. `.liveStream`: The mode for performing object detection on a live stream of
 *      input data, such as from the camera.
 */
@property(nonatomic) MPPRunningMode runningMode;

/**
 * An object that confirms to `ObjectDetectorLiveStreamDelegate` protocol. This object must
 * implement `objectDetector(_:didFinishDetectionWithResult:timestampInMilliseconds:error:)` to
 * receive the results of performing asynchronous object detection on images (i.e, when
 * `runningMode` = `.liveStream`).
 */
@property(nonatomic, weak, nullable) id<MPPObjectDetectorLiveStreamDelegate>
    objectDetectorLiveStreamDelegate;

/**
 * The locale to use for display names specified through the TFLite Model Metadata, if any. Defaults
 * to English.
 */
@property(nonatomic, copy) NSString *displayNamesLocale;

/**
 * The maximum number of top-scored classification results to return. If < 0, all available results
 * will be returned. If 0, an invalid argument error is returned.
 */
@property(nonatomic) NSInteger maxResults;

/**
 * Score threshold to override the one provided in the model metadata (if any). Results below this
 * value are rejected.
 */
@property(nonatomic) float scoreThreshold;

/**
 * The allowlist of category names. If non-empty, detection results whose category name is not in
 * this set will be filtered out. Duplicate or unknown category names are ignored. Mutually
 * exclusive with categoryDenylist.
 */
@property(nonatomic, copy) NSArray<NSString *> *categoryAllowlist;

/**
 * The denylist of category names. If non-empty, detection results whose category name is in this
 * set will be filtered out. Duplicate or unknown category names are ignored. Mutually exclusive
 * with categoryAllowlist.
 */
@property(nonatomic, copy) NSArray<NSString *> *categoryDenylist;

@end

NS_ASSUME_NONNULL_END
