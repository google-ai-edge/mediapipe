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
#import "mediapipe/tasks/ios/vision/image_classifier/sources/MPPImageClassifierResult.h"

NS_ASSUME_NONNULL_BEGIN

@class MPPImageClassifier;

/**
 * This protocol defines an interface for the delegates of `ImageClassifier` object to receive
 * results of asynchronous classification of images (i.e, when `runningMode` = `.liveStream`).
 *
 * The delegate of `ImageClassifier` must adopt `ImageClassifierLiveStreamDelegate` protocol.
 * The methods in this protocol are optional.
 */
NS_SWIFT_NAME(ImageClassifierLiveStreamDelegate)
@protocol MPPImageClassifierLiveStreamDelegate <NSObject>

@optional
/**
 * This method notifies a delegate that the results of asynchronous classification of
 * an image submitted to the `ImageClassifier` is available.
 *
 * This method is called on a private serial queue created by the `ImageClassifier`
 * for performing the asynchronous delegates calls.
 *
 * @param imageClassifier The image classifier which performed the classification.
 * This is useful to test equality when there are multiple instances of `ImageClassifier`.
 * @param result An `ImageClassifierResult` object that contains a list of image classifications.
 * @param timestampInMilliseconds The timestamp (in milliseconds) which indicates when the input
 * image was sent to the image classifier.
 * @param error An optional error parameter populated when there is an error in performing image
 * classification on the input live stream image data.
 */
- (void)imageClassifier:(MPPImageClassifier *)imageClassifier
    didFinishClassificationWithResult:(nullable MPPImageClassifierResult *)result
              timestampInMilliseconds:(NSInteger)timestampInMilliseconds
                                error:(nullable NSError *)error
    NS_SWIFT_NAME(imageClassifier(_:didFinishClassification:timestampInMilliseconds:error:));
@end

/**
 * Options for setting up a `ImageClassifier`.
 */
NS_SWIFT_NAME(ImageClassifierOptions)
@interface MPPImageClassifierOptions : MPPTaskOptions <NSCopying>

/**
 * Running mode of the image classifier task. Defaults to `.image`.
 * `ImageClassifier` can be created with one of the following running modes:
 *  1. `.image`: The mode for performing classification on single image inputs.
 *  2. `.video`: The mode for performing classification on the decoded frames of a
 *      video.
 *  3. `.liveStream`: The mode for performing classification on a live stream of input
 *      data, such as from the camera.
 */
@property(nonatomic) MPPRunningMode runningMode;

/**
 * An object that confirms to `ImageClassifierLiveStreamDelegate` protocol. This object must
 * implement `objectDetector(_:didFinishDetectionWithResult:timestampInMilliseconds:error:)` to
 * receive the results of asynchronous classification on images (i.e, when `runningMode =
 * .liveStream`).
 */
@property(nonatomic, weak, nullable) id<MPPImageClassifierLiveStreamDelegate>
    imageClassifierLiveStreamDelegate;

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
