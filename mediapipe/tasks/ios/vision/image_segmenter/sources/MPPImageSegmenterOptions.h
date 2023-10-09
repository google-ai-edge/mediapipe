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
#import "mediapipe/tasks/ios/vision/image_segmenter/sources/MPPImageSegmenterResult.h"

NS_ASSUME_NONNULL_BEGIN

@class MPPImageSegmenter;

/**
 * This protocol defines an interface for the delegates of `ImageSegmenter` object to receive
 * results of performing asynchronous segmentation on images (i.e, when `runningMode` =
 * `liveStream`).
 *
 * The delegate of `ImageSegmenter` must adopt `ImageSegmenterLiveStreamDelegate` protocol.
 * The methods in this protocol are optional.
 */
NS_SWIFT_NAME(ImageSegmenterLiveStreamDelegate)
@protocol MPPImageSegmenterLiveStreamDelegate <NSObject>

@optional

/**
 * This method notifies a delegate that the results of asynchronous segmentation of
 * an image submitted to the `ImageSegmenter` is available.
 *
 * This method is called on a private serial dispatch queue created by the `ImageSegmenter`
 * for performing the asynchronous delegates calls.
 *
 * @param imageSegmenter The image segmenter which performed the segmentation. This is useful to
 * test equality when there are multiple instances of `ImageSegmenter`.
 * @param result The `ImageSegmenterResult` object that contains a list of category or confidence
 * masks and optional quality scores.
 * @param timestampInMilliseconds The timestamp (in milliseconds) which indicates when the input
 * image was sent to the image segmenter.
 * @param error An optional error parameter populated when there is an error in performing
 * segmentation on the input live stream image data.
 */
- (void)imageSegmenter:(MPPImageSegmenter *)imageSegmenter
    didFinishSegmentationWithResult:(nullable MPPImageSegmenterResult *)result
            timestampInMilliseconds:(NSInteger)timestampInMilliseconds
                              error:(nullable NSError *)error
    NS_SWIFT_NAME(imageSegmenter(_:didFinishSegmentation:timestampInMilliseconds:error:));
@end

/** Options for setting up a `ImageSegmenter`. */
NS_SWIFT_NAME(ImageSegmenterOptions)
@interface MPPImageSegmenterOptions : MPPTaskOptions <NSCopying>

/**
 * Running mode of the image segmenter task. Defaults to `image`.
 * `ImageSegmenter` can be created with one of the following running modes:
 *  1. `image`: The mode for performing segmentation on single image inputs.
 *  2. `video`: The mode for performing segmentation on the decoded frames of a
 *      video.
 *  3. `liveStream`: The mode for performing segmentation on a live stream of
 *      input data, such as from the camera.
 */
@property(nonatomic) MPPRunningMode runningMode;

/**
 * An object that confirms to `ImageSegmenterLiveStreamDelegate` protocol. This object must
 * implement `imageSegmenter(_:didFinishSegmentationWithResult:timestampInMilliseconds:error:)` to
 * receive the results of performing asynchronous segmentation on images (i.e, when `runningMode` =
 * `liveStream`).
 */
@property(nonatomic, weak, nullable) id<MPPImageSegmenterLiveStreamDelegate>
    imageSegmenterLiveStreamDelegate;

/**
 * The locale to use for display names specified through the TFLite Model Metadata, if any. Defaults
 * to English.
 */
@property(nonatomic, copy) NSString *displayNamesLocale;

/** Represents whether to output confidence masks. */
@property(nonatomic) BOOL shouldOutputConfidenceMasks;

/** Represents whether to output category mask. */
@property(nonatomic) BOOL shouldOutputCategoryMask;

@end

NS_ASSUME_NONNULL_END
