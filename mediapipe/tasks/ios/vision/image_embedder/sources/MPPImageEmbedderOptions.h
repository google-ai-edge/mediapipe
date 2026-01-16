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
#import "mediapipe/tasks/ios/vision/image_embedder/sources/MPPImageEmbedderResult.h"

NS_ASSUME_NONNULL_BEGIN

@class MPPImageEmbedder;

/**
 * This protocol defines an interface for the delegates of `ImageEmbedder` object to receive
 * results of asynchronous embedding extraction on images (i.e, when `runningMode` = `.liveStream`).
 *
 * The delegate of `ImageEmbedder` must adopt `ImageEmbedderLiveStreamDelegate` protocol.
 * The methods in this protocol are optional.
 */
NS_SWIFT_NAME(ImageEmbedderLiveStreamDelegate)
@protocol MPPImageEmbedderLiveStreamDelegate <NSObject>

@optional
/**
 * This method notifies a delegate that the results of asynchronous embedding extraction on
 * an image submitted to the `ImageEmbedder` is available.
 *
 * This method is called on a private serial queue created by the `ImageEmbedder`
 * for performing the asynchronous delegates calls.
 *
 * @param imageEmbedder The image embedder which performed the embedding extraction.
 * This is useful to test equality when there are multiple instances of `ImageEmbedder`.
 * @param result An `ImageEmbedderResult` object that contains a list of generated image embeddings.
 * @param timestampInMilliseconds The timestamp (in milliseconds) which indicates when the input
 * image was sent to the image embedder.
 * @param error An optional error parameter populated when there is an error in performing embedding
 * extraction on the input live stream image data.
 */
- (void)imageEmbedder:(MPPImageEmbedder *)imageEmbedder
    didFinishEmbeddingWithResult:(nullable MPPImageEmbedderResult *)result
         timestampInMilliseconds:(NSInteger)timestampInMilliseconds
                           error:(nullable NSError *)error
    NS_SWIFT_NAME(imageEmbedder(_:didFinishEmbedding:timestampInMilliseconds:error:));
@end

/**
 * Options for setting up a `ImageEmbedder`.
 */
NS_SWIFT_NAME(ImageEmbedderOptions)
@interface MPPImageEmbedderOptions : MPPTaskOptions <NSCopying>

/**
 * Running mode of the image embedder task. Defaults to `.image`.
 * `ImageEmbedder` can be created with one of the following running modes:
 *  1. `.image`: The mode for performing embedding extraction on single image inputs.
 *  2. `.video`: The mode for performing embedding extraction on the decoded frames of a
 *      video.
 *  3. `.liveStream`: The mode for performing embedding extraction on a live stream of input
 *      data, such as from the camera.
 */
@property(nonatomic) MPPRunningMode runningMode;

/**
 * An object that confirms to `ImageEmbedderLiveStreamDelegate` protocol. This object must
 * implement `imageEmbedder(_:didFinishEmbeddingWithResult:timestampInMilliseconds:error:)` to
 * receive the results of asynchronous embedding extraction on images (i.e, when `runningMode =
 * .liveStream`).
 */
@property(nonatomic, weak, nullable) id<MPPImageEmbedderLiveStreamDelegate>
    imageEmbedderLiveStreamDelegate;

/**
 * @brief Sets whether L2 normalization should be performed on the returned embeddings.
 * Use this option only if the model does not already contain a native L2_NORMALIZATION TF Lite Op.
 * In most cases, this is already the case and L2 norm is thus achieved through TF Lite inference.
 *
 * `NO` by default.
 */
@property(nonatomic) BOOL l2Normalize;

/**
 * @brief Sets whether the returned embedding should be quantized to bytes via scalar quantization.
 * Embeddings are implicitly assumed to be unit-norm and therefore any dimensions is guaranteed to
 * have value in [-1.0, 1.0]. Use the `l2Normalize` property if this is not the case.
 *
 * `NO` by default.
 */
@property(nonatomic) BOOL quantize;

@end

NS_ASSUME_NONNULL_END
