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

#import "mediapipe/tasks/ios/vision/core/sources/MPPImage.h"
#import "mediapipe/tasks/ios/vision/image_embedder/sources/MPPImageEmbedderOptions.h"
#import "mediapipe/tasks/ios/vision/image_embedder/sources/MPPImageEmbedderResult.h"

NS_ASSUME_NONNULL_BEGIN

/**
 * @brief Performs embedding extraction on images.
 *
 * The API expects a TFLite model with optional, but strongly recommended,
 * [TFLite Model Metadata.](https://www.tensorflow.org/lite/convert/metadata").
 *
 * The API supports models with one image input tensor and one or more output tensors. To be more
 * specific, here are the requirements.
 *
 * Input image tensor
 *  (kTfLiteUInt8/kTfLiteFloat32)
 *  - image input of size `[batch x height x width x channels]`.
 *  - batch inference is not supported (`batch` is required to be 1).
 *  - only RGB inputs are supported (`channels` is required to be 3).
 *  - if type is kTfLiteFloat32, NormalizationOptions are required to be attached to the metadata
 * for input normalization.
 *
 * At least one output tensor (kTfLiteUInt8/kTfLiteFloat32) with shape `[1 x N]` where N
 * is the number of dimensions in the produced embeddings.
 */
NS_SWIFT_NAME(ImageEmbedder)
@interface MPPImageEmbedder : NSObject

/**
 * Creates a new instance of `ImageEmbedder` from an absolute path to a TensorFlow Lite model file
 * stored locally on the device and the default `ImageEmbedderOptions`.
 *
 * @param modelPath An absolute path to a TensorFlow Lite model file stored locally on the device.
 *
 * @return A new instance of `ImageEmbedder` with the given model path. `nil` if there is an
 * error in initializing the image embedder.
 */
- (nullable instancetype)initWithModelPath:(NSString *)modelPath error:(NSError **)error;

/**
 * Creates a new instance of `ImageEmbedder` from the given `ImageEmbedderOptions`.
 *
 * @param options The options of type `ImageEmbedderOptions` to use for configuring the
 * `ImageEmbedder`.
 *
 * @return A new instance of `ImageEmbedder` with the given options. `nil` if there is an error in
 * initializing the image embedder.
 */
- (nullable instancetype)initWithOptions:(MPPImageEmbedderOptions *)options
                                   error:(NSError **)error NS_DESIGNATED_INITIALIZER;

/**
 * Performs embedding extraction on the provided `MPImage` using the whole image as region of
 * interest. Rotation will be applied according to the `orientation` property of the provided
 * `MPImage`. Only use this method when the `ImageEmbedder` is created with running mode, `.image`.
 *
 * This method supports embedding extraction on RGBA images. If your `MPImage` has a
 * source type of `.pixelBuffer` or `.sampleBuffer`, the underlying pixel buffer must use
 * `kCVPixelFormatType_32BGRA` as its pixel format.
 *
 * If your `MPImage` has a source type of `.image` ensure that the color space is RGB with an Alpha
 * channel.
 *
 * @param image The `MPImage` on which embedding extraction is to be performed.
 *
 * @return  An `ImageEmbedderResult` object that contains a list of embedding extraction.
 */
- (nullable MPPImageEmbedderResult *)embedImage:(MPPImage *)image
                                          error:(NSError **)error NS_SWIFT_NAME(embed(image:));

/**
 * Performs embedding extraction on the provided `MPImage` cropped to the specified region of
 * interest. Rotation will be applied on the cropped image according to the `orientation` property
 * of the provided `MPImage`. Only use this method when the `ImageEmbedder` is created with running
 * mode, `.image`.
 *
 * This method supports embedding extraction on RGBA images. If your `MPImage` has a
 * source type of `.pixelBuffer` or `.sampleBuffer`, the underlying pixel buffer must use
 * `kCVPixelFormatType_32BGRA` as its pixel format.
 *
 * If your `MPImage` has a source type of `.image` ensure that the color space is RGB with an Alpha
 * channel.
 *
 * @param image The `MPImage` on which embedding extraction is to be performed.
 * @param roi A `CGRect` specifying the region of interest within the given `MPImage`, on which
 * embedding extraction should be performed.
 *
 * @return  An `ImageEmbedderResult` object that contains a list of generated image embeddings.
 */
- (nullable MPPImageEmbedderResult *)embedImage:(MPPImage *)image
                               regionOfInterest:(CGRect)roi
                                          error:(NSError **)error
    NS_SWIFT_NAME(embed(image:regionOfInterest:));

/**
 * Performs embedding extraction on the provided video frame of type `MPImage` using the whole image
 * as region of interest. Rotation will be applied according to the `orientation` property of the
 * provided `MPImage`. Only use this method when the `ImageEmbedder` is created with running mode
 * `.video`.
 *
 * It's required to provide the video frame's timestamp (in milliseconds). The input timestamps must
 * be monotonically increasing.
 *
 * This method supports embedding extraction on RGBA images. If your `MPImage` has a
 * source type of `.pixelBuffer` or `.sampleBuffer`, the underlying pixel buffer must use
 * `kCVPixelFormatType_32BGRA` as its pixel format.
 *
 * If your `MPImage` has a source type of `.image` ensure that the color space is RGB with an Alpha
 * channel.
 *
 * @param image The `MPImage` on which embedding extraction is to be performed.
 * @param timestampInMilliseconds The video frame's timestamp (in milliseconds). The input
 * timestamps must be monotonically increasing.
 *
 * @return  An `ImageEmbedderResult` object that contains a list of generated image embeddings.
 */
- (nullable MPPImageEmbedderResult *)embedVideoFrame:(MPPImage *)image
                             timestampInMilliseconds:(NSInteger)timestampInMilliseconds
                                               error:(NSError **)error
    NS_SWIFT_NAME(embed(videoFrame:timestampInMilliseconds:));

/**
 * Performs embedding extraction on the provided video frame of type `MPImage` cropped to the
 * specified region of interest. Rotation will be applied according to the `orientation` property of
 * the provided `MPImage`. Only use this method when the `ImageEmbedder` is created with `.video`.
 *
 * It's required to provide the video frame's timestamp (in milliseconds). The input timestamps must
 * be monotonically increasing.
 *
 * This method supports embedding extraction on RGBA images. If your `MPImage` has a
 * source type of `.pixelBuffer` or `.sampleBuffer`, the underlying pixel buffer must use
 * `kCVPixelFormatType_32BGRA` as its pixel format.
 *
 * If your `MPImage` has a source type of `.image` ensure that the color space is RGB with an Alpha
 * channel.
 *
 * @param image A live stream image data of type `MPImage` on which embedding extraction is to be
 * performed.
 * @param timestampInMilliseconds The video frame's timestamp (in milliseconds). The input
 * timestamps must be monotonically increasing.
 * @param roi A `CGRect` specifying the region of interest within the video frame of type
 * `MPImage`, on which embedding extraction should be performed.
 *
 * @return  An `ImageEmbedderResult` object that contains a list of generated image embeddings.
 */
- (nullable MPPImageEmbedderResult *)embedVideoFrame:(MPPImage *)image
                             timestampInMilliseconds:(NSInteger)timestampInMilliseconds
                                    regionOfInterest:(CGRect)roi
                                               error:(NSError **)error
    NS_SWIFT_NAME(embed(videoFrame:timestampInMilliseconds:regionOfInterest:));

/**
 * Sends live stream image data of type `MPImage` to perform embedding extraction using the whole
 * image as region of interest. Rotation will be applied according to the `orientation` property of
 * the provided `MPImage`. Only use this method when the `ImageEmbedder` is created with running
 * mode `.liveStream`.
 *
 * The object which needs to be continuously notified of the available results of image
 * embedding extraction must confirm to `ImageEmbedderLiveStreamDelegate` protocol and implement the
 * `imageEmbedder(_:didFinishEmbeddingWithResult:timestampInMilliseconds:error:)` delegate
 * method.
 *
 * It's required to provide a timestamp (in milliseconds) to indicate when the input image is sent
 * to the image embedder. The input timestamps must be monotonically increasing.
 *
 * This method supports embedding extraction on RGBA images. If your `MPImage` has a
 * source type of `.pixelBuffer` or `.sampleBuffer`, the underlying pixel buffer must use
 * `kCVPixelFormatType_32BGRA` as its pixel format.
 *
 * If the input `MPImage` has a source type of `.image` ensure that the color space is RGB with an
 * Alpha channel.
 *
 * If this method is used for embedding live camera frames using `AVFoundation`, ensure that you
 * request `AVCaptureVideoDataOutput` to output frames in `kCMPixelFormat_32BGRA` using its
 * `videoSettings` property.
 *
 * @param image A live stream image data of type `MPImage` on which embedding extraction is to be
 * performed.
 * @param timestampInMilliseconds The timestamp (in milliseconds) which indicates when the input
 * image is sent to the image embedder. The input timestamps must be monotonically increasing.
 *
 * @return `true` if the image was sent to the task successfully, otherwise `false`.
 */
- (BOOL)embedAsyncImage:(MPPImage *)image
    timestampInMilliseconds:(NSInteger)timestampInMilliseconds
                      error:(NSError **)error
    NS_SWIFT_NAME(embedAsync(image:timestampInMilliseconds:));

/**
 * Sends live stream image data of type `MPImage` to perform embedding extraction, cropped to the
 * specified region of interest.. Rotation will be applied according to the `orientation` property
 * of the provided `MPImage`. Only use this method when the `ImageEmbedder` is created with
 * `.liveStream`.
 *
 * The object which needs to be continuously notified of the available results of image embedding
 * extraction must confirm to `ImageEmbedderLiveStreamDelegate` protocol and implement the
 * `imageEmbedder(_:didFinishEmbeddingWithResult:timestampInMilliseconds:error:)` delegate
 * method.
 *
 * It's required to provide a timestamp (in milliseconds) to indicate when the input image is sent
 * to the image embedder. The input timestamps must be monotonically increasing.
 *
 * This method supports embedding extraction on RGBA images. If your `MPImage` has a
 * source type of `.pixelBuffer` or `.sampleBuffer`, the underlying pixel buffer must use
 * `kCVPixelFormatType_32BGRA` as its pixel format.
 *
 * If the input `MPImage` has a source type of `.image` ensure that the color space is RGB with an
 * Alpha channel.
 *
 * If this method is used for embedding live camera frames using `AVFoundation`, ensure that you
 * request `AVCaptureVideoDataOutput` to output frames in `kCMPixelFormat_32BGRA` using its
 * `videoSettings` property.
 *
 * @param image A live stream image data of type `MPImage` on which embedding extraction is to be
 * performed.
 * @param timestampInMilliseconds The timestamp (in milliseconds) which indicates when the input
 * image is sent to the image embedder. The input timestamps must be monotonically increasing.
 * @param roi A `CGRect` specifying the region of interest within the given live stream image data
 * of type `MPImage`, on which embedding extraction should be performed.
 *
 * @return `true` if the image was sent to the task successfully, otherwise `false`.
 */
- (BOOL)embedAsyncImage:(MPPImage *)image
    timestampInMilliseconds:(NSInteger)timestampInMilliseconds
           regionOfInterest:(CGRect)roi
                      error:(NSError **)error
    NS_SWIFT_NAME(embedAsync(image:timestampInMilliseconds:regionOfInterest:));

- (instancetype)init NS_UNAVAILABLE;

/**
 * Utility function to compute[cosine similarity](https://en.wikipedia.org/wiki/Cosine_similarity)
 * between two `MPPEmbedding` objects.
 *
 * @param embedding1 One of the two `MPPEmbedding`s between whom cosine similarity is to be
 * computed.
 * @param embedding2 One of the two `MPPEmbedding`s between whom cosine similarity is to be
 * computed.
 * @param error An optional error parameter populated when there is an error in calculating cosine
 * similarity between two embeddings.
 *
 * @return An `NSNumber` which holds the cosine similarity of type `double`.
 */
+ (nullable NSNumber *)cosineSimilarityBetweenEmbedding1:(MPPEmbedding *)embedding1
                                           andEmbedding2:(MPPEmbedding *)embedding2
                                                   error:(NSError **)error
    NS_SWIFT_NAME(cosineSimilarity(embedding1:embedding2:));

+ (instancetype)new NS_UNAVAILABLE;

@end

NS_ASSUME_NONNULL_END
