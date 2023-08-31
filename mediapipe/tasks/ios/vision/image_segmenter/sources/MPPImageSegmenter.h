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
#import "mediapipe/tasks/ios/vision/image_segmenter/sources/MPPImageSegmenterOptions.h"
#import "mediapipe/tasks/ios/vision/image_segmenter/sources/MPPImageSegmenterResult.h"

NS_ASSUME_NONNULL_BEGIN

/**
 * @brief Class that performs segmentation on images.
 *
 * The API expects a TFLite model with mandatory TFLite Model Metadata.
 */
NS_SWIFT_NAME(ImageSegmenter)
@interface MPPImageSegmenter : NSObject

/**
 * Creates a new instance of `MPPImageSegmenter` from an absolute path to a TensorFlow Lite model
 * file stored locally on the device and the default `MPPImageSegmenterOptions`.
 *
 * @param modelPath An absolute path to a TensorFlow Lite model file stored locally on the device.
 * @param error An optional error parameter populated when there is an error in initializing the
 * image segmenter.
 *
 * @return A new instance of `MPPImageSegmenter` with the given model path. `nil` if there is an
 * error in initializing the image segmenter.
 */
- (nullable instancetype)initWithModelPath:(NSString *)modelPath error:(NSError **)error;

/**
 * Creates a new instance of `MPPImageSegmenter` from the given `MPPImageSegmenterOptions`.
 *
 * @param options The options of type `MPPImageSegmenterOptions` to use for configuring the
 * `MPPImageSegmenter`.
 * @param error An optional error parameter populated when there is an error in initializing the
 * image segmenter.
 *
 * @return A new instance of `MPPImageSegmenter` with the given options. `nil` if there is an error
 * in initializing the image segmenter.
 */
- (nullable instancetype)initWithOptions:(MPPImageSegmenterOptions *)options
                                   error:(NSError **)error NS_DESIGNATED_INITIALIZER;

/**
 * Performs segmentation on the provided MPPImage using the whole image as region of interest.
 * Rotation will be applied according to the `orientation` property of the provided `MPPImage`. Only
 * use this method when the `MPPImageSegmenter` is created with `MPPRunningModeImage`.
 *
 * This method supports RGBA images. If your `MPPImage` has a source type of
 * `MPPImageSourceTypePixelBuffer` or `MPPImageSourceTypeSampleBuffer`, the underlying pixel buffer
 * must have one of the following pixel format types:
 * 1. kCVPixelFormatType_32BGRA
 * 2. kCVPixelFormatType_32RGBA
 *
 * If your `MPPImage` has a source type of `MPPImageSourceTypeImage` ensure that the color space is
 * RGB with an Alpha channel.
 *
 * @param image The `MPPImage` on which segmentation is to be performed.
 * @param error An optional error parameter populated when there is an error in performing
 * segmentation on the input image.
 *
 * @return An `MPPImageSegmenterResult` that contains the segmented masks.
 */
- (nullable MPPImageSegmenterResult *)segmentImage:(MPPImage *)image
                                             error:(NSError *)error NS_SWIFT_NAME(segment(image:));

/**
 * Performs segmentation on the provided MPPImage using the whole image as region of interest and
 * invokes the given completion handler block with the response. The method returns synchronously
 * once the completion handler returns.
 *
 * Rotation will be applied according to the `orientation` property of the provided
 * `MPPImage`. Only use this method when the `MPPImageSegmenter` is created with
 * `MPPRunningModeImage`.
 *
 * This method supports RGBA images. If your `MPPImage` has a source type of
 * `MPPImageSourceTypePixelBuffer` or `MPPImageSourceTypeSampleBuffer`, the underlying pixel buffer
 * must have one of the following pixel format types:
 * 1. kCVPixelFormatType_32BGRA
 * 2. kCVPixelFormatType_32RGBA
 *
 * If your `MPPImage` has a source type of `MPPImageSourceTypeImage` ensure that the color space is
 * RGB with an Alpha channel.
 *
 * @param image The `MPPImage` on which segmentation is to be performed.
 * @param completionHandler A block to be invoked with the results of performing segmentation on the
 * image. The block takes two arguments, the optional `MPPImageSegmenterResult` that contains the
 * segmented masks if the segmentation was successful and an optional error populated upon failure.
 * The lifetime of the returned masks is only guaranteed for the duration of the block.
 */
- (void)segmentImage:(MPPImage *)image
    withCompletionHandler:(void (^)(MPPImageSegmenterResult *_Nullable result,
                                    NSError *_Nullable error))completionHandler
    NS_SWIFT_NAME(segment(image:completion:));

/**
 * Performs segmentation on the provided video frame of type `MPPImage` using the whole image as
 * region of interest.
 *
 * Rotation will be applied according to the `orientation` property of the provided `MPPImage`. Only
 * use this method when the `MPPImageSegmenter` is created with `MPPRunningModeVideo`.
 *
 * This method supports RGBA images. If your `MPPImage` has a source type of
 * `MPPImageSourceTypePixelBuffer` or `MPPImageSourceTypeSampleBuffer`, the underlying pixel buffer
 * must have one of the following pixel format types:
 * 1. kCVPixelFormatType_32BGRA
 * 2. kCVPixelFormatType_32RGBA
 *
 * If your `MPPImage` has a source type of `MPPImageSourceTypeImage` ensure that the color space is
 * RGB with an Alpha channel.
 *
 * @param image The `MPPImage` on which segmentation is to be performed.
 * @param timestampInMilliseconds The video frame's timestamp (in milliseconds). The input
 * timestamps must be monotonically increasing.
 * @param error An optional error parameter populated when there is an error in performing
 * segmentation on the input image.
 *
 * @return An `MPPImageSegmenterResult` that contains a the segmented masks.
 */
- (nullable MPPImageSegmenterResult *)segmentVideoFrame:(MPPImage *)image
                                timestampInMilliseconds:(NSInteger)timestampInMilliseconds
                                                  error:(NSError **)error
    NS_SWIFT_NAME(segment(videoFrame:timestampInMilliseconds:));

/**
 * Performs segmentation on the provided video frame of type `MPPImage` using the whole image as
 * region of interest invokes the given completion handler block with the response. The method
 * returns synchronously once the completion handler returns.
 *
 * Rotation will be applied according to the `orientation` property of the provided `MPPImage`. Only
 * use this method when the `MPPImageSegmenter` is created with `MPPRunningModeVideo`.
 *
 * This method supports RGBA images. If your `MPPImage` has a source type of
 * `MPPImageSourceTypePixelBuffer` or `MPPImageSourceTypeSampleBuffer`, the underlying pixel buffer
 * must have one of the following pixel format types:
 * 1. kCVPixelFormatType_32BGRA
 * 2. kCVPixelFormatType_32RGBA
 *
 * If your `MPPImage` has a source type of `MPPImageSourceTypeImage` ensure that the color space is
 * RGB with an Alpha channel.
 *
 * @param image The `MPPImage` on which segmentation is to be performed.
 * @param timestampInMilliseconds The video frame's timestamp (in milliseconds). The input
 * timestamps must be monotonically increasing.
 * @param completionHandler A block to be invoked with the results of performing segmentation on the
 * image. The block takes two arguments, the optional `MPPImageSegmenterResult` that contains the
 * segmented masks if the segmentation was successful and an optional error only populated upon
 * failure. The lifetime of the returned masks is only guaranteed for the duration of the block.
 */
- (void)segmentVideoFrame:(MPPImage *)image
    timestampInMilliseconds:(NSInteger)timestampInMilliseconds
      withCompletionHandler:(void (^)(MPPImageSegmenterResult *_Nullable result,
                                      NSError *_Nullable error))completionHandler
    NS_SWIFT_NAME(segment(videoFrame:timestampInMilliseconds:completion:));

/**
 * Sends live stream image data of type `MPPImage` to perform segmentation using the whole image as
 * region of interest.
 *
 * Rotation will be applied according to the `orientation` property of the provided `MPPImage`. Only
 * use this method when the `MPPImageSegmenter` is created with`MPPRunningModeLiveStream`.
 *
 * The object which needs to be continuously notified of the available results of image segmentation
 * must confirm to `MPPImageSegmenterLiveStreamDelegate` protocol and implement the
 *`imageSegmenter:didFinishSegmentationWithResult:timestampInMilliseconds:error:` delegate method.
 *
 * It's required to provide a timestamp (in milliseconds) to indicate when the input image is sent
 * to the segmenter. The input timestamps must be monotonically increasing.
 *
 * This method supports RGBA images. If your `MPPImage` has a source type of
 *`MPPImageSourceTypePixelBuffer` or `MPPImageSourceTypeSampleBuffer`, the underlying pixel buffer
 * must have one of the following pixel format types:
 * 1. kCVPixelFormatType_32BGRA
 * 2. kCVPixelFormatType_32RGBA
 *
 * If the input `MPPImage` has a source type of `MPPImageSourceTypeImage` ensure that the color
 * space is RGB with an Alpha channel.
 *
 * If this method is used for classifying live camera frames using `AVFoundation`, ensure that you
 * request `AVCaptureVideoDataOutput` to output frames in `kCMPixelFormat_32RGBA` using its
 * `videoSettings` property.
 *
 * @param image A live stream image data of type `MPPImage` on which segmentation is to be
 * performed.
 * @param timestampInMilliseconds The timestamp (in milliseconds) which indicates when the input
 * image is sent to the segmenter. The input timestamps must be monotonically increasing.
 * @param error An optional error parameter populated when there is an error when sending the input
 * image to the graph.
 *
 * @return `YES` if the image was sent to the task successfully, otherwise `NO`.
 */
- (BOOL)segmentAsyncInImage:(MPPImage *)image
    timestampInMilliseconds:(NSInteger)timestampInMilliseconds
                      error:(NSError **)error
    NS_SWIFT_NAME(segmentAsync(image:timestampInMilliseconds:));

- (instancetype)init NS_UNAVAILABLE;

+ (instancetype)new NS_UNAVAILABLE;

@end

NS_ASSUME_NONNULL_END
