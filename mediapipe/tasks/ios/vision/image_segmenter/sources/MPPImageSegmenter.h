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
 * Get the category label list of the `ImageSegmenter` can recognize. For CATEGORY_MASK type, the
 * index in the category mask corresponds to the category in the label list. For CONFIDENCE_MASK
 * type, the output mask list at index corresponds to the category in the label list. If there is no
 * labelmap provided in the model file, empty array is returned.
 */
@property(nonatomic, readonly) NSArray<NSString *> *labels;

/**
 * Creates a new instance of `ImageSegmenter` from an absolute path to a TensorFlow Lite model
 * file stored locally on the device and the default `ImageSegmenterOptions`.
 *
 * @param modelPath An absolute path to a TensorFlow Lite model file stored locally on the device.
 *
 * @return A new instance of `ImageSegmenter` with the given model path. `nil` if there is an
 * error in initializing the image segmenter.
 */
- (nullable instancetype)initWithModelPath:(NSString *)modelPath error:(NSError **)error;

/**
 * Creates a new instance of `ImageSegmenter` from the given `ImageSegmenterOptions`.
 *
 * @param options The options of type `ImageSegmenterOptions` to use for configuring the
 * `ImageSegmenter`.
 *
 * @return A new instance of `ImageSegmenter` with the given options. `nil` if there is an error
 * in initializing the image segmenter.
 */
- (nullable instancetype)initWithOptions:(MPPImageSegmenterOptions *)options
                                   error:(NSError **)error NS_DESIGNATED_INITIALIZER;

/**
 * Performs segmentation on the provided MPPImage using the whole image as region of interest.
 * Rotation will be applied according to the `orientation` property of the provided `MPImage`. Only
 * use this method when the `ImageSegmenter` is created with running mode, `image`.
 *
 * This method supports segmentation of RGBA images. If your `MPImage` has a source type of
 * `.pixelBuffer` or `.sampleBuffer`, the underlying pixel buffer must use
 * `kCVPixelFormatType_32BGRA` as its pixel format.
 *
 * If your `MPImage` has a source type of `.image` ensure that the color space is RGB with an Alpha
 * channel.
 *
 * @param image The `MPImage` on which segmentation is to be performed.
 *
 * @return An `ImageSegmenterResult` that contains the segmented masks.
 */
- (nullable MPPImageSegmenterResult *)segmentImage:(MPPImage *)image
                                             error:(NSError **)error NS_SWIFT_NAME(segment(image:));

/**
 * Performs segmentation on the provided MPPImage using the whole image as region of interest and
 * invokes the given completion handler block with the response. The method returns synchronously
 * once the completion handler returns.
 *
 * Rotation will be applied according to the `orientation` property of the provided `MPImage`. Only
 * use this method when the `ImageSegmenter` is created with running mode, `image`.
 *
 * This method supports segmentation of RGBA images. If your `MPImage` has a source type of
 * `.pixelBuffer` or `.sampleBuffer`, the underlying pixel buffer must use
 * `kCVPixelFormatType_32BGRA` as its pixel format.
 *
 * If your `MPImage` has a source type of `image` ensure that the color space is RGB with an Alpha
 * channel.
 *
 * @param image The `MPImage` on which segmentation is to be performed.
 * @param completionHandler A block to be invoked with the results of performing segmentation on the
 * image. The block takes two arguments, the optional `ImageSegmenterResult` that contains the
 * segmented masks if the segmentation was successful and an optional error populated upon failure.
 * The lifetime of the returned masks is only guaranteed for the duration of the block.
 */
- (void)segmentImage:(MPPImage *)image
    withCompletionHandler:(void (^)(MPPImageSegmenterResult *_Nullable result,
                                    NSError *_Nullable error))completionHandler
    NS_SWIFT_NAME(segment(image:completion:));

/**
 * Performs segmentation on the provided video frame of type `MPImage` using the whole image as
 * region of interest.
 *
 * Rotation will be applied according to the `orientation` property of the provided `MPImage`. Only
 * use this method when the `ImageSegmenter` is created with `video`.
 *
 * This method supports segmentation of RGBA images. If your `MPImage` has a source type of
 * `.pixelBuffer` or `.sampleBuffer`, the underlying pixel buffer must use
 * `kCVPixelFormatType_32BGRA` as its pixel format.
 *
 * If your `MPImage` has a source type of `image` ensure that the color space is RGB with an Alpha
 * channel.
 *
 * @param image The `MPImage` on which segmentation is to be performed.
 * @param timestampInMilliseconds The video frame's timestamp (in milliseconds). The input
 * timestamps must be monotonically increasing.
 *
 * @return An `ImageSegmenterResult` that contains a the segmented masks.
 */
- (nullable MPPImageSegmenterResult *)segmentVideoFrame:(MPPImage *)image
                                timestampInMilliseconds:(NSInteger)timestampInMilliseconds
                                                  error:(NSError **)error
    NS_SWIFT_NAME(segment(videoFrame:timestampInMilliseconds:));

/**
 * Performs segmentation on the provided video frame of type `MPImage` using the whole image as
 * region of interest invokes the given completion handler block with the response. The method
 * returns synchronously once the completion handler returns.
 *
 * Rotation will be applied according to the `orientation` property of the provided `MPImage`. Only
 * use this method when the `ImageSegmenter` is created with running mode, `video`.
 *
 * This method supports segmentation of RGBA images. If your `MPImage` has a source type of
 * `.pixelBuffer` or `.sampleBuffer`, the underlying pixel buffer must use
 * `kCVPixelFormatType_32BGRA` as its pixel format.
 *
 * If your `MPImage` has a source type of `image` ensure that the color space is RGB with an Alpha
 * channel.
 *
 * @param image The `MPImage` on which segmentation is to be performed.
 * @param timestampInMilliseconds The video frame's timestamp (in milliseconds). The input
 * timestamps must be monotonically increasing.
 * @param completionHandler A block to be invoked with the results of performing segmentation on the
 * image. The block takes two arguments, the optional `ImageSegmenterResult` that contains the
 * segmented masks if the segmentation was successful and an optional error only populated upon
 * failure. The lifetime of the returned masks is only guaranteed for the duration of the block.
 */
- (void)segmentVideoFrame:(MPPImage *)image
    timestampInMilliseconds:(NSInteger)timestampInMilliseconds
      withCompletionHandler:(void (^)(MPPImageSegmenterResult *_Nullable result,
                                      NSError *_Nullable error))completionHandler
    NS_SWIFT_NAME(segment(videoFrame:timestampInMilliseconds:completion:));

/**
 * Sends live stream image data of type `MPImage` to perform segmentation using the whole image as
 *region of interest.
 *
 * Rotation will be applied according to the `orientation` property of the provided `MPImage`. Only
 *use this method when the `ImageSegmenter` is created with running mode, `liveStream`.
 *
 * The object which needs to be continuously notified of the available results of image segmentation
 * must confirm to `ImageSegmenterLiveStreamDelegate` protocol and implement the
 * `imageSegmenter(_:didFinishSegmentationWithResult:timestampInMilliseconds:error:)` delegate
 * method.
 *
 * It's required to provide a timestamp (in milliseconds) to indicate when the input image is sent
 * to the segmenter. The input timestamps must be monotonically increasing.
 *
 * This method supports segmentation of RGBA images. If your `MPImage` has a source type of
 *`.pixelBuffer` or `.sampleBuffer`, the underlying pixel buffer must use
 *`kCVPixelFormatType_32BGRA` as its pixel format.
 *
 * If the input `MPImage` has a source type of `image` ensure that the color space is RGB with an
 * Alpha channel.
 *
 * If this method is used for classifying live camera frames using `AVFoundation`, ensure that you
 * request `AVCaptureVideoDataOutput` to output frames in `kCMPixelFormat_32BGRA` using its
 * `videoSettings` property.
 *
 * @param image A live stream image data of type `MPImage` on which segmentation is to be
 * performed.
 * @param timestampInMilliseconds The timestamp (in milliseconds) which indicates when the input
 * image is sent to the segmenter. The input timestamps must be monotonically increasing.
 *
 * @return `YES` if the image was sent to the task successfully, otherwise `NO`.
 */
- (BOOL)segmentAsyncImage:(MPPImage *)image
    timestampInMilliseconds:(NSInteger)timestampInMilliseconds
                      error:(NSError **)error
    NS_SWIFT_NAME(segmentAsync(image:timestampInMilliseconds:));

- (instancetype)init NS_UNAVAILABLE;

+ (instancetype)new NS_UNAVAILABLE;

@end

NS_ASSUME_NONNULL_END
