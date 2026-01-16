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
#import "mediapipe/tasks/ios/vision/image_classifier/sources/MPPImageClassifierOptions.h"
#import "mediapipe/tasks/ios/vision/image_classifier/sources/MPPImageClassifierResult.h"

NS_ASSUME_NONNULL_BEGIN

/**
 * @brief Performs classification on images.
 *
 * The API expects a TFLite model with optional, but strongly recommended,
 * [TFLite Model Metadata.](https://www.tensorflow.org/lite/convert/metadata").
 *
 * The API supports models with one image input tensor and one or more output tensors. To be more
 * specific, here are the requirements.
 *
 * Input tensor
 *  (kTfLiteUInt8/kTfLiteFloat32)
 *  - image input of size `[batch x height x width x channels]`.
 *  - batch inference is not supported (`batch` is required to be 1).
 *  - only RGB inputs are supported (`channels` is required to be 3).
 *  - if type is kTfLiteFloat32, NormalizationOptions are required to be attached to the metadata
 * for input normalization.
 *
 * At least one output tensor with:
 *  (kTfLiteUInt8/kTfLiteFloat32)
 *  - `N `classes and either 2 or 4 dimensions, i.e. `[1 x N]` or `[1 x 1 x 1 x N]`
 *  - optional (but recommended) label map(s) as AssociatedFiles with type TENSOR_AXIS_LABELS,
 *    containing one label per line. The first such AssociatedFile (if any) is used to fill the
 *    `class_name` field of the results. The `display_name` field is filled from the AssociatedFile
 *    (if any) whose locale matches the `display_names_locale` field of the `ImageClassifierOptions`
 *    used at creation time ("en" by default, i.e. English). If none of these are available, only
 *    the `index` field of the results will be filled.
 *  - optional score calibration can be attached using ScoreCalibrationOptions and an AssociatedFile
 *    with type TENSOR_AXIS_SCORE_CALIBRATION. See metadata_schema.fbs [1] for more details.
 */
NS_SWIFT_NAME(ImageClassifier)
@interface MPPImageClassifier : NSObject

/**
 * Creates a new instance of `ImageClassifier` from an absolute path to a TensorFlow Lite model file
 * stored locally on the device and the default `ImageClassifierOptions`.
 *
 * @param modelPath An absolute path to a TensorFlow Lite model file stored locally on the device.
 *
 * @return A new instance of `ImageClassifier` with the given model path. `nil` if there is an
 * error in initializing the image classifier.
 */
- (nullable instancetype)initWithModelPath:(NSString *)modelPath error:(NSError **)error;

/**
 * Creates a new instance of `ImageClassifier` from the given `ImageClassifierOptions`.
 *
 * @param options The options of type `ImageClassifierOptions` to use for configuring the
 * `ImageClassifier`.
 *
 * @return A new instance of `ImageClassifier` with the given options. `nil` if there is an error in
 * initializing the image classifier.
 */
- (nullable instancetype)initWithOptions:(MPPImageClassifierOptions *)options
                                   error:(NSError **)error NS_DESIGNATED_INITIALIZER;

/**
 * Performs image classification on the provided `MPImage` using the whole image as region of
 * interest. Rotation will be applied according to the `orientation` property of the provided
 * `MPImage`. Only use this method when the `ImageClassifier` is created with running mode,
 * `.image`.
 *
 * This method supports classification of RGBA images. If your `MPImage` has a source type
 * of `.pixelBuffer` or `.sampleBuffer`, the underlying pixel buffer must use
 * `kCVPixelFormatType_32BGRA` as its pixel format.
 *
 * If your `MPImage` has a source type of `.image` ensure that the color space is RGB with an Alpha
 * channel.
 *
 * @param image The `MPImage` on which image classification is to be performed.
 *
 * @return  An `ImageClassifierResult` object that contains a list of image classifications.
 */
- (nullable MPPImageClassifierResult *)classifyImage:(MPPImage *)image
                                               error:(NSError **)error
    NS_SWIFT_NAME(classify(image:));

/**
 * Performs image classification on the provided `MPImage` cropped to the specified region of
 * interest. Rotation will be applied on the cropped image according to the `orientation` property
 * of the provided `MPImage`. Only use this method when the `ImageClassifier` is created with
 * running mode, `.image`.
 *
 * This method supports classification of RGBA images. If your `MPImage` has a source type
 * of `.pixelBuffer` or `.sampleBuffer`, the underlying pixel buffer must use
 * `kCVPixelFormatType_32BGRA` as its pixel format.
 *
 * If your `MPImage` has a source type of `.image` ensure that the color space is RGB with an Alpha
 * channel.
 *
 * @param image The `MPImage` on which image classification is to be performed.
 * @param roi A `CGRect` specifying the region of interest within the given `MPImage`, on which
 * image classification should be performed.
 *
 * @return  An `ImageClassifierResult` object that contains a list of image classifications.
 */
- (nullable MPPImageClassifierResult *)classifyImage:(MPPImage *)image
                                    regionOfInterest:(CGRect)roi
                                               error:(NSError **)error
    NS_SWIFT_NAME(classify(image:regionOfInterest:));

/**
 * Performs image classification on the provided video frame of type `MPImage` using the whole
 * image as region of interest. Rotation will be applied according to the `orientation` property of
 * the provided `MPImage`. Only use this method when the `ImageClassifier` is created with
 * running mode `.video`.
 *
 * It's required to provide the video frame's timestamp (in milliseconds). The input timestamps must
 * be monotonically increasing.
 *
 * This method supports classification of RGBA images. If your `MPImage` has a source type
 * of `.pixelBuffer` or `.sampleBuffer`, the underlying pixel buffer must use
 * `kCVPixelFormatType_32BGRA` as its pixel format.
 *
 * If your `MPImage` has a source type of `.image` ensure that the color space is RGB with an Alpha
 * channel.
 *
 * @param image The `MPImage` on which image classification is to be performed.
 * @param timestampInMilliseconds The video frame's timestamp (in milliseconds). The input
 * timestamps must be monotonically increasing.
 *
 * @return  An `ImageClassifierResult` object that contains a list of image classifications.
 */
- (nullable MPPImageClassifierResult *)classifyVideoFrame:(MPPImage *)image
                                  timestampInMilliseconds:(NSInteger)timestampInMilliseconds
                                                    error:(NSError **)error
    NS_SWIFT_NAME(classify(videoFrame:timestampInMilliseconds:));

/**
 * Performs image classification on the provided video frame of type `MPImage` cropped to the
 * specified region of interest. Rotation will be applied according to the `orientation` property of
 * the provided `MPImage`. Only use this method when the `ImageClassifier` is created with `.video`.
 *
 * It's required to provide the video frame's timestamp (in milliseconds). The input timestamps must
 * be monotonically increasing.
 *
 * This method supports classification of RGBA images. If your `MPImage` has a source type
 * of `.pixelBuffer` or `.sampleBuffer`, the underlying pixel buffer must use
 * `kCVPixelFormatType_32BGRA` as its pixel format.
 *
 * If your `MPImage` has a source type of `.image` ensure that the color space is RGB with an Alpha
 * channel.
 *
 * @param image A live stream image data of type `MPImage` on which image classification is to be
 * performed.
 * @param timestampInMilliseconds The video frame's timestamp (in milliseconds). The input
 * timestamps must be monotonically increasing.
 * @param roi A `CGRect` specifying the region of interest within the video frame of type
 * `MPImage`, on which image classification should be performed.
 *
 * @return  An `ImageClassifierResult` object that contains a list of image classifications.
 */
- (nullable MPPImageClassifierResult *)classifyVideoFrame:(MPPImage *)image
                                  timestampInMilliseconds:(NSInteger)timestampInMilliseconds
                                         regionOfInterest:(CGRect)roi
                                                    error:(NSError **)error
    NS_SWIFT_NAME(classify(videoFrame:timestampInMilliseconds:regionOfInterest:));

/**
 * Sends live stream image data of type `MPImage` to perform image classification using the whole
 * image as region of interest. Rotation will be applied according to the `orientation` property of
 * the provided `MPImage`. Only use this method when the `ImageClassifier` is created with running
 * mode `.liveStream`.
 *
 * The object which needs to be continuously notified of the available results of image
 * classification must confirm to `ImageClassifierLiveStreamDelegate` protocol and implement the
 * `imageClassifier(_:didFinishClassificationWithResult:timestampInMilliseconds:error:)`
 * delegate method.
 *
 * It's required to provide a timestamp (in milliseconds) to indicate when the input image is sent
 * to the image classifier. The input timestamps must be monotonically increasing.
 *
 * This method supports classification of RGBA images. If your `MPImage` has a source type
 * of `.pixelBuffer` or `.sampleBuffer`, the underlying pixel buffer must use
 * `kCVPixelFormatType_32BGRA` as its pixel format.
 *
 * If the input `MPImage` has a source type of `.image` ensure that the color space is RGB with an
 * Alpha channel.
 *
 * If this method is used for classifying live camera frames using `AVFoundation`, ensure that you
 * request `AVCaptureVideoDataOutput` to output frames in `kCMPixelFormat_32BGRA` using its
 * `videoSettings` property.
 *
 * @param image A live stream image data of type `MPImage` on which image classification is to be
 * performed.
 * @param timestampInMilliseconds The timestamp (in milliseconds) which indicates when the input
 * image is sent to the image classifier. The input timestamps must be monotonically increasing.
 *
 * @return `true` if the image was sent to the task successfully, otherwise `false`.
 */
- (BOOL)classifyAsyncImage:(MPPImage *)image
    timestampInMilliseconds:(NSInteger)timestampInMilliseconds
                      error:(NSError **)error
    NS_SWIFT_NAME(classifyAsync(image:timestampInMilliseconds:));

/**
 * Sends live stream image data of type `MPImage` to perform image classification, cropped to the
 * specified region of interest.. Rotation will be applied according to the `orientation` property
 * of the provided `MPImage`. Only use this method when the `ImageClassifier` is created with
 * `.liveStream`.
 *
 * The object which needs to be continuously notified of the available results of image
 * classification must confirm to `ImageClassifierLiveStreamDelegate` protocol and implement the
 * `imageClassifier(_:didFinishClassificationWithResult:timestampInMilliseconds:error:)` delegate
 * method.
 *
 * It's required to provide a timestamp (in milliseconds) to indicate when the input image is sent
 * to the image classifier. The input timestamps must be monotonically increasing.
 *
 * This method supports classification of RGBA images. If your `MPImage` has a source type
 * of `.pixelBuffer` or `.sampleBuffer`, the underlying pixel buffer must use
 * `kCVPixelFormatType_32BGRA` as its pixel format.
 *
 * If the input `MPImage` has a source type of `.image` ensure that the color space is RGB with an
 * Alpha channel.
 *
 * If this method is used for classifying live camera frames using `AVFoundation`, ensure that you
 * request `AVCaptureVideoDataOutput` to output frames in `kCMPixelFormat_32BGRA` using its
 * `videoSettings` property.
 *
 * @param image A live stream image data of type `MPImage` on which image classification is to be
 * performed.
 * @param timestampInMilliseconds The timestamp (in milliseconds) which indicates when the input
 * image is sent to the image classifier. The input timestamps must be monotonically increasing.
 * @param roi A `CGRect` specifying the region of interest within the given live stream image data
 * of type `MPImage`, on which image classification should be performed.
 *
 * @return `true` if the image was sent to the task successfully, otherwise `false`.
 */
- (BOOL)classifyAsyncImage:(MPPImage *)image
    timestampInMilliseconds:(NSInteger)timestampInMilliseconds
           regionOfInterest:(CGRect)roi
                      error:(NSError **)error
    NS_SWIFT_NAME(classifyAsync(image:timestampInMilliseconds:regionOfInterest:));

- (instancetype)init NS_UNAVAILABLE;

+ (instancetype)new NS_UNAVAILABLE;

@end

NS_ASSUME_NONNULL_END
