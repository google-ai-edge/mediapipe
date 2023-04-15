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
 * Creates a new instance of `MPPImageClassifier` from an absolute path to a TensorFlow Lite model
 * file stored locally on the device and the default `MPPImageClassifierOptions`.
 *
 * @param modelPath An absolute path to a TensorFlow Lite model file stored locally on the device.
 * @param error An optional error parameter populated when there is an error in initializing the
 * image classifier.
 *
 * @return A new instance of `MPPImageClassifier` with the given model path. `nil` if there is an
 * error in initializing the image classifier.
 */
- (nullable instancetype)initWithModelPath:(NSString *)modelPath error:(NSError **)error;

/**
 * Creates a new instance of `MPPImageClassifier` from the given `MPPImageClassifierOptions`.
 *
 * @param options The options of type `MPPImageClassifierOptions` to use for configuring the
 * `MPPImageClassifier`.
 * @param error An optional error parameter populated when there is an error in initializing the
 * image classifier.
 *
 * @return A new instance of `MPPImageClassifier` with the given options. `nil` if there is an error
 * in initializing the image classifier.
 */
- (nullable instancetype)initWithOptions:(MPPImageClassifierOptions *)options
                                   error:(NSError **)error NS_DESIGNATED_INITIALIZER;

/**
 * Performs image classification on the provided MPPImage using the whole image as region of
 * interest. Rotation will be applied according to the `orientation` property of the provided
 * `MPPImage`. Only use this method when the `MPPImageClassifier` is created with
 * `MPPRunningModeImage`.
 *
 * @param image The `MPPImage` on which image classification is to be performed.
 * @param error An optional error parameter populated when there is an error in performing image
 * classification on the input image.
 *
 * @return  An `MPPImageClassifierResult` object that contains a list of image classifications.
 */
- (nullable MPPImageClassifierResult *)classifyImage:(MPPImage *)image
                                               error:(NSError **)error
    NS_SWIFT_NAME(classify(image:));

/**
 * Performs image classification on the provided `MPPImage` cropped to the specified region of
 * interest. Rotation will be applied on the cropped image according to the `orientation` property
 * of the provided `MPPImage`. Only use this method when the `MPPImageClassifier` is created with
 * `MPPRunningModeImage`.
 *
 * @param image The `MPPImage` on which image classification is to be performed.
 * @param roi A `CGRect` specifying the region of interest within the given `MPPImage`, on which
 * image classification should be performed.
 * @param error An optional error parameter populated when there is an error in performing image
 * classification on the input image.
 *
 * @return  An `MPPImageClassifierResult` object that contains a list of image classifications.
 */
- (nullable MPPImageClassifierResult *)classifyImage:(MPPImage *)image
                                    regionOfInterest:(CGRect)roi
                                               error:(NSError **)error
    NS_SWIFT_NAME(classify(image:regionOfInterest:));

/**
 * Performs image classification on the provided video frame of type `MPPImage` using the whole
 * image as region of interest. Rotation will be applied according to the `orientation` property of
 * the provided `MPPImage`. Only use this method when the `MPPImageClassifier` is created with
 * `MPPRunningModeVideo`.
 *
 * @param image The `MPPImage` on which image classification is to be performed.
 * @param timestampInMilliseconds The video frame's timestamp (in milliseconds). The input
 * timestamps must be monotonically increasing.
 * @param error An optional error parameter populated when there is an error in performing image
 * classification on the input video frame.
 *
 * @return  An `MPPImageClassifierResult` object that contains a list of image classifications.
 */
- (nullable MPPImageClassifierResult *)classifyVideoFrame:(MPPImage *)image
                                  timestampInMilliseconds:(NSInteger)timestampInMilliseconds
                                                    error:(NSError **)error
    NS_SWIFT_NAME(classify(videoFrame:timestampInMilliseconds:));

/**
 * Performs image classification on the provided video frame of type `MPPImage` cropped to the
 * specified region of interest. Rotation will be applied according to the `orientation` property of
 * the provided `MPPImage`. Only use this method when the `MPPImageClassifier` is created with
 * `MPPRunningModeVideo`.
 *
 * It's required to provide the video frame's timestamp (in milliseconds). The input timestamps must
 * be monotonically increasing.
 *
 * @param image A live stream image data of type `MPPImage` on which image classification is to be
 * performed.
 * @param timestampInMilliseconds The video frame's timestamp (in milliseconds). The input
 * timestamps must be monotonically increasing.
 * @param roi A `CGRect` specifying the region of interest within the video frame of type
 * `MPPImage`, on which image classification should be performed.
 * @param error An optional error parameter populated when there is an error in performing image
 * classification on the input video frame.
 *
 * @return  An `MPPImageClassifierResult` object that contains a list of image classifications.
 */
- (nullable MPPImageClassifierResult *)classifyVideoFrame:(MPPImage *)image
                                  timestampInMilliseconds:(NSInteger)timestampInMilliseconds
                                         regionOfInterest:(CGRect)roi
                                                    error:(NSError **)error
    NS_SWIFT_NAME(classify(videoFrame:timestampInMilliseconds:regionOfInterest:));

/**
 * Sends live stream image data of type `MPPImage` to perform image classification using the whole
 * image as region of interest. Rotation will be applied according to the `orientation` property of
 * the provided `MPPImage`. Only use this method when the `MPPImageClassifier` is created with
 * `MPPRunningModeLiveStream`. Results are provided asynchronously via the `completion` callback
 * provided in the `MPPImageClassifierOptions`.
 *
 * It's required to provide a timestamp (in milliseconds) to indicate when the input image is sent
 * to the image classifier. The input timestamps must be monotonically increasing.
 *
 * @param image A live stream image data of type `MPPImage` on which image classification is to be
 * performed.
 * @param timestampInMilliseconds The timestamp (in milliseconds) which indicates when the input
 * image is sent to the image classifier. The input timestamps must be monotonically increasing.
 * @param error An optional error parameter populated when there is an error in performing image
 * classification on the input live stream image data.
 *
 * @return `YES` if the image was sent to the task successfully, otherwise `NO`.
 */
- (BOOL)classifyAsyncImage:(MPPImage *)image
    timestampInMilliseconds:(NSInteger)timestampInMilliseconds
                      error:(NSError **)error
    NS_SWIFT_NAME(classifyAsync(image:timestampInMilliseconds:));

/**
 * Sends live stream image data of type `MPPImage` to perform image classification, cropped to the
 * specified region of interest.. Rotation will be applied according to the `orientation` property
 * of the provided `MPPImage`. Only use this method when the `MPPImageClassifier` is created with
 * `MPPRunningModeLiveStream`. Results are provided asynchronously via the `completion` callback
 * provided in the `MPPImageClassifierOptions`.
 *
 * It's required to provide a timestamp (in milliseconds) to indicate when the input image is sent
 * to the image classifier. The input timestamps must be monotonically increasing.
 *
 * @param image A live stream image data of type `MPPImage` on which image classification is to be
 * performed.
 * @param timestampInMilliseconds The timestamp (in milliseconds) which indicates when the input
 * image is sent to the image classifier. The input timestamps must be monotonically increasing.
 * @param roi A `CGRect` specifying the region of interest within the given live stream image data
 * of type `MPPImage`, on which image classification should be performed.
 * @param error An optional error parameter populated when there is an error in performing image
 * classification on the input live stream image data.
 *
 * @return `YES` if the image was sent to the task successfully, otherwise `NO`.
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
