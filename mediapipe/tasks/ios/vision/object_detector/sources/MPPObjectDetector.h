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
#import "mediapipe/tasks/ios/vision/object_detector/sources/MPPObjectDetectionResult.h"
#import "mediapipe/tasks/ios/vision/object_detector/sources/MPPObjectDetectorOptions.h"

NS_ASSUME_NONNULL_BEGIN

/**
 * @brief Class that performs object detection on images.
 *
 * The API expects a TFLite model with mandatory TFLite Model Metadata.
 *
 * The API supports models with one image input tensor and one or more output tensors. To be more
 * specific, here are the requirements:
 *
 * Input tensor
 *  (kTfLiteUInt8/kTfLiteFloat32)
 *  - image input of size `[batch x height x width x channels]`.
 *  - batch inference is not supported (`batch` is required to be 1).
 *  - only RGB inputs are supported (`channels` is required to be 3).
 *  - if type is kTfLiteFloat32, NormalizationOptions are required to be attached to the metadata
 * for input normalization.
 *
 * Output tensors must be the 4 outputs of a `DetectionPostProcess` op, i.e:(kTfLiteFloat32)
 *  (kTfLiteUInt8/kTfLiteFloat32)
 *  - locations tensor of size `[num_results x 4]`, the inner array representing bounding boxes
 *    in the form [top, left, right, bottom].
 *  - BoundingBoxProperties are required to be attached to the metadata and must specify
 *    type=BOUNDARIES and coordinate_type=RATIO.
 *  (kTfLiteFloat32)
 *  - classes tensor of size `[num_results]`, each value representing the integer index of a
 *  class.
 *  - optional (but recommended) label map(s) can be attached as AssociatedFiles with type
 *    TENSOR_VALUE_LABELS, containing one label per line. The first such AssociatedFile (if any) is
 *    used to fill the `class_name` field of the results. The `display_name` field is filled from
 *    the AssociatedFile (if any) whose locale matches the `display_names_locale` field of the
 *    `ObjectDetectorOptions` used at creation time ("en" by default, i.e. English). If none of
 *    these are available, only the `index` field of the results will be filled.
 *  (kTfLiteFloat32)
 *  - scores tensor of size `[num_results]`, each value representing the score of the detected
 *    object.
 *  - optional score calibration can be attached using ScoreCalibrationOptions and an
 *    AssociatedFile with type TENSOR_AXIS_SCORE_CALIBRATION. See metadata_schema.fbs [1] for more
 *    details.
 *  (kTfLiteFloat32)
 *  - integer num_results as a tensor of size `[1]`
 */
NS_SWIFT_NAME(ObjectDetector)
@interface MPPObjectDetector : NSObject

/**
 * Creates a new instance of `MPPObjectDetector` from an absolute path to a TensorFlow Lite model
 * file stored locally on the device and the default `MPPObjectDetector`.
 *
 * @param modelPath An absolute path to a TensorFlow Lite model file stored locally on the device.
 * @param error An optional error parameter populated when there is an error in initializing the
 * object detector.
 *
 * @return A new instance of `MPPObjectDetector` with the given model path. `nil` if there is an
 * error in initializing the object detector.
 */
- (nullable instancetype)initWithModelPath:(NSString *)modelPath error:(NSError **)error;

/**
 * Creates a new instance of `MPPObjectDetector` from the given `MPPObjectDetectorOptions`.
 *
 * @param options The options of type `MPPObjectDetectorOptions` to use for configuring the
 * `MPPImageClassifMPPObjectDetectorier`.
 * @param error An optional error parameter populated when there is an error in initializing the
 * object detector.
 *
 * @return A new instance of `MPPObjectDetector` with the given options. `nil` if there is an error
 * in initializing the object detector.
 */
- (nullable instancetype)initWithOptions:(MPPObjectDetectorOptions *)options
                                   error:(NSError **)error NS_DESIGNATED_INITIALIZER;

/**
 * Performs object detection on the provided MPPImage using the whole image as region of
 * interest. Rotation will be applied according to the `orientation` property of the provided
 * `MPPImage`. Only use this method when the `MPPObjectDetector` is created with
 * `MPPRunningModeImage`.
 *
 * This method supports classification of RGBA images. If your `MPPImage` has a source type of
 * `MPPImageSourceTypePixelBuffer` or `MPPImageSourceTypeSampleBuffer`, the underlying pixel buffer
 * must have one of the following pixel format types:
 * 1. kCVPixelFormatType_32BGRA
 * 2. kCVPixelFormatType_32RGBA
 *
 * If your `MPPImage` has a source type of `MPPImageSourceTypeImage` ensure that the color space is
 * RGB with an Alpha channel.
 *
 * @param image The `MPPImage` on which object detection is to be performed.
 * @param error An optional error parameter populated when there is an error in performing object
 * detection on the input image.
 *
 * @return An `MPPObjectDetectionResult` object that contains a list of detections, each detection
 * has a bounding box that is expressed in the unrotated input frame of reference coordinates
 * system, i.e. in `[0,image_width) x [0,image_height)`, which are the dimensions of the underlying
 * image data.
 */
- (nullable MPPObjectDetectionResult *)detectInImage:(MPPImage *)image
                                               error:(NSError **)error
    NS_SWIFT_NAME(detect(image:));

/**
 * Performs object detection on the provided video frame of type `MPPImage` using the whole
 * image as region of interest. Rotation will be applied according to the `orientation` property of
 * the provided `MPPImage`. Only use this method when the `MPPObjectDetector` is created with
 * `MPPRunningModeVideo`.
 *
 * This method supports classification of RGBA images. If your `MPPImage` has a source type of
 * `MPPImageSourceTypePixelBuffer` or `MPPImageSourceTypeSampleBuffer`, the underlying pixel buffer
 * must have one of the following pixel format types:
 * 1. kCVPixelFormatType_32BGRA
 * 2. kCVPixelFormatType_32RGBA
 *
 * If your `MPPImage` has a source type of `MPPImageSourceTypeImage` ensure that the color space is
 * RGB with an Alpha channel.
 *
 * @param image The `MPPImage` on which object detection is to be performed.
 * @param timestampInMilliseconds The video frame's timestamp (in milliseconds). The input
 * timestamps must be monotonically increasing.
 * @param error An optional error parameter populated when there is an error in performing object
 * detection on the input image.
 *
 * @return An `MPPObjectDetectionResult` object that contains a list of detections, each detection
 * has a bounding box that is expressed in the unrotated input frame of reference coordinates
 * system, i.e. in `[0,image_width) x [0,image_height)`, which are the dimensions of the underlying
 * image data.
 */
- (nullable MPPObjectDetectionResult *)detectInVideoFrame:(MPPImage *)image
                                  timestampInMilliseconds:(NSInteger)timestampInMilliseconds
                                                    error:(NSError **)error
    NS_SWIFT_NAME(detect(videoFrame:timestampInMilliseconds:));

/**
 * Sends live stream image data of type `MPPImage` to perform object detection using the whole
 * image as region of interest. Rotation will be applied according to the `orientation` property of
 * the provided `MPPImage`. Only use this method when the `MPPObjectDetector` is created with
 * `MPPRunningModeLiveStream`.
 *
 * The object which needs to be continuously notified of the available results of object
 * detection must confirm to `MPPObjectDetectorLiveStreamDelegate` protocol and implement the
 * `objectDetector:didFinishDetectionWithResult:timestampInMilliseconds:error:` delegate method.
 *
 * It's required to provide a timestamp (in milliseconds) to indicate when the input image is sent
 * to the object detector. The input timestamps must be monotonically increasing.
 *
 * This method supports classification of RGBA images. If your `MPPImage` has a source type of
 * `MPPImageSourceTypePixelBuffer` or `MPPImageSourceTypeSampleBuffer`, the underlying pixel buffer
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
 * @param image A live stream image data of type `MPPImage` on which object detection is to be
 * performed.
 * @param timestampInMilliseconds The timestamp (in milliseconds) which indicates when the input
 * image is sent to the object detector. The input timestamps must be monotonically increasing.
 * @param error An optional error parameter populated when there is an error in performing object
 * detection on the input live stream image data.
 *
 * @return `YES` if the image was sent to the task successfully, otherwise `NO`.
 */
- (BOOL)detectAsyncInImage:(MPPImage *)image
    timestampInMilliseconds:(NSInteger)timestampInMilliseconds
                      error:(NSError **)error
    NS_SWIFT_NAME(detectAsync(image:timestampInMilliseconds:));

- (instancetype)init NS_UNAVAILABLE;

+ (instancetype)new NS_UNAVAILABLE;

@end

NS_ASSUME_NONNULL_END
