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
#import "mediapipe/tasks/ios/vision/face_detector/sources/MPPFaceDetectorOptions.h"
#import "mediapipe/tasks/ios/vision/face_detector/sources/MPPFaceDetectorResult.h"

NS_ASSUME_NONNULL_BEGIN

/**
 * @brief Class that performs face detection on images.
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
 *  - scores tensor of size `[num_results]`, each value representing the score of the detected
 *    face.
 *  - optional score calibration can be attached using ScoreCalibrationOptions and an
 *    AssociatedFile with type TENSOR_AXIS_SCORE_CALIBRATION. See metadata_schema.fbs [1] for more
 *    details.
 *  (kTfLiteFloat32)
 *  - integer num_results as a tensor of size `[1]`
 */
NS_SWIFT_NAME(FaceDetector)
@interface MPPFaceDetector : NSObject

/**
 * Creates a new instance of `FaceDetector` from an absolute path to a TensorFlow Lite model
 * file stored locally on the device and the default `FaceDetector`.
 *
 * @param modelPath An absolute path to a TensorFlow Lite model file stored locally on the device.
 *
 * @return A new instance of `FaceDetector` with the given model path. `nil` if there is an
 * error in initializing the face detector.
 */
- (nullable instancetype)initWithModelPath:(NSString *)modelPath error:(NSError **)error;

/**
 * Creates a new instance of `FaceDetector` from the given `FaceDetectorOptions`.
 *
 * @param options The options of type `FaceDetectorOptions` to use for configuring the
 * `FaceDetector`.
 *
 * @return A new instance of `FaceDetector` with the given options. `nil` if there is an error
 * in initializing the face detector.
 */
- (nullable instancetype)initWithOptions:(MPPFaceDetectorOptions *)options
                                   error:(NSError **)error NS_DESIGNATED_INITIALIZER;

/**
 * Performs face detection on the provided `MPImage` using the whole image as region of
 * interest. Rotation will be applied according to the `orientation` property of the provided
 * `MPImage`. Only use this method when the `FaceDetector` is created with running mode `.image`.
 *
 * This method supports performing face detection on RGBA images. If your `MPImage` has a source
 * type of `.pixelBuffer` or `.sampleBuffer`, the underlying pixel buffer must use
 * `kCVPixelFormatType_32BGRA` as its pixel format.
 *
 * If your `MPImage` has a source type of `.image` ensure that the color space is
 * RGB with an Alpha channel.
 *
 * @param image The `MPImage` on which face detection is to be performed.
 *
 * @return An `FaceDetectorResult` face that contains a list of detections, each detection
 * has a bounding box that is expressed in the unrotated input frame of reference coordinates
 * system, i.e. in `[0,image_width) x [0,image_height)`, which are the dimensions of the underlying
 * image data.
 */
- (nullable MPPFaceDetectorResult *)detectImage:(MPPImage *)image
                                          error:(NSError **)error NS_SWIFT_NAME(detect(image:));

/**
 * Performs face detection on the provided video frame of type `MPImage` using the whole
 * image as region of interest. Rotation will be applied according to the `orientation` property of
 * the provided `MPImage`. Only use this method when the `FaceDetector` is created with running
 * mode `.video`.
 *
 * This method supports performing face detection on RGBA images. If your `MPImage` has a source
 * type of `.pixelBuffer` or `.sampleBuffer`, the underlying pixel buffer must use
 * `kCVPixelFormatType_32BGRA` as its pixel format.
 *
 * If your `MPImage` has a source type of `.image` ensure that the color space is RGB with an Alpha
 * channel.
 *
 * @param image The `MPImage` on which face detection is to be performed.
 * @param timestampInMilliseconds The video frame's timestamp (in milliseconds). The input
 * timestamps must be monotonically increasing.
 *
 * @return An `FaceDetectorResult` face that contains a list of detections, each detection
 * has a bounding box that is expressed in the unrotated input frame of reference coordinates
 * system, i.e. in `[0,image_width) x [0,image_height)`, which are the dimensions of the underlying
 * image data.
 */
- (nullable MPPFaceDetectorResult *)detectVideoFrame:(MPPImage *)image
                             timestampInMilliseconds:(NSInteger)timestampInMilliseconds
                                               error:(NSError **)error
    NS_SWIFT_NAME(detect(videoFrame:timestampInMilliseconds:));

/**
 * Sends live stream image data of type `MPImage` to perform face detection using the whole
 * image as region of interest. Rotation will be applied according to the `orientation` property of
 * the provided `MPImage`. Only use this method when the `FaceDetector` is created with
 * `.liveStream`.
 *
 * The object which needs to be continuously notified of the available results of face
 * detection must confirm to `FaceDetectorLiveStreamDelegate` protocol and implement the
 * `faceDetector(_:didFinishDetectionWithResult:timestampInMilliseconds:error:)` delegate method.
 *
 * It's required to provide a timestamp (in milliseconds) to indicate when the input image is sent
 * to the face detector. The input timestamps must be monotonically increasing.
 *
 * This method supports performing face detection on RGBA images. If your `MPImage` has a source
 * type of `.pixelBuffer` or `.sampleBuffer`, the underlying pixel buffer must use
 * `kCVPixelFormatType_32BGRA` as its pixel format.
 *
 * If the input `MPImage` has a source type of `.image` ensure that the color
 * space is RGB with an Alpha channel.
 *
 * If this method is used for classifying live camera frames using `AVFoundation`, ensure that you
 * request `AVCaptureVideoDataOutput` to output frames in `kCMPixelFormat_32BGRA` using its
 * `videoSettings` property.
 *
 * @param image A live stream image data of type `MPImage` on which face detection is to be
 * performed.
 * @param timestampInMilliseconds The timestamp (in milliseconds) which indicates when the input
 * image is sent to the face detector. The input timestamps must be monotonically increasing.
 *
 * @return `true` if the image was sent to the task successfully, otherwise `false`.
 */
- (BOOL)detectAsyncImage:(MPPImage *)image
    timestampInMilliseconds:(NSInteger)timestampInMilliseconds
                      error:(NSError **)error
    NS_SWIFT_NAME(detectAsync(image:timestampInMilliseconds:));

- (instancetype)init NS_UNAVAILABLE;

+ (instancetype)new NS_UNAVAILABLE;

@end

NS_ASSUME_NONNULL_END
