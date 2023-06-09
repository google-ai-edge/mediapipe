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

#import "mediapipe/tasks/ios/components/containers/sources/MPPConnection.h"
#import "mediapipe/tasks/ios/vision/core/sources/MPPImage.h"
#import "mediapipe/tasks/ios/vision/face_landmarker/sources/MPPFaceLandmarkerOptions.h"
#import "mediapipe/tasks/ios/vision/face_landmarker/sources/MPPFaceLandmarkerResult.h"

NS_ASSUME_NONNULL_BEGIN

/**
 * @brief Class that performs face landmark detection on images.
 *
 * The API expects a TFLite model with mandatory TFLite Model Metadata.
 */
NS_SWIFT_NAME(FaceLandmarker)
@interface MPPFaceLandmarker : NSObject

/**
 * Creates a new instance of `MPPFaceLandmarker` from an absolute path to a TensorFlow Lite model
 * file stored locally on the device and the default `MPPFaceLandmarker`.
 *
 * @param modelPath An absolute path to a TensorFlow Lite model file stored locally on the device.
 * @param error An optional error parameter populated when there is an error in initializing the
 * face landmaker.
 *
 * @return A new instance of `MPPFaceLandmarker` with the given model path. `nil` if there is an
 * error in initializing the face landmaker.
 */
- (nullable instancetype)initWithModelPath:(NSString *)modelPath error:(NSError **)error;

/**
 * Creates a new instance of `MPPFaceLandmarker` from the given `MPPFaceLandmarkerOptions`.
 *
 * @param options The options of type `MPPFaceLandmarkerOptions` to use for configuring the
 * `MPPFaceLandmarker`.
 * @param error An optional error parameter populated when there is an error in initializing the
 * face landmaker.
 *
 * @return A new instance of `MPPFaceLandmarker` with the given options. `nil` if there is an error
 * in initializing the face landmaker.
 */
- (nullable instancetype)initWithOptions:(MPPFaceLandmarkerOptions *)options
                                   error:(NSError **)error NS_DESIGNATED_INITIALIZER;

/**
 * Performs face landmark detection on the provided MPPImage using the whole image as region of
 * interest. Rotation will be applied according to the `orientation` property of the provided
 * `MPPImage`. Only use this method when the `MPPFaceLandmarker` is created with
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
 * @param image The `MPPImage` on which face landmark detection is to be performed.
 * @param error An optional error parameter populated when there is an error in performing face
 * landmark detection on the input image.
 *
 * @return An `MPPFaceLandmarkerResult` that contains a list of landmarks.
 */
- (nullable MPPFaceLandmarkerResult *)detectInImage:(MPPImage *)image
                                              error:(NSError **)error NS_SWIFT_NAME(detect(image:));

/**
 * Performs face landmark detection on the provided video frame of type `MPPImage` using the whole
 * image as region of interest. Rotation will be applied according to the `orientation` property of
 * the provided `MPPImage`. Only use this method when the `MPPFaceLandmarker` is created with
 * `MPPRunningModeVideo`.
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
 * @param image The `MPPImage` on which face landmark detection is to be performed.
 * @param timestampInMilliseconds The video frame's timestamp (in milliseconds). The input
 * timestamps must be monotonically increasing.
 * @param error An optional error parameter populated when there is an error in performing face
 * landmark detection on the input image.
 *
 * @return An `MPPFaceLandmarkerResult` that contains a list of landmarks.
 */
- (nullable MPPFaceLandmarkerResult *)detectInVideoFrame:(MPPImage *)image
                                 timestampInMilliseconds:(NSInteger)timestampInMilliseconds
                                                   error:(NSError **)error
    NS_SWIFT_NAME(detect(videoFrame:timestampInMilliseconds:));

/**
 * Sends live stream image data of type `MPPImage` to perform face landmark detection using the
 * whole image as region of interest. Rotation will be applied according to the `orientation`
 * property of the provided `MPPImage`. Only use this method when the `MPPFaceLandmarker` is created
 * with `MPPRunningModeLiveStream`.
 *
 * The object which needs to be continuously notified of the available results of face
 * detection must confirm to `MPPFaceLandmarkerLiveStreamDelegate` protocol and implement the
 * `faceLandmarker:didFinishDetectionWithResult:timestampInMilliseconds:error:` delegate method.
 *
 * It's required to provide a timestamp (in milliseconds) to indicate when the input image is sent
 * to the face detector. The input timestamps must be monotonically increasing.
 *
 * This method supports RGBA images. If your `MPPImage` has a source type of
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
 * @param image A live stream image data of type `MPPImage` on which face landmark detection is to
 * be performed.
 * @param timestampInMilliseconds The timestamp (in milliseconds) which indicates when the input
 * image is sent to the face detector. The input timestamps must be monotonically increasing.
 * @param error An optional error parameter populated when there is an error when sending the input
 * image to the graph.
 *
 * @return `YES` if the image was sent to the task successfully, otherwise `NO`.
 */
- (BOOL)detectAsyncInImage:(MPPImage *)image
    timestampInMilliseconds:(NSInteger)timestampInMilliseconds
                      error:(NSError **)error
    NS_SWIFT_NAME(detectAsync(image:timestampInMilliseconds:));

/**
 * Returns the connections between all the landmarks in the lips.
 *
 * @return An array of connections between all the landmarks in the lips.
 */
+ (NSArray<MPPConnection *> *)lipsConnections;

/**
 * Returns the connections between all the landmarks in the left eye.
 *
 * @return An array of connections between all the landmarks in the left eye.
 */
+ (NSArray<MPPConnection *> *)leftEyeConnections;

/**
 * Returns the connections between all the landmarks in the left eyebrow.
 *
 * @return An array of connections between all the landmarks in the left eyebrow.
 */
+ (NSArray<MPPConnection *> *)leftEyebrowConnections;

/**
 * Returns the connections between all the landmarks in the left iris.
 *
 * @return An array of connections between all the landmarks in the left iris.
 */
+ (NSArray<MPPConnection *> *)leftIrisConnections;

/**
 * Returns the connections between all the landmarks in the right eye.
 *
 * @return An array of connections between all the landmarks in the right eyr.
 */
+ (NSArray<MPPConnection *> *)rightEyeConnections;

/**
 * Returns the connections between all the landmarks in the right eyebrow.
 *
 * @return An array of connections between all the landmarks in the right eyebrow.
 */
+ (NSArray<MPPConnection *> *)rightEyebrowConnections;

/**
 * Returns the connections between all the landmarks in the right iris.
 *
 * @return An array of connections between all the landmarks in the right iris.
 */
+ (NSArray<MPPConnection *> *)rightIrisConnections;

/**
 * Returns the connections between all the landmarks of the face oval.
 *
 * @return An array of connections between all the landmarks of the face oval.
 */
+ (NSArray<MPPConnection *> *)faceOvalConnections;

/**
 * Returns the connections between making up the contours of the face.
 *
 * @return An array of connections between all the contours of the face.
 */
+ (NSArray<MPPConnection *> *)contoursConnections;

/**
 * Returns the connections between all the landmarks making up the tesselation of the face.
 *
 * @return An array of connections between all the landmarks  making up the tesselation of the face.
 */
+ (NSArray<MPPConnection *> *)tesselationConnections;

/**
 * Returns the connections between all the landmarks in the face.
 *
 * @return An array of connections between all the landmarks in the face.
 */
+ (NSArray<MPPConnection *> *)faceConnections;

- (instancetype)init NS_UNAVAILABLE;

+ (instancetype)new NS_UNAVAILABLE;

@end

NS_ASSUME_NONNULL_END
