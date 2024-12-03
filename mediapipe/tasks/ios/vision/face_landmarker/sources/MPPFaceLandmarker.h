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
 * Creates a new instance of `FaceLandmarker` from an absolute path to a TensorFlow Lite model
 * file stored locally on the device and the default `FaceLandmarker`.
 *
 * @param modelPath An absolute path to a TensorFlow Lite model file stored locally on the device.
 *
 * @return A new instance of `FaceLandmarker` with the given model path. `nil` if there is an
 * error in initializing the face landmaker.
 */
- (nullable instancetype)initWithModelPath:(NSString *)modelPath error:(NSError **)error;

/**
 * Creates a new instance of `FaceLandmarker` from the given `FaceLandmarkerOptions`.
 *
 * @param options The options of type `FaceLandmarkerOptions` to use for configuring the
 * `FaceLandmarker`.
 *
 * @return A new instance of `FaceLandmarker` with the given options. `nil` if there is an error
 * in initializing the face landmaker.
 */
- (nullable instancetype)initWithOptions:(MPPFaceLandmarkerOptions *)options
                                   error:(NSError **)error NS_DESIGNATED_INITIALIZER;

/**
 * Performs face landmark detection on the provided `MPImage` using the whole image as region of
 * interest. Rotation will be applied according to the `orientation` property of the provided
 * `MPImage`. Only use this method when the `FaceLandmarker` is created with `.image`.
 *
 * This method supports performing face landmark detection on RGBA images. If your `MPImage` has a
 * source type of `.pixelBuffer` or `.sampleBuffer`, the underlying pixel buffer must use
 * `kCVPixelFormatType_32BGRA` as its pixel format.
 *
 * If your `MPImage` has a source type of `.image` ensure that the color space is RGB with an
 * Alpha channel.
 *
 * @param image The `MPImage` on which face landmark detection is to be performed.
 *
 * @return An `FaceLandmarkerResult` that contains a list of landmarks. `nil` if there is an error
 * in initializing the face landmaker.
 */
- (nullable MPPFaceLandmarkerResult *)detectImage:(MPPImage *)image
                                            error:(NSError **)error NS_SWIFT_NAME(detect(image:));

/**
 * Performs face landmark detection on the provided video frame of type `MPImage` using the whole
 * image as region of interest. Rotation will be applied according to the `orientation` property of
 * the provided `MPImage`. Only use this method when the `FaceLandmarker` is created with running
 * mode `.video`.
 *
 * This method supports performing face landmark detection on RGBA images. If your `MPImage` has a
 * source type of `.pixelBuffer` or `.sampleBuffer`, the underlying pixel buffer must use
 * `kCVPixelFormatType_32BGRA` as its pixel format.
 *
 * If your `MPImage` has a source type of `.image` ensure that the color space is RGB with an Alpha
 * channel.
 *
 * @param image The `MPImage` on which face landmark detection is to be performed.
 * @param timestampInMilliseconds The video frame's timestamp (in milliseconds). The input
 * timestamps must be monotonically increasing.
 *
 * @return An `FaceLandmarkerResult` that contains a list of landmarks. `nil` if there is an
 * error in initializing the face landmaker.
 */
- (nullable MPPFaceLandmarkerResult *)detectVideoFrame:(MPPImage *)image
                               timestampInMilliseconds:(NSInteger)timestampInMilliseconds
                                                 error:(NSError **)error
    NS_SWIFT_NAME(detect(videoFrame:timestampInMilliseconds:));

/**
 * Sends live stream image data of type `MPImage` to perform face landmark detection using the
 * whole image as region of interest. Rotation will be applied according to the `orientation`
 * property of the provided `MPImage`. Only use this method when the `FaceLandmarker` is created
 * with `.liveStream`.
 *
 * The object which needs to be continuously notified of the available results of face
 * detection must confirm to `FaceLandmarkerLiveStreamDelegate` protocol and implement the
 * `faceLandmarker(_:didFinishDetectionWithResult:timestampInMilliseconds:error:)` delegate method.
 *
 * It's required to provide a timestamp (in milliseconds) to indicate when the input image is sent
 * to the face detector. The input timestamps must be monotonically increasing.
 *
 * This method supports performing face landmark detection on RGBA images. If your `MPImage` has a
 * source type of `.pixelBuffer` or `.sampleBuffer`, the underlying pixel buffer must use
 * `kCVPixelFormatType_32BGRA` as its pixel format.
 *
 * If the input `MPImage` has a source type of `.image` ensure that the color space is RGB with an
 * Alpha channel.
 *
 * If this method is used for classifying live camera frames using `AVFoundation`, ensure that you
 * request `AVCaptureVideoDataOutput` to output frames in `kCMPixelFormat_32BGRA` using its
 * `videoSettings` property.
 *
 * @param image A live stream image data of type `MPImage` on which face landmark detection is to be
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
