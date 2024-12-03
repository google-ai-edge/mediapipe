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
#import "mediapipe/tasks/ios/vision/hand_landmarker/sources/MPPHandLandmarkerOptions.h"
#import "mediapipe/tasks/ios/vision/hand_landmarker/sources/MPPHandLandmarkerResult.h"

NS_ASSUME_NONNULL_BEGIN

/**
 * The enum containing the 21 hand landmarks.
 */
typedef NS_ENUM(NSUInteger, MPPHandLandmark) {
  MPPHandLandmarkWrist,

  MPPHandLandmarkThumbCMC,

  MPPHandLandmarkThumbMCP,

  MPPHandLandmarkThumbIP,

  MPPHandLandmarkThumbTIP,

  MPPHandLandmarkIndexFingerMCP,

  MPPHandLandmarkIndexFingerPIP,

  MPPHandLandmarkIndexFingerDIP,

  MPPHandLandmarkIndexFingerTIP,

  MPPHandLandmarkMiddleFingerMCP,

  MPPHandLandmarkMiddleFingerPIP,

  MPPHandLandmarkMiddleFingerDIP,

  MPPHandLandmarkMiddleFingerTIP,

  MPPHandLandmarkRingFingerMCP,

  MPPHandLandmarkRingFingerPIP,

  MPPHandLandmarkRingFingerDIP,

  MPPHandLandmarkRingFingerTIP,

  MPPHandLandmarkPinkyMCP,

  MPPHandLandmarkPinkyPIP,

  MPPHandLandmarkPinkyDIP,

  MPPHandLandmarkPinkyTIP,

} NS_SWIFT_NAME(HandLandmark);

/**
 * @brief Performs hand landmarks detection on images.
 *
 * This API expects a pre-trained hand landmarks model asset bundle.
 */
NS_SWIFT_NAME(HandLandmarker)
@interface MPPHandLandmarker : NSObject

/** The array of connections between the landmarks in the palm. */
@property(class, nonatomic, readonly) NSArray<MPPConnection *> *handPalmConnections;

/** The array of connections between the landmarks in the thumb. */
@property(class, nonatomic, readonly) NSArray<MPPConnection *> *handThumbConnections;

/** The array of connections between the landmarks in the index finger. */
@property(class, nonatomic, readonly) NSArray<MPPConnection *> *handIndexFingerConnections;

/** The array of connections between the landmarks in the middle finger. */
@property(class, nonatomic, readonly) NSArray<MPPConnection *> *handMiddleFingerConnections;

/** The array of connections between the landmarks in the ring finger. */
@property(class, nonatomic, readonly) NSArray<MPPConnection *> *handRingFingerConnections;

/** The array of connections between the landmarks in the pinky. */
@property(class, nonatomic, readonly) NSArray<MPPConnection *> *handPinkyConnections;

/** The array of connections between all the landmarks in the hand. */
@property(class, nonatomic, readonly) NSArray<MPPConnection *> *handConnections;

/**
 * Creates a new instance of `HandLandmarker` from an absolute path to a model asset bundle stored
 * locally on the device and the default `HandLandmarkerOptions`.
 *
 * @param modelPath An absolute path to a model asset bundle stored locally on the device.
 * @param error An optional error parameter populated when there is an error in initializing the
 * hand landmarker.
 *
 * @return A new instance of `HandLandmarker` with the given model path. `nil` if there is an error
 * in initializing the hand landmarker.
 */
- (nullable instancetype)initWithModelPath:(NSString *)modelPath error:(NSError **)error;

/**
 * Creates a new instance of `HandLandmarker` from the given `HandLandmarkerOptions`.
 *
 * @param options The options of type `HandLandmarkerOptions` to use for configuring the
 * `HandLandmarker`.
 * @param error An optional error parameter populated when there is an error in initializing the
 * hand landmarker.
 *
 * @return A new instance of `HandLandmarker` with the given options. `nil` if there is an error in
 * initializing the hand landmarker.
 */
- (nullable instancetype)initWithOptions:(MPPHandLandmarkerOptions *)options
                                   error:(NSError **)error NS_DESIGNATED_INITIALIZER;

/**
 * Performs hand landmarks detection on the provided `MPImage` using the whole image as region of
 * interest. Rotation will be applied according to the `orientation` property of the provided
 * `MPImage`. Only use this method when the `HandLandmarker` is created with running mode, `.image`.
 *
 * This method supports performing hand landmarks detection on RGBA images. If your `MPImage` has a
 * source type of `.pixelBuffer` or `.sampleBuffer`, the underlying pixel buffer must use
 * `kCVPixelFormatType_32BGRA` as its pixel format.
 *
 * If your `MPImage` has a source type of `.image` ensure that the color space is RGB with an Alpha
 * channel.
 *
 * @param image The `MPImage` on which hand landmarks detection is to be performed.
 * @param error An optional error parameter populated when there is an error in performing hand
 * landmarks detection on the input image.
 *
 * @return  An `HandLandmarkerResult` object that contains the hand hand landmarks detection
 * results.
 */
- (nullable MPPHandLandmarkerResult *)detectImage:(MPPImage *)image
                                            error:(NSError **)error NS_SWIFT_NAME(detect(image:));

/**
 * Performs hand landmarks detection on the provided video frame of type `MPImage` using the whole
 * image as region of interest. Rotation will be applied according to the `orientation` property of
 * the provided `MPImage`. Only use this method when the `HandLandmarker` is created with running
 * mode, `.video`.
 *
 * It's required to provide the video frame's timestamp (in milliseconds). The input timestamps must
 * be monotonically increasing.
 *
 * This method supports performing hand landmarks detection on RGBA images. If your `MPImage` has a
 * source type of `.pixelBuffer` or `.sampleBuffer`, the underlying pixel buffer must use
 * `kCVPixelFormatType_32BGRA` as its pixel format.
 *
 * If your `MPImage` has a source type of `.image` ensure that the color space is RGB with an Alpha
 * channel.
 *
 * @param image The `MPImage` on which hand landmarks detection is to be performed.
 * @param timestampInMilliseconds The video frame's timestamp (in milliseconds). The input
 * timestamps must be monotonically increasing.
 * @param error An optional error parameter populated when there is an error in performing hand
 * landmarks detection on the input video frame.
 *
 * @return  An `HandLandmarkerResult` object that contains the hand hand landmarks detection
 * results.
 */
- (nullable MPPHandLandmarkerResult *)detectVideoFrame:(MPPImage *)image
                               timestampInMilliseconds:(NSInteger)timestampInMilliseconds
                                                 error:(NSError **)error
    NS_SWIFT_NAME(detect(videoFrame:timestampInMilliseconds:));

/**
 * Sends live stream image data of type `MPImage` to perform hand landmarks detection using the
 * whole image as region of interest. Rotation will be applied according to the `orientation`
 * property of the provided `MPImage`. Only use this method when the `HandLandmarker` is created
 * with running mode, `.liveStream`.
 *
 * The object which needs to be continuously notified of the available results of hand landmarks
 * detection must confirm to `HandLandmarkerLiveStreamDelegate` protocol and implement the
 * `handLandmarker(_:didFinishDetectionWithResult:timestampInMilliseconds:error:)` delegate method.
 *
 * It's required to provide a timestamp (in milliseconds) to indicate when the input image is sent
 * to the hand landmarker. The input timestamps must be monotonically increasing.
 *
 * This method supports performing hand landmarks detection on RGBA images. If your `MPImage` has a
 * source type of `.pixelBuffer` or `.sampleBuffer`, the underlying pixel buffer must use
 * `kCVPixelFormatType_32BGRA` as its pixel format.
 *
 * If the input `MPImage` has a source type of `.image` ensure that the color space is RGB with an
 * Alpha channel.
 *
 * If this method is used for performing hand landmarks detection on live camera frames using
 * `AVFoundation`, ensure that you request `AVCaptureVideoDataOutput` to output frames in
 * `kCMPixelFormat_32BGRA` using its `videoSettings` property.
 *
 * @param image A live stream image data of type `MPImage` on which hand landmarks detection is to
 * be performed.
 * @param timestampInMilliseconds The timestamp (in milliseconds) which indicates when the input
 * image is sent to the hand landmarker. The input timestamps must be monotonically increasing.
 * @param error An optional error parameter populated when there is an error in performing hand
 * landmarks detection on the input live stream image data.
 *
 * @return `YES` if the image was sent to the task successfully, otherwise `NO`.
 */
- (BOOL)detectAsyncImage:(MPPImage *)image
    timestampInMilliseconds:(NSInteger)timestampInMilliseconds
                      error:(NSError **)error
    NS_SWIFT_NAME(detectAsync(image:timestampInMilliseconds:));

- (instancetype)init NS_UNAVAILABLE;

+ (instancetype)new NS_UNAVAILABLE;

@end

NS_ASSUME_NONNULL_END
