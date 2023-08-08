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
#import "mediapipe/tasks/ios/vision/pose_landmarker/sources/MPPPoseLandmarkerOptions.h"
#import "mediapipe/tasks/ios/vision/pose_landmarker/sources/MPPPoseLandmarkerResult.h"

NS_ASSUME_NONNULL_BEGIN

/**
 * @brief Performs pose landmarks detection on images.
 *
 * This API expects a pre-trained pose landmarks model asset bundle.
 */
NS_SWIFT_NAME(PoseLandmarker)
@interface MPPPoseLandmarker : NSObject

/** The array of connections between all the landmarks in the pose. */
@property (class, nonatomic, readonly) NSArray<MPPConnection *> *poseLandmarksConnections;

/**
 * Creates a new instance of `MPPPoseLandmarker` from an absolute path to a model asset bundle
 * stored locally on the device and the default `MPPPoseLandmarkerOptions`.
 *
 * @param modelPath An absolute path to a model asset bundle stored locally on the device.
 * @param error An optional error parameter populated when there is an error in initializing the
 * pose landmarker.
 *
 * @return A new instance of `MPPPoseLandmarker` with the given model path. `nil` if there is an
 * error in initializing the pose landmarker.
 */
- (nullable instancetype)initWithModelPath:(NSString *)modelPath error:(NSError **)error;

/**
 * Creates a new instance of `MPPPoseLandmarker` from the given `MPPPoseLandmarkerOptions`.
 *
 * @param options The options of type `MPPPoseLandmarkerOptions` to use for configuring the
 * `MPPPoseLandmarker`.
 * @param error An optional error parameter populated when there is an error in initializing the
 * pose landmarker.
 *
 * @return A new instance of `MPPPoseLandmarker` with the given options. `nil` if there is an
 * error in initializing the pose landmarker.
 */
- (nullable instancetype)initWithOptions:(MPPPoseLandmarkerOptions *)options
                                   error:(NSError **)error NS_DESIGNATED_INITIALIZER;

/**
 * Performs pose landmarks detection on the provided `MPPImage` using the whole image as region of
 * interest. Rotation will be applied according to the `orientation` property of the provided
 * `MPPImage`. Only use this method when the `MPPPoseLandmarker` is created with
 * `MPPRunningModeImage`.
 *
 * This method supports performing pose landmarks detection on RGBA images. If your `MPPImage` has a
 * source type of `MPPImageSourceTypePixelBuffer` or `MPPImageSourceTypeSampleBuffer`, the
 * underlying pixel buffer must have one of the following pixel format types:
 * 1. kCVPixelFormatType_32BGRA
 * 2. kCVPixelFormatType_32RGBA
 *
 * If your `MPPImage` has a source type of `MPPImageSourceTypeImage` ensure that the color space is
 * RGB with an Alpha channel.
 *
 * @param image The `MPPImage` on which pose landmarks detection is to be performed.
 * @param error An optional error parameter populated when there is an error in performing pose
 * landmarks detection on the input image.
 *
 * @return  An `MPPPoseLandmarkerResult` object that contains the pose pose landmarks detection
 * results.
 */
- (nullable MPPPoseLandmarkerResult *)detectInImage:(MPPImage *)image
                                              error:(NSError **)error NS_SWIFT_NAME(detect(image:));

/**
 * Performs pose landmarks detection on the provided video frame of type `MPPImage` using the whole
 * image as region of interest. Rotation will be applied according to the `orientation` property of
 * the provided `MPPImage`. Only use this method when the `MPPPoseLandmarker` is created with
 * `MPPRunningModeVideo`.
 *
 * It's required to provide the video frame's timestamp (in milliseconds). The input timestamps must
 * be monotonically increasing.
 *
 * This method supports performing pose landmarks detection on RGBA images. If your `MPPImage` has a
 * source type of `MPPImageSourceTypePixelBuffer` or `MPPImageSourceTypeSampleBuffer`, the
 * underlying pixel buffer must have one of the following pixel format types:
 * 1. kCVPixelFormatType_32BGRA
 * 2. kCVPixelFormatType_32RGBA
 *
 * If your `MPPImage` has a source type of `MPPImageSourceTypeImage` ensure that the color space is
 * RGB with an Alpha channel.
 *
 * @param image The `MPPImage` on which pose landmarks detection is to be performed.
 * @param timestampInMilliseconds The video frame's timestamp (in milliseconds). The input
 * timestamps must be monotonically increasing.
 * @param error An optional error parameter populated when there is an error in performing pose
 * landmarks detection on the input video frame.
 *
 * @return  An `MPPPoseLandmarkerResult` object that contains the pose pose landmarks detection
 * results.
 */
- (nullable MPPPoseLandmarkerResult *)detectInVideoFrame:(MPPImage *)image
                                 timestampInMilliseconds:(NSInteger)timestampInMilliseconds
                                                   error:(NSError **)error NS_SWIFT_NAME(detect(videoFrame:timestampInMilliseconds:));

/**
 * Sends live stream image data of type `MPPImage` to perform pose landmarks detection using the
 * whole image as region of interest. Rotation will be applied according to the `orientation`
 * property of the provided `MPPImage`. Only use this method when the `MPPPoseLandmarker` is created
 * with `MPPRunningModeLiveStream`.
 *
 * The object which needs to be continuously notified of the available results of pose landmarks
 * detection must confirm to `MPPPoseLandmarkerLiveStreamDelegate` protocol and implement the
 * `poseLandmarker:didFinishDetectionWithResult:timestampInMilliseconds:error:`
 * delegate method.
 *
 * It's required to provide a timestamp (in milliseconds) to indicate when the input image is sent
 * to the pose landmarker. The input timestamps must be monotonically increasing.
 *
 * This method supports performing pose landmarks detection on RGBA images. If your `MPPImage` has a
 * source type of `MPPImageSourceTypePixelBuffer` or `MPPImageSourceTypeSampleBuffer`, the
 * underlying pixel buffer must have one of the following pixel format types:
 * 1. kCVPixelFormatType_32BGRA
 * 2. kCVPixelFormatType_32RGBA
 *
 * If the input `MPPImage` has a source type of `MPPImageSourceTypeImage` ensure that the color
 * space is RGB with an Alpha channel.
 *
 * If this method is used for performing pose landmarks detection on live camera frames using
 * `AVFoundation`, ensure that you request `AVCaptureVideoDataOutput` to output frames in
 * `kCMPixelFormat_32RGBA` using its `videoSettings` property.
 *
 * @param image A live stream image data of type `MPPImage` on which pose landmarks detection is to
 * be performed.
 * @param timestampInMilliseconds The timestamp (in milliseconds) which indicates when the input
 * image is sent to the pose landmarker. The input timestamps must be monotonically increasing.
 * @param error An optional error parameter populated when there is an error in performing pose
 * landmarks detection on the input live stream image data.
 *
 * @return `YES` if the image was sent to the task successfully, otherwise `NO`.
 */
- (BOOL)detectAsyncInImage: (MPPImage *)image
   timestampInMilliseconds: (NSInteger)timestampInMilliseconds
                     error: (NSError **)error NS_SWIFT_NAME(detectAsync(image:timestampInMilliseconds:));

- (instancetype)init NS_UNAVAILABLE;

+ (instancetype)new NS_UNAVAILABLE;

@end

NS_ASSUME_NONNULL_END
