// Copyright 2024 The MediaPipe Authors.
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
#import "mediapipe/tasks/ios/vision/holistic_landmarker/sources/MPPHolisticLandmarkerOptions.h"
#import "mediapipe/tasks/ios/vision/holistic_landmarker/sources/MPPHolisticLandmarkerResult.h"

NS_ASSUME_NONNULL_BEGIN

/**
 * @brief Performs holistic landmarks detection on images.
 *
 * This API expects a pre-trained holistic landmarks model asset bundle.
 */
NS_SWIFT_NAME(HolisticLandmarker)
@interface MPPHolisticLandmarker : NSObject

/**
 * Creates a new instance of `HolisticLandmarker` from an absolute path to a model asset bundle
 * stored locally on the device and the default `HolisticLandmarkerOptions`.
 *
 * @param modelPath An absolute path to a model asset bundle stored locally on the device.
 * @param error An optional error parameter populated when there is an error in initializing the
 * holistic landmarker.
 *
 * @return A new instance of `HolisticLandmarker` with the given model path. `nil` if there is an
 * error in initializing the holistic landmarker.
 */
- (nullable instancetype)initWithModelPath:(NSString *)modelPath error:(NSError **)error;

/**
 * Creates a new instance of `HolisticLandmarker` from the given `HolisticLandmarkerOptions`.
 *
 * @param options The options of type `HolisticLandmarkerOptions` to use for configuring the
 * `HolisticLandmarker`.
 * @param error An optional error parameter populated when there is an error in initializing the
 * holistic landmarker.
 *
 * @return A new instance of `HolisticLandmarker` with the given options. `nil` if there is an error
 * in initializing the holistic landmarker.
 */
- (nullable instancetype)initWithOptions:(MPPHolisticLandmarkerOptions *)options
                                   error:(NSError **)error NS_DESIGNATED_INITIALIZER;

/**
 * Performs holistic landmarks detection on the provided `MPImage` using the whole image as region
 * ofinterest. Rotation will be applied according to the `orientation` property of the
 * providedm`MPImage`. Only use this method when the `HolisticLandmarker` is created with running
 * mode, `.image`.
 *
 * This method supports performing holistic landmarks detection on RGBA images. If your `MPImage`
 * has a source type of `.pixelBuffer` or `.sampleBuffer`, the underlying pixel buffer must use
 * `kCVPixelFormatType_32BGRA` as its pixel format.
 *
 * If your `MPImage` has a source type of `.image` ensure that the color space is RGB with an Alpha
 * channel.
 *
 * @param image The `MPImage` on which holistic landmarks detection is to be performed.
 * @param error An optional error parameter populated when there is an error in performing holistic
 * landmarks detection on the input image.
 *
 * @return  An `HolisticLandmarkerResult` object that contains the holistic landmarks detection
 * results.
 */
- (nullable MPPHolisticLandmarkerResult *)detectImage:(MPPImage *)image
                                                error:(NSError **)error
    NS_SWIFT_NAME(detect(image:));

/**
 * Performs holistic landmarks detection on the provided video frame of type `MPImage` using the
 * whole image as region of interest. Rotation will be applied according to the `orientation`
 * property of the provided `MPImage`. Only use this method when the `HolisticLandmarker` is created
 * with running mode, `.video`.
 *
 * It's required to provide the video frame's timestamp (in milliseconds). The input timestamps must
 * be monotonically increasing.
 *
 * This method supports performing holistic landmarks detection on RGBA images. If your `MPImage`
 * has a source type of `.pixelBuffer` or `.sampleBuffer`, the underlying pixel buffer must use
 * `kCVPixelFormatType_32BGRA` as its pixel format.
 *
 * If your `MPImage` has a source type of `.image` ensure that the color space is RGB with an Alpha
 * channel.
 *
 * @param image The `MPImage` on which holistic landmarks detection is to be performed.
 * @param timestampInMilliseconds The video frame's timestamp (in milliseconds). The input
 * timestamps must be monotonically increasing.
 * @param error An optional error parameter populated when there is an error in performing holistic
 * landmarks detection on the input video frame.
 *
 * @return  An `HolisticLandmarkerResult` object that contains the holistic landmarks detection
 * results.
 */
- (nullable MPPHolisticLandmarkerResult *)detectVideoFrame:(MPPImage *)image
                                   timestampInMilliseconds:(NSInteger)timestampInMilliseconds
                                                     error:(NSError **)error
    NS_SWIFT_NAME(detect(videoFrame:timestampInMilliseconds:));

/**
 * Sends live stream image data of type `MPImage` to perform holistic landmarks detection using the
 * whole image as region of interest. Rotation will be applied according to the `orientation`
 * property of the provided `MPImage`. Only use this method when the `HolisticLandmarker` is created
 * with running mode, `.liveStream`.
 *
 * The object which needs to be continuously notified of the available results of holistic landmarks
 * detection must confirm to `HolisticLandmarkerLiveStreamDelegate` protocol and implement the
 * `holisticLandmarker(_:didFinishDetectionWithResult:timestampInMilliseconds:error:)` delegate
 * method.
 *
 * It's required to provide a timestamp (in milliseconds) to indicate when the input image is sent
 * to the holistic landmarker. The input timestamps must be monotonically increasing.
 *
 * This method supports performing holistic landmarks detection on RGBA images. If your `MPImage`
 * has a source type of `.pixelBuffer` or `.sampleBuffer`, the underlying pixel buffer must use
 * `kCVPixelFormatType_32BGRA` as its pixel format.
 *
 * If the input `MPImage` has a source type of `.image` ensure that the color space is RGB with an
 * Alpha channel.
 *
 * If this method is used for performing holistic landmarks detection on live camera frames using
 * `AVFoundation`, ensure that you request `AVCaptureVideoDataOutput` to output frames in
 * `kCMPixelFormat_32BGRA` using its `videoSettings` property.
 *
 * @param image A live stream image data of type `MPImage` on which holistic landmarks detection is
 * to be performed.
 * @param timestampInMilliseconds The timestamp (in milliseconds) which indicates when the input
 * image is sent to the holistic landmarker. The input timestamps must be monotonically increasing.
 * @param error An optional error parameter populated when there is an error in performing holistic
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
