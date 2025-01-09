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
#import "mediapipe/tasks/ios/vision/gesture_recognizer/sources/MPPGestureRecognizerOptions.h"
#import "mediapipe/tasks/ios/vision/gesture_recognizer/sources/MPPGestureRecognizerResult.h"

NS_ASSUME_NONNULL_BEGIN

/**
 * @brief Performs gesture recognition on images.
 *
 * This API expects a pre-trained TFLite hand gesture recognizer model or a custom one created using
 * MediaPipe Solutions Model Maker. See
 * https://developers.google.com/mediapipe/solutions/model_maker.
 */
NS_SWIFT_NAME(GestureRecognizer)
@interface MPPGestureRecognizer : NSObject

/**
 * Creates a new instance of `GestureRecognizer` from an absolute path to a TensorFlow Lite model
 * file stored locally on the device and the default `GestureRecognizerOptions`.
 *
 * @param modelPath An absolute path to a TensorFlow Lite model file stored locally on the device.
 * @param error An optional error parameter populated when there is an error in initializing the
 * gesture recognizer.
 *
 * @return A new instance of `GestureRecognizer` with the given model path. `nil` if there is an
 * error in initializing the gesture recognizer.
 */
- (nullable instancetype)initWithModelPath:(NSString *)modelPath error:(NSError **)error;

/**
 * Creates a new instance of `GestureRecognizer` from the given `GestureRecognizerOptions`.
 *
 * @param options The options of type `GestureRecognizerOptions` to use for configuring the
 * `GestureRecognizer`.
 * @param error An optional error parameter populated when there is an error in initializing the
 * gesture recognizer.
 *
 * @return A new instance of `GestureRecognizer` with the given options. `nil` if there is an error
 * in initializing the gesture recognizer.
 */
- (nullable instancetype)initWithOptions:(MPPGestureRecognizerOptions *)options
                                   error:(NSError **)error NS_DESIGNATED_INITIALIZER;

/**
 * Performs gesture recognition on the provided `MPImage` using the whole image as region of
 * interest. Rotation will be applied according to the `orientation` property of the provided
 * `MPImage`. Only use this method when the `GestureRecognizer` is created with running mode,
 * `.image`.
 *
 * This method supports performing gesture recognition on RGBA images. If your `MPImage` has a
 * source type of `.pixelBuffer` or `.sampleBuffer`, the underlying pixel buffer must use
 * `kCVPixelFormatType_32BGRA` as its pixel format.
 *
 * If your `MPImage` has a source type of `.image` ensure that the color space is RGB with an Alpha
 * channel.
 *
 * @param image The `MPImage` on which gesture recognition is to be performed.
 * @param error An optional error parameter populated when there is an error in performing gesture
 * recognition on the input image.
 *
 * @return  An `GestureRecognizerResult` object that contains the hand gesture recognition
 * results.
 */
- (nullable MPPGestureRecognizerResult *)recognizeImage:(MPPImage *)image
                                                  error:(NSError **)error
    NS_SWIFT_NAME(recognize(image:));

/**
 * Performs gesture recognition on the provided video frame of type `MPImage` using the whole image
 * as region of interest. Rotation will be applied according to the `orientation` property of the
 * provided `MPImage`. Only use this method when the `GestureRecognizer` is created with running
 * mode, `.video`.
 *
 * It's required to provide the video frame's timestamp (in milliseconds). The input timestamps must
 * be monotonically increasing.
 *
 * This method supports performing gesture recognition on RGBA images. If your `MPImage` has a
 * source type of `.pixelBuffer` or `.sampleBuffer`, the underlying pixel buffer must use
 * `kCVPixelFormatType_32BGRA` as its pixel format.
 *
 * If your `MPImage` has a source type of `.image` ensure that the color space is RGB with an Alpha
 * channel.
 *
 * @param image The `MPImage` on which gesture recognition is to be performed.
 * @param timestampInMilliseconds The video frame's timestamp (in milliseconds). The input
 * timestamps must be monotonically increasing.
 * @param error An optional error parameter populated when there is an error in performing gesture
 * recognition on the input video frame.
 *
 * @return  An `GestureRecognizerResult` object that contains the hand gesture recognition
 * results.
 */
- (nullable MPPGestureRecognizerResult *)recognizeVideoFrame:(MPPImage *)image
                                     timestampInMilliseconds:(NSInteger)timestampInMilliseconds
                                                       error:(NSError **)error
    NS_SWIFT_NAME(recognize(videoFrame:timestampInMilliseconds:));

/**
 * Sends live stream image data of type `MPImage` to perform gesture recognition using the whole
 * image as region of interest. Rotation will be applied according to the `orientation` property of
 * the provided `MPImage`. Only use this method when the `GestureRecognizer` is created with running
 * mode, `.liveStream`.
 *
 * The object which needs to be continuously notified of the available results of gesture
 * recognition must confirm to `GestureRecognizerLiveStreamDelegate` protocol and implement the
 * `gestureRecognizer(_:didFinishRecognitionWithResult:timestampInMilliseconds:error:)`
 * delegate method.
 *
 * It's required to provide a timestamp (in milliseconds) to indicate when the input image is sent
 * to the gesture recognizer. The input timestamps must be monotonically increasing.
 *
 * This method supports performing gesture recognition on RGBA images. If your `MPImage` has a
 * source type of `.pixelBuffer` or `.sampleBuffer`, the underlying pixel buffer must use
 * `kCVPixelFormatType_32BGRA` as its pixel format.
 *
 * If the input `MPImage` has a source type of `.image` ensure that the color space is RGB with an
 * Alpha channel.
 *
 * If this method is used for performing gesture recognition on live camera frames using
 * `AVFoundation`, ensure that you request `AVCaptureVideoDataOutput` to output frames in
 * `kCMPixelFormat_32BGRA` using its `videoSettings` property.
 *
 * @param image A live stream image data of type `MPImage` on which gesture recognition is to be
 * performed.
 * @param timestampInMilliseconds The timestamp (in milliseconds) which indicates when the input
 * image is sent to the gesture recognizer. The input timestamps must be monotonically increasing.
 * @param error An optional error parameter populated when there is an error in performing gesture
 * recognition on the input live stream image data.
 *
 * @return `YES` if the image was sent to the task successfully, otherwise `NO`.
 */
- (BOOL)recognizeAsyncImage:(MPPImage *)image
    timestampInMilliseconds:(NSInteger)timestampInMilliseconds
                      error:(NSError **)error
    NS_SWIFT_NAME(recognizeAsync(image:timestampInMilliseconds:));

- (instancetype)init NS_UNAVAILABLE;

+ (instancetype)new NS_UNAVAILABLE;

@end

NS_ASSUME_NONNULL_END
