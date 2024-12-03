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

#import "mediapipe/tasks/ios/components/containers/sources/MPPRegionOfInterest.h"
#import "mediapipe/tasks/ios/vision/core/sources/MPPImage.h"
#import "mediapipe/tasks/ios/vision/interactive_segmenter/sources/MPPInteractiveSegmenterOptions.h"
#import "mediapipe/tasks/ios/vision/interactive_segmenter/sources/MPPInteractiveSegmenterResult.h"

NS_ASSUME_NONNULL_BEGIN

/**
 * @brief Class that performs interactive segmentation on images.
 *
 * Users can represent user interaction through `RegionOfInterest`, which gives a hint to
 * `InteractiveSegmenter` to perform segmentation focusing on the given region of interest.
 *
 * The API expects a TFLite model with mandatory TFLite Model Metadata.
 *
 * Input tensor:
 *  (kTfLiteUInt8/kTfLiteFloat32)
 *  - image input of size `[batch x height x width x channels]`.
 *  - batch inference is not supported (`batch` is required to be 1).
 *  - RGB and greyscale inputs are supported (`channels` is required to be
 *  1 or 3).
 *  - if type is kTfLiteFloat32, NormalizationOptions are required to be attached to the metadata
 * for input normalization. Output tensors: (kTfLiteUInt8/kTfLiteFloat32)
 *  - list of segmented masks.
 *  - if `output_type` is CATEGORY_MASK, uint8 Image, Image vector of size 1.
 *  - if `output_type` is CONFIDENCE_MASK, float32 Image list of size `channels`.
 *  - batch is always 1.
 *
 * An example of such model can be found at:
 * https://tfhub.dev/tensorflow/lite-model/deeplabv3/1/metadata/2
 */
NS_SWIFT_NAME(InteractiveSegmenter)
@interface MPPInteractiveSegmenter : NSObject

/**
 * Get the category label list of the `InteractiveSegmenter` can recognize. For CATEGORY_MASK type,
 * the index in the category mask corresponds to the category in the label list. For CONFIDENCE_MASK
 * type, the output mask list at index corresponds to the category in the label list. If there is no
 * labelmap provided in the model file, empty array is returned.
 */
@property(nonatomic, readonly) NSArray<NSString *> *labels;

/**
 * Creates a new instance of `InteractiveSegmenter` from an absolute path to a TensorFlow Lite model
 * file stored locally on the device and the default `InteractiveSegmenterOptions`.
 *
 * @param modelPath An absolute path to a TensorFlow Lite model file stored locally on the device.
 *
 * @return A new instance of `InteractiveSegmenter` with the given model path. `nil` if there is an
 * error in initializing the interactive segmenter.
 */
- (nullable instancetype)initWithModelPath:(NSString *)modelPath error:(NSError **)error;

/**
 * Creates a new instance of `InteractiveSegmenter` from the given `InteractiveSegmenterOptions`.
 *
 * @param options The options of type `InteractiveSegmenterOptions` to use for configuring the
 * `InteractiveSegmenter`.
 *
 * @return A new instance of `InteractiveSegmenter` with the given options. `nil` if there is an
 * error in initializing the interactive segmenter.
 */
- (nullable instancetype)initWithOptions:(MPPInteractiveSegmenterOptions *)options
                                   error:(NSError **)error NS_DESIGNATED_INITIALIZER;

/**
 * Performs segmentation on the provided MPPImage using the specified user's region of interest.
 * Rotation will be applied according to the `orientation` property of the provided `MPImage`.
 *
 * This method supports interactive segmentation of RGBA images. If your `MPImage` has a source type
 * of `.pixelBuffer` or `.sampleBuffer`, the underlying pixel buffer must use
 * `kCVPixelFormatType_32BGRA` as its pixel format.
 *
 * If your `MPImage` has a source type of `.image` ensure that the color space is RGB with an Alpha
 * channel.
 *
 * @param image The `MPImage` on which segmentation is to be performed.
 *
 * @return An `InteractiveSegmenterResult` that contains the segmented masks.
 */
- (nullable MPPInteractiveSegmenterResult *)segmentImage:(MPPImage *)image
                                        regionOfInterest:(MPPRegionOfInterest *)regionOfInterest
                                                   error:(NSError **)error
    NS_SWIFT_NAME(segment(image:regionOfInterest:));

/**
 * Performs segmentation on the provided MPPImage using the specified user's region of interest and
 * invokes the given completion handler block with the response. The method returns synchronously
 * once the completion handler returns.
 *
 * Rotation will be applied according to the `orientation` property of the provided `MPImage`.
 *
 * This method supports interactive segmentation of RGBA images. If your `MPImage` has a source type
 * of `.pixelBuffer` or `.sampleBuffer`, the underlying pixel buffer must use
 * `kCVPixelFormatType_32BGRA` as its pixel format.
 *
 * If your `MPImage` has a source type of `image` ensure that the color space is RGB with an Alpha
 * channel.
 *
 * @param image The `MPImage` on which segmentation is to be performed.
 * @param completionHandler A block to be invoked with the results of performing segmentation on the
 * image. The block takes two arguments, the optional `InteractiveSegmenterResult` that contains the
 * segmented masks if the segmentation was successful and an optional error populated upon failure.
 * The lifetime of the returned masks is only guaranteed for the duration of the block.
 */
- (void)segmentImage:(MPPImage *)image
         regionOfInterest:(MPPRegionOfInterest *)regionOfInterest
    withCompletionHandler:(void (^)(MPPInteractiveSegmenterResult *_Nullable result,
                                    NSError *_Nullable error))completionHandler
    NS_SWIFT_NAME(segment(image:regionOfInterest:completion:));

- (instancetype)init NS_UNAVAILABLE;

+ (instancetype)new NS_UNAVAILABLE;

@end

NS_ASSUME_NONNULL_END
