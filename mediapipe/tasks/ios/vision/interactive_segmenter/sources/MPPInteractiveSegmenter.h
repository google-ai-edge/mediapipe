// Copyright 2026 The MediaPipe Authors.
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
#import "mediapipe/tasks/ios/vision/interactive_segmenter/sources/MPPInteractiveSegmenterOptions.h"
#import "mediapipe/tasks/ios/vision/interactive_segmenter/sources/MPPStroke.h"

NS_ASSUME_NONNULL_BEGIN

/**
 * @brief Class that performs interactive segmentation on images using split mode.
 *
 * This API optimizes performance by splitting the model into an encoder (feature extraction
 * that runs once per image) and a decoder (lightweight heatmap generation that runs per user
 * stroke).
 */
NS_SWIFT_NAME(InteractiveSegmenter)
@interface MPPInteractiveSegmenter : NSObject

/**
 * Creates a new instance of `InteractiveSegmenter` from the given
 * `InteractiveSegmenterOptions`.
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
 * Sets the image to be segmented. This runs the encoder part of the split model.
 * This method must be called before calling `segmentWithStrokes:error:`.
 *
 * @param image The `MPImage` on which segmentation is to be performed.
 *
 * @return `YES` if the image was successfully set and encoded, `NO` otherwise.
 */
- (BOOL)setImage:(MPPImage *)image error:(NSError **)error NS_SWIFT_NAME(set(image:));

/**
 * Performs segmentation on the previously set image using the provided user's strokes.
 * This runs the decoder part of the split model.
 *
 * @param strokes An array of `MPPStroke` objects representing the user interaction.
 *
 * @return An `MPImage` that contains the segmented mask.
 */
- (nullable MPPImage *)segment:(NSArray<MPPStroke *> *)strokes
                         error:(NSError **)error NS_SWIFT_NAME(segment(strokes:));

- (instancetype)init NS_UNAVAILABLE;

+ (instancetype)new NS_UNAVAILABLE;

@end

NS_ASSUME_NONNULL_END
