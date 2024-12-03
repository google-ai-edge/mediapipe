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

#include "mediapipe/framework/formats/image.h"
#include "mediapipe/framework/formats/image_frame.h"

#import "mediapipe/tasks/ios/vision/core/sources/MPPImage.h"

NS_ASSUME_NONNULL_BEGIN

/**
 * Helper utility for converting `MPPImage` into a `mediapipe::ImageFrame`.
 */
@interface MPPImage (Utils)
/**
 * Converts the `MPPImage` into a `mediapipe::ImageFrame`.
 * Irrespective of whether the underlying buffer is grayscale, RGBA, BGRA etc., the `MPPImage` is
 * converted to an RGBA format. In case of grayscale images, the mono channel is duplicated in the
 * R, G, B channels.
 *
 * @param error Pointer to the memory location where errors if any should be saved. If @c NULL, no
 * error will be saved.
 *
 * @return An std::unique_ptr<mediapipe::ImageFrame> or `nullptr` in case of errors.
 */
- (std::unique_ptr<mediapipe::ImageFrame>)imageFrameWithError:(NSError **)error;

/**
 * Initializes an `MPPImage` object with the pixels from the given `mediapipe::Image` and source
 * type and orientation from the given source image.
 *
 * Only supports initialization from `mediapipe::Image` of format RGBA.
 * If `shouldCopyPixelData` is set to `YES`, the newly created `MPPImage` stores a reference to a
 * deep copied pixel data of the given `image`. Since deep copies are expensive, it is recommended
 * to not set `shouldCopyPixelData` unless the `MPPImage` must outlive the passed in `image`.
 *
 * @param image The `mediapipe::Image` whose pixels are used for creating the `MPPImage`.
 * @param sourceImage The `MPPImage` whose `orientation` and `imageSourceType` are used for creating
 * the new `MPPImage`.
 * @param shouldCopyPixelData `BOOL` that determines if the newly created `MPPImage` stores a
 * reference to a deep copied pixel data of the given `image` or not.
 *
 * @param error Pointer to the memory location where errors if any should be saved. If @c NULL, no
 * error will be saved.
 *
 * @return A new `MPImage` instance with the pixels from the given `mediapipe::Image` and meta data
 * equal to the `sourceImage`. `nil` if there is an error in initializing the `MPPImage`.
 */
- (nullable instancetype)initWithCppImage:(const mediapipe::Image &)image
           cloningPropertiesOfSourceImage:(MPPImage *)sourceImage
                      shouldCopyPixelData:(BOOL)shouldCopyPixelData
                                    error:(NSError **)error;

@end

NS_ASSUME_NONNULL_END
