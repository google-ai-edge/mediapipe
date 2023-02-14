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

NS_ASSUME_NONNULL_BEGIN

/** 
 * Helper utility for performing operations on MPPImage specific to the MediaPipe Vision library.
 */
@interface MPPImage (Utils)

/** Bitmap size of the image. */
@property(nonatomic, readonly) CGSize bitmapSize;

/**
 * Returns the underlying uint8 pixel buffer of an `MPPImage`.
 * Irrespective of whether the underlying buffer is grayscale, RGB, RGBA, BGRA etc., the pixel
 * data is converted to an RGB format. In case of grayscale images, the mono channel is duplicated
 * in the R, G, B channels.
 *
 * @param error Pointer to the memory location where errors if any should be
 * saved. If @c NULL, no error will be saved.
 *
 * @return The underlying pixel buffer of the `MPPImage` or nil in case of errors.
 */
- (nullable uint8_t *)rgbPixelDataWithError:(NSError **)error;

@end

NS_ASSUME_NONNULL_END
