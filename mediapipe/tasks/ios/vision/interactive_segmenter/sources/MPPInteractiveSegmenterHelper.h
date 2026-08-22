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

#include <memory>
#include "mediapipe/framework/formats/image.h"
#include "mediapipe/framework/formats/image_frame.h"

NS_ASSUME_NONNULL_BEGIN

/**
 * @brief Extracts a C++ `mediapipe::ImageFrame` from an `MPPImage`.
 *
 * @param image The source `MPPImage` to extract the frame from.
 * @param error Pointer to an `NSError` to be populated on failure.
 *
 * @return A unique pointer to the extracted `mediapipe::ImageFrame`, or nullptr if extraction
 * fails.
 */
std::unique_ptr<mediapipe::ImageFrame> MPPGetImageFrameFromImage(MPPImage *image, NSError **error);

/**
 * @brief Creates an `MPPImage` from a C++ `mediapipe::Image`.
 *
 * @param image The C++ `mediapipe::Image` to convert.
 * @param sourceImage The source `MPPImage` whose properties (like orientation) are cloned.
 * @param error Pointer to an `NSError` to be populated on failure.
 *
 * @return The created `MPPImage`, or nil if creation fails.
 */
MPPImage *_Nullable MPPCreateImageFromCppImage(const mediapipe::Image &image, MPPImage *sourceImage,
                                               NSError **error);

NS_ASSUME_NONNULL_END
