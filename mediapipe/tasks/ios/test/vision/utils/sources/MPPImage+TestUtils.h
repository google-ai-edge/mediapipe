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

#import "mediapipe/tasks/ios/test/utils/sources/MPPFileInfo.h"
#import "mediapipe/tasks/ios/vision/core/sources/MPPImage.h"

NS_ASSUME_NONNULL_BEGIN

/**
 * Helper utility for initializing `MPPImage` for MediaPipe iOS vision library tests.
 */
@interface MPPImage (TestUtils)

/**
 * Loads an image from a file in an app bundle into a `MPPImage` object of source type
 * `MPPImageSourceTypeImage`.
 *
 * @param fileInfo The file info specifying the name and extension of the image
 * file in the bundle.
 *
 * @return The `MPPImage` object contains the loaded image. This method returns
 * nil if it cannot load the image.
 */
+ (MPPImage *)imageWithFileInfo:(MPPFileInfo *)fileInfo NS_SWIFT_NAME(image(withFileInfo:));

/**
 * Loads an image from a file in an app bundle into a `MPPImage` object with the specified
 * orientation and source type `MPPImageSourceTypeImage`.
 *
 * @param fileInfo The file info specifying the name and extension of the image file in the bundle.
 *
 * @return The `MPPImage` object contains the loaded image. This method returns nil if it cannot
 * load the image.
 */
+ (MPPImage *)imageWithFileInfo:(MPPFileInfo *)fileInfo
                    orientation:(UIImageOrientation)orientation
    NS_SWIFT_NAME(image(withFileInfo:orientation:));

/**
 * Loads an image from a file in an app bundle into a `MPPImage` object with the specified
 * source type.
 *
 * For source type `MPPImageSourceTypeSampleBuffer`, the method returns an `MPImage` whose sample
 * buffer has timing info, `kCMTimingInfoInvalid`. The underlying pixel buffer of the returned image
 * will be of type `kCVPixelFormatType32BGRA`.
 *
 * @param fileInfo The file info specifying the name and extension of the image file in the bundle.
 * @param sourceType The expected `MPPImageSourceType` of the `MPPImage` created by this method.
 *
 * @return The `MPPImage` object contains the loaded image. This method returns nil if it cannot
 * load the image.
 */
+ (MPPImage *)imageWithFileInfo:(MPPFileInfo *)fileInfo
                     sourceType:(MPPImageSourceType)sourceType
    NS_SWIFT_NAME(image(withFileInfo:sourceType:));

// TODO: Remove after all tests are migrated
/**
 * Loads an image from a file in an app bundle into a `MPPImage` object.
 *
 * @param classObject The specified class associated with the bundle containing the file to be
 * loaded.
 * @param name Name of the image file.
 * @param type Extension of the image file.
 *
 * @return The `MPPImage` object contains the loaded image. This method returns nil if it cannot
 * load the image.
 */
+ (nullable MPPImage *)imageFromBundleWithClass:(Class)classObject
                                       fileName:(NSString *)name
                                         ofType:(NSString *)type
    NS_SWIFT_NAME(imageFromBundle(class:filename:type:));

// TODO: Remove after all tests are migrated
/**
 * Loads an image from a file in an app bundle into a `MPPImage` object with the specified
 * orientation.
 *
 * @param classObject The specified class associated with the bundle containing the file to be
 * loaded.
 * @param name Name of the image file.
 * @param type Extension of the image file.
 * @param orientation Orientation of the image.
 *
 * @return The `MPPImage` object contains the loaded image. This method returns nil if it cannot
 * load the image.
 */
+ (nullable MPPImage *)imageFromBundleWithClass:(Class)classObject
                                       fileName:(NSString *)name
                                         ofType:(NSString *)type
                                    orientation:(UIImageOrientation)imageOrientation
    NS_SWIFT_NAME(imageFromBundle(class:filename:type:orientation:));

@end

NS_ASSUME_NONNULL_END
