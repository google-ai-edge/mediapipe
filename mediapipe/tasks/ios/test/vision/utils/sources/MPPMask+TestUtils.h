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
#import <UIKit/UIKit.h>

#import "mediapipe/tasks/ios/test/utils/sources/MPPFileInfo.h"
#import "mediapipe/tasks/ios/vision/core/sources/MPPMask.h"

NS_ASSUME_NONNULL_BEGIN

/**
 * Helper utility for initializing `MPPMask` for MediaPipe iOS vision library tests.
 */
@interface MPPMask (TestUtils)

/**
 * Loads an image from a file in an app bundle and Creates an `MPPMask` of type
 * `MPPMaskDataTypeUInt8` using the gray scale pixel data of a `UIImage` loaded from a file with the
 * given `MPPFileInfo`.
 *
 * @param fileInfo The file info specifying the name and type of the image file in the app bundle.
 *
 * @return The `MPPMask` with the pixel data of the loaded image. This method returns `nil` if there
 * is an error in loading the image correctly.
 */
- (nullable instancetype)initWithImageFileInfo:(MPPFileInfo *)fileInfo;

@end

NS_ASSUME_NONNULL_END
