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

#import "mediapipe/tasks/ios/test/vision/utils/sources/MPPMask+TestUtils.h"

@implementation MPPMask (TestUtils)

- (instancetype)initWithImageFileInfo:(MPPFileInfo *)fileInfo {
  UIImage *image = [[UIImage alloc] initWithContentsOfFile:fileInfo.path];

  if (!image.CGImage) {
    return nil;
  }

  size_t width = CGImageGetWidth(image.CGImage);
  size_t height = CGImageGetHeight(image.CGImage);

  NSInteger bitsPerComponent = 8;

  UInt8 *pixelData = NULL;

  CGColorSpaceRef colorSpace = CGColorSpaceCreateDeviceGray();

  // For a gray scale image (single component) with no alpha, the bitmap info is
  // `kCGImageAlphaNone` in combination with bytesPerRow being equal to width.
  CGContextRef context = CGBitmapContextCreate(nil, width, height, bitsPerComponent, width,
                                               colorSpace, kCGImageAlphaNone);

  if (!context) {
    CGColorSpaceRelease(colorSpace);
    return nil;
  }

  CGContextDrawImage(context, CGRectMake(0, 0, width, height), image.CGImage);
  pixelData = (UInt8 *)CGBitmapContextGetData(context);

  // A copy is needed to ensure that the pixel data outlives the `CGContextRelease` call.
  // Alternative is to make the context, color space instance variables and release them in
  // `dealloc()`. Since Categories don't allow adding instance variables, choosing to copy rather
  // than creating a new custom class similar to `MPPMask` only for the tests.
  MPPMask *mask = [self initWithUInt8Data:pixelData width:width height:height shouldCopy:YES];

  CGColorSpaceRelease(colorSpace);
  CGContextRelease(context);

  return mask;
}

@end
