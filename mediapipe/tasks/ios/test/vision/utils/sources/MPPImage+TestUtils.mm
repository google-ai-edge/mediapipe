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

#include <iostream>
#import "mediapipe/tasks/ios/test/vision/utils/sources/MPPImage+TestUtils.h"

#import <CoreMedia/CoreMedia.h>

namespace {
static void FreeRefConReleaseCallback(void *refCon, const void *baseAddress) { free(refCon); }
}  // namespace

@interface UIImage (FileUtils)

@property(readonly, nonatomic) CVPixelBufferRef pixelBuffer;

// TODO: Remove this method after all tests are migrated to the new methods.
+ (nullable UIImage *)imageFromBundleWithClass:(Class)classObject
                                      fileName:(NSString *)name
                                        ofType:(NSString *)type;

@end

@implementation UIImage (FileUtils)

+ (nullable UIImage *)imageFromBundleWithClass:(Class)classObject
                                      fileName:(NSString *)name
                                        ofType:(NSString *)type {
  NSString *imagePath = [[NSBundle bundleForClass:classObject] pathForResource:name ofType:type];
  if (!imagePath) return nil;

  return [[UIImage alloc] initWithContentsOfFile:imagePath];
}

- (CVPixelBufferRef)pixelBuffer {
  if (!self.CGImage) {
    return nullptr;
  }

  size_t width = CGImageGetWidth(self.CGImage);
  size_t height = CGImageGetHeight(self.CGImage);

  NSInteger bitsPerComponent = 8;
  NSInteger channelCount = 4;
  size_t bytesPerRow = channelCount * width;

  CGColorSpaceRef colorSpace = CGColorSpaceCreateDeviceRGB();

  if (!colorSpace) {
    return nullptr;
  }

  // To create a `CVPixelBuffer` from `CGImage`, the underlying buffer of the `CGImage` is extracted
  // in the format `kCVPixelFormatType32BGRA`. since `CVPixelBuffer`s only support the pixel format
  // `kCVPixelFormatType32BGRA` for 32 bit RGBA images (All RGB images are stored with an alpha
  // value of 255.0 by iOS).
  //
  // `kCGImageAlphaPremultipliedFirst | kCGBitmapByteOrder32Little` specifies a pixel format of
  // `kCVPixelFormatType32BGRA`. `kCGImageAlphaPremultipliedFirst` specifies that Alpha will be next
  // to R and the R, G, B values will be pre multiplied with alpha. Images with alpha != 255 are
  // stored with the R, G, B values premultiplied with alpha by iOS. `kCGBitmapByteOrder32Little`
  // specifies that B will be stored before R.

  CGBitmapInfo bitMapinfo = kCGImageAlphaPremultipliedFirst | kCGBitmapByteOrder32Little;

  CGContextRef context = CGBitmapContextCreate(nil, width, height, bitsPerComponent, bytesPerRow,
                                               colorSpace, bitMapinfo);

  void *copiedData = nullptr;

  if (context) {
    CGContextDrawImage(context, CGRectMake(0, 0, width, height), self.CGImage);
    void *srcData = CGBitmapContextGetData(context);
    if (srcData) {
      // The pixel data of the `CGImage` extracted using the `context` is only retained in memory
      // until the context is released. Hence the data is copied to a new buffer and this buffer is
      // used to create the `CVPixelBuffer` to ensure it outlives the created `context`.
      copiedData = malloc(height * bytesPerRow * sizeof(UInt8));
      memcpy(copiedData, srcData, sizeof(UInt8) * height * bytesPerRow);
    }
    CGContextRelease(context);
  }

  CGColorSpaceRelease(colorSpace);

  CVPixelBufferRef outputBuffer = nullptr;

  if (copiedData) {
    // A callback frunction to cleanup the memory of the copied buffer is provided when creating the
    // `CVPixelBuffer`.
    (void)CVPixelBufferCreateWithBytes(nullptr, width, height, kCVPixelFormatType_32BGRA,
                                       copiedData, bytesPerRow, FreeRefConReleaseCallback,
                                       copiedData, nullptr, &outputBuffer);
  }

  return outputBuffer;
}

@end

@implementation MPPImage (TestUtils)

+ (MPPImage *)imageWithFileInfo:(MPPFileInfo *)fileInfo {
  if (!fileInfo.path) return nil;

  UIImage *image = [[UIImage alloc] initWithContentsOfFile:fileInfo.path];

  if (!image) return nil;

  return [[MPPImage alloc] initWithUIImage:image error:nil];
}

+ (MPPImage *)imageWithFileInfo:(MPPFileInfo *)fileInfo
                    orientation:(UIImageOrientation)orientation {
  if (!fileInfo.path) return nil;

  UIImage *image = [[UIImage alloc] initWithContentsOfFile:fileInfo.path];

  if (!image) return nil;

  return [[MPPImage alloc] initWithUIImage:image orientation:orientation error:nil];
}

+ (MPPImage *)imageWithFileInfo:(MPPFileInfo *)fileInfo sourceType:(MPPImageSourceType)sourceType {
  switch (sourceType) {
    case MPPImageSourceTypeImage:
      return [MPPImage imageWithFileInfo:fileInfo];
    case MPPImageSourceTypePixelBuffer:
      return [MPPImage imageOfPixelBufferSourceTypeWithFileInfo:fileInfo];
    case MPPImageSourceTypeSampleBuffer:
      return [MPPImage imageOfSampleBufferSourceTypeWithFileInfo:fileInfo
                                                      timingInfo:&kCMTimingInfoInvalid];
  }
  return nil;
}

+ (MPPImage *)imageOfPixelBufferSourceTypeWithFileInfo:(MPPFileInfo *)fileInfo {
  CVPixelBufferRef pixelBuffer = [MPPImage pixelBufferWithFileInfo:fileInfo];

  if (!pixelBuffer) {
    return nil;
  }

  MPPImage *image = [[MPPImage alloc] initWithPixelBuffer:pixelBuffer error:nil];

  CVPixelBufferRelease(pixelBuffer);

  return image;
}

+ (MPPImage *)imageOfSampleBufferSourceTypeWithFileInfo:(MPPFileInfo *)fileInfo
                                             timingInfo:(const CMSampleTimingInfo *)timingInfo {
  CVPixelBufferRef pixelBuffer = [MPPImage pixelBufferWithFileInfo:fileInfo];

  if (!pixelBuffer) {
    return nil;
  }

  CMFormatDescriptionRef formatDescription;
  CMVideoFormatDescriptionCreateForImageBuffer(kCFAllocatorDefault, pixelBuffer,
                                               &formatDescription);

  CMSampleBufferRef sampleBuffer;
  CMSampleBufferCreateReadyWithImageBuffer(kCFAllocatorDefault, pixelBuffer, formatDescription,
                                           timingInfo, &sampleBuffer);
  CFRelease(formatDescription);

  MPPImage *image = [[MPPImage alloc] initWithSampleBuffer:sampleBuffer error:nil];

  CVPixelBufferRelease(pixelBuffer);
  CFRelease(sampleBuffer);

  return image;
}

+ (CVPixelBufferRef)pixelBufferWithFileInfo:(MPPFileInfo *)fileInfo {
  if (!fileInfo.path) return nil;

  // To create an `MPPImage` of source type
  // `MPPImageSourceTypePixelBuffer`/`MPPImageSourceTypeSampleBuffer`, a `UIImage` is first created
  // from the provided file path. A `CVPixelBuffer` can then be easily extracted from the `UIImage`
  // which in turn will be used to create the `MPPImage`. In real world use cases, `MPPImage`s from
  // files are intended to be initialized using `UIImage`s with a source type of
  // `MPPImageSourceTypeImage`. `MPPImageSourceTypePixelBuffer` is expected to be used when the
  // application receives a `CVPixelBuffer` after some processing or from the camera.
  //
  // Since image files are the only sources used in tests, the aforementioned approach is followed
  // to enable testing `MPPImage`s of source type `MPPImageSourceTypePixelBuffer`.
  UIImage *image = [[UIImage alloc] initWithContentsOfFile:fileInfo.path];

  if (!image) return nil;

  return image.pixelBuffer;
}

// TODO: Remove after all tests are migrated
+ (nullable MPPImage *)imageFromBundleWithClass:(Class)classObject
                                       fileName:(NSString *)name
                                         ofType:(NSString *)type {
  UIImage *image = [UIImage imageFromBundleWithClass:classObject fileName:name ofType:type];

  return [[MPPImage alloc] initWithUIImage:image error:nil];
}

// TODO: Remove after all tests are migrated
+ (nullable MPPImage *)imageFromBundleWithClass:(Class)classObject
                                       fileName:(NSString *)name
                                         ofType:(NSString *)type
                                    orientation:(UIImageOrientation)imageOrientation {
  UIImage *image = [UIImage imageFromBundleWithClass:classObject fileName:name ofType:type];

  return [[MPPImage alloc] initWithUIImage:image orientation:imageOrientation error:nil];
}

@end
