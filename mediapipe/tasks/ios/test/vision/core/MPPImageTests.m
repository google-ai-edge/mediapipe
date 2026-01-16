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

#import "mediapipe/tasks/ios/common/sources/MPPCommon.h"
#import "mediapipe/tasks/ios/vision/core/sources/MPPImage.h"

#import <Accelerate/Accelerate.h>
#import <CoreGraphics/CoreGraphics.h>
#import <CoreMedia/CoreMedia.h>
#import <CoreVideo/CoreVideo.h>
#import <XCTest/XCTest.h>
#import <UIKit/UIKit.h>

NS_ASSUME_NONNULL_BEGIN

static NSString *const kTestImageName = @"burger";
static NSString *const kTestImageType = @"jpg";
static CGFloat kTestImageWidthInPixels = 480.0f;
static CGFloat kTestImageHeightInPixels = 325.0f;
static NSString *const kExpectedErrorDomain = @"com.google.mediapipe.tasks";

#define AssertEqualErrors(error, expectedError)                                               \
  XCTAssertNotNil(error);                                                                     \
  XCTAssertEqualObjects(error.domain, expectedError.domain);                                  \
  XCTAssertEqual(error.code, expectedError.code);                                             \
  XCTAssertNotEqual(                                                                          \
      [error.localizedDescription rangeOfString:expectedError.localizedDescription].location, \
      NSNotFound)

/** Unit tests for `MPPImage`. */
@interface MPPImageTests : XCTestCase

/** Test image. */
@property(nonatomic, nullable) UIImage *image;

@end

@implementation MPPImageTests

#pragma mark - Tests

- (void)setUp {
  [super setUp];
  NSString *imageName = [[NSBundle bundleForClass:[self class]] pathForResource:kTestImageName
                                                                         ofType:kTestImageType];
  self.image = [[UIImage alloc] initWithContentsOfFile:imageName];
}

- (void)tearDown {
  self.image = nil;
  [super tearDown];
}

- (void)assertMPPImage:(nullable MPPImage *)mppImage
         hasSourceType:(MPPImageSourceType)sourceType
        hasOrientation:(UIImageOrientation)expectedOrientation
                 width:(CGFloat)expectedWidth
                height:(CGFloat)expectedHeight {
  XCTAssertNotNil(mppImage);
  XCTAssertEqual(mppImage.imageSourceType, sourceType);
  XCTAssertEqual(mppImage.orientation, expectedOrientation);
  XCTAssertEqualWithAccuracy(mppImage.width, expectedWidth, FLT_EPSILON);
  XCTAssertEqualWithAccuracy(mppImage.height, expectedHeight, FLT_EPSILON);
}

- (void)assertInitFailsWithImage:(nullable MPPImage *)mppImage
                           error:(NSError *)error
                   expectedError:(NSError *)expectedError {
  XCTAssertNil(mppImage);
  XCTAssertNotNil(error);
  AssertEqualErrors(error, expectedError);
}

- (void)testInitWithImageSucceeds {
  MPPImage *mppImage = [[MPPImage alloc] initWithUIImage:self.image error:nil];
  [self assertMPPImage:mppImage
         hasSourceType:MPPImageSourceTypeImage
        hasOrientation:self.image.imageOrientation
                 width:kTestImageWidthInPixels
                height:kTestImageHeightInPixels];
}

- (void)testInitWithImageAndOrientation {
  UIImageOrientation orientation = UIImageOrientationRight;

  MPPImage *mppImage = [[MPPImage alloc] initWithUIImage:self.image
                                             orientation:orientation
                                                   error:nil];
  [self assertMPPImage:mppImage
         hasSourceType:MPPImageSourceTypeImage
        hasOrientation:orientation
                 width:kTestImageWidthInPixels
                height:kTestImageHeightInPixels];
}

- (void)testInitWithImage_nilImage {
  NSError *error;

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wnonnull"
  MPPImage *mppImage = [[MPPImage alloc] initWithUIImage:nil error:&error];
#pragma clang diagnostic pop

  [self
      assertInitFailsWithImage:mppImage
                         error:error
                 expectedError:[NSError errorWithDomain:kExpectedErrorDomain
                                                   code:MPPTasksErrorCodeInvalidArgumentError
                                               userInfo:@{
                                                 NSLocalizedDescriptionKey : @"Image cannot be nil."
                                               }]];
}

- (void)testInitWithImageAndOrientation_nilImage {
  NSError *error;

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wnonnull"
  MPPImage *mppImage = [[MPPImage alloc] initWithUIImage:nil
                                             orientation:UIImageOrientationRight
                                                   error:&error];
#pragma clang diagnostic pop

  [self
      assertInitFailsWithImage:mppImage
                         error:error
                 expectedError:[NSError errorWithDomain:kExpectedErrorDomain
                                                   code:MPPTasksErrorCodeInvalidArgumentError
                                               userInfo:@{
                                                 NSLocalizedDescriptionKey : @"Image cannot be nil."
                                               }]];
}

- (void)testInitWithSampleBuffer {
  CMSampleBufferRef sampleBuffer = [self sampleBuffer];

  MPPImage *mppImage = [[MPPImage alloc] initWithSampleBuffer:sampleBuffer error:nil];
  [self assertMPPImage:mppImage
         hasSourceType:MPPImageSourceTypeSampleBuffer
        hasOrientation:UIImageOrientationUp
                 width:kTestImageWidthInPixels
                height:kTestImageHeightInPixels];
}

- (void)testInitWithSampleBufferAndOrientation {
  UIImageOrientation orientation = UIImageOrientationRight;
  CMSampleBufferRef sampleBuffer = [self sampleBuffer];

  MPPImage *mppImage = [[MPPImage alloc] initWithSampleBuffer:sampleBuffer
                                                  orientation:orientation
                                                        error:nil];
  [self assertMPPImage:mppImage
         hasSourceType:MPPImageSourceTypeSampleBuffer
        hasOrientation:orientation
                 width:kTestImageWidthInPixels
                height:kTestImageHeightInPixels];
}

- (void)testInitWithSampleBuffer_nilImage {
  NSError *error;

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wnonnull"
  MPPImage *mppImage = [[MPPImage alloc] initWithSampleBuffer:nil error:&error];
#pragma clang diagnostic pop

  [self
      assertInitFailsWithImage:mppImage
                         error:error
                 expectedError:
                     [NSError errorWithDomain:kExpectedErrorDomain
                                         code:MPPTasksErrorCodeInvalidArgumentError
                                     userInfo:@{
                                       NSLocalizedDescriptionKey :
                                           @"Sample buffer is not valid. Invoking "
                                           @"CMSampleBufferIsValid(sampleBuffer) must return true."
                                     }]];
}

- (void)testInitWithSampleBufferAndOrientation_nilImage {
  NSError *error;

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wnonnull"
  MPPImage *mppImage = [[MPPImage alloc] initWithSampleBuffer:nil
                                                  orientation:UIImageOrientationRight
                                                        error:&error];
#pragma clang diagnostic pop

  [self
      assertInitFailsWithImage:mppImage
                         error:error
                 expectedError:
                     [NSError errorWithDomain:kExpectedErrorDomain
                                         code:MPPTasksErrorCodeInvalidArgumentError
                                     userInfo:@{
                                       NSLocalizedDescriptionKey :
                                           @"Sample buffer is not valid. Invoking "
                                           @"CMSampleBufferIsValid(sampleBuffer) must return true."
                                     }]];
}

- (void)testInitWithPixelBuffer {
  CMSampleBufferRef sampleBuffer = [self sampleBuffer];
  CVPixelBufferRef pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer);

  MPPImage *mppImage = [[MPPImage alloc] initWithPixelBuffer:pixelBuffer error:nil];
  [self assertMPPImage:mppImage
         hasSourceType:MPPImageSourceTypePixelBuffer
        hasOrientation:UIImageOrientationUp
                 width:kTestImageWidthInPixels
                height:kTestImageHeightInPixels];
}

- (void)testInitWithPixelBufferAndOrientation {
  UIImageOrientation orientation = UIImageOrientationRight;

  CMSampleBufferRef sampleBuffer = [self sampleBuffer];
  CVPixelBufferRef pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer);

  MPPImage *mppImage = [[MPPImage alloc] initWithPixelBuffer:pixelBuffer
                                                 orientation:orientation
                                                       error:nil];
  [self assertMPPImage:mppImage
         hasSourceType:MPPImageSourceTypePixelBuffer
        hasOrientation:orientation
                 width:kTestImageWidthInPixels
                height:kTestImageHeightInPixels];
}

- (void)testInitWithPixelBuffer_nilImage {
  NSError *error;

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wnonnull"
  MPPImage *mppImage = [[MPPImage alloc] initWithPixelBuffer:nil error:&error];
#pragma clang diagnostic pop

  [self assertInitFailsWithImage:mppImage
                           error:error
                   expectedError:[NSError errorWithDomain:kExpectedErrorDomain
                                                     code:MPPTasksErrorCodeInvalidArgumentError
                                                 userInfo:@{
                                                   NSLocalizedDescriptionKey :
                                                       @"Pixel Buffer cannot be nil."
                                                 }]];
}

- (void)testInitWithPixelBufferAndOrientation_nilImage {
  NSError *error;

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wnonnull"
  MPPImage *mppImage = [[MPPImage alloc] initWithPixelBuffer:nil
                                                 orientation:UIImageOrientationRight
                                                       error:&error];
#pragma clang diagnostic pop

  [self assertInitFailsWithImage:mppImage
                           error:error
                   expectedError:[NSError errorWithDomain:kExpectedErrorDomain
                                                     code:MPPTasksErrorCodeInvalidArgumentError
                                                 userInfo:@{
                                                   NSLocalizedDescriptionKey :
                                                       @"Pixel Buffer cannot be nil."
                                                 }]];
}

#pragma mark - Private

/**
 * Converts the input image in RGBA space into a `CMSampleBuffer`.
 *
 * @return `CMSampleBuffer` converted from the given `UIImage`.
 */
- (CMSampleBufferRef)sampleBuffer {
  // Rotate the image and convert from RGBA to BGRA.
  CGImageRef CGImage = self.image.CGImage;
  size_t width = CGImageGetWidth(CGImage);
  size_t height = CGImageGetHeight(CGImage);
  size_t bpr = CGImageGetBytesPerRow(CGImage);

  CGDataProviderRef provider = CGImageGetDataProvider(CGImage);
  NSData *imageRGBAData = (id)CFBridgingRelease(CGDataProviderCopyData(provider));
  const uint8_t order[4] = {2, 1, 0, 3};

  NSData *imageBGRAData = nil;
  unsigned char *bgraPixel = (unsigned char *)malloc([imageRGBAData length]);
  if (bgraPixel) {
    vImage_Buffer src;
    src.height = height;
    src.width = width;
    src.rowBytes = bpr;
    src.data = (void *)[imageRGBAData bytes];

    vImage_Buffer dest;
    dest.height = height;
    dest.width = width;
    dest.rowBytes = bpr;
    dest.data = bgraPixel;

    // Specify ordering changes in map.
    vImage_Error error = vImagePermuteChannels_ARGB8888(&src, &dest, order, kvImageNoFlags);

    // Package the result.
    if (error == kvImageNoError) {
      imageBGRAData = [NSData dataWithBytes:bgraPixel length:[imageRGBAData length]];
    }

    // Memory cleanup.
    free(bgraPixel);
  }

  if (imageBGRAData == nil) {
    XCTFail(@"Failed to convert input image.");
  }

  // Write data to `CMSampleBuffer`.
  NSDictionary *options = @{
    (__bridge NSString *)kCVPixelBufferCGImageCompatibilityKey : @(YES),
    (__bridge NSString *)kCVPixelBufferCGBitmapContextCompatibilityKey : @(YES)
  };
  CVPixelBufferRef pixelBuffer;
  CVReturn status = CVPixelBufferCreateWithBytes(
      kCFAllocatorDefault, width, height, kCVPixelFormatType_32BGRA, (void *)[imageBGRAData bytes],
      bpr, NULL, nil, (__bridge CFDictionaryRef)options, &pixelBuffer);

  if (status != kCVReturnSuccess) {
    XCTFail(@"Failed to create pixel buffer.");
  }

  CVPixelBufferLockBaseAddress(pixelBuffer, 0);
  CMVideoFormatDescriptionRef videoInfo = NULL;
  CMVideoFormatDescriptionCreateForImageBuffer(kCFAllocatorDefault, pixelBuffer, &videoInfo);

  CMSampleBufferRef buffer;
  CMSampleBufferCreateForImageBuffer(kCFAllocatorDefault, pixelBuffer, true, NULL, NULL, videoInfo,
                                     &kCMTimingInfoInvalid, &buffer);

  CVPixelBufferUnlockBaseAddress(pixelBuffer, 0);

  return buffer;
}

@end

NS_ASSUME_NONNULL_END
