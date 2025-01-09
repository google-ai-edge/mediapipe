// Copyright 2024 The MediaPipe Authors.
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
#import "mediapipe/tasks/ios/common/utils/sources/NSString+Helpers.h"
#import "mediapipe/tasks/ios/test/vision/utils/sources/MPPImage+TestUtils.h"
#import "mediapipe/tasks/ios/vision/core/utils/sources/MPPImage+Utils.h"

#include "mediapipe/framework/deps/file_path.h"
#include "mediapipe/tasks/cc/vision/utils/image_utils.h"

#import <Accelerate/Accelerate.h>
#import <CoreGraphics/CoreGraphics.h>
#import <CoreMedia/CoreMedia.h>
#import <CoreVideo/CoreVideo.h>
#import <UIKit/UIKit.h>
#import <XCTest/XCTest.h>

NS_ASSUME_NONNULL_BEGIN

static MPPFileInfo *const kBurgerImageFileInfo = [[MPPFileInfo alloc] initWithName:@"burger"
                                                                              type:@"jpg"];

static NSString *const kExpectedErrorDomain = @"com.google.mediapipe.tasks";

#define AssertEqualErrors(error, expectedError)              \
  XCTAssertNotNil(error);                                    \
  XCTAssertEqualObjects(error.domain, expectedError.domain); \
  XCTAssertEqual(error.code, expectedError.code);            \
  XCTAssertEqualObjects(error.localizedDescription, expectedError.localizedDescription)

#define AssertEqualMPImages(image, expectedImage)               \
  XCTAssertEqual(image.width, expectedImage.width);             \
  XCTAssertEqual(image.height, expectedImage.height);           \
  XCTAssertEqual(image.orientation, expectedImage.orientation); \
  XCTAssertEqual(image.imageSourceType, expectedImage.imageSourceType);

namespace {
using ::mediapipe::Image;
using ::mediapipe::ImageFrame;

Image CppImageWithMPImage(MPPImage *image) {
  std::unique_ptr<ImageFrame> imageFrame = [image imageFrameWithError:nil];
  return Image(std::move(imageFrame));
}

}  // namespace

/** Unit tests for `MPPImage+Utils`. */
@interface MPPImageUtilsTests : XCTestCase

@end

@implementation MPPImageUtilsTests

#pragma mark - Tests

- (void)setUp {
  // Prevent illegal memory access when trying to compare pixel buffers when image files are not found.
  self.continueAfterFailure = NO;
}

#pragma mark - Tests for Initializig `MPPImage`s with MediaPipe C++ Images

- (void)testInitMPImageWithCppImageAndNilSourceImageFails {
  // Initialize the source MPPImage whose properties will be used to initialize an MPPImage from a
  // C++ `Image`.
  MPPImage *sourceImage = [MPPImage imageWithFileInfo:kBurgerImageFileInfo];
  XCTAssertNotNil(sourceImage);

  // Create C++ `Image` from the source image.
  Image sourceCppImage = CppImageWithMPImage(sourceImage);

  // Create `MPPImage` from C++ `Image` with properties of the `sourceImage`.
  NSError *error = nil;
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wnonnull"
  MPPImage *image = [[MPPImage alloc] initWithCppImage:sourceCppImage
                        cloningPropertiesOfSourceImage:nil
                                   shouldCopyPixelData:YES
                                                 error:&error];
#pragma clang diagnostic pop
  XCTAssertNil(image);

  AssertEqualErrors(
      error,
      [NSError errorWithDomain:kExpectedErrorDomain
                          code:MPPTasksErrorCodeInvalidArgumentError
                      userInfo:@{NSLocalizedDescriptionKey : @"Source image cannot be nil."}]);
}

- (void)testInitMPImageOfSourceTypeUIImageWithCppImageSucceeds {
  // Initialize the source MPPImage whose properties will be used to initialize an MPPImage from a
  // C++ `Image`.
  MPPImage *sourceImage = [MPPImage imageWithFileInfo:kBurgerImageFileInfo];
  XCTAssertNotNil(sourceImage);

  // Create C++ `Image` from the source image.
  Image sourceCppImage = CppImageWithMPImage(sourceImage);

  // Create `MPPImage` from C++ `Image` with properties of the `sourceImage`.
  MPPImage *image = [[MPPImage alloc] initWithCppImage:sourceCppImage
                        cloningPropertiesOfSourceImage:sourceImage
                                   shouldCopyPixelData:YES
                                                 error:nil];

  // Check if newly created image has the same properties as the source image.
  AssertEqualMPImages(image, sourceImage);

  // Check if contents of the pixel buffers of the created `MPPImage` and the C++ image from which
  // it was created are equal.
  XCTAssertTrue(image.image.CGImage != nullptr);
  [MPPImageUtilsTests assertUnderlyingBufferOfCGImage:image.image.CGImage
                                      equalToCppImage:sourceCppImage];
}

- (void)testInitMPImageOfSourceTypeUIImageWithCppImageNoCopySucceeds {
  // Initialize the source MPPImage whose properties will be used to initialize an MPPImage from a
  // C++ `Image`.
  MPPImage *sourceImage = [MPPImage imageWithFileInfo:kBurgerImageFileInfo];
  XCTAssertNotNil(sourceImage);

  // Create C++ `Image` from the source image.
  Image sourceCppImage = CppImageWithMPImage(sourceImage);

  // Create `MPPImage` from C++ `Image` with properties of the `sourceImage`.
  MPPImage *image = [[MPPImage alloc] initWithCppImage:sourceCppImage
                        cloningPropertiesOfSourceImage:sourceImage
                                   shouldCopyPixelData:NO
                                                 error:nil];

  // Check if newly created image has the same properties as the source image.
  AssertEqualMPImages(image, sourceImage);

  // Check if contents of the pixel buffers of the created `MPPImage` and the C++ image from which
  // it was created are equal.
  XCTAssertTrue(image.image.CGImage != nullptr);
  [MPPImageUtilsTests assertUnderlyingBufferOfCGImage:image.image.CGImage
                                      equalToCppImage:sourceCppImage];
}

- (void)testInitMPImageOfSourceTypePixelBufferWithCPPImageSucceeds {
  // Initialize the source MPPImage whose properties will be used to initialize an MPPImage from a
  // C++ `Image`.
  MPPImage *sourceImage = [MPPImage imageWithFileInfo:kBurgerImageFileInfo
                                           sourceType:MPPImageSourceTypePixelBuffer];
  XCTAssertNotNil(sourceImage);

  // Create C++ `Image` from the source image.
  Image sourceCppImage = CppImageWithMPImage(sourceImage);

  // Create `MPPImage` from C++ `Image` with properties of the `sourceImage`.
  MPPImage *image = [[MPPImage alloc] initWithCppImage:sourceCppImage
                        cloningPropertiesOfSourceImage:sourceImage
                                   shouldCopyPixelData:YES
                                                 error:nil];

  XCTAssertTrue(image.pixelBuffer != nullptr);
  AssertEqualMPImages(image, sourceImage);

  CVPixelBufferRef pixelBuffer = image.pixelBuffer;

  [MPPImageUtilsTests assertUnderlyingBufferOfCVPixelBuffer:pixelBuffer
                                            equalToCPPImage:sourceCppImage];
}

- (void)testInitMPImageOfSourceTypePixelBufferWithCPPImageNoCopyFails {
  // Initialize the source MPPImage whose properties will be used to initialize an MPPImage from a
  // C++ `Image`.
  MPPImage *sourceImage = [MPPImage imageWithFileInfo:kBurgerImageFileInfo
                                           sourceType:MPPImageSourceTypePixelBuffer];
  XCTAssertNotNil(sourceImage);

  // Create C++ `Image` from the source image.
  Image sourceCppImage = CppImageWithMPImage(sourceImage);

  NSError *error;
  MPPImage *image = [[MPPImage alloc] initWithCppImage:sourceCppImage
                        cloningPropertiesOfSourceImage:sourceImage
                                   shouldCopyPixelData:NO
                                                 error:&error];

  XCTAssertNil(image);
  AssertEqualErrors(
      error,
      [NSError
          errorWithDomain:kExpectedErrorDomain
                     code:MPPTasksErrorCodeInvalidArgumentError
                 userInfo:@{
                   NSLocalizedDescriptionKey :
                       @"When the source type is pixel buffer, you cannot request uncopied data."
                 }]);
}

- (void)testInitMPImageOfSourceTypeSampleBufferWithCPPImageSuceeds {
  // Initialize the source MPPImage whose properties will be used to initialize an MPPImage from a
  // C++ `Image`.
  MPPImage *sourceImage = [MPPImage imageWithFileInfo:kBurgerImageFileInfo
                                           sourceType:MPPImageSourceTypeSampleBuffer];
  XCTAssertNotNil(sourceImage);

  // Create C++ `Image` from the source image.
  Image sourceCppImage = CppImageWithMPImage(sourceImage);

  NSError *error;
  MPPImage *image = [[MPPImage alloc] initWithCppImage:sourceCppImage
                        cloningPropertiesOfSourceImage:sourceImage
                                   shouldCopyPixelData:YES
                                                 error:&error];

  XCTAssertTrue(image.sampleBuffer != nullptr);
  AssertEqualMPImages(image, sourceImage);

  CVPixelBufferRef pixelBuffer = (CVPixelBufferRef)CMSampleBufferGetImageBuffer(image.sampleBuffer);
  [MPPImageUtilsTests assertUnderlyingBufferOfCVPixelBuffer:pixelBuffer
                                            equalToCPPImage:sourceCppImage];
}

- (void)testInitMPImageOfSourceTypeSampleBufferWithCPPImageNoCopyFails {
  // Initialize the source MPPImage whose properties will be used to initialize an MPPImage from a
  // C++ `Image`.

  MPPImage *sourceImage = [MPPImage imageWithFileInfo:kBurgerImageFileInfo
                                           sourceType:MPPImageSourceTypeSampleBuffer];
  XCTAssertNotNil(sourceImage);

  // Create C++ `Image` from the source image.
  Image sourceCppImage = CppImageWithMPImage(sourceImage);

  NSError *error;
  MPPImage *image = [[MPPImage alloc] initWithCppImage:sourceCppImage
                        cloningPropertiesOfSourceImage:sourceImage
                                   shouldCopyPixelData:NO
                                                 error:&error];

  XCTAssertNil(image);
  AssertEqualErrors(
      error,
      [NSError
          errorWithDomain:kExpectedErrorDomain
                     code:MPPTasksErrorCodeInvalidArgumentError
                 userInfo:@{
                   NSLocalizedDescriptionKey :
                       @"When the source type is sample buffer, you cannot request uncopied data."
                 }]);
}

#pragma mark - Helper Methods

+ (void)assertUnderlyingBufferOfCGImage:(_Nonnull CGImageRef)cgImage
                        equalToCppImage:(const Image &)cppImage {
  // Using common method for both copy and no Copy scenario without testing equality of the base
  // addresses of the pixel buffers of the `CGImage` and the C++ Image. `CGDataProviderCopyData` is
  // currently the only documented way to access the underlying bytes of a `CGImage` and according
  // to Apple's official documentation, this method should return copied bytes of the `CGImage`.
  // Thus, in theory, testing equality of the base addresses of the copied bytes of the `CGImage`
  // and the C++ `Image` is pointless.
  //
  // But debugging `CGDataProviderCopyData` output always returned the original underlying bytes of
  // the `CGImage` without a copy. The base address of the `CGImage` buffer was found to be equal to
  // the base address of the C++ `Image` buffer in the no copy scenario and unequal in the copy
  // scenario. This verifies that the copy and no copy scenarios work as expected. Since this isn't
  // the expected behaviour of `CGDataProviderCopyData` according to Apple's official documentation,
  // the equality checks of the base addresses of the pixel buffers of the `CGImage` and C++ Image
  // have been omitted for the time being.
  CFDataRef resultImageData = CGDataProviderCopyData(CGImageGetDataProvider(cgImage));
  const UInt8 *resultImagePixels = CFDataGetBytePtr(resultImageData);

  ImageFrame *cppImageFrame = cppImage.GetImageFrameSharedPtr().get();

  XCTAssertEqual(cppImageFrame->Width(), CGImageGetWidth(cgImage));
  XCTAssertEqual(cppImageFrame->Height(), CGImageGetHeight(cgImage));
  XCTAssertEqual(cppImageFrame->ByteDepth() * 8, CGImageGetBitsPerComponent(cgImage));

  const UInt8 *cppImagePixels = cppImageFrame->PixelData();

  NSInteger consistentPixels = 0;
  for (int i = 0; i < cppImageFrame->Height() * cppImageFrame->WidthStep(); ++i) {
    consistentPixels += resultImagePixels[i] == cppImagePixels[i] ? 1 : 0;
  }

  XCTAssertEqual(consistentPixels, cppImageFrame->Height() * cppImageFrame->WidthStep());

  CFRelease(resultImageData);
}

+ (void)assertUnderlyingBufferOfCVPixelBuffer:(_Nonnull CVPixelBufferRef &)pixelBuffer
                              equalToCPPImage:(const Image &)cppImage {
  ImageFrame *cppImageFrame = cppImage.GetImageFrameSharedPtr().get();
  XCTAssertEqual(cppImageFrame->Width(), CVPixelBufferGetWidth(pixelBuffer));
  XCTAssertEqual(cppImageFrame->Height(), CVPixelBufferGetHeight(pixelBuffer));
  XCTAssertEqual(cppImageFrame->WidthStep(), CVPixelBufferGetBytesPerRow(pixelBuffer));

  const UInt8 *cppImagePixels = cppImageFrame->PixelData();

  CVPixelBufferLockBaseAddress(pixelBuffer, 0);
  UInt8 *resultImagePixels = (UInt8 *)CVPixelBufferGetBaseAddress(pixelBuffer);

  // Ensure that the underlying buffer of the created `MPPImage` is copied. In case of
  // `CVPixelBuffer`s this is straightforward to test.
  XCTAssertNotEqual(resultImagePixels, cppImagePixels);

  NSInteger consistentPixels = 0;

  // MediaPipe images only support inference of RGBA images. Thus `[MPPImage imageFrameWithError:]`
  // returns RGBA image frames irrespective of the order of the channels in the `MPPImage`. The
  // `MPPImage` being tested here has pixel ordering of BGRA since it is created using a
  // `CVPixelBuffer` that supports only BGRA images. The pixel equality testing code below takes
  // into account the differenc in the channel ordering.
  const int kRIndexInRGBA = 0, kBIndexInRGBA = 2;
  const int kRIndexInBGRA = 2, kBIndexInBGRA = 0;
  const int kGIndex = 1, kAlphaIndex = 3;

  for (int i = 0; i < cppImageFrame->Width() * cppImageFrame->Height(); ++i) {
    consistentPixels +=
        resultImagePixels[i * 4 + kBIndexInBGRA] == cppImagePixels[i * 4 + kBIndexInRGBA] ? 1 : 0;
    consistentPixels +=
        resultImagePixels[i * 4 + kGIndex] == cppImagePixels[i * 4 + kGIndex] ? 1 : 0;
    consistentPixels +=
        resultImagePixels[i * 4 + kRIndexInBGRA] == cppImagePixels[i * 4 + kRIndexInRGBA] ? 1 : 0;
    consistentPixels +=
        resultImagePixels[i * 4 + kAlphaIndex] == cppImagePixels[i * 4 + kAlphaIndex] ? 1 : 0;
  }
  CVPixelBufferUnlockBaseAddress(pixelBuffer, 0);

  XCTAssertEqual(consistentPixels, cppImageFrame->Height() * cppImageFrame->WidthStep());
}

@end

NS_ASSUME_NONNULL_END
