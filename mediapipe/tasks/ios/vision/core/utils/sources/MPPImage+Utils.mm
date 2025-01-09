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

#import "mediapipe/tasks/ios/vision/core/utils/sources/MPPImage+Utils.h"

#import "mediapipe/tasks/ios/common/sources/MPPCommon.h"
#import "mediapipe/tasks/ios/common/utils/sources/MPPCommonUtils.h"

#import <Accelerate/Accelerate.h>
#import <CoreGraphics/CoreGraphics.h>
#import <CoreImage/CoreImage.h>
#import <CoreVideo/CoreVideo.h>

#include <cstdint>
#include <memory>

#include "mediapipe/framework/formats/image_format.pb.h"

namespace {
using ::mediapipe::ImageFormat;
using ::mediapipe::ImageFrame;

vImage_Buffer CreateEmptyVImageBufferFromImageFrame(ImageFrame &imageFrame, bool shouldAllocate) {
  UInt8 *data = shouldAllocate
                    ? (UInt8 *)malloc(imageFrame.Height() * imageFrame.WidthStep() * sizeof(UInt8))
                    : nullptr;
  return {.data = data,
          .height = static_cast<vImagePixelCount>(imageFrame.Height()),
          .width = static_cast<vImagePixelCount>(imageFrame.Width()),
          .rowBytes = static_cast<size_t>(imageFrame.WidthStep())};
}

vImage_Buffer CreateVImageBufferFromImageFrame(ImageFrame &imageFrame) {
  vImage_Buffer imageBuffer = CreateEmptyVImageBufferFromImageFrame(imageFrame, false);
  imageBuffer.data = imageFrame.MutablePixelData();
  return imageBuffer;
}

vImage_Buffer allocatedVImageBuffer(vImagePixelCount width, vImagePixelCount height,
                                    size_t rowBytes) {
  UInt8 *data = (UInt8 *)malloc(height * rowBytes * sizeof(UInt8));
  return {.data = data, .height = height, .width = width, .rowBytes = rowBytes};
}

static void FreeRefConReleaseCallback(void *refCon, const void *baseAddress) {
  free((void *)baseAddress);
}

static void FreeCGDataProviderReleaseCallback(void *info, const void *data, size_t size) {
  free((void *)data);
}
}  // namespace

@interface MPPPixelDataUtils : NSObject

+ (std::unique_ptr<ImageFrame>)imageFrameFromPixelData:(uint8_t *)pixelData
                                             withWidth:(size_t)width
                                                height:(size_t)height
                                                stride:(size_t)stride
                                     pixelBufferFormat:(OSType)pixelBufferFormatType
                                                 error:(NSError **)error;

+ (UInt8 *)rgbaPixelDataFromImageFrame:(ImageFrame &)imageFrame
                            shouldCopy:(BOOL)shouldCopy
                                 error:(NSError **)error;

@end

@interface MPPCVPixelBufferUtils : NSObject

+ (std::unique_ptr<ImageFrame>)imageFrameFromCVPixelBuffer:(CVPixelBufferRef)pixelBuffer
                                                     error:(NSError **)error;

// This method is used to create CVPixelBuffer from output images of tasks like `FaceStylizer` only
// when the input `MPImage` source type is `pixelBuffer`.
// Always copies the pixel data of the image frame to the created `CVPixelBuffer`.
//
// The only possible 32 RGBA pixel format of input `CVPixelBuffer` is `kCVPixelFormatType_32BGRA`.
// But Mediapipe does not support inference on images of format `BGRA`. Hence the channels of the
// underlying pixel data of `CVPixelBuffer` are permuted to the supported RGBA format before passing
// them to the task for inference. The pixel format of the output images of any MediaPipe task will
// be the same as the pixel format of the input image. (RGBA in this case).
//
// Since creation of `CVPixelBuffer` from the output image pixels with a format of
// `kCVPixelFormatType_32RGBA` is not possible, the channels of the output C++ image `RGBA` have to
// be permuted to the format `BGRA`. When the pixels are copied to create `CVPixelBuffer` this does
// not pose a challenge.
//
// TODO: Investigate if permuting channels of output `mediapipe::Image` in place is possible for
// creating `CVPixelBuffer`s without copying the underlying pixels.
+ (CVPixelBufferRef)cvPixelBufferFromImageFrame:(ImageFrame &)imageFrame error:(NSError **)error;
@end

@interface MPPCGImageUtils : NSObject

+ (std::unique_ptr<ImageFrame>)imageFrameFromCGImage:(CGImageRef)cgImage error:(NSError **)error;
+ (CGImageRef)cgImageFromImageFrame:(std::shared_ptr<ImageFrame>)imageFrame
                shouldCopyPixelData:(BOOL)shouldCopyPixelData
                              error:(NSError **)error;

@end

@interface UIImage (ImageFrameUtils)

- (std::unique_ptr<ImageFrame>)imageFrameWithError:(NSError **)error;

@end

@implementation MPPPixelDataUtils : NSObject

+ (std::unique_ptr<ImageFrame>)imageFrameFromPixelData:(uint8_t *)pixelData
                                             withWidth:(size_t)width
                                                height:(size_t)height
                                                stride:(size_t)stride
                                     pixelBufferFormat:(OSType)pixelBufferFormatType
                                                 error:(NSError **)error {
  NSInteger destinationChannelCount = 4;
  size_t destinationBytesPerRow = width * destinationChannelCount;

  ImageFormat::Format imageFormat = ImageFormat::SRGBA;

  vImage_Buffer srcBuffer = {.data = pixelData,
                             .height = (vImagePixelCount)height,
                             .width = (vImagePixelCount)width,
                             .rowBytes = stride};

  vImage_Buffer destBuffer;

  vImage_Error convertError = kvImageNoError;

  // Convert the raw pixel data to RGBA format and un-premultiply the alpha from the R, G, B values
  // since MediaPipe C++ APIs only accept un pre-multiplied channels.
  //
  // This method is commonly used for `MPImage`s of all source types. Hence supporting BGRA and RGBA
  // formats. Only `pixelBuffer` source type is restricted to `BGRA` format.
  switch (pixelBufferFormatType) {
    case kCVPixelFormatType_32RGBA: {
      destBuffer = allocatedVImageBuffer((vImagePixelCount)width, (vImagePixelCount)height,
                                         destinationBytesPerRow);
      convertError = vImageUnpremultiplyData_RGBA8888(&srcBuffer, &destBuffer, kvImageNoFlags);
      break;
    }
    case kCVPixelFormatType_32BGRA: {
      // Permute channels to `RGBA` since MediaPipe tasks don't support inference on images of
      // format `BGRA`.
      const uint8_t permute_map[4] = {2, 1, 0, 3};
      destBuffer = allocatedVImageBuffer((vImagePixelCount)width, (vImagePixelCount)height,
                                         destinationBytesPerRow);
      convertError =
          vImagePermuteChannels_ARGB8888(&srcBuffer, &destBuffer, permute_map, kvImageNoFlags);
      if (convertError == kvImageNoError) {
        convertError = vImageUnpremultiplyData_RGBA8888(&destBuffer, &destBuffer, kvImageNoFlags);
      }
      break;
    }
    default: {
      [MPPCommonUtils createCustomError:error
                               withCode:MPPTasksErrorCodeInvalidArgumentError
                            description:@"Some internal error occured."];
      return nullptr;
    }
  }

  if (convertError != kvImageNoError) {
    [MPPCommonUtils createCustomError:error
                             withCode:MPPTasksErrorCodeInternalError
                          description:@"Some error occured while preprocessing the input image. "
                                      @"Please verify that the image is not corrupted."];
    return nullptr;
  }

  // Uses default deleter
  return std::make_unique<ImageFrame>(imageFormat, width, height, destinationBytesPerRow,
                                      static_cast<uint8_t *>(destBuffer.data));
}

+ (UInt8 *)rgbaPixelDataFromImageFrame:(ImageFrame &)imageFrame
                            shouldCopy:(BOOL)shouldCopy
                                 error:(NSError **)error {
  vImage_Buffer sourceBuffer = CreateVImageBufferFromImageFrame(imageFrame);

  // Pre-multiply the raw pixels from a `mediapipe::Image` before creating a `CGImage` to ensure
  // that pixels are displayed correctly irrespective of their alpha values.
  vImage_Buffer destinationBuffer;
  vImage_Error vImageOperationError;

  switch (imageFrame.Format()) {
    case ImageFormat::SRGBA: {
      destinationBuffer =
          shouldCopy ? CreateEmptyVImageBufferFromImageFrame(imageFrame, true) : sourceBuffer;
      vImageOperationError =
          vImagePremultiplyData_RGBA8888(&sourceBuffer, &destinationBuffer, kvImageNoFlags);
      break;
    }
    case ImageFormat::SRGB: {
      // Some tasks like the Face Stylizer output RGB images inspite of the input being restricted
      // to RGBA format. iOS does not allow creation of 24 bit images (RGB). All native image
      // formats supported by `MPPImage` only allow creation of 32 bit images (RGBA). Hence,
      // uncopied pixel buffer APIs cannot be supported. RGB pixel buffers must be copied to
      // RGBA buffers with an alpha of 1.0.
      if (!shouldCopy) {
        [MPPCommonUtils createCustomError:error
                                 withCode:MPPTasksErrorCodeInternalError
                              description:@"An error occured while processing the output image "
                                          @"pixels of the vision task."];
        return nullptr;
      }

      const vImagePixelCount channelCount = 4;
      destinationBuffer = allocatedVImageBuffer(imageFrame.Width(), imageFrame.Height(),
                                                imageFrame.Width() * channelCount);

      const Pixel_8 alpha = 255;

      vImageOperationError = vImageConvert_RGB888toRGBA8888(&sourceBuffer, nil, alpha,
                                                            &destinationBuffer, NO, kvImageNoFlags);
      break;
    }
    default: {
      [MPPCommonUtils createCustomError:error
                               withCode:MPPTasksErrorCodeInternalError
                            description:@"An error occured while processing the output image "
                                        @"pixels of the vision task."];
      return nullptr;
    }
  }

  if (vImageOperationError != kvImageNoError) {
    // Freeing allocated memory if one of the vImage operations fail. In practice, the operations
    // performed by this method never fail since image parameters are evaluated before invoking
    // them.
    // Placed here for an extra layer of safety and correctness.
    if (shouldCopy) {
      free(destinationBuffer.data);
    }

    [MPPCommonUtils
        createCustomError:error
                 withCode:MPPTasksErrorCodeInternalError
              description:
                  @"An error occured while processing the output image pixels of the vision task."];

    return nullptr;
  }

  return (UInt8 *)destinationBuffer.data;
}

@end

@implementation MPPCVPixelBufferUtils

+ (std::unique_ptr<ImageFrame>)imageFrameFromCVPixelBuffer:(CVPixelBufferRef)pixelBuffer
                                                     error:(NSError **)error {
  OSType pixelBufferFormat = CVPixelBufferGetPixelFormatType(pixelBuffer);
  std::unique_ptr<ImageFrame> imageFrame = nullptr;

  switch (pixelBufferFormat) {
    // Core Video only supports pixel data of order BGRA for 32 bit RGBA images.
    // Thus other formats like `kCVPixelFormatType_32BGRA` don't need to be accounted for.
    case kCVPixelFormatType_32BGRA: {
      CVPixelBufferLockBaseAddress(pixelBuffer, 0);
      imageFrame = [MPPPixelDataUtils
          imageFrameFromPixelData:(uint8_t *)CVPixelBufferGetBaseAddress(pixelBuffer)
                        withWidth:CVPixelBufferGetWidth(pixelBuffer)
                           height:CVPixelBufferGetHeight(pixelBuffer)
                           stride:CVPixelBufferGetBytesPerRow(pixelBuffer)
                pixelBufferFormat:pixelBufferFormat
                            error:error];
      CVPixelBufferUnlockBaseAddress(pixelBuffer, 0);
      break;
    }
    default: {
      [MPPCommonUtils createCustomError:error
                               withCode:MPPTasksErrorCodeInvalidArgumentError
                            description:@"Unsupported pixel format for CVPixelBuffer. Expected "
                                        @"kCVPixelFormatType_32BGRA"];
    }
  }

  return imageFrame;
}

+ (CVPixelBufferRef)cvPixelBufferFromImageFrame:(ImageFrame &)imageFrame error:(NSError **)error {
  switch (imageFrame.Format()) {
    case ImageFormat::SRGBA:
    case ImageFormat::SRGB:
      break;
    default: {
      [MPPCommonUtils createCustomError:error
                               withCode:MPPTasksErrorCodeInternalError
                            description:@"An error occured while creating a CVPixelBuffer from the "
                                        @"output image of the vision task."];
      return nullptr;
    }
  }

  UInt8 *pixelData = [MPPPixelDataUtils rgbaPixelDataFromImageFrame:imageFrame
                                                         shouldCopy:YES
                                                              error:error];

  if (!pixelData) {
    return nullptr;
  }

  const uint8_t permute_map[4] = {2, 1, 0, 3};
  const int channelCount = 4;
  const int bytesPerRow = imageFrame.Width() * channelCount;

  vImage_Buffer sourceBuffer = {.data = pixelData,
                                .height = static_cast<vImagePixelCount>(imageFrame.Height()),
                                .width = static_cast<vImagePixelCount>(imageFrame.Width()),
                                .rowBytes = static_cast<size_t>(bytesPerRow)};

  if (vImagePermuteChannels_ARGB8888(&sourceBuffer, &sourceBuffer, permute_map, kvImageNoFlags) ==
      kvImageNoError) {
    CVPixelBufferRef outputBuffer;

    OSType pixelBufferFormatType = kCVPixelFormatType_32BGRA;

    // Since data is copied, pass in a release callback that will be invoked when the pixel buffer
    // is destroyed.
    if (CVPixelBufferCreateWithBytes(kCFAllocatorDefault, imageFrame.Width(), imageFrame.Height(),
                                     pixelBufferFormatType, pixelData, bytesPerRow,
                                     FreeRefConReleaseCallback, pixelData, nullptr,
                                     &outputBuffer) == kCVReturnSuccess) {
      return outputBuffer;
    }
  }

  [MPPCommonUtils createCustomError:error
                           withCode:MPPTasksErrorCodeInternalError
                        description:@"An error occured while creating a CVPixelBuffer from the "
                                    @"output image of the vision task."];
  return nullptr;
}

@end

@implementation MPPCGImageUtils

+ (std::unique_ptr<ImageFrame>)imageFrameFromCGImage:(CGImageRef)cgImage error:(NSError **)error {
  size_t width = CGImageGetWidth(cgImage);
  size_t height = CGImageGetHeight(cgImage);

  NSInteger bitsPerComponent = 8;
  NSInteger channelCount = 4;
  size_t bytesPerRow = channelCount * width;

  std::unique_ptr<ImageFrame> imageFrame = nullptr;

  CGColorSpaceRef colorSpace = CGColorSpaceCreateDeviceRGB();
  // iOS infers bytesPerRow if it is set to 0.
  // See https://developer.apple.com/documentation/coregraphics/1455939-cgbitmapcontextcreate
  // But for segmentation test image, this was not the case.
  // Hence setting it to the value of channelCount*width.
  // kCGImageAlphaPremultipliedLast specifies that Alpha will always be next to B and the R, G, B
  // values will be pre multiplied with alpha. Images with alpha != 255 are stored with the R, G, B
  // values premultiplied with alpha by iOS. Hence `kCGImageAlphaPremultipliedLast` ensures all
  // kinds of images (alpha from 0 to 255) are correctly accounted for by iOS.
  // kCGBitmapByteOrder32Big specifies that R will be stored before B.
  // In combination they signify a pixelFormat of kCVPixelFormatType32RGBA.
  CGBitmapInfo bitMapinfoFor32RGBA = kCGImageAlphaPremultipliedLast | kCGBitmapByteOrder32Big;
  CGContextRef context = CGBitmapContextCreate(nil, width, height, bitsPerComponent, bytesPerRow,
                                               colorSpace, bitMapinfoFor32RGBA);

  if (context) {
    CGContextDrawImage(context, CGRectMake(0, 0, width, height), cgImage);
    uint8_t *srcData = (uint8_t *)CGBitmapContextGetData(context);

    if (srcData) {
      // We have drawn the image as an RGBA image with 8 bitsPerComponent and hence can safely input
      // a pixel format of type kCVPixelFormatType_32RGBA for conversion by vImage.
      imageFrame = [MPPPixelDataUtils imageFrameFromPixelData:srcData
                                                    withWidth:width
                                                       height:height
                                                       stride:bytesPerRow
                                            pixelBufferFormat:kCVPixelFormatType_32RGBA
                                                        error:error];
    }

    CGContextRelease(context);
  }

  CGColorSpaceRelease(colorSpace);

  return imageFrame;
}

+ (CGImageRef)cgImageFromImageFrame:(std::shared_ptr<ImageFrame>)imageFrame
                shouldCopyPixelData:(BOOL)shouldCopyPixelData
                              error:(NSError **)error {
  CGBitmapInfo bitmapInfo = kCGImageAlphaNoneSkipLast | kCGBitmapByteOrderDefault;

  ImageFrame *internalImageFrame = imageFrame.get();

  UInt8 *pixelData = [MPPPixelDataUtils rgbaPixelDataFromImageFrame:*internalImageFrame
                                                         shouldCopy:shouldCopyPixelData
                                                              error:error];

  if (!pixelData) {
    return nullptr;
  }

  switch (internalImageFrame->Format()) {
    case ImageFormat::SRGBA:
    case ImageFormat::SRGB: {
      bitmapInfo = kCGImageAlphaPremultipliedLast | kCGBitmapByteOrder32Big;
      break;
    }
    default:
      [MPPCommonUtils createCustomError:error
                               withCode:MPPTasksErrorCodeInternalError
                            description:@"An error occured while creating a CGImage from the "
                                        @"output image of the vision task."];
      return nullptr;
  }

  const int channelCount = 4;
  const size_t bytesPerRow = size_t(internalImageFrame->Width() * channelCount);

  CGDataProviderReleaseDataCallback imagePixelsReleaseCallback =
      shouldCopyPixelData ? FreeCGDataProviderReleaseCallback : nullptr;

  CGDataProviderRef provider = CGDataProviderCreateWithData(
      pixelData, pixelData, bytesPerRow * internalImageFrame->Height(), imagePixelsReleaseCallback);

  CGColorSpaceRef colorSpace = CGColorSpaceCreateDeviceRGB();

  CGImageRef cgImageRef = nullptr;

  if (provider && colorSpace) {
    size_t bitsPerComponent = 8;
    cgImageRef =
        CGImageCreate(internalImageFrame->Width(), internalImageFrame->Height(), bitsPerComponent,
                      bitsPerComponent * channelCount, bytesPerRow, colorSpace, bitmapInfo,
                      provider, nullptr, YES, kCGRenderingIntentDefault);
  }

  // Can safely pass `NULL` to these functions according to iOS docs.
  CGDataProviderRelease(provider);
  CGColorSpaceRelease(colorSpace);

  if (!cgImageRef) {
    [MPPCommonUtils createCustomError:error
                             withCode:MPPTasksErrorCodeInternalError
                          description:@"An error occured while converting the output image of the "
                                      @"vision task to a CGImage."];
  }

  return cgImageRef;
}

@end

@implementation UIImage (ImageFrameUtils)

- (std::unique_ptr<ImageFrame>)imageFrameFromCIImageWithError:(NSError **)error {
  if (self.CIImage.pixelBuffer) {
    return [MPPCVPixelBufferUtils imageFrameFromCVPixelBuffer:self.CIImage.pixelBuffer error:error];

  } else if (self.CIImage.CGImage) {
    return [MPPCGImageUtils imageFrameFromCGImage:self.CIImage.CGImage error:error];
  } else {
    [MPPCommonUtils createCustomError:error
                             withCode:MPPTasksErrorCodeInvalidArgumentError
                          description:@"CIImage should have CGImage or CVPixelBuffer info."];
  }

  return nullptr;
}

- (std::unique_ptr<ImageFrame>)imageFrameWithError:(NSError **)error {
  if (self.CGImage) {
    return [MPPCGImageUtils imageFrameFromCGImage:self.CGImage error:error];
  } else if (self.CIImage) {
    return [self imageFrameFromCIImageWithError:error];
  } else {
    [MPPCommonUtils createCustomError:error
                             withCode:MPPTasksErrorCodeInvalidArgumentError
                          description:@"UIImage should be initialized from"
                                       " CIImage or CGImage."];
  }

  return nullptr;
}

@end

@implementation MPPImage (Utils)

- (nullable instancetype)initWithCppImage:(const mediapipe::Image &)image
           cloningPropertiesOfSourceImage:(MPPImage *)sourceImage
                      shouldCopyPixelData:(BOOL)shouldCopyPixelData
                                    error:(NSError **)error {
  if (!sourceImage) {
    [MPPCommonUtils createCustomError:error
                             withCode:MPPTasksErrorCodeInvalidArgumentError
                          description:@"Source image cannot be nil."];
    return nil;
  }

  if (!sourceImage) {
    [MPPCommonUtils createCustomError:error
                             withCode:MPPTasksErrorCodeInvalidArgumentError
                          description:@"Source image cannot be nil."];
    return nil;
  }

  switch (sourceImage.imageSourceType) {
    case MPPImageSourceTypeImage: {
      CGImageRef cgImageRef = [MPPCGImageUtils cgImageFromImageFrame:image.GetImageFrameSharedPtr()
                                                 shouldCopyPixelData:shouldCopyPixelData
                                                               error:error];

      // `[UIImage imageWithCGImage]` seems to be returning an autoreleased object. Thus ARC only
      // deallocates it when the autoreleasepool to which the image was added is drained. This may
      // happen only much later during the life cycle of the app. For a standalone inference this
      // isn't a concern. If this method is invoked in a loop, the unreleased UIImage's accumulate
      // in memory. They get destroyed all at once when all iterations of the loop are completed.
      // For infinite loops like the camera callback, this results in an increase in memory
      // footprint over time. To avoid this, the`UIImage` is being created in an @autoreleasepool
      // block which results in the image being released as soon as the block completes. The MPImage
      // retains the UIImage, to keep the image alive during its lifetime.
      // (Until caller keeps a reference to the result retured by the task)
      //
      // Reference: `Use Local Autorelease Pool Blocks to Reduce Peak Memory Footprint`
      // https://developer.apple.com/library/archive/documentation/Cocoa/Conceptual/MemoryMgmt/Articles/mmAutoreleasePools.html.
      MPPImage *mpImage;
      @autoreleasepool {
        UIImage *uiImage = [UIImage imageWithCGImage:cgImageRef];
        mpImage = [self initWithUIImage:uiImage orientation:sourceImage.orientation error:nil];
        CGImageRelease(cgImageRef);
      }

      return mpImage;
    }
    case MPPImageSourceTypePixelBuffer: {
      if (!shouldCopyPixelData) {
        // TODO: Investigate possibility of permuting channels of `mediapipe::Image` returned by
        // vision tasks in place to ensure that we can support creating `CVPixelBuffer`s without
        // copying the pixel data.
        [MPPCommonUtils
            createCustomError:error
                     withCode:MPPTasksErrorCodeInvalidArgumentError
                  description:
                      @"When the source type is pixel buffer, you cannot request uncopied data."];
        return nil;
      }

      CVPixelBufferRef pixelBuffer =
          [MPPCVPixelBufferUtils cvPixelBufferFromImageFrame:*(image.GetImageFrameSharedPtr())
                                                       error:error];
      MPPImage *mpImage = [self initWithPixelBuffer:pixelBuffer
                                        orientation:sourceImage.orientation
                                              error:nil];
      CVPixelBufferRelease(pixelBuffer);
      return mpImage;
    }
    case MPPImageSourceTypeSampleBuffer: {
      if (!shouldCopyPixelData) {
        // TODO: Investigate possibility of permuting channels of `mediapipe::Image` returned by
        // vision tasks in place to ensure that we can support creating `CVPixelBuffer`s without
        // copying the pixel data.
        [MPPCommonUtils
            createCustomError:error
                     withCode:MPPTasksErrorCodeInvalidArgumentError
                  description:
                      @"When the source type is sample buffer, you cannot request uncopied data."];
        return nil;
      }
      CMSampleTimingInfo timingInfo;
      if (CMSampleBufferGetSampleTimingInfo(sourceImage.sampleBuffer, 0, &timingInfo) != 0) {
        [MPPCommonUtils createCustomError:error
                                 withCode:MPPTasksErrorCodeInvalidArgumentError
                              description:@"Some error occured while fetching the sample timing "
                                          @"info of the CMSampleBuffer."];
        return nil;
      }

      CVPixelBufferRef pixelBuffer =
          [MPPCVPixelBufferUtils cvPixelBufferFromImageFrame:*(image.GetImageFrameSharedPtr())
                                                       error:error];
      CMFormatDescriptionRef formatDescription;
      CMVideoFormatDescriptionCreateForImageBuffer(kCFAllocatorDefault, pixelBuffer,
                                                   &formatDescription);

      CMSampleBufferRef sampleBuffer;

      // This call takes ownership of the pixelBuffer. Docs are not very clear about this.
      CMSampleBufferCreateReadyWithImageBuffer(kCFAllocatorDefault, pixelBuffer, formatDescription,
                                               &timingInfo, &sampleBuffer);
      CFRelease(formatDescription);

      MPPImage *mpImage = [self initWithSampleBuffer:sampleBuffer
                                         orientation:sourceImage.orientation
                                               error:nil];

      // Can safely release here since CMSampleBuffer takes ownership of the pixelBuffer.
      CVPixelBufferRelease(pixelBuffer);
      CFRelease(sampleBuffer);
      return mpImage;
    }
    default:
      return nil;
  }
}

- (std::unique_ptr<ImageFrame>)imageFrameWithError:(NSError **)error {
  switch (self.imageSourceType) {
    case MPPImageSourceTypeSampleBuffer: {
      CVPixelBufferRef sampleImagePixelBuffer = CMSampleBufferGetImageBuffer(self.sampleBuffer);
      return [MPPCVPixelBufferUtils imageFrameFromCVPixelBuffer:sampleImagePixelBuffer error:error];
    }
    case MPPImageSourceTypePixelBuffer: {
      return [MPPCVPixelBufferUtils imageFrameFromCVPixelBuffer:self.pixelBuffer error:error];
    }
    case MPPImageSourceTypeImage: {
      return [self.image imageFrameWithError:error];
    }
    default:
      [MPPCommonUtils createCustomError:error
                               withCode:MPPTasksErrorCodeInvalidArgumentError
                            description:@"Invalid source type for MPPImage."];
  }

  return nullptr;
}

@end
