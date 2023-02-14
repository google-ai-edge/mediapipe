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

#import "mediapipe/tasks/ios/vision/core/sources/MPPImage.h"
#import "mediapipe/tasks/ios/common/sources/MPPCommon.h"
#import "mediapipe/tasks/ios/common/utils/sources/MPPCommonUtils.h"

NS_ASSUME_NONNULL_BEGIN

@implementation MPPImage

- (nullable instancetype)initWithUIImage:(UIImage *)image error:(NSError **)error {
  return [self initWithUIImage:image orientation:image.imageOrientation error:error];
}

- (nullable instancetype)initWithUIImage:(UIImage *)image
                             orientation:(UIImageOrientation)orientation
                                   error:(NSError **)error {
  if (image == nil) {
    [MPPCommonUtils createCustomError:error
                             withCode:MPPTasksErrorCodeInvalidArgumentError
                          description:@"Image cannot be nil."];
    return nil;
  }
  if (image.CGImage == NULL) {
    [MPPCommonUtils createCustomError:error
                             withCode:MPPTasksErrorCodeInvalidArgumentError
                          description:@"Image does not have a valid underlying CGImage. "
                                      @"image.CGImage must be non nil."];
    return nil;
  }

  self = [super init];
  if (self) {
    _imageSourceType = MPPImageSourceTypeImage;
    _orientation = orientation;
    _image = image;
    _width = image.size.width * image.scale;
    _height = image.size.height * image.scale;
  }
  return self;
}

- (nullable instancetype)initWithPixelBuffer:(CVPixelBufferRef)pixelBuffer error:(NSError **)error {
  return [self initWithPixelBuffer:pixelBuffer orientation:UIImageOrientationUp error:error];
}

- (nullable instancetype)initWithPixelBuffer:(CVPixelBufferRef)pixelBuffer
                                 orientation:(UIImageOrientation)orientation
                                       error:(NSError **)error {
  if (pixelBuffer == NULL) {
    [MPPCommonUtils createCustomError:error
                             withCode:MPPTasksErrorCodeInvalidArgumentError
                          description:@"Pixel Buffer cannot be nil."];
    return nil;
  }

  self = [super init];
  if (self != nil) {
    _imageSourceType = MPPImageSourceTypePixelBuffer;
    _orientation = orientation;
    CVPixelBufferRetain(pixelBuffer);
    _pixelBuffer = pixelBuffer;
    _width = CVPixelBufferGetWidth(pixelBuffer);
    _height = CVPixelBufferGetHeight(pixelBuffer);
  }
  return self;
}

- (nullable instancetype)initWithSampleBuffer:(CMSampleBufferRef)sampleBuffer
                                        error:(NSError **)error {
  return [self initWithSampleBuffer:sampleBuffer orientation:UIImageOrientationUp error:error];
}

- (nullable instancetype)initWithSampleBuffer:(CMSampleBufferRef)sampleBuffer
                                  orientation:(UIImageOrientation)orientation
                                        error:(NSError **)error {
  if (!CMSampleBufferIsValid(sampleBuffer)) {
    [MPPCommonUtils createCustomError:error
                             withCode:MPPTasksErrorCodeInvalidArgumentError
                          description:@"Sample buffer is not valid. Invoking "
                                      @"CMSampleBufferIsValid(sampleBuffer) must return true."];
    return nil;
  }

  CVImageBufferRef imageBuffer = CMSampleBufferGetImageBuffer(sampleBuffer);
  if (imageBuffer == NULL) {
    return nil;
  }

  self = [super init];
  if (self != nil) {
    _imageSourceType = MPPImageSourceTypeSampleBuffer;
    _orientation = orientation;
    CFRetain(sampleBuffer);
    _sampleBuffer = sampleBuffer;
    _width = CVPixelBufferGetWidth(imageBuffer);
    _height = CVPixelBufferGetHeight(imageBuffer);
  }
  return self;
}

- (void)dealloc {
  if (_sampleBuffer != NULL) {
    CFRelease(_sampleBuffer);
  }
  if (_pixelBuffer != NULL) {
    CVPixelBufferRelease(_pixelBuffer);
  }
}

@end

NS_ASSUME_NONNULL_END
