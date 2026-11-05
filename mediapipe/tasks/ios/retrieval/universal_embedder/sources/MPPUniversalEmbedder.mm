// Copyright 2026 The MediaPipe Authors. All Rights Reserved.
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

#import "mediapipe/tasks/ios/retrieval/universal_embedder/sources/MPPUniversalEmbedder.h"

#import <CoreMedia/CoreMedia.h>
#import <CoreVideo/CoreVideo.h>
#import <UIKit/UIKit.h>

#import "mediapipe/tasks/ios/audio/core/sources/MPPAudioData.h"
#import "mediapipe/tasks/ios/audio/core/sources/MPPFloatBuffer.h"
#import "mediapipe/tasks/ios/common/sources/MPPCommon.h"
#import "mediapipe/tasks/ios/common/utils/sources/MPPCommonUtils.h"
#import "mediapipe/tasks/ios/components/containers/sources/MPPEmbeddingResult.h"
#import "mediapipe/tasks/ios/components/containers/utils/sources/MPPEmbeddingResult+Helpers.h"
#import "mediapipe/tasks/ios/components/utils/sources/MPPCosineSimilarity.h"
#import "mediapipe/tasks/ios/core/sources/MPPTaskPart.h"
#import "mediapipe/tasks/ios/core/utils/sources/MPPBaseOptions+Helpers.h"
#import "mediapipe/tasks/ios/retrieval/semantic_retriever/sources/MPPAudioLoader.h"
#import "mediapipe/tasks/ios/vision/core/utils/sources/MPPImage+Utils.h"

#include <vector>

#include "mediapipe/tasks/c/retrieval/universal_embedder/universal_embedder.h"

namespace {

MpDelegate CDelegateFromMPPDelegate(MPPDelegate delegate) {
  switch (delegate) {
    case MPPDelegateGPU:
      return MP_DELEGATE_GPU;
    case MPPDelegateCPU:
      return MP_DELEGATE_CPU;
    default:
      return MP_DELEGATE_DEFAULT;
  }
}

MpActivationDataType CActivationFromMPPActivation(MPPActivationDataType activation) {
  switch (activation) {
    case MPPActivationDataTypeFloat32:
      return kMpActivationDataTypeFloat32;
    case MPPActivationDataTypeFloat16:
      return kMpActivationDataTypeFloat16;
    case MPPActivationDataTypeInt16:
      return kMpActivationDataTypeInt16;
    case MPPActivationDataTypeInt8:
      return kMpActivationDataTypeInt8;
    case MPPActivationDataTypeDefault:
    default:
      return kMpActivationDataTypeDefault;
  }
}

}  // namespace

@interface MPPUniversalEmbedder () {
  MpUniversalEmbedderPtr _universalEmbedder;
}

- (nullable NSData *)uncompressedTGADataFromPixelBuffer:(CVPixelBufferRef)pixelBuffer
                                                  error:(NSError **)error;
- (nullable NSData *)imageDataFromMPPImage:(MPPImage *)image error:(NSError **)error;

@end

@implementation MPPUniversalEmbedder

- (nullable instancetype)initWithModelPath:(NSString *)modelPath error:(NSError **)error {
  MPPUniversalEmbedderOptions *options = [[MPPUniversalEmbedderOptions alloc] init];
  options.baseOptions.modelAssetPath = modelPath;
  return [self initWithOptions:options error:error];
}

- (nullable instancetype)initWithOptions:(MPPUniversalEmbedderOptions *)options
                                   error:(NSError **)error {
  self = [super init];
  if (self) {
    MpUniversalEmbedderOptions cOptions = {};
    [options.baseOptions copyToCOptions:&cOptions.base_options];
    cOptions.l2_normalize = options.l2Normalize;
    cOptions.max_input_length = static_cast<int>(options.maxInputLength);
    cOptions.vision_tokens_per_image = static_cast<int>(options.visionTokensPerImage);
    cOptions.activation_data_type = CActivationFromMPPActivation(options.activationDataType);
    cOptions.cache_dir = options.cacheDir.length > 0 ? options.cacheDir.UTF8String : nullptr;
    cOptions.text_delegate = CDelegateFromMPPDelegate(options.textDelegate == MPPDelegateDefault
                                                          ? options.baseOptions.delegate
                                                          : options.textDelegate);
    cOptions.vision_delegate = CDelegateFromMPPDelegate(options.visionDelegate == MPPDelegateDefault
                                                            ? options.baseOptions.delegate
                                                            : options.visionDelegate);
    cOptions.audio_delegate = CDelegateFromMPPDelegate(options.audioDelegate == MPPDelegateDefault
                                                           ? options.baseOptions.delegate
                                                           : options.audioDelegate);

    char *cError = nullptr;
    MpStatus status = MpUniversalEmbedderCreate(&cOptions, &_universalEmbedder, &cError);
    [MPPBaseOptions freeCOptions:&cOptions.base_options];
    if (![MPPCommonUtils checkCError:status errorMessage:cError toError:error]) {
      return nil;
    }
  }
  return self;
}

- (void)dealloc {
  if (_universalEmbedder) {
    MpUniversalEmbedderClose(_universalEmbedder, nullptr);
    _universalEmbedder = nullptr;
  }
}

- (nullable MPPEmbeddingResult *)embedText:(NSString *)text error:(NSError **)error {
  if (text == nil) {
    [MPPCommonUtils createCustomError:error
                             withCode:MPPTasksErrorCodeInvalidArgumentError
                          description:@"Text cannot be nil."];
    return nil;
  }

  MpUniversalEmbedderResult cResult = {};
  char *cError = nullptr;
  MpStatus status =
      MpUniversalEmbedderEmbedText(_universalEmbedder, text.UTF8String, &cResult, &cError);
  if (![MPPCommonUtils checkCError:status errorMessage:cError toError:error]) {
    return nil;
  }

  MPPEmbeddingResult *result = [MPPEmbeddingResult embeddingResultWithCResult:cResult];
  MpUniversalEmbedderCloseResult(&cResult);
  return result;
}

- (nullable MPPEmbeddingResult *)embedImageBytes:(NSData *)imageBytes error:(NSError **)error {
  if (imageBytes == nil) {
    [MPPCommonUtils createCustomError:error
                             withCode:MPPTasksErrorCodeInvalidArgumentError
                          description:@"Image bytes cannot be nil."];
    return nil;
  }

  MpUniversalEmbedderResult cResult = {};
  char *cError = nullptr;
  MpStatus status =
      MpUniversalEmbedderEmbedImage(_universalEmbedder, (const char *)imageBytes.bytes,
                                    (int)imageBytes.length, &cResult, &cError);
  if (![MPPCommonUtils checkCError:status errorMessage:cError toError:error]) {
    return nil;
  }

  MPPEmbeddingResult *result = [MPPEmbeddingResult embeddingResultWithCResult:cResult];
  MpUniversalEmbedderCloseResult(&cResult);
  return result;
}

- (nullable NSData *)uncompressedTGADataFromPixelBuffer:(CVPixelBufferRef)pixelBuffer
                                                  error:(NSError **)error {
  CVPixelBufferLockBaseAddress(pixelBuffer, kCVPixelBufferLock_ReadOnly);

  uint8_t *baseAddress = (uint8_t *)CVPixelBufferGetBaseAddress(pixelBuffer);
  const size_t width = CVPixelBufferGetWidth(pixelBuffer);
  const size_t height = CVPixelBufferGetHeight(pixelBuffer);
  const size_t bytesPerRow = CVPixelBufferGetBytesPerRow(pixelBuffer);
  const size_t rowSize = width * 4;  // 4 bytes per pixel (BGRA)
  const size_t pixelSize = width * height * 4;

  // Initialize TGA Header
  uint8_t header[18] = {0};
  header[2] = 2;              // Uncompressed True-Color Image
  header[12] = width & 0xFF;  // Width LSB
  header[13] = (width >> 8) & 0xFF;
  header[14] = height & 0xFF;  // Height LSB
  header[15] = (height >> 8) & 0xFF;
  header[16] = 32;    // 32-bits per pixel (BGRA)
  header[17] = 0x28;  // Top-Left origin flag (0x20) + 8-bit alpha (0x08)

  // Allocate a single contiguous block of memory for both header and pixels
  NSMutableData *tgaData = [NSMutableData dataWithLength:18 + pixelSize];
  uint8_t *dstBytes = (uint8_t *)tgaData.mutableBytes;

  // Copy the 18-byte header
  memcpy(dstBytes, header, 18);

  // Fast Contiguous Copy Check
  if (bytesPerRow == rowSize) {
    memcpy(dstBytes + 18, baseAddress, pixelSize);
  } else {
    uint8_t *dstPixels = dstBytes + 18;
    for (size_t y = 0; y < height; y++) {
      memcpy(dstPixels + y * rowSize, baseAddress + y * bytesPerRow, rowSize);
    }
  }

  CVPixelBufferUnlockBaseAddress(pixelBuffer, kCVPixelBufferLock_ReadOnly);
  return tgaData;
}

- (nullable NSData *)imageDataFromMPPImage:(MPPImage *)image error:(NSError **)error {
  if (image.imageSourceType == MPPImageSourceTypePixelBuffer) {
    return [self uncompressedTGADataFromPixelBuffer:image.pixelBuffer error:error];
  } else if (image.imageSourceType == MPPImageSourceTypeSampleBuffer) {
    CVImageBufferRef imageBuffer = CMSampleBufferGetImageBuffer(image.sampleBuffer);
    if (imageBuffer != nullptr) {
      return [self uncompressedTGADataFromPixelBuffer:imageBuffer error:error];
    }
  }

  UIImage *uiImage = [image toUIImageWithError:error];
  if (uiImage == nil) {
    return nil;
  }
  NSData *imageData = UIImageJPEGRepresentation(uiImage, 0.9);
  if (imageData == nil) {
    imageData = UIImagePNGRepresentation(uiImage);
  }
  if (imageData == nil) {
    [MPPCommonUtils createCustomError:error
                             withCode:MPPTasksErrorCodeInvalidArgumentError
                          description:@"Failed to compress UIImage to JPEG/PNG."];
    return nil;
  }
  return imageData;
}

- (nullable MPPEmbeddingResult *)embedImage:(MPPImage *)image error:(NSError **)error {
  if (image == nil) {
    [MPPCommonUtils createCustomError:error
                             withCode:MPPTasksErrorCodeInvalidArgumentError
                          description:@"Image cannot be nil."];
    return nil;
  }
  NSData *imageData = [self imageDataFromMPPImage:image error:error];
  if (imageData == nil) {
    return nil;
  }
  return [self embedImageBytes:imageData error:error];
}

- (nullable MPPEmbeddingResult *)embedAudioFloatBuffer:(MPPFloatBuffer *)audioData
                                                 error:(NSError **)error {
  if (audioData == nil) {
    [MPPCommonUtils createCustomError:error
                             withCode:MPPTasksErrorCodeInvalidArgumentError
                          description:@"Audio data cannot be nil."];
    return nil;
  }

  MpUniversalEmbedderResult cResult = {};
  char *cError = nullptr;
  MpStatus status = MpUniversalEmbedderEmbedAudio(
      _universalEmbedder, audioData.data, static_cast<int>(audioData.length), &cResult, &cError);
  if (![MPPCommonUtils checkCError:status errorMessage:cError toError:error]) {
    return nil;
  }

  MPPEmbeddingResult *result = [MPPEmbeddingResult embeddingResultWithCResult:cResult];
  MpUniversalEmbedderCloseResult(&cResult);
  return result;
}

- (nullable MPPEmbeddingResult *)embedAudio:(MPPAudioData *)audioData error:(NSError **)error {
  if (audioData == nil) {
    [MPPCommonUtils createCustomError:error
                             withCode:MPPTasksErrorCodeInvalidArgumentError
                          description:@"Audio data cannot be nil."];
    return nil;
  }
  return [self embedAudioFloatBuffer:audioData.buffer error:error];
}

- (nullable MPPEmbeddingResult *)embedContent:(NSArray<id> *)content error:(NSError **)error {
  if (content == nil) {
    [MPPCommonUtils createCustomError:error
                             withCode:MPPTasksErrorCodeInvalidArgumentError
                          description:@"Content cannot be nil."];
    return nil;
  }

  std::vector<MpUniversalEmbedderPart> cParts;
  cParts.reserve(content.count);
  NS_VALID_UNTIL_END_OF_SCOPE NSMutableArray<id> *retainedData = [NSMutableArray array];

  for (id part in content) {
    MpUniversalEmbedderPart cPart = {};
    if ([part isKindOfClass:[NSString class]]) {
      NSString *text = (NSString *)part;
      [retainedData addObject:text];
      cPart.type = kMpUniversalEmbedderPartTypeText;
      cPart.value.text = text.UTF8String;
    } else if ([part isKindOfClass:[MPPTextPart class]]) {
      NSString *text = ((MPPTextPart *)part).text;
      if (text == nil) {
        [MPPCommonUtils createCustomError:error
                                 withCode:MPPTasksErrorCodeInvalidArgumentError
                              description:@"Text part cannot be nil."];
        return nil;
      }
      [retainedData addObject:text];
      cPart.type = kMpUniversalEmbedderPartTypeText;
      cPart.value.text = text.UTF8String;
    } else if ([part isKindOfClass:[MPPImage class]]) {
      MPPImage *image = (MPPImage *)part;
      NSData *imageData = [self imageDataFromMPPImage:image error:error];
      if (imageData == nil) {
        return nil;
      }
      [retainedData addObject:imageData];
      cPart.type = kMpUniversalEmbedderPartTypeImage;
      cPart.value.image.bytes = (const char *)imageData.bytes;
      cPart.value.image.size = (int)imageData.length;
    } else if ([part isKindOfClass:[MPPImagePart class]]) {
      MPPImagePart *imagePart = (MPPImagePart *)part;
      NSData *imageData = imagePart.data;
      if (imageData == nil || imageData.length == 0) {
        NSString *filePath = imagePart.filePath;
        NSError *loadError = nil;
        imageData = [NSData dataWithContentsOfFile:filePath options:0 error:&loadError];
        if (imageData == nil) {
          if (error) {
            *error = loadError;
          }
          return nil;
        }
      }
      [retainedData addObject:imageData];
      cPart.type = kMpUniversalEmbedderPartTypeImage;
      cPart.value.image.bytes = (const char *)imageData.bytes;
      cPart.value.image.size = (int)imageData.length;
    } else if ([part isKindOfClass:[MPPAudioData class]]) {
      MPPAudioData *audioData = (MPPAudioData *)part;
      MPPFloatBuffer *floatBuffer = audioData.buffer;
      if (floatBuffer == nil) {
        [MPPCommonUtils createCustomError:error
                                 withCode:MPPTasksErrorCodeInvalidArgumentError
                              description:@"Audio data buffer cannot be nil."];
        return nil;
      }
      [retainedData addObject:floatBuffer];
      cPart.type = kMpUniversalEmbedderPartTypeAudio;
      cPart.value.audio.samples = floatBuffer.data;
      cPart.value.audio.size = (int)floatBuffer.length;
    } else if ([part isKindOfClass:[MPPAudioPart class]]) {
      MPPAudioPart *audioPart = (MPPAudioPart *)part;
      MPPAudioData *audioData = nil;
      if (audioPart.data != nil) {
        audioData = [MPPAudioLoader loadAudioFromData:audioPart.data error:error];
      } else {
        audioData = [MPPAudioLoader loadAudioFromFilePath:audioPart.filePath error:error];
      }
      if (audioData == nil) {
        return nil;
      }
      MPPFloatBuffer *floatBuffer = audioData.buffer;
      if (floatBuffer == nil) {
        [MPPCommonUtils createCustomError:error
                                 withCode:MPPTasksErrorCodeInvalidArgumentError
                              description:@"Audio data buffer cannot be nil."];
        return nil;
      }
      [retainedData addObject:floatBuffer];
      cPart.type = kMpUniversalEmbedderPartTypeAudio;
      cPart.value.audio.samples = floatBuffer.data;
      cPart.value.audio.size = (int)floatBuffer.length;
    } else {
      [MPPCommonUtils createCustomError:error
                               withCode:MPPTasksErrorCodeInvalidArgumentError
                            description:@"Unsupported part type in content list."];
      return nil;
    }
    cParts.push_back(cPart);
  }

  MpUniversalEmbedderResult cResult = {};
  char *cError = nullptr;
  MpStatus status = MpUniversalEmbedderEmbedContent(_universalEmbedder, cParts.data(),
                                                    (int)cParts.size(), &cResult, &cError);
  if (![MPPCommonUtils checkCError:status errorMessage:cError toError:error]) {
    return nil;
  }

  MPPEmbeddingResult *result = [MPPEmbeddingResult embeddingResultWithCResult:cResult];
  MpUniversalEmbedderCloseResult(&cResult);
  return result;
}

+ (nullable NSNumber *)cosineSimilarityBetweenEmbedding1:(MPPEmbedding *)embedding1
                                           andEmbedding2:(MPPEmbedding *)embedding2
                                                   error:(NSError **)error {
  return [MPPCosineSimilarity computeBetweenEmbedding1:embedding1
                                         andEmbedding2:embedding2
                                                 error:error];
}

@end
