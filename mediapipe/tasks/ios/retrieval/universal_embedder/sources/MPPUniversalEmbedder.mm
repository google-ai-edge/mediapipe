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

#import "mediapipe/tasks/ios/common/sources/MPPCommon.h"
#import "mediapipe/tasks/ios/common/utils/sources/MPPCommonUtils.h"
#import "mediapipe/tasks/ios/components/utils/sources/MPPCosineSimilarity.h"
#import "mediapipe/tasks/ios/retrieval/semantic_retriever/sources/MPPAudioLoader.h"
#import "mediapipe/tasks/ios/vision/core/utils/sources/MPPImage+Utils.h"

#include <atomic>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "mediapipe/framework/timestamp.h"
#include "mediapipe/tasks/cc/components/containers/embedding_result.h"
#include "mediapipe/tasks/cc/core/base_options.h"
#include "mediapipe/tasks/cc/core/logging/factory/logging_factory.h"
#include "mediapipe/tasks/cc/core/logging/tasks_logger.h"
#include "mediapipe/tasks/cc/retrieval/universal_embedder/universal_embedder.h"

@interface MPPUniversalEmbedder () {
  std::unique_ptr<mediapipe::tasks::retrieval::universal_embedder::UniversalEmbedder>
      _universalEmbedder;
  std::unique_ptr<mediapipe::tasks::core::logging::TasksLogger> _statsLogger;
  std::atomic<int64_t> _syntheticTimestamp;
}

+ (MPPEmbeddingResult *)embeddingResultWithCppEmbeddingResult:
    (const mediapipe::tasks::components::containers::EmbeddingResult &)cppResult;

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
    auto cppOptions = std::make_unique<
        mediapipe::tasks::retrieval::universal_embedder::UniversalEmbedderOptions>();
    cppOptions->base_options.model_asset_path = options.baseOptions.modelAssetPath.UTF8String;
    if (options.baseOptions.delegate == MPPDelegateGPU) {
      cppOptions->base_options.delegate = mediapipe::tasks::core::BaseOptions::GPU;
    } else {
      cppOptions->base_options.delegate = mediapipe::tasks::core::BaseOptions::CPU;
    }
    cppOptions->l2_normalize = options.l2Normalize;
    if (options.maxInputLength > 0) {
      cppOptions->max_input_length = (int)options.maxInputLength;
    }
    if (options.visionTokensPerImage > 0) {
      cppOptions->vision_tokens_per_image = (int)options.visionTokensPerImage;
    }
    switch (options.activationDataType) {
      case MPPActivationDataTypeFloat32:
        cppOptions->activation_data_type =
            mediapipe::tasks::retrieval::universal_embedder::ActivationDataType::FLOAT32;
        break;
      case MPPActivationDataTypeFloat16:
        cppOptions->activation_data_type =
            mediapipe::tasks::retrieval::universal_embedder::ActivationDataType::FLOAT16;
        break;
      case MPPActivationDataTypeInt16:
        cppOptions->activation_data_type =
            mediapipe::tasks::retrieval::universal_embedder::ActivationDataType::INT16;
        break;
      case MPPActivationDataTypeInt8:
        cppOptions->activation_data_type =
            mediapipe::tasks::retrieval::universal_embedder::ActivationDataType::INT8;
        break;
      case MPPActivationDataTypeDefault:
        break;
    }
    if (options.cacheDir.length > 0) {
      cppOptions->cache_dir = std::string(options.cacheDir.UTF8String);
    }

    auto cppEmbedderStatus =
        mediapipe::tasks::retrieval::universal_embedder::UniversalEmbedder::Create(
            std::move(cppOptions));
    if (![MPPCommonUtils checkCppError:cppEmbedderStatus.status() toError:error]) {
      return nil;
    }
    _universalEmbedder = std::move(*cppEmbedderStatus);

    NSString *appId = [MPPCommonUtils appID];
    NSString *appVersion = [MPPCommonUtils appVersion];
    NSString *iosVersion = [MPPCommonUtils osVersion];

    mediapipe::tasks::core::logging::LoggingOptions loggingOptions = {
        .task_name = "UniversalEmbedder",
        .task_running_mode = mediapipe::tasks::core::RunningMode::kUnspecified,
        .host_environment = mediapipe::tasks::core::HostEnvironment::HOST_ENVIRONMENT_IOS,
        .host_system = mediapipe::tasks::core::HostSystem::HOST_SYSTEM_IOS,
        .host_version = iosVersion ? iosVersion.UTF8String : "",
        .app_id = appId ? appId.UTF8String : "",
        .app_version = appVersion ? appVersion.UTF8String : ""};

    _statsLogger = mediapipe::tasks::core::logging::CreateTasksLogger(loggingOptions);
    _statsLogger->LogSessionStart();
    _syntheticTimestamp = 0;
  }
  return self;
}

- (void)dealloc {
  if (_statsLogger) {
    _statsLogger->LogSessionEnd();
  }
}

- (nullable MPPEmbeddingResult *)embedText:(NSString *)text error:(NSError **)error {
  if (text == nil) {
    [MPPCommonUtils createCustomError:error
                             withCode:MPPTasksErrorCodeInvalidArgumentError
                          description:@"Text cannot be nil."];
    return nil;
  }
  int64_t timestampVal = _syntheticTimestamp++;
  mediapipe::Timestamp timestamp(timestampVal);
  _statsLogger->RecordCpuInputArrival(timestamp);

  auto cppResultStatus = _universalEmbedder->EmbedText(std::string(text.UTF8String));
  if (![MPPCommonUtils checkCppError:cppResultStatus.status() toError:error]) {
    _statsLogger->RecordInvocationEnd(timestamp);
    return nil;
  }

  _statsLogger->RecordInvocationEnd(timestamp);
  return [MPPUniversalEmbedder embeddingResultWithCppEmbeddingResult:*cppResultStatus];
}

- (nullable MPPEmbeddingResult *)embedImageBytes:(NSData *)imageBytes error:(NSError **)error {
  if (imageBytes == nil) {
    [MPPCommonUtils createCustomError:error
                             withCode:MPPTasksErrorCodeInvalidArgumentError
                          description:@"Image bytes cannot be nil."];
    return nil;
  }
  int64_t timestampVal = _syntheticTimestamp++;
  mediapipe::Timestamp timestamp(timestampVal);
  _statsLogger->RecordCpuInputArrival(timestamp);

  absl::string_view bytesView((const char *)imageBytes.bytes, imageBytes.length);
  auto cppResultStatus = _universalEmbedder->EmbedImage(bytesView);
  if (![MPPCommonUtils checkCppError:cppResultStatus.status() toError:error]) {
    _statsLogger->RecordInvocationEnd(timestamp);
    return nil;
  }

  _statsLogger->RecordInvocationEnd(timestamp);
  return [MPPUniversalEmbedder embeddingResultWithCppEmbeddingResult:*cppResultStatus];
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
    // Copy the entire pixel array in a single, lightning-fast memcpy block
    memcpy(dstBytes + 18, baseAddress, pixelSize);
  } else {
    // Fallback if the pixel buffer has alignment padding bytes at the end of each row
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

  // Fallback to UIImage compression
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
  int64_t timestampVal = _syntheticTimestamp++;
  mediapipe::Timestamp timestamp(timestampVal);
  _statsLogger->RecordCpuInputArrival(timestamp);

  std::vector<float> cppAudio(audioData.data, audioData.data + audioData.length);
  auto cppResultStatus = _universalEmbedder->EmbedAudio(cppAudio);
  if (![MPPCommonUtils checkCppError:cppResultStatus.status() toError:error]) {
    _statsLogger->RecordInvocationEnd(timestamp);
    return nil;
  }

  _statsLogger->RecordInvocationEnd(timestamp);
  return [MPPUniversalEmbedder embeddingResultWithCppEmbeddingResult:*cppResultStatus];
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
  int64_t timestampVal = _syntheticTimestamp++;
  mediapipe::Timestamp timestamp(timestampVal);
  _statsLogger->RecordCpuInputArrival(timestamp);

  std::vector<mediapipe::tasks::retrieval::universal_embedder::UniversalEmbedder::Part> cppParts;
  for (id part in content) {
    if ([part isKindOfClass:[NSString class]]) {
      NSString *text = (NSString *)part;
      cppParts.push_back(
          mediapipe::tasks::retrieval::universal_embedder::UniversalEmbedder::TextPart{
              .text = text.UTF8String});
    } else if ([part isKindOfClass:[MPPTextPart class]]) {
      NSString *text = ((MPPTextPart *)part).text;
      cppParts.push_back(
          mediapipe::tasks::retrieval::universal_embedder::UniversalEmbedder::TextPart{
              .text = text.UTF8String});
    } else if ([part isKindOfClass:[MPPImage class]]) {
      MPPImage *image = (MPPImage *)part;
      NSData *imageData = [self imageDataFromMPPImage:image error:error];
      if (imageData == nil) {
        _statsLogger->RecordInvocationEnd(timestamp);
        return nil;
      }
      cppParts.push_back(
          mediapipe::tasks::retrieval::universal_embedder::UniversalEmbedder::ImagePart{
              .image_bytes = std::string((const char *)imageData.bytes, imageData.length)});
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
          _statsLogger->RecordInvocationEnd(timestamp);
          return nil;
        }
      }
      cppParts.push_back(
          mediapipe::tasks::retrieval::universal_embedder::UniversalEmbedder::ImagePart{
              .image_bytes = std::string((const char *)imageData.bytes, imageData.length)});
    } else if ([part isKindOfClass:[MPPAudioData class]]) {
      MPPAudioData *audioData = (MPPAudioData *)part;
      MPPFloatBuffer *floatBuffer = audioData.buffer;
      std::vector<float> audioSamples(floatBuffer.data, floatBuffer.data + floatBuffer.length);
      cppParts.push_back(
          mediapipe::tasks::retrieval::universal_embedder::UniversalEmbedder::AudioPart{
              .audio_data = std::move(audioSamples)});
    } else if ([part isKindOfClass:[MPPAudioPart class]]) {
      MPPAudioPart *audioPart = (MPPAudioPart *)part;
      MPPAudioData *audioData = nil;
      if (audioPart.data != nil) {
        audioData = [MPPAudioLoader loadAudioFromData:audioPart.data error:error];
      } else {
        audioData = [MPPAudioLoader loadAudioFromFilePath:audioPart.filePath error:error];
      }
      if (audioData == nil) {
        _statsLogger->RecordInvocationEnd(timestamp);
        return nil;
      }
      MPPFloatBuffer *floatBuffer = audioData.buffer;
      std::vector<float> audioSamples(floatBuffer.data, floatBuffer.data + floatBuffer.length);
      cppParts.push_back(
          mediapipe::tasks::retrieval::universal_embedder::UniversalEmbedder::AudioPart{
              .audio_data = std::move(audioSamples)});
    } else {
      [MPPCommonUtils createCustomError:error
                               withCode:MPPTasksErrorCodeInvalidArgumentError
                            description:@"Unsupported part type in content list."];
      _statsLogger->RecordInvocationEnd(timestamp);
      return nil;
    }
  }

  auto cppResultStatus = _universalEmbedder->EmbedContent(cppParts);
  if (![MPPCommonUtils checkCppError:cppResultStatus.status() toError:error]) {
    _statsLogger->RecordInvocationEnd(timestamp);
    return nil;
  }

  _statsLogger->RecordInvocationEnd(timestamp);
  return [MPPUniversalEmbedder embeddingResultWithCppEmbeddingResult:*cppResultStatus];
}

+ (nullable NSNumber *)cosineSimilarityBetweenEmbedding1:(MPPEmbedding *)embedding1
                                           andEmbedding2:(MPPEmbedding *)embedding2
                                                   error:(NSError **)error {
  return [MPPCosineSimilarity computeBetweenEmbedding1:embedding1
                                         andEmbedding2:embedding2
                                                 error:error];
}

+ (MPPEmbeddingResult *)embeddingResultWithCppEmbeddingResult:
    (const mediapipe::tasks::components::containers::EmbeddingResult &)cppResult {
  NSMutableArray<MPPEmbedding *> *embeddings =
      [NSMutableArray arrayWithCapacity:cppResult.embeddings.size()];
  for (const auto &cppEmbedding : cppResult.embeddings) {
    NSMutableArray<NSNumber *> *floatEmbedding = nil;
    if (!cppEmbedding.float_embedding.empty()) {
      floatEmbedding = [NSMutableArray arrayWithCapacity:cppEmbedding.float_embedding.size()];
      for (float val : cppEmbedding.float_embedding) {
        [floatEmbedding addObject:@(val)];
      }
    }
    NSMutableArray<NSNumber *> *quantizedEmbedding = nil;
    if (!cppEmbedding.quantized_embedding.empty()) {
      quantizedEmbedding =
          [NSMutableArray arrayWithCapacity:cppEmbedding.quantized_embedding.size()];
      for (size_t i = 0; i < cppEmbedding.quantized_embedding.size(); ++i) {
        [quantizedEmbedding addObject:@((uint8_t)cppEmbedding.quantized_embedding[i])];
      }
    }
    NSString *headName = cppEmbedding.head_name.has_value()
                             ? [NSString stringWithUTF8String:cppEmbedding.head_name->c_str()]
                             : nil;
    MPPEmbedding *embedding = [[MPPEmbedding alloc] initWithFloatEmbedding:floatEmbedding
                                                        quantizedEmbedding:quantizedEmbedding
                                                                 headIndex:cppEmbedding.head_index
                                                                  headName:headName];
    [embeddings addObject:embedding];
  }
  NSInteger timestampMs =
      cppResult.timestamp_ms.has_value() ? (NSInteger)*cppResult.timestamp_ms : 0;
  return [[MPPEmbeddingResult alloc] initWithEmbeddings:embeddings
                                timestampInMilliseconds:timestampMs];
}

@end
