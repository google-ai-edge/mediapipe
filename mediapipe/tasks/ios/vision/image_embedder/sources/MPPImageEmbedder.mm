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

#import "mediapipe/tasks/ios/vision/image_embedder/sources/MPPImageEmbedder.h"

#import <UIKit/UIKit.h>

#import "mediapipe/tasks/ios/common/sources/MPPCommon.h"
#import "mediapipe/tasks/ios/common/utils/sources/MPPCommonUtils.h"
#import "mediapipe/tasks/ios/components/utils/sources/MPPCosineSimilarity.h"
#import "mediapipe/tasks/ios/core/sources/MPPTaskPart.h"
#import "mediapipe/tasks/ios/vision/core/sources/MPPVisionTaskRunner.h"
#import "mediapipe/tasks/ios/vision/image_embedder/utils/sources/MPPImageEmbedderOptions+Helpers.h"
#import "mediapipe/tasks/ios/vision/image_embedder/utils/sources/MPPImageEmbedderResult+Helpers.h"

#include "mediapipe/tasks/c/vision/core/image_processing_options.h"
#include "mediapipe/tasks/c/vision/image_embedder/image_embedder.h"

static NSString *const kTaskName = @"imageEmbedder";

@interface MPPImageEmbedder () {
  MpImageEmbedderPtr _embedder;
  MPPVisionTaskRunner *_taskRunner;
  dispatch_queue_t _callbackQueue;
  int _callbackSlot;
}
@property(nonatomic, weak) id<MPPImageEmbedderLiveStreamDelegate> imageEmbedderLiveStreamDelegate;

- (void)handleLiveStreamResult:(const ImageEmbedderResult *)cResult
                        status:(MpStatus)status
                     timestamp:(int64_t)timestampInMilliseconds;
@end

namespace {
using mediapipe::tasks::ios::vision::core::VisionCallbackPool;
using CallbackPool = VisionCallbackPool<MPPImageEmbedder, ImageEmbedderResult,
                                        ImageEmbedderOptions::result_callback_fn>;

static void OnLiveStreamResult(MPPImageEmbedder *task, MpStatus status,
                               const ImageEmbedderResult *result, MpImagePtr image,
                               int64_t timestamp_ms) {
  if (task) {
    [task handleLiveStreamResult:result status:status timestamp:timestamp_ms];
  }
}
}  // namespace

@implementation MPPImageEmbedder

#pragma mark - Public

- (instancetype)initWithOptions:(MPPImageEmbedderOptions *)options error:(NSError **)error {
  self = [super init];
  if (self) {
    _callbackSlot = -1;
    _taskRunner = [[MPPVisionTaskRunner alloc]
        initWithRunningMode:options.runningMode
                 roiAllowed:YES
            normRectAllowed:YES
                hasDelegate:(options.imageEmbedderLiveStreamDelegate != nil)
                      error:error];
    if (!_taskRunner) {
      return nil;
    }

    if (options.imageEmbedderLiveStreamDelegate) {
      _imageEmbedderLiveStreamDelegate = options.imageEmbedderLiveStreamDelegate;
      _callbackQueue = dispatch_queue_create(
          [MPPVisionTaskRunner uniqueDispatchQueueNameWithSuffix:kTaskName], nullptr);
      _callbackSlot = CallbackPool::AllocateSlot(self, &OnLiveStreamResult);
      if (_callbackSlot < 0) {
        [MPPCommonUtils createCustomError:error
                                 withCode:MPPTasksErrorCodeInternalError
                              description:@"Failed to allocate callback slot."];
        return nil;
      }
    }

    ImageEmbedderOptions cOptions = {};
    [options copyToCOptions:&cOptions];
    if (_callbackSlot >= 0) {
      cOptions.result_callback = CallbackPool::GetCallback(_callbackSlot);
    }

    char *errorMsg = nullptr;
    MpStatus status = MpImageEmbedderCreate(&cOptions, &_embedder, &errorMsg);
    [MPPImageEmbedderOptions freeCOptions:&cOptions];

    if (![MPPCommonUtils checkCError:status errorMessage:errorMsg toError:error]) {
      if (_callbackSlot >= 0) {
        CallbackPool::FreeSlot(_callbackSlot);
        _callbackSlot = -1;
      }
      return nil;
    }
  }
  return self;
}

- (instancetype)initWithModelPath:(NSString *)modelPath error:(NSError **)error {
  MPPImageEmbedderOptions *options = [[MPPImageEmbedderOptions alloc] init];
  options.baseOptions.modelAssetPath = modelPath;
  return [self initWithOptions:options error:error];
}

- (nullable MPPImageEmbedderResult *)embedImage:(MPPImage *)image
                               regionOfInterest:(CGRect)roi
                                          error:(NSError **)error {
  if (![_taskRunner validateRunningMode:MPPRunningModeImage error:error]) {
    return nil;
  }
  struct MpImageProcessingOptions imageProcessingOptions = {};
  ScopedMpImage mpImage = [_taskRunner createMpImageWithMPPImage:image
                                                regionOfInterest:roi
                                          imageProcessingOptions:&imageProcessingOptions
                                                           error:error];
  if (!mpImage) {
    return nil;
  }
  ImageEmbedderResult cResult = {};
  char *errorMsg = nullptr;
  MpStatus status =
      MpImageEmbedderEmbedImage(_embedder, mpImage, &imageProcessingOptions, &cResult, &errorMsg);
  if (![MPPCommonUtils checkCError:status errorMessage:errorMsg toError:error]) {
    return nil;
  }
  MPPImageEmbedderResult *result = [MPPImageEmbedderResult imageEmbedderResultWithCResult:cResult
                                                                  timestampInMilliseconds:0];
  MpImageEmbedderCloseResult(&cResult);
  return result;
}

- (nullable MPPImageEmbedderResult *)embedImage:(MPPImage *)image error:(NSError **)error {
  return [self embedImage:image regionOfInterest:CGRectZero error:error];
}

- (nullable MPPImageEmbedderResult *)embedVideoFrame:(MPPImage *)image
                             timestampInMilliseconds:(NSInteger)timestampInMilliseconds
                                    regionOfInterest:(CGRect)roi
                                               error:(NSError **)error {
  if (![_taskRunner validateRunningMode:MPPRunningModeVideo error:error]) {
    return nil;
  }
  struct MpImageProcessingOptions imageProcessingOptions = {};
  ScopedMpImage mpImage = [_taskRunner createMpImageWithMPPImage:image
                                                regionOfInterest:roi
                                          imageProcessingOptions:&imageProcessingOptions
                                                           error:error];
  if (!mpImage) {
    return nil;
  }
  ImageEmbedderResult cResult = {};
  char *errorMsg = nullptr;
  MpStatus status = MpImageEmbedderEmbedForVideo(_embedder, mpImage, &imageProcessingOptions,
                                                 static_cast<int64_t>(timestampInMilliseconds),
                                                 &cResult, &errorMsg);
  if (![MPPCommonUtils checkCError:status errorMessage:errorMsg toError:error]) {
    return nil;
  }
  MPPImageEmbedderResult *result =
      [MPPImageEmbedderResult imageEmbedderResultWithCResult:cResult
                                     timestampInMilliseconds:timestampInMilliseconds];
  MpImageEmbedderCloseResult(&cResult);
  return result;
}

- (nullable MPPImageEmbedderResult *)embedVideoFrame:(MPPImage *)image
                             timestampInMilliseconds:(NSInteger)timestampInMilliseconds
                                               error:(NSError **)error {
  return [self embedVideoFrame:image
       timestampInMilliseconds:timestampInMilliseconds
              regionOfInterest:CGRectZero
                         error:error];
}

- (BOOL)embedAsyncImage:(MPPImage *)image
    timestampInMilliseconds:(NSInteger)timestampInMilliseconds
           regionOfInterest:(CGRect)roi
                      error:(NSError **)error {
  if (![_taskRunner validateRunningMode:MPPRunningModeLiveStream error:error]) {
    return NO;
  }
  struct MpImageProcessingOptions imageProcessingOptions = {};
  ScopedMpImage mpImage = [_taskRunner createMpImageWithMPPImage:image
                                                regionOfInterest:roi
                                          imageProcessingOptions:&imageProcessingOptions
                                                           error:error];
  if (!mpImage) {
    return NO;
  }
  char *errorMsg = nullptr;
  MpStatus status =
      MpImageEmbedderEmbedAsync(_embedder, mpImage, &imageProcessingOptions,
                                static_cast<int64_t>(timestampInMilliseconds), &errorMsg);
  return [MPPCommonUtils checkCError:status errorMessage:errorMsg toError:error];
}

- (BOOL)embedAsyncImage:(MPPImage *)image
    timestampInMilliseconds:(NSInteger)timestampInMilliseconds
                      error:(NSError **)error {
  return [self embedAsyncImage:image
       timestampInMilliseconds:timestampInMilliseconds
              regionOfInterest:CGRectZero
                         error:error];
}

- (void)dealloc {
  MpImageEmbedderPtr embedder = _embedder;
  _embedder = nullptr;
  int callbackSlot = _callbackSlot;
  _callbackSlot = -1;
  dispatch_queue_t callbackQueue = _callbackQueue;
  _callbackQueue = nil;

  if (callbackSlot >= 0) {
    dispatch_queue_t cleanupQueue =
        callbackQueue ?: dispatch_get_global_queue(QOS_CLASS_DEFAULT, 0);
    dispatch_async(cleanupQueue, ^{
      if (embedder) {
        MpImageEmbedderClose(embedder, nullptr);
      }
      CallbackPool::FreeSlot(callbackSlot);
    });
  } else if (embedder) {
    MpImageEmbedderClose(embedder, nullptr);
  }
}

+ (nullable NSNumber *)cosineSimilarityBetweenEmbedding1:(MPPEmbedding *)embedding1
                                           andEmbedding2:(MPPEmbedding *)embedding2
                                                   error:(NSError **)error {
  return [MPPCosineSimilarity computeBetweenEmbedding1:embedding1
                                         andEmbedding2:embedding2
                                                 error:error];
}

- (nullable MPPEmbeddingResult *)embedContent:(NSArray<id> *)content error:(NSError **)error {
  for (id part in content) {
    if ([part isKindOfClass:[MPPImage class]]) {
      MPPImageEmbedderResult *result = [self embedImage:(MPPImage *)part error:error];
      if (result == nil) {
        return nil;
      }
      return result.embeddingResult;
    } else if ([part isKindOfClass:[MPPImagePart class]]) {
      NSString *filePath = ((MPPImagePart *)part).filePath;
      NSData *imageBytes = [NSData dataWithContentsOfFile:filePath];
      if (!imageBytes) {
        [MPPCommonUtils
            createCustomError:error
                     withCode:MPPTasksErrorCodeInvalidArgumentError
                  description:[NSString stringWithFormat:@"Failed to load image from path: %@",
                                                         filePath]];
        return nil;
      }
      UIImage *uiImage = [UIImage imageWithData:imageBytes];
      if (!uiImage) {
        [MPPCommonUtils
            createCustomError:error
                     withCode:MPPTasksErrorCodeInvalidArgumentError
                  description:[NSString stringWithFormat:@"Failed to decode image from path: %@",
                                                         filePath]];
        return nil;
      }
      MPPImage *mppImage = [[MPPImage alloc] initWithUIImage:uiImage error:error];
      if (!mppImage) {
        return nil;
      }
      MPPImageEmbedderResult *result = [self embedImage:mppImage error:error];
      if (result == nil) {
        return nil;
      }
      return result.embeddingResult;
    }
  }
  [MPPCommonUtils
      createCustomError:error
               withCode:MPPTasksErrorCodeInvalidArgumentError
            description:@"Content is empty or does not contain any supported image elements."];
  return nil;
}

#pragma mark - Private

- (void)handleLiveStreamResult:(const ImageEmbedderResult *)cResult
                        status:(MpStatus)status
                     timestamp:(int64_t)timestampInMilliseconds {
  if (![self.imageEmbedderLiveStreamDelegate
          respondsToSelector:@selector(imageEmbedder:didFinishEmbeddingWithResult:
                                       timestampInMilliseconds:error:)]) {
    return;
  }

  NSError *callbackError = nil;
  MPPImageEmbedderResult *result = nil;

  if (status != kMpOk) {
    [MPPCommonUtils checkCError:status errorMessage:nullptr toError:&callbackError];
  } else if (cResult) {
    result = [MPPImageEmbedderResult imageEmbedderResultWithCResult:*cResult
                                            timestampInMilliseconds:timestampInMilliseconds];
  }

  dispatch_queue_t callbackQueue = _callbackQueue;
  if (!callbackQueue) {
    return;
  }
  __weak id<MPPImageEmbedderLiveStreamDelegate> delegate = self.imageEmbedderLiveStreamDelegate;
  MPPImageEmbedder *__weak weakSelf = self;

  dispatch_async(callbackQueue, ^{
    MPPImageEmbedder *strongSelf = weakSelf;
    if (strongSelf) {
      [delegate imageEmbedder:strongSelf
          didFinishEmbeddingWithResult:result
               timestampInMilliseconds:timestampInMilliseconds
                                 error:callbackError];
    }
  });
}

@end
