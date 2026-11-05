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

#import "mediapipe/tasks/ios/vision/image_classifier/sources/MPPImageClassifier.h"

#import "mediapipe/tasks/ios/common/sources/MPPCommon.h"
#import "mediapipe/tasks/ios/common/utils/sources/MPPCommonUtils.h"
#import "mediapipe/tasks/ios/vision/core/sources/MPPVisionTaskRunner.h"
#import "mediapipe/tasks/ios/vision/image_classifier/utils/sources/MPPImageClassifierOptions+Helpers.h"
#import "mediapipe/tasks/ios/vision/image_classifier/utils/sources/MPPImageClassifierResult+Helpers.h"

#include "mediapipe/tasks/c/vision/core/image_processing_options.h"
#include "mediapipe/tasks/c/vision/image_classifier/image_classifier.h"

static NSString *const kTaskName = @"imageClassifier";

@interface MPPImageClassifier () {
  MpImageClassifierPtr _classifier;
  MPPVisionTaskRunner *_taskRunner;
  dispatch_queue_t _callbackQueue;
  int _callbackSlot;
}
@property(nonatomic, weak) id<MPPImageClassifierLiveStreamDelegate>
    imageClassifierLiveStreamDelegate;

- (void)handleLiveStreamResult:(const MpImageClassifierResult *)cResult
                        status:(MpStatus)status
                     timestamp:(int64_t)timestampInMilliseconds;

@end

namespace {
using mediapipe::tasks::ios::vision::core::VisionCallbackPool;
using CallbackPool = VisionCallbackPool<MPPImageClassifier, MpImageClassifierResult,
                                        MpImageClassifierOptions::result_callback_fn>;

static void OnLiveStreamResult(MPPImageClassifier *task, MpStatus status,
                               const MpImageClassifierResult *result, MpImagePtr image,
                               int64_t timestamp_ms) {
  if (task) {
    [task handleLiveStreamResult:result status:status timestamp:timestamp_ms];
  }
}
}  // namespace

@implementation MPPImageClassifier

#pragma mark - Public

- (instancetype)initWithOptions:(MPPImageClassifierOptions *)options error:(NSError **)error {
  self = [super init];
  if (self) {
    _callbackSlot = -1;
    _taskRunner = [[MPPVisionTaskRunner alloc]
        initWithRunningMode:options.runningMode
                 roiAllowed:YES
            normRectAllowed:YES
                hasDelegate:(options.imageClassifierLiveStreamDelegate != nil)
                      error:error];
    if (!_taskRunner) {
      return nil;
    }

    if (options.imageClassifierLiveStreamDelegate) {
      _imageClassifierLiveStreamDelegate = options.imageClassifierLiveStreamDelegate;
      _callbackQueue = dispatch_queue_create(
          [MPPVisionTaskRunner uniqueDispatchQueueNameWithSuffix:kTaskName], NULL);
      _callbackSlot = CallbackPool::AllocateSlot(self, &OnLiveStreamResult);
      if (_callbackSlot < 0) {
        [MPPCommonUtils createCustomError:error
                                 withCode:MPPTasksErrorCodeInternalError
                              description:@"Failed to allocate callback slot."];
        return nil;
      }
    }

    MpImageClassifierOptions cOptions = {};
    [options copyToCOptions:&cOptions];
    if (_callbackSlot >= 0) {
      cOptions.result_callback = CallbackPool::GetCallback(_callbackSlot);
    }

    char *errorMsg = nullptr;
    MpStatus status = MpImageClassifierCreate(&cOptions, &_classifier, &errorMsg);
    [MPPImageClassifierOptions freeCOptions:&cOptions];

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
  MPPImageClassifierOptions *options = [[MPPImageClassifierOptions alloc] init];
  options.baseOptions.modelAssetPath = modelPath;
  return [self initWithOptions:options error:error];
}

- (nullable MPPImageClassifierResult *)classifyImage:(MPPImage *)image
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
  MpImageClassifierResult cResult = {};
  char *errorMsg = nullptr;
  MpStatus status = MpImageClassifierClassifyImage(_classifier, mpImage, &imageProcessingOptions,
                                                   &cResult, &errorMsg);
  if (![MPPCommonUtils checkCError:status errorMessage:errorMsg toError:error]) {
    return nil;
  }
  MPPImageClassifierResult *result =
      [MPPImageClassifierResult imageClassifierResultWithCResult:cResult timestampInMilliseconds:0];
  MpImageClassifierCloseResult(&cResult);
  return result;
}

- (nullable MPPImageClassifierResult *)classifyImage:(MPPImage *)image error:(NSError **)error {
  return [self classifyImage:image regionOfInterest:CGRectZero error:error];
}

- (nullable MPPImageClassifierResult *)classifyVideoFrame:(MPPImage *)image
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
  MpImageClassifierResult cResult = {};
  char *errorMsg = nullptr;
  MpStatus status = MpImageClassifierClassifyForVideo(_classifier, mpImage, &imageProcessingOptions,
                                                      static_cast<int64_t>(timestampInMilliseconds),
                                                      &cResult, &errorMsg);
  if (![MPPCommonUtils checkCError:status errorMessage:errorMsg toError:error]) {
    return nil;
  }
  MPPImageClassifierResult *result =
      [MPPImageClassifierResult imageClassifierResultWithCResult:cResult
                                         timestampInMilliseconds:timestampInMilliseconds];
  MpImageClassifierCloseResult(&cResult);
  return result;
}

- (nullable MPPImageClassifierResult *)classifyVideoFrame:(MPPImage *)image
                                  timestampInMilliseconds:(NSInteger)timestampInMilliseconds
                                                    error:(NSError **)error {
  return [self classifyVideoFrame:image
          timestampInMilliseconds:timestampInMilliseconds
                 regionOfInterest:CGRectZero
                            error:error];
}

- (BOOL)classifyAsyncImage:(MPPImage *)image
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
      MpImageClassifierClassifyAsync(_classifier, mpImage, &imageProcessingOptions,
                                     static_cast<int64_t>(timestampInMilliseconds), &errorMsg);
  return [MPPCommonUtils checkCError:status errorMessage:errorMsg toError:error];
}

- (BOOL)classifyAsyncImage:(MPPImage *)image
    timestampInMilliseconds:(NSInteger)timestampInMilliseconds
                      error:(NSError **)error {
  return [self classifyAsyncImage:image
          timestampInMilliseconds:timestampInMilliseconds
                 regionOfInterest:CGRectZero
                            error:error];
}

- (void)dealloc {
  MpImageClassifierPtr classifier = _classifier;
  _classifier = nullptr;
  int callbackSlot = _callbackSlot;
  _callbackSlot = -1;
  dispatch_queue_t callbackQueue = _callbackQueue;
  _callbackQueue = nil;

  if (callbackSlot >= 0) {
    dispatch_queue_t cleanupQueue =
        callbackQueue ?: dispatch_get_global_queue(QOS_CLASS_DEFAULT, 0);
    dispatch_async(cleanupQueue, ^{
      if (classifier) {
        MpImageClassifierClose(classifier, nullptr);
      }
      CallbackPool::FreeSlot(callbackSlot);
    });
  } else if (classifier) {
    MpImageClassifierClose(classifier, nullptr);
  }
}

#pragma mark - Private

- (void)handleLiveStreamResult:(const MpImageClassifierResult *)cResult
                        status:(MpStatus)status
                     timestamp:(int64_t)timestampInMilliseconds {
  if (![self.imageClassifierLiveStreamDelegate
          respondsToSelector:@selector(imageClassifier:didFinishClassificationWithResult:
                                       timestampInMilliseconds:error:)]) {
    return;
  }

  NSError *callbackError = nil;
  MPPImageClassifierResult *result = nil;

  if (status != kMpOk) {
    [MPPCommonUtils checkCError:status errorMessage:nullptr toError:&callbackError];
  } else if (cResult) {
    result = [MPPImageClassifierResult imageClassifierResultWithCResult:*cResult
                                                timestampInMilliseconds:timestampInMilliseconds];
  }

  dispatch_queue_t callbackQueue = _callbackQueue;
  if (!callbackQueue) {
    return;
  }
  __weak id<MPPImageClassifierLiveStreamDelegate> delegate = self.imageClassifierLiveStreamDelegate;
  MPPImageClassifier *__weak weakSelf = self;

  dispatch_async(callbackQueue, ^{
    MPPImageClassifier *strongSelf = weakSelf;
    if (strongSelf) {
      [delegate imageClassifier:strongSelf
          didFinishClassificationWithResult:result
                    timestampInMilliseconds:timestampInMilliseconds
                                      error:callbackError];
    }
  });
}

@end
