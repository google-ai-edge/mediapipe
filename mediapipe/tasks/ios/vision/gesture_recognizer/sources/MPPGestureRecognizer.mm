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

#import "mediapipe/tasks/ios/vision/gesture_recognizer/sources/MPPGestureRecognizer.h"

#import "mediapipe/tasks/ios/common/sources/MPPCommon.h"
#import "mediapipe/tasks/ios/common/utils/sources/MPPCommonUtils.h"
#import "mediapipe/tasks/ios/vision/core/sources/MPPVisionTaskRunner.h"
#import "mediapipe/tasks/ios/vision/core/utils/sources/MPPImage+Utils.h"
#import "mediapipe/tasks/ios/vision/gesture_recognizer/utils/sources/MPPGestureRecognizerOptions+Helpers.h"
#import "mediapipe/tasks/ios/vision/gesture_recognizer/utils/sources/MPPGestureRecognizerResult+Helpers.h"

#include "mediapipe/tasks/c/vision/core/image_processing_options.h"
#include "mediapipe/tasks/c/vision/gesture_recognizer/gesture_recognizer.h"

static NSString *const kTaskName = @"gestureRecognizer";

@interface MPPGestureRecognizer () {
  MpGestureRecognizerPtr _recognizer;
  MPPVisionTaskRunner *_taskRunner;
  dispatch_queue_t _callbackQueue;
  __weak id<MPPGestureRecognizerLiveStreamDelegate> _gestureRecognizerLiveStreamDelegate;
  int _callbackSlot;
}

- (void)handleLiveStreamResult:(const MpGestureRecognizerResult *)cResult
                        status:(MpStatus)status
                     timestamp:(int64_t)timestampInMilliseconds;

@end

namespace {
using CallbackPool = mediapipe::tasks::ios::vision::core::VisionCallbackPool<
    MPPGestureRecognizer, MpGestureRecognizerResult,
    MpGestureRecognizerOptions::result_callback_fn>;

static void OnLiveStreamResult(MPPGestureRecognizer *task, MpStatus status,
                               const MpGestureRecognizerResult *result, MpImagePtr image,
                               int64_t timestamp_ms) {
  if (task) {
    [task handleLiveStreamResult:result status:status timestamp:timestamp_ms];
  }
}
}  // namespace

@implementation MPPGestureRecognizer

#pragma mark - Public

- (instancetype)initWithOptions:(MPPGestureRecognizerOptions *)options error:(NSError **)error {
  self = [super init];
  if (self) {
    _callbackSlot = -1;
    _taskRunner = [[MPPVisionTaskRunner alloc]
        initWithRunningMode:options.runningMode
                 roiAllowed:NO
            normRectAllowed:YES
                hasDelegate:(options.gestureRecognizerLiveStreamDelegate != nil)
                      error:error];
    if (!_taskRunner) {
      return nil;
    }

    if (options.gestureRecognizerLiveStreamDelegate) {
      _gestureRecognizerLiveStreamDelegate = options.gestureRecognizerLiveStreamDelegate;
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

    MpGestureRecognizerOptions cOptions = {};
    [options copyToCOptions:&cOptions];
    if (_callbackSlot >= 0) {
      cOptions.result_callback = CallbackPool::GetCallback(_callbackSlot);
    }

    char *errorMsg = nullptr;
    MpStatus status = MpGestureRecognizerCreate(&cOptions, &_recognizer, &errorMsg);
    [MPPGestureRecognizerOptions freeCOptions:&cOptions];

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
  MPPGestureRecognizerOptions *options = [[MPPGestureRecognizerOptions alloc] init];
  options.baseOptions.modelAssetPath = modelPath;
  return [self initWithOptions:options error:error];
}

- (nullable MPPGestureRecognizerResult *)recognizeImage:(MPPImage *)image error:(NSError **)error {
  if (![_taskRunner validateRunningMode:MPPRunningModeImage error:error]) {
    return nil;
  }
  struct MpImageProcessingOptions imageProcessingOptions = {};
  ScopedMpImage cImage = [_taskRunner createMpImageWithMPPImage:image
                                               regionOfInterest:CGRectZero
                                         imageProcessingOptions:&imageProcessingOptions
                                                          error:error];
  if (!cImage) {
    return nil;
  }

  MpGestureRecognizerResult cResult = {};
  char *errorMsg = nullptr;
  MpStatus status = MpGestureRecognizerRecognizeImage(_recognizer, cImage, &imageProcessingOptions,
                                                      &cResult, &errorMsg);
  if (![MPPCommonUtils checkCError:status errorMessage:errorMsg toError:error]) {
    return nil;
  }

  MPPGestureRecognizerResult *result =
      [MPPGestureRecognizerResult gestureRecognizerResultWithCResult:cResult
                                             timestampInMilliseconds:0];
  MpGestureRecognizerCloseResult(&cResult);
  return result;
}

- (nullable MPPGestureRecognizerResult *)recognizeVideoFrame:(MPPImage *)image
                                     timestampInMilliseconds:(NSInteger)timestampInMilliseconds
                                                       error:(NSError **)error {
  if (![_taskRunner validateRunningMode:MPPRunningModeVideo error:error]) {
    return nil;
  }
  struct MpImageProcessingOptions imageProcessingOptions = {};
  ScopedMpImage cImage = [_taskRunner createMpImageWithMPPImage:image
                                               regionOfInterest:CGRectZero
                                         imageProcessingOptions:&imageProcessingOptions
                                                          error:error];
  if (!cImage) {
    return nil;
  }

  MpGestureRecognizerResult cResult = {};
  char *errorMsg = nullptr;
  MpStatus status = MpGestureRecognizerRecognizeForVideo(
      _recognizer, cImage, &imageProcessingOptions, timestampInMilliseconds, &cResult, &errorMsg);
  if (![MPPCommonUtils checkCError:status errorMessage:errorMsg toError:error]) {
    return nil;
  }

  MPPGestureRecognizerResult *result =
      [MPPGestureRecognizerResult gestureRecognizerResultWithCResult:cResult
                                             timestampInMilliseconds:timestampInMilliseconds];
  MpGestureRecognizerCloseResult(&cResult);
  return result;
}

- (BOOL)recognizeAsyncImage:(MPPImage *)image
    timestampInMilliseconds:(NSInteger)timestampInMilliseconds
                      error:(NSError **)error {
  if (![_taskRunner validateRunningMode:MPPRunningModeLiveStream error:error]) {
    return NO;
  }
  struct MpImageProcessingOptions imageProcessingOptions = {};
  ScopedMpImage cImage = [_taskRunner createMpImageWithMPPImage:image
                                               regionOfInterest:CGRectZero
                                         imageProcessingOptions:&imageProcessingOptions
                                                          error:error];
  if (!cImage) {
    return NO;
  }

  char *errorMsg = nullptr;
  MpStatus status = MpGestureRecognizerRecognizeAsync(_recognizer, cImage, &imageProcessingOptions,
                                                      timestampInMilliseconds, &errorMsg);
  return [MPPCommonUtils checkCError:status errorMessage:errorMsg toError:error];
}

- (void)dealloc {
  MpGestureRecognizerPtr recognizer = _recognizer;
  _recognizer = nullptr;
  int callbackSlot = _callbackSlot;
  _callbackSlot = -1;
  dispatch_queue_t callbackQueue = _callbackQueue;
  _callbackQueue = nil;

  if (callbackSlot >= 0) {
    dispatch_queue_t cleanupQueue =
        callbackQueue ?: dispatch_get_global_queue(QOS_CLASS_DEFAULT, 0);
    dispatch_async(cleanupQueue, ^{
      if (recognizer) {
        MpGestureRecognizerClose(recognizer, nullptr);
      }
      CallbackPool::FreeSlot(callbackSlot);
    });
  } else if (recognizer) {
    MpGestureRecognizerClose(recognizer, nullptr);
  }
}

#pragma mark - Private

- (void)handleLiveStreamResult:(const MpGestureRecognizerResult *)cResult
                        status:(MpStatus)status
                     timestamp:(int64_t)timestampInMilliseconds {
  if (![_gestureRecognizerLiveStreamDelegate
          respondsToSelector:@selector(gestureRecognizer:didFinishRecognitionWithResult:
                                       timestampInMilliseconds:error:)]) {
    return;
  }

  NSError *callbackError = nil;
  MPPGestureRecognizerResult *result = nil;

  if (status != kMpOk) {
    [MPPCommonUtils checkCError:status errorMessage:nullptr toError:&callbackError];
  } else if (cResult) {
    result =
        [MPPGestureRecognizerResult gestureRecognizerResultWithCResult:*cResult
                                               timestampInMilliseconds:timestampInMilliseconds];
  }

  dispatch_queue_t callbackQueue = _callbackQueue;
  if (!callbackQueue) {
    return;
  }
  __weak id<MPPGestureRecognizerLiveStreamDelegate> delegate = _gestureRecognizerLiveStreamDelegate;
  MPPGestureRecognizer *__weak weakSelf = self;

  dispatch_async(callbackQueue, ^{
    MPPGestureRecognizer *strongSelf = weakSelf;
    if (strongSelf) {
      [delegate gestureRecognizer:strongSelf
          didFinishRecognitionWithResult:result
                 timestampInMilliseconds:timestampInMilliseconds
                                   error:callbackError];
    }
  });
}

@end
