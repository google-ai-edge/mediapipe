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

#import "mediapipe/tasks/ios/vision/pose_landmarker/sources/MPPPoseLandmarker.h"

#import "mediapipe/tasks/ios/common/sources/MPPCommon.h"
#import "mediapipe/tasks/ios/common/utils/sources/MPPCommonUtils.h"
#import "mediapipe/tasks/ios/vision/core/sources/MPPVisionTaskRunner.h"
#import "mediapipe/tasks/ios/vision/core/utils/sources/MPPImage+Utils.h"
#import "mediapipe/tasks/ios/vision/pose_landmarker/sources/MPPPoseLandmarksConnections.h"
#import "mediapipe/tasks/ios/vision/pose_landmarker/utils/sources/MPPPoseLandmarkerOptions+Helpers.h"
#import "mediapipe/tasks/ios/vision/pose_landmarker/utils/sources/MPPPoseLandmarkerResult+Helpers.h"

#include "mediapipe/tasks/c/vision/core/image_processing_options.h"
#include "mediapipe/tasks/c/vision/pose_landmarker/pose_landmarker.h"

static NSString *const kTaskName = @"poseLandmarker";

@interface MPPPoseLandmarker () {
  MpPoseLandmarkerPtr _landmarker;
  MPPVisionTaskRunner *_taskRunner;
  dispatch_queue_t _callbackQueue;
  __weak id<MPPPoseLandmarkerLiveStreamDelegate> _poseLandmarkerLiveStreamDelegate;
  int _callbackSlot;
}

- (void)handleLiveStreamResult:(const MpPoseLandmarkerResult *)cResult
                        status:(MpStatus)status
                     timestamp:(int64_t)timestampInMilliseconds;

@end

namespace {
using CallbackPool = mediapipe::tasks::ios::vision::core::VisionCallbackPool<
    MPPPoseLandmarker, MpPoseLandmarkerResult, MpPoseLandmarkerOptions::result_callback_fn>;

static void OnLiveStreamResult(MPPPoseLandmarker *task, MpStatus status,
                               const MpPoseLandmarkerResult *result, MpImagePtr image,
                               int64_t timestamp_ms) {
  if (task) {
    [task handleLiveStreamResult:result status:status timestamp:timestamp_ms];
  }
}
}  // namespace

@implementation MPPPoseLandmarker

#pragma mark - Public

- (instancetype)initWithOptions:(MPPPoseLandmarkerOptions *)options error:(NSError **)error {
  self = [super init];
  if (self) {
    _callbackSlot = -1;
    _taskRunner = [[MPPVisionTaskRunner alloc]
        initWithRunningMode:options.runningMode
                 roiAllowed:NO
            normRectAllowed:YES
                hasDelegate:(options.poseLandmarkerLiveStreamDelegate != nil)
                      error:error];
    if (!_taskRunner) {
      return nil;
    }

    if (options.poseLandmarkerLiveStreamDelegate) {
      _poseLandmarkerLiveStreamDelegate = options.poseLandmarkerLiveStreamDelegate;
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

    MpPoseLandmarkerOptions cOptions = {};
    [options copyToCOptions:&cOptions];
    if (_callbackSlot >= 0) {
      cOptions.result_callback = CallbackPool::GetCallback(_callbackSlot);
    }

    char *errorMsg = nullptr;
    MpStatus status = MpPoseLandmarkerCreate(&cOptions, &_landmarker, &errorMsg);
    [MPPPoseLandmarkerOptions freeCOptions:&cOptions];

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
  MPPPoseLandmarkerOptions *options = [[MPPPoseLandmarkerOptions alloc] init];
  options.baseOptions.modelAssetPath = modelPath;
  return [self initWithOptions:options error:error];
}

- (nullable MPPPoseLandmarkerResult *)detectImage:(MPPImage *)image error:(NSError **)error {
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

  MpPoseLandmarkerResult cResult = {};
  char *errorMsg = nullptr;
  MpStatus status = MpPoseLandmarkerDetectImage(_landmarker, cImage, &imageProcessingOptions,
                                                &cResult, &errorMsg);
  if (![MPPCommonUtils checkCError:status errorMessage:errorMsg toError:error]) {
    return nil;
  }

  MPPPoseLandmarkerResult *result = [MPPPoseLandmarkerResult poseLandmarkerResultWithCResult:cResult
                                                                     timestampInMilliseconds:0];
  MpPoseLandmarkerCloseResult(&cResult);
  return result;
}

- (nullable MPPPoseLandmarkerResult *)detectVideoFrame:(MPPImage *)image
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

  MpPoseLandmarkerResult cResult = {};
  char *errorMsg = nullptr;
  MpStatus status = MpPoseLandmarkerDetectForVideo(_landmarker, cImage, &imageProcessingOptions,
                                                   timestampInMilliseconds, &cResult, &errorMsg);
  if (![MPPCommonUtils checkCError:status errorMessage:errorMsg toError:error]) {
    return nil;
  }

  MPPPoseLandmarkerResult *result =
      [MPPPoseLandmarkerResult poseLandmarkerResultWithCResult:cResult
                                       timestampInMilliseconds:timestampInMilliseconds];
  MpPoseLandmarkerCloseResult(&cResult);
  return result;
}

- (BOOL)detectAsyncImage:(MPPImage *)image
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
  MpStatus status = MpPoseLandmarkerDetectAsync(_landmarker, cImage, &imageProcessingOptions,
                                                timestampInMilliseconds, &errorMsg);
  return [MPPCommonUtils checkCError:status errorMessage:errorMsg toError:error];
}

+ (NSArray<MPPConnection *> *)poseLandmarks {
  return MPPPoseLandmarksConnections;
}

- (void)dealloc {
  MpPoseLandmarkerPtr landmarker = _landmarker;
  _landmarker = nullptr;
  int callbackSlot = _callbackSlot;
  _callbackSlot = -1;
  dispatch_queue_t callbackQueue = _callbackQueue;
  _callbackQueue = nil;

  if (callbackSlot >= 0) {
    dispatch_queue_t cleanupQueue =
        callbackQueue ?: dispatch_get_global_queue(QOS_CLASS_DEFAULT, 0);
    dispatch_async(cleanupQueue, ^{
      if (landmarker) {
        MpPoseLandmarkerClose(landmarker, nullptr);
      }
      CallbackPool::FreeSlot(callbackSlot);
    });
  } else if (landmarker) {
    MpPoseLandmarkerClose(landmarker, nullptr);
  }
}

#pragma mark - Private

- (void)handleLiveStreamResult:(const MpPoseLandmarkerResult *)cResult
                        status:(MpStatus)status
                     timestamp:(int64_t)timestampInMilliseconds {
  if (![_poseLandmarkerLiveStreamDelegate
          respondsToSelector:@selector(poseLandmarker:didFinishDetectionWithResult:
                                       timestampInMilliseconds:error:)]) {
    return;
  }

  NSError *callbackError = nil;
  MPPPoseLandmarkerResult *result = nil;

  if (status != kMpOk) {
    [MPPCommonUtils checkCError:status errorMessage:nullptr toError:&callbackError];
  } else if (cResult) {
    result = [MPPPoseLandmarkerResult poseLandmarkerResultWithCResult:*cResult
                                              timestampInMilliseconds:timestampInMilliseconds];
  }

  dispatch_queue_t callbackQueue = _callbackQueue;
  if (!callbackQueue) {
    return;
  }
  __weak id<MPPPoseLandmarkerLiveStreamDelegate> delegate = _poseLandmarkerLiveStreamDelegate;
  MPPPoseLandmarker *__weak weakSelf = self;

  dispatch_async(callbackQueue, ^{
    MPPPoseLandmarker *strongSelf = weakSelf;
    if (strongSelf) {
      [delegate poseLandmarker:strongSelf
          didFinishDetectionWithResult:result
               timestampInMilliseconds:timestampInMilliseconds
                                 error:callbackError];
    }
  });
}

@end
