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

#import "mediapipe/tasks/ios/vision/holistic_landmarker/sources/MPPHolisticLandmarker.h"

#import "mediapipe/tasks/ios/common/sources/MPPCommon.h"
#import "mediapipe/tasks/ios/common/utils/sources/MPPCommonUtils.h"
#import "mediapipe/tasks/ios/vision/core/sources/MPPVisionTaskRunner.h"
#import "mediapipe/tasks/ios/vision/core/utils/sources/MPPImage+Utils.h"
#import "mediapipe/tasks/ios/vision/holistic_landmarker/utils/sources/MPPHolisticLandmarkerOptions+Helpers.h"
#import "mediapipe/tasks/ios/vision/holistic_landmarker/utils/sources/MPPHolisticLandmarkerResult+Helpers.h"

#include "mediapipe/tasks/c/vision/core/image_processing_options.h"
#include "mediapipe/tasks/c/vision/holistic_landmarker/holistic_landmarker.h"

static NSString *const kTaskName = @"holisticLandmarker";

@interface MPPHolisticLandmarker () {
  MpHolisticLandmarkerPtr _landmarker;
  MPPVisionTaskRunner *_taskRunner;
  dispatch_queue_t _callbackQueue;
  __weak id<MPPHolisticLandmarkerLiveStreamDelegate> _holisticLandmarkerLiveStreamDelegate;
  int _callbackSlot;
}

- (void)handleLiveStreamResult:(const MpHolisticLandmarkerResult *)cResult
                        status:(MpStatus)status
                     timestamp:(int64_t)timestampInMilliseconds;

@end

namespace {
using CallbackPool = mediapipe::tasks::ios::vision::core::VisionCallbackPool<
    MPPHolisticLandmarker, MpHolisticLandmarkerResult,
    MpHolisticLandmarkerOptions::result_callback_fn>;

static void OnLiveStreamResult(MPPHolisticLandmarker *task, MpStatus status,
                               const MpHolisticLandmarkerResult *result, MpImagePtr image,
                               int64_t timestamp_ms) {
  if (task) {
    [task handleLiveStreamResult:result status:status timestamp:timestamp_ms];
  }
}
}  // namespace

@implementation MPPHolisticLandmarker

#pragma mark - Public

- (instancetype)initWithOptions:(MPPHolisticLandmarkerOptions *)options error:(NSError **)error {
  self = [super init];
  if (self) {
    _callbackSlot = -1;
    if (!options) {
      [MPPCommonUtils createCustomError:error
                               withCode:MPPTasksErrorCodeInvalidArgumentError
                            description:@"options cannot be nil."];
      return nil;
    }
    _taskRunner = [[MPPVisionTaskRunner alloc]
        initWithRunningMode:options.runningMode
                 roiAllowed:NO
            normRectAllowed:NO
                hasDelegate:(options.holisticLandmarkerLiveStreamDelegate != nil)
                      error:error];
    if (!_taskRunner) {
      return nil;
    }

    if (options.holisticLandmarkerLiveStreamDelegate) {
      _holisticLandmarkerLiveStreamDelegate = options.holisticLandmarkerLiveStreamDelegate;
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

    MpHolisticLandmarkerOptions cOptions = {};
    [options copyToCOptions:&cOptions];
    if (_callbackSlot >= 0) {
      cOptions.result_callback = CallbackPool::GetCallback(_callbackSlot);
    }

    char *errorMsg = nullptr;
    MpStatus status = MpHolisticLandmarkerCreate(&cOptions, &_landmarker, &errorMsg);
    [MPPHolisticLandmarkerOptions freeCOptions:&cOptions];

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
  MPPHolisticLandmarkerOptions *options = [[MPPHolisticLandmarkerOptions alloc] init];
  options.baseOptions.modelAssetPath = modelPath;
  return [self initWithOptions:options error:error];
}

- (nullable MPPHolisticLandmarkerResult *)detectImage:(MPPImage *)image error:(NSError **)error {
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

  MpHolisticLandmarkerResult cResult = {};
  char *errorMsg = nullptr;
  MpStatus status = MpHolisticLandmarkerDetectImage(_landmarker, cImage, &imageProcessingOptions,
                                                    &cResult, &errorMsg);
  if (![MPPCommonUtils checkCError:status errorMessage:errorMsg toError:error]) {
    return nil;
  }

  MPPHolisticLandmarkerResult *result =
      [MPPHolisticLandmarkerResult holisticLandmarkerResultWithCResult:cResult
                                               timestampInMilliseconds:0];
  MpHolisticLandmarkerCloseResult(&cResult);
  return result;
}

- (nullable MPPHolisticLandmarkerResult *)detectVideoFrame:(MPPImage *)image
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

  MpHolisticLandmarkerResult cResult = {};
  char *errorMsg = nullptr;
  MpStatus status = MpHolisticLandmarkerDetectForVideo(
      _landmarker, cImage, &imageProcessingOptions, timestampInMilliseconds, &cResult, &errorMsg);
  if (![MPPCommonUtils checkCError:status errorMessage:errorMsg toError:error]) {
    return nil;
  }

  MPPHolisticLandmarkerResult *result =
      [MPPHolisticLandmarkerResult holisticLandmarkerResultWithCResult:cResult
                                               timestampInMilliseconds:timestampInMilliseconds];
  MpHolisticLandmarkerCloseResult(&cResult);
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
  MpStatus status = MpHolisticLandmarkerDetectAsync(_landmarker, cImage, &imageProcessingOptions,
                                                    timestampInMilliseconds, &errorMsg);
  return [MPPCommonUtils checkCError:status errorMessage:errorMsg toError:error];
}

- (void)dealloc {
  MpHolisticLandmarkerPtr landmarker = _landmarker;
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
        MpHolisticLandmarkerClose(landmarker, nullptr);
      }
      CallbackPool::FreeSlot(callbackSlot);
    });
  } else if (landmarker) {
    MpHolisticLandmarkerClose(landmarker, nullptr);
  }
}

#pragma mark - Private

- (void)handleLiveStreamResult:(const MpHolisticLandmarkerResult *)cResult
                        status:(MpStatus)status
                     timestamp:(int64_t)timestampInMilliseconds {
  if (![_holisticLandmarkerLiveStreamDelegate
          respondsToSelector:@selector(holisticLandmarker:didFinishDetectionWithResult:
                                       timestampInMilliseconds:error:)]) {
    return;
  }

  NSError *callbackError = nil;
  MPPHolisticLandmarkerResult *result = nil;

  if (status != kMpOk) {
    [MPPCommonUtils checkCError:status errorMessage:nullptr toError:&callbackError];
  } else if (cResult) {
    result =
        [MPPHolisticLandmarkerResult holisticLandmarkerResultWithCResult:*cResult
                                                 timestampInMilliseconds:timestampInMilliseconds];
  }

  dispatch_queue_t callbackQueue = _callbackQueue;
  if (!callbackQueue) {
    return;
  }
  __weak id<MPPHolisticLandmarkerLiveStreamDelegate> delegate =
      _holisticLandmarkerLiveStreamDelegate;
  MPPHolisticLandmarker *__weak weakSelf = self;

  dispatch_async(callbackQueue, ^{
    MPPHolisticLandmarker *strongSelf = weakSelf;
    if (strongSelf) {
      [delegate holisticLandmarker:strongSelf
          didFinishDetectionWithResult:result
               timestampInMilliseconds:timestampInMilliseconds
                                 error:callbackError];
    }
  });
}

@end
