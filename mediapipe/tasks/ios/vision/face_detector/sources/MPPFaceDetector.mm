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

#import "mediapipe/tasks/ios/vision/face_detector/sources/MPPFaceDetector.h"

#import "mediapipe/tasks/ios/common/sources/MPPCommon.h"
#import "mediapipe/tasks/ios/common/utils/sources/MPPCommonUtils.h"
#import "mediapipe/tasks/ios/vision/core/sources/MPPVisionTaskRunner.h"
#import "mediapipe/tasks/ios/vision/face_detector/utils/sources/MPPFaceDetectorOptions+Helpers.h"
#import "mediapipe/tasks/ios/vision/face_detector/utils/sources/MPPFaceDetectorResult+Helpers.h"

#include "mediapipe/tasks/c/vision/core/image_processing_options.h"
#include "mediapipe/tasks/c/vision/face_detector/face_detector.h"

static NSString *const kTaskName = @"faceDetector";

@interface MPPFaceDetector () {
  MpFaceDetectorPtr _detector;
  MPPVisionTaskRunner *_taskRunner;
  dispatch_queue_t _callbackQueue;
  int _callbackSlot;
}
@property(nonatomic, weak) id<MPPFaceDetectorLiveStreamDelegate> faceDetectorLiveStreamDelegate;

- (void)handleLiveStreamResult:(const MpFaceDetectorResult *)cResult
                        status:(MpStatus)status
                     timestamp:(int64_t)timestampInMilliseconds;
@end

namespace {
using mediapipe::tasks::ios::vision::core::VisionCallbackPool;
using CallbackPool = VisionCallbackPool<MPPFaceDetector, MpFaceDetectorResult,
                                        MpFaceDetectorOptions::result_callback_fn>;

static void OnLiveStreamResult(MPPFaceDetector *task, MpStatus status,
                               const MpFaceDetectorResult *result, MpImagePtr image,
                               int64_t timestamp_ms) {
  if (task) {
    [task handleLiveStreamResult:result status:status timestamp:timestamp_ms];
  }
}
}  // namespace

@implementation MPPFaceDetector

- (instancetype)initWithOptions:(MPPFaceDetectorOptions *)options error:(NSError **)error {
  self = [super init];
  if (self) {
    _callbackSlot = -1;
    _taskRunner = [[MPPVisionTaskRunner alloc]
        initWithRunningMode:options.runningMode
                 roiAllowed:NO
            normRectAllowed:YES
                hasDelegate:(options.faceDetectorLiveStreamDelegate != nil)
                      error:error];
    if (!_taskRunner) {
      return nil;
    }

    if (options.faceDetectorLiveStreamDelegate) {
      _faceDetectorLiveStreamDelegate = options.faceDetectorLiveStreamDelegate;
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

    MpFaceDetectorOptions cOptions = {};
    [options copyToCOptions:&cOptions];
    if (_callbackSlot >= 0) {
      cOptions.result_callback = CallbackPool::GetCallback(_callbackSlot);
    }

    char *errorMsg = nullptr;
    MpStatus status = MpFaceDetectorCreate(&cOptions, &_detector, &errorMsg);
    [MPPFaceDetectorOptions freeCOptions:&cOptions];

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
  MPPFaceDetectorOptions *options = [[MPPFaceDetectorOptions alloc] init];
  options.baseOptions.modelAssetPath = modelPath;
  return [self initWithOptions:options error:error];
}

- (nullable MPPFaceDetectorResult *)detectImage:(MPPImage *)image error:(NSError **)error {
  if (![_taskRunner validateRunningMode:MPPRunningModeImage error:error]) {
    return nil;
  }
  struct MpImageProcessingOptions imageProcessingOptions = {};
  ScopedMpImage mpImage = [_taskRunner createMpImageWithMPPImage:image
                                                regionOfInterest:CGRectZero
                                          imageProcessingOptions:&imageProcessingOptions
                                                           error:error];
  if (!mpImage) {
    return nil;
  }
  MpFaceDetectorResult cResult = {};
  char *errorMsg = nullptr;
  MpStatus status =
      MpFaceDetectorDetectImage(_detector, mpImage, &imageProcessingOptions, &cResult, &errorMsg);
  if (![MPPCommonUtils checkCError:status errorMessage:errorMsg toError:error]) {
    return nil;
  }
  MPPFaceDetectorResult *result = [MPPFaceDetectorResult faceDetectorResultWithCResult:cResult
                                                               timestampInMilliseconds:0];
  MpFaceDetectorCloseResult(&cResult);
  return result;
}

- (nullable MPPFaceDetectorResult *)detectVideoFrame:(MPPImage *)image
                             timestampInMilliseconds:(NSInteger)timestampInMilliseconds
                                               error:(NSError **)error {
  if (![_taskRunner validateRunningMode:MPPRunningModeVideo error:error]) {
    return nil;
  }
  struct MpImageProcessingOptions imageProcessingOptions = {};
  ScopedMpImage mpImage = [_taskRunner createMpImageWithMPPImage:image
                                                regionOfInterest:CGRectZero
                                          imageProcessingOptions:&imageProcessingOptions
                                                           error:error];
  if (!mpImage) {
    return nil;
  }
  MpFaceDetectorResult cResult = {};
  char *errorMsg = nullptr;
  MpStatus status = MpFaceDetectorDetectForVideo(_detector, mpImage, &imageProcessingOptions,
                                                 static_cast<int64_t>(timestampInMilliseconds),
                                                 &cResult, &errorMsg);
  if (![MPPCommonUtils checkCError:status errorMessage:errorMsg toError:error]) {
    return nil;
  }
  MPPFaceDetectorResult *result =
      [MPPFaceDetectorResult faceDetectorResultWithCResult:cResult
                                   timestampInMilliseconds:timestampInMilliseconds];
  MpFaceDetectorCloseResult(&cResult);
  return result;
}

- (BOOL)detectAsyncImage:(MPPImage *)image
    timestampInMilliseconds:(NSInteger)timestampInMilliseconds
                      error:(NSError **)error {
  if (![_taskRunner validateRunningMode:MPPRunningModeLiveStream error:error]) {
    return NO;
  }
  struct MpImageProcessingOptions imageProcessingOptions = {};
  ScopedMpImage mpImage = [_taskRunner createMpImageWithMPPImage:image
                                                regionOfInterest:CGRectZero
                                          imageProcessingOptions:&imageProcessingOptions
                                                           error:error];
  if (!mpImage) {
    return NO;
  }
  char *errorMsg = nullptr;
  MpStatus status =
      MpFaceDetectorDetectAsync(_detector, mpImage, &imageProcessingOptions,
                                static_cast<int64_t>(timestampInMilliseconds), &errorMsg);
  return [MPPCommonUtils checkCError:status errorMessage:errorMsg toError:error];
}

- (void)dealloc {
  MpFaceDetectorPtr detector = _detector;
  _detector = nullptr;
  int callbackSlot = _callbackSlot;
  _callbackSlot = -1;
  dispatch_queue_t callbackQueue = _callbackQueue;
  _callbackQueue = nil;

  if (callbackSlot >= 0) {
    dispatch_queue_t cleanupQueue =
        callbackQueue ?: dispatch_get_global_queue(QOS_CLASS_DEFAULT, 0);
    dispatch_async(cleanupQueue, ^{
      if (detector) {
        MpFaceDetectorClose(detector, nullptr);
      }
      CallbackPool::FreeSlot(callbackSlot);
    });
  } else if (detector) {
    MpFaceDetectorClose(detector, nullptr);
  }
}

#pragma mark - Private

- (void)handleLiveStreamResult:(const MpFaceDetectorResult *)cResult
                        status:(MpStatus)status
                     timestamp:(int64_t)timestampInMilliseconds {
  if (![self.faceDetectorLiveStreamDelegate
          respondsToSelector:@selector(faceDetector:didFinishDetectionWithResult:
                                       timestampInMilliseconds:error:)]) {
    return;
  }

  NSError *callbackError = nil;
  MPPFaceDetectorResult *result = nil;

  if (status != kMpOk) {
    [MPPCommonUtils checkCError:status errorMessage:nullptr toError:&callbackError];
  } else if (cResult) {
    result = [MPPFaceDetectorResult faceDetectorResultWithCResult:*cResult
                                          timestampInMilliseconds:timestampInMilliseconds];
  }

  dispatch_queue_t callbackQueue = _callbackQueue;
  if (!callbackQueue) {
    return;
  }
  __weak id<MPPFaceDetectorLiveStreamDelegate> delegate = self.faceDetectorLiveStreamDelegate;
  MPPFaceDetector *__weak weakSelf = self;

  dispatch_async(callbackQueue, ^{
    MPPFaceDetector *strongSelf = weakSelf;
    if (strongSelf) {
      [delegate faceDetector:strongSelf
          didFinishDetectionWithResult:result
               timestampInMilliseconds:timestampInMilliseconds
                                 error:callbackError];
    }
  });
}

@end
