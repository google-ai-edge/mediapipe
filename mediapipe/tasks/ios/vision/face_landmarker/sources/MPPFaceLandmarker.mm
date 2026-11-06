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

#import "mediapipe/tasks/ios/vision/face_landmarker/sources/MPPFaceLandmarker.h"

#import "mediapipe/tasks/ios/common/sources/MPPCommon.h"
#import "mediapipe/tasks/ios/common/utils/sources/MPPCommonUtils.h"
#import "mediapipe/tasks/ios/vision/core/sources/MPPVisionTaskRunner.h"
#import "mediapipe/tasks/ios/vision/core/utils/sources/MPPImage+Utils.h"
#import "mediapipe/tasks/ios/vision/face_landmarker/sources/MPPFaceLandmarksConnections.h"
#import "mediapipe/tasks/ios/vision/face_landmarker/utils/sources/MPPFaceLandmarkerOptions+Helpers.h"
#import "mediapipe/tasks/ios/vision/face_landmarker/utils/sources/MPPFaceLandmarkerResult+Helpers.h"

#include "mediapipe/tasks/c/vision/core/image_processing_options.h"
#include "mediapipe/tasks/c/vision/face_landmarker/face_landmarker.h"

static NSString *const kTaskName = @"faceLandmarker";

@interface MPPFaceLandmarker () {
  MpFaceLandmarkerPtr _landmarker;
  MPPVisionTaskRunner *_taskRunner;
  dispatch_queue_t _callbackQueue;
  __weak id<MPPFaceLandmarkerLiveStreamDelegate> _faceLandmarkerLiveStreamDelegate;
  int _callbackSlot;
}

- (void)handleLiveStreamResult:(const MpFaceLandmarkerResult *)cResult
                        status:(MpStatus)status
                     timestamp:(int64_t)timestampInMilliseconds;

@end

namespace {
using CallbackPool = mediapipe::tasks::ios::vision::core::VisionCallbackPool<
    MPPFaceLandmarker, MpFaceLandmarkerResult, MpFaceLandmarkerOptions::result_callback_fn>;

static void OnLiveStreamResult(MPPFaceLandmarker *task, MpStatus status,
                               const MpFaceLandmarkerResult *result, MpImagePtr image,
                               int64_t timestamp_ms) {
  if (task) {
    [task handleLiveStreamResult:result status:status timestamp:timestamp_ms];
  }
}
}  // namespace

@implementation MPPFaceLandmarker

#pragma mark - Public

- (instancetype)initWithOptions:(MPPFaceLandmarkerOptions *)options error:(NSError **)error {
  self = [super init];
  if (self) {
    _callbackSlot = -1;
    _taskRunner = [[MPPVisionTaskRunner alloc]
        initWithRunningMode:options.runningMode
                 roiAllowed:NO
            normRectAllowed:YES
                hasDelegate:(options.faceLandmarkerLiveStreamDelegate != nil)
                      error:error];
    if (!_taskRunner) {
      return nil;
    }

    if (options.faceLandmarkerLiveStreamDelegate) {
      _faceLandmarkerLiveStreamDelegate = options.faceLandmarkerLiveStreamDelegate;
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

    MpFaceLandmarkerOptions cOptions = {};
    [options copyToCOptions:&cOptions];
    if (_callbackSlot >= 0) {
      cOptions.result_callback = CallbackPool::GetCallback(_callbackSlot);
    }

    char *errorMsg = nullptr;
    MpStatus status = MpFaceLandmarkerCreate(&cOptions, &_landmarker, &errorMsg);
    [MPPFaceLandmarkerOptions freeCOptions:&cOptions];

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
  MPPFaceLandmarkerOptions *options = [[MPPFaceLandmarkerOptions alloc] init];
  options.baseOptions.modelAssetPath = modelPath;
  return [self initWithOptions:options error:error];
}

- (nullable MPPFaceLandmarkerResult *)detectImage:(MPPImage *)image error:(NSError **)error {
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

  MpFaceLandmarkerResult cResult = {};
  char *errorMsg = nullptr;
  MpStatus status = MpFaceLandmarkerDetectImage(_landmarker, cImage, &imageProcessingOptions,
                                                &cResult, &errorMsg);
  if (![MPPCommonUtils checkCError:status errorMessage:errorMsg toError:error]) {
    return nil;
  }

  MPPFaceLandmarkerResult *result = [MPPFaceLandmarkerResult faceLandmarkerResultWithCResult:cResult
                                                                     timestampInMilliseconds:0];
  MpFaceLandmarkerCloseResult(&cResult);
  return result;
}

- (nullable MPPFaceLandmarkerResult *)detectVideoFrame:(MPPImage *)image
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

  MpFaceLandmarkerResult cResult = {};
  char *errorMsg = nullptr;
  MpStatus status = MpFaceLandmarkerDetectForVideo(_landmarker, cImage, &imageProcessingOptions,
                                                   timestampInMilliseconds, &cResult, &errorMsg);
  if (![MPPCommonUtils checkCError:status errorMessage:errorMsg toError:error]) {
    return nil;
  }

  MPPFaceLandmarkerResult *result =
      [MPPFaceLandmarkerResult faceLandmarkerResultWithCResult:cResult
                                       timestampInMilliseconds:timestampInMilliseconds];
  MpFaceLandmarkerCloseResult(&cResult);
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
  MpStatus status = MpFaceLandmarkerDetectAsync(_landmarker, cImage, &imageProcessingOptions,
                                                timestampInMilliseconds, &errorMsg);
  return [MPPCommonUtils checkCError:status errorMessage:errorMsg toError:error];
}

+ (NSArray<MPPConnection *> *)lipsConnections {
  return MPPFaceLandmarksLips;
}

+ (NSArray<MPPConnection *> *)leftEyeConnections {
  return MPPFaceLandmarksLeftEye;
}

+ (NSArray<MPPConnection *> *)leftEyebrowConnections {
  return MPPFaceLandmarksLeftEyebrow;
}

+ (NSArray<MPPConnection *> *)leftIrisConnections {
  return MPPFaceLandmarksLeftIris;
}

+ (NSArray<MPPConnection *> *)rightEyeConnections {
  return MPPFaceLandmarksRightEye;
}

+ (NSArray<MPPConnection *> *)rightEyebrowConnections {
  return MPPFaceLandmarksRightEyebrow;
}

+ (NSArray<MPPConnection *> *)rightIrisConnections {
  return MPPFaceLandmarksLeftIris;
}

+ (NSArray<MPPConnection *> *)faceOvalConnections {
  return MPPFaceLandmarksFaceOval;
}

+ (NSArray<MPPConnection *> *)contoursConnections {
  return MPPFaceLandmarksContours;
}

+ (NSArray<MPPConnection *> *)tesselationConnections {
  return MPPFaceLandmarksTesselation;
}

+ (NSArray<MPPConnection *> *)faceConnections {
  return MPPFaceConnections;
}

- (void)dealloc {
  MpFaceLandmarkerPtr landmarker = _landmarker;
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
        MpFaceLandmarkerClose(landmarker, nullptr);
      }
      CallbackPool::FreeSlot(callbackSlot);
    });
  } else if (landmarker) {
    MpFaceLandmarkerClose(landmarker, nullptr);
  }
}

#pragma mark - Private

- (void)handleLiveStreamResult:(const MpFaceLandmarkerResult *)cResult
                        status:(MpStatus)status
                     timestamp:(int64_t)timestampInMilliseconds {
  if (![_faceLandmarkerLiveStreamDelegate
          respondsToSelector:@selector(faceLandmarker:didFinishDetectionWithResult:
                                       timestampInMilliseconds:error:)]) {
    return;
  }

  NSError *callbackError = nil;
  MPPFaceLandmarkerResult *result = nil;

  if (status != kMpOk) {
    [MPPCommonUtils checkCError:status errorMessage:nullptr toError:&callbackError];
  } else if (cResult) {
    result = [MPPFaceLandmarkerResult faceLandmarkerResultWithCResult:*cResult
                                              timestampInMilliseconds:timestampInMilliseconds];
  }

  dispatch_queue_t callbackQueue = _callbackQueue;
  if (!callbackQueue) {
    return;
  }
  __weak id<MPPFaceLandmarkerLiveStreamDelegate> delegate = _faceLandmarkerLiveStreamDelegate;
  MPPFaceLandmarker *__weak weakSelf = self;

  dispatch_async(callbackQueue, ^{
    MPPFaceLandmarker *strongSelf = weakSelf;
    if (strongSelf) {
      [delegate faceLandmarker:strongSelf
          didFinishDetectionWithResult:result
               timestampInMilliseconds:timestampInMilliseconds
                                 error:callbackError];
    }
  });
}

@end
