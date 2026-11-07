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

#import "mediapipe/tasks/ios/vision/image_segmenter/sources/MPPImageSegmenter.h"

#import "mediapipe/tasks/ios/common/sources/MPPCommon.h"
#import "mediapipe/tasks/ios/common/utils/sources/MPPCommonUtils.h"
#import "mediapipe/tasks/ios/vision/core/sources/MPPVisionTaskRunner.h"
#import "mediapipe/tasks/ios/vision/image_segmenter/utils/sources/MPPImageSegmenterOptions+Helpers.h"
#import "mediapipe/tasks/ios/vision/image_segmenter/utils/sources/MPPImageSegmenterResult+Helpers.h"

#include "mediapipe/tasks/c/core/common.h"
#include "mediapipe/tasks/c/vision/core/image_processing_options.h"
#include "mediapipe/tasks/c/vision/image_segmenter/image_segmenter.h"

static NSString *const kTaskName = @"imageSegmenter";

@interface MPPImageSegmenter () {
  MpImageSegmenterPtr _segmenter;
  MPPVisionTaskRunner *_taskRunner;
  dispatch_queue_t _callbackQueue;
  int _callbackSlot;
  NSArray<NSString *> *_labels;
}
@property(nonatomic, weak) id<MPPImageSegmenterLiveStreamDelegate> imageSegmenterLiveStreamDelegate;

- (BOOL)populateCResult:(MpImageSegmenterResult *)cResult
              withImage:(MPPImage *)image
                  error:(NSError **)error;
- (BOOL)populateCResult:(MpImageSegmenterResult *)cResult
             withVideoFrame:(MPPImage *)image
    timestampInMilliseconds:(NSInteger)timestampInMilliseconds
                      error:(NSError **)error;
- (void)handleLiveStreamResult:(const MpImageSegmenterResult *)cResult
                         status:(MpStatus)status
                      timestamp:(int64_t)timestampInMilliseconds;
@end

namespace {
using mediapipe::tasks::ios::vision::core::VisionCallbackPool;
using CallbackPool = VisionCallbackPool<MPPImageSegmenter, MpImageSegmenterResult,
                                        MpImageSegmenterOptions::result_callback_fn>;

static void OnLiveStreamResult(MPPImageSegmenter *task, MpStatus status,
                               const MpImageSegmenterResult *result, MpImagePtr image,
                               int64_t timestamp_ms) {
  if (task) {
    [task handleLiveStreamResult:result status:status timestamp:timestamp_ms];
  }
}
}  // namespace

@implementation MPPImageSegmenter

- (instancetype)initWithOptions:(MPPImageSegmenterOptions *)options error:(NSError **)error {
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
            normRectAllowed:YES
                hasDelegate:(options.imageSegmenterLiveStreamDelegate != nil)
                      error:error];
    if (!_taskRunner) {
      return nil;
    }

    if (options.imageSegmenterLiveStreamDelegate) {
      _imageSegmenterLiveStreamDelegate = options.imageSegmenterLiveStreamDelegate;
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

    MpImageSegmenterOptions cOptions = {};
    [options copyToCOptions:&cOptions];
    if (_callbackSlot >= 0) {
      cOptions.result_callback = CallbackPool::GetCallback(_callbackSlot);
    }

    char *errorMsg = nullptr;
    MpStatus status = MpImageSegmenterCreate(&cOptions, &_segmenter, &errorMsg);
    [MPPImageSegmenterOptions freeCOptions:&cOptions];

    if (![MPPCommonUtils checkCError:status errorMessage:errorMsg toError:error]) {
      if (_callbackSlot >= 0) {
        CallbackPool::FreeSlot(_callbackSlot);
        _callbackSlot = -1;
      }
      return nil;
    }

    MpStringList labelList = {};
    errorMsg = nullptr;
    status = MpImageSegmenterGetLabels(_segmenter, &labelList, &errorMsg);
    if (![MPPCommonUtils checkCError:status errorMessage:errorMsg toError:error]) {
      return nil;
    }
    NSMutableArray<NSString *> *labels =
        [NSMutableArray arrayWithCapacity:labelList.num_strings];
    for (int i = 0; i < labelList.num_strings; ++i) {
      NSString *label = (labelList.strings && labelList.strings[i])
                            ? [NSString stringWithUTF8String:labelList.strings[i]]
                            : nil;
      [labels addObject:label ?: @""];
    }
    _labels = [labels copy];
    MpStringListFree(&labelList);
  }
  return self;
}

- (instancetype)initWithModelPath:(NSString *)modelPath error:(NSError **)error {
  MPPImageSegmenterOptions *options = [[MPPImageSegmenterOptions alloc] init];
  options.baseOptions.modelAssetPath = modelPath;
  return [self initWithOptions:options error:error];
}

- (nullable MPPImageSegmenterResult *)segmentImage:(MPPImage *)image error:(NSError **)error {
  MpImageSegmenterResult cResult = {};
  if (![self populateCResult:&cResult withImage:image error:error]) {
    return nil;
  }
  MPPImageSegmenterResult *result =
      [MPPImageSegmenterResult imageSegmenterResultWithCResult:cResult
                                       timestampInMilliseconds:0
                                      shouldCopyMaskPacketData:YES];
  MpImageSegmenterCloseResult(&cResult);
  return result;
}

- (void)segmentImage:(MPPImage *)image
    withCompletionHandler:(void (^)(MPPImageSegmenterResult *_Nullable result,
                                    NSError *_Nullable error))completionHandler {
  if (!completionHandler) {
    return;
  }
  NSError *error = nil;
  MpImageSegmenterResult cResult = {};
  if (![self populateCResult:&cResult withImage:image error:&error]) {
    completionHandler(nil, error);
    return;
  }
  MPPImageSegmenterResult *result =
      [MPPImageSegmenterResult imageSegmenterResultWithCResult:cResult
                                       timestampInMilliseconds:0
                                      shouldCopyMaskPacketData:NO];
  completionHandler(result, nil);
  MpImageSegmenterCloseResult(&cResult);
}

- (nullable MPPImageSegmenterResult *)segmentVideoFrame:(MPPImage *)image
                                timestampInMilliseconds:(NSInteger)timestampInMilliseconds
                                                  error:(NSError **)error {
  MpImageSegmenterResult cResult = {};
  if (![self populateCResult:&cResult
              withVideoFrame:image
     timestampInMilliseconds:timestampInMilliseconds
                       error:error]) {
    return nil;
  }
  MPPImageSegmenterResult *result =
      [MPPImageSegmenterResult imageSegmenterResultWithCResult:cResult
                                       timestampInMilliseconds:timestampInMilliseconds
                                      shouldCopyMaskPacketData:YES];
  MpImageSegmenterCloseResult(&cResult);
  return result;
}

- (void)segmentVideoFrame:(MPPImage *)image
    timestampInMilliseconds:(NSInteger)timestampInMilliseconds
      withCompletionHandler:(void (^)(MPPImageSegmenterResult *_Nullable result,
                                      NSError *_Nullable error))completionHandler {
  if (!completionHandler) {
    return;
  }
  NSError *error = nil;
  MpImageSegmenterResult cResult = {};
  if (![self populateCResult:&cResult
              withVideoFrame:image
     timestampInMilliseconds:timestampInMilliseconds
                       error:&error]) {
    completionHandler(nil, error);
    return;
  }
  MPPImageSegmenterResult *result =
      [MPPImageSegmenterResult imageSegmenterResultWithCResult:cResult
                                       timestampInMilliseconds:timestampInMilliseconds
                                      shouldCopyMaskPacketData:NO];
  completionHandler(result, nil);
  MpImageSegmenterCloseResult(&cResult);
}

- (BOOL)segmentAsyncImage:(MPPImage *)image
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
      MpImageSegmenterSegmentAsync(_segmenter, mpImage, &imageProcessingOptions,
                                   static_cast<int64_t>(timestampInMilliseconds), &errorMsg);
  return [MPPCommonUtils checkCError:status errorMessage:errorMsg toError:error];
}

- (void)dealloc {
  MpImageSegmenterPtr segmenter = _segmenter;
  _segmenter = nullptr;
  int callbackSlot = _callbackSlot;
  _callbackSlot = -1;
  dispatch_queue_t callbackQueue = _callbackQueue;
  _callbackQueue = nil;

  if (callbackSlot >= 0) {
    dispatch_queue_t cleanupQueue =
        callbackQueue ?: dispatch_get_global_queue(QOS_CLASS_DEFAULT, 0);
    dispatch_async(cleanupQueue, ^{
      if (segmenter) {
        MpImageSegmenterClose(segmenter, nullptr);
      }
      CallbackPool::FreeSlot(callbackSlot);
    });
  } else if (segmenter) {
    MpImageSegmenterClose(segmenter, nullptr);
  }
}

#pragma mark - Private

- (BOOL)populateCResult:(MpImageSegmenterResult *)cResult
              withImage:(MPPImage *)image
                  error:(NSError **)error {
  if (![_taskRunner validateRunningMode:MPPRunningModeImage error:error]) {
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
  MpStatus status = MpImageSegmenterSegmentImage(_segmenter, mpImage, &imageProcessingOptions,
                                                 cResult, &errorMsg);
  return [MPPCommonUtils checkCError:status errorMessage:errorMsg toError:error];
}

- (BOOL)populateCResult:(MpImageSegmenterResult *)cResult
             withVideoFrame:(MPPImage *)image
    timestampInMilliseconds:(NSInteger)timestampInMilliseconds
                      error:(NSError **)error {
  if (![_taskRunner validateRunningMode:MPPRunningModeVideo error:error]) {
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
  MpStatus status = MpImageSegmenterSegmentForVideo(_segmenter, mpImage, &imageProcessingOptions,
                                                    static_cast<int64_t>(timestampInMilliseconds),
                                                    cResult, &errorMsg);
  return [MPPCommonUtils checkCError:status errorMessage:errorMsg toError:error];
}

- (void)handleLiveStreamResult:(const MpImageSegmenterResult *)cResult
                        status:(MpStatus)status
                     timestamp:(int64_t)timestampInMilliseconds {
  if (![self.imageSegmenterLiveStreamDelegate
          respondsToSelector:@selector(imageSegmenter:didFinishSegmentationWithResult:
                                       timestampInMilliseconds:error:)]) {
    return;
  }

  NSError *callbackError = nil;
  MPPImageSegmenterResult *result = nil;

  if (status != kMpOk) {
    [MPPCommonUtils checkCError:status errorMessage:nullptr toError:&callbackError];
  } else if (cResult) {
    result = [MPPImageSegmenterResult imageSegmenterResultWithCResult:*cResult
                                              timestampInMilliseconds:timestampInMilliseconds
                                             shouldCopyMaskPacketData:YES];
  }

  dispatch_queue_t callbackQueue = _callbackQueue;
  if (!callbackQueue) {
    return;
  }
  __weak id<MPPImageSegmenterLiveStreamDelegate> delegate = self.imageSegmenterLiveStreamDelegate;
  MPPImageSegmenter *__weak weakSelf = self;

  dispatch_async(callbackQueue, ^{
    MPPImageSegmenter *strongSelf = weakSelf;
    if (strongSelf) {
      [delegate imageSegmenter:strongSelf
          didFinishSegmentationWithResult:result
                  timestampInMilliseconds:timestampInMilliseconds
                                    error:callbackError];
    }
  });
}

@end
