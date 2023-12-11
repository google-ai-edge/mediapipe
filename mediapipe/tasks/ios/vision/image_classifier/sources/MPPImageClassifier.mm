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

#import "mediapipe/tasks/ios/common/utils/sources/MPPCommonUtils.h"
#import "mediapipe/tasks/ios/common/utils/sources/NSString+Helpers.h"
#import "mediapipe/tasks/ios/core/sources/MPPTaskInfo.h"
#import "mediapipe/tasks/ios/vision/core/sources/MPPVisionPacketCreator.h"
#import "mediapipe/tasks/ios/vision/core/sources/MPPVisionTaskRunner.h"
#import "mediapipe/tasks/ios/vision/image_classifier/utils/sources/MPPImageClassifierOptions+Helpers.h"
#import "mediapipe/tasks/ios/vision/image_classifier/utils/sources/MPPImageClassifierResult+Helpers.h"

#include "mediapipe/tasks/cc/components/containers/proto/classifications.pb.h"

namespace {
using ::mediapipe::NormalizedRect;
using ::mediapipe::Packet;
using ::mediapipe::Timestamp;
using ::mediapipe::tasks::core::PacketMap;
using ::mediapipe::tasks::core::PacketsCallback;
}  // namespace

static NSString *const kClassificationsStreamName = @"classifications_out";
static NSString *const kClassificationsTag = @"CLASSIFICATIONS";
static NSString *const kImageInStreamName = @"image_in";
static NSString *const kImageOutStreamName = @"image_out";
static NSString *const kImageTag = @"IMAGE";
static NSString *const kNormRectStreamName = @"norm_rect_in";
static NSString *const kNormRectTag = @"NORM_RECT";
static NSString *const kTaskGraphName =
    @"mediapipe.tasks.vision.image_classifier.ImageClassifierGraph";
static NSString *const kTaskName = @"imageClassifier";

static const int kMicrosecondsPerMillisecond = 1000;

#define InputPacketMap(imagePacket, normalizedRectPacket) \
  {                                                       \
    {kImageInStreamName.cppString, imagePacket}, {        \
      kNormRectStreamName.cppString, normalizedRectPacket \
    }                                                     \
  }

#define ImageClassifierResultWithOutputPacketMap(outputPacketMap)                                 \
  (                                                                                               \
    [MPPImageClassifierResult                                                                     \
        imageClassifierResultWithClassificationsPacket:outputPacketMap[kClassificationsStreamName \
                                                                           .cppString]]           \
  )

@interface MPPImageClassifier () {
  /** iOS Vision Task Runner */
  MPPVisionTaskRunner *_visionTaskRunner;
  dispatch_queue_t _callbackQueue;
}
@property(nonatomic, weak) id<MPPImageClassifierLiveStreamDelegate>
    imageClassifierLiveStreamDelegate;
@end

@implementation MPPImageClassifier

#pragma mark - Public

- (instancetype)initWithOptions:(MPPImageClassifierOptions *)options error:(NSError **)error {
  self = [super init];
  if (self) {
    MPPTaskInfo *taskInfo = [[MPPTaskInfo alloc]
        initWithTaskGraphName:kTaskGraphName
                 inputStreams:@[
                   [NSString stringWithFormat:@"%@:%@", kImageTag, kImageInStreamName],
                   [NSString stringWithFormat:@"%@:%@", kNormRectTag, kNormRectStreamName]
                 ]
                outputStreams:@[
                  [NSString
                      stringWithFormat:@"%@:%@", kClassificationsTag, kClassificationsStreamName],
                  [NSString stringWithFormat:@"%@:%@", kImageTag, kImageOutStreamName]
                ]
                  taskOptions:options
           enableFlowLimiting:options.runningMode == MPPRunningModeLiveStream
                        error:error];

    if (!taskInfo) {
      return nil;
    }

    PacketsCallback packetsCallback = nullptr;

    if (options.imageClassifierLiveStreamDelegate) {
      _imageClassifierLiveStreamDelegate = options.imageClassifierLiveStreamDelegate;

      // Create a private serial dispatch queue in which the deleagte method will be called
      // asynchronously. This is to ensure that if the client performs a long running operation in
      // the delegate method, the queue on which the C++ callbacks is invoked is not blocked and is
      // freed up to continue with its operations.
      _callbackQueue = dispatch_queue_create(
          [MPPVisionTaskRunner uniqueDispatchQueueNameWithSuffix:kTaskName], NULL);

      // Capturing `self` as weak in order to avoid `self` being kept in memory
      // and cause a retain cycle, after self is set to `nil`.
      MPPImageClassifier *__weak weakSelf = self;
      packetsCallback = [=](absl::StatusOr<PacketMap> liveStreamResult) {
        [weakSelf processLiveStreamResult:liveStreamResult];
      };
    }

    _visionTaskRunner = [[MPPVisionTaskRunner alloc] initWithTaskInfo:taskInfo
                                                          runningMode:options.runningMode
                                                           roiAllowed:YES
                                                      packetsCallback:std::move(packetsCallback)
                                                 imageInputStreamName:kImageInStreamName
                                              normRectInputStreamName:kNormRectStreamName
                                                                error:error];

    if (!_visionTaskRunner) {
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
  std::optional<PacketMap> outputPacketMap = [_visionTaskRunner processImage:image
                                                            regionOfInterest:roi
                                                                       error:error];

  return [MPPImageClassifier imageClassifierResultWithOptionalOutputPacketMap:outputPacketMap];
}

- (nullable MPPImageClassifierResult *)classifyImage:(MPPImage *)image error:(NSError **)error {
  return [self classifyImage:image regionOfInterest:CGRectZero error:error];
}

- (nullable MPPImageClassifierResult *)classifyVideoFrame:(MPPImage *)image
                                  timestampInMilliseconds:(NSInteger)timestampInMilliseconds
                                         regionOfInterest:(CGRect)roi
                                                    error:(NSError **)error {
  std::optional<PacketMap> outputPacketMap =
      [_visionTaskRunner processVideoFrame:image
                          regionOfInterest:roi
                   timestampInMilliseconds:timestampInMilliseconds
                                     error:error];

  return [MPPImageClassifier imageClassifierResultWithOptionalOutputPacketMap:outputPacketMap];
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
  return [_visionTaskRunner processLiveStreamImage:image
                                  regionOfInterest:roi
                           timestampInMilliseconds:timestampInMilliseconds
                                             error:error];
}

- (BOOL)classifyAsyncImage:(MPPImage *)image
    timestampInMilliseconds:(NSInteger)timestampInMilliseconds
                      error:(NSError **)error {
  return [self classifyAsyncImage:image
          timestampInMilliseconds:timestampInMilliseconds
                 regionOfInterest:CGRectZero
                            error:error];
}

#pragma mark - Private

- (void)processLiveStreamResult:(absl::StatusOr<PacketMap>)liveStreamResult {
  if (![self.imageClassifierLiveStreamDelegate
          respondsToSelector:@selector
          (imageClassifier:didFinishClassificationWithResult:timestampInMilliseconds:error:)]) {
    return;
  }

  NSError *callbackError = nil;
  if (![MPPCommonUtils checkCppError:liveStreamResult.status() toError:&callbackError]) {
    dispatch_async(_callbackQueue, ^{
      [self.imageClassifierLiveStreamDelegate imageClassifier:self
                            didFinishClassificationWithResult:nil
                                      timestampInMilliseconds:Timestamp::Unset().Value()
                                                        error:callbackError];
    });
    return;
  }

  PacketMap &outputPacketMap = liveStreamResult.value();
  if (outputPacketMap[kImageOutStreamName.cppString].IsEmpty()) {
    return;
  }

  MPPImageClassifierResult *result = ImageClassifierResultWithOutputPacketMap(outputPacketMap);

  NSInteger timestampInMilliseconds =
      outputPacketMap[kImageOutStreamName.cppString].Timestamp().Value() /
      kMicrosecondsPerMillisecond;
  dispatch_async(_callbackQueue, ^{
    [self.imageClassifierLiveStreamDelegate imageClassifier:self
                          didFinishClassificationWithResult:result
                                    timestampInMilliseconds:timestampInMilliseconds
                                                      error:callbackError];
  });
}

+ (nullable MPPImageClassifierResult *)imageClassifierResultWithOptionalOutputPacketMap:
    (std::optional<PacketMap>)outputPacketMap {
  if (!outputPacketMap.has_value()) {
    return nil;
  }

  return ImageClassifierResultWithOutputPacketMap(outputPacketMap.value());
}

@end
