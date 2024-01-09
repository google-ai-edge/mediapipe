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

#import "mediapipe/tasks/ios/common/utils/sources/MPPCommonUtils.h"
#import "mediapipe/tasks/ios/common/utils/sources/NSString+Helpers.h"
#import "mediapipe/tasks/ios/components/utils/sources/MPPCosineSimilarity.h"
#import "mediapipe/tasks/ios/core/sources/MPPTaskInfo.h"
#import "mediapipe/tasks/ios/vision/core/sources/MPPVisionPacketCreator.h"
#import "mediapipe/tasks/ios/vision/core/sources/MPPVisionTaskRunner.h"
#import "mediapipe/tasks/ios/vision/image_embedder/utils/sources/MPPImageEmbedderOptions+Helpers.h"
#import "mediapipe/tasks/ios/vision/image_embedder/utils/sources/MPPImageEmbedderResult+Helpers.h"

namespace {
using ::mediapipe::Timestamp;
using ::mediapipe::tasks::core::PacketMap;
using ::mediapipe::tasks::core::PacketsCallback;
}  // namespace

static NSString *const kEmbeddingsOutStreamName = @"embeddings_out";
static NSString *const kEmbeddingsTag = @"EMBEDDINGS";
static NSString *const kImageInStreamName = @"image_in";
static NSString *const kImageOutStreamName = @"image_out";
static NSString *const kImageTag = @"IMAGE";
static NSString *const kNormRectStreamName = @"norm_rect_in";
static NSString *const kNormRectTag = @"NORM_RECT";
static NSString *const kTaskGraphName = @"mediapipe.tasks.vision.image_embedder.ImageEmbedderGraph";
static NSString *const kTaskName = @"imageEmbedder";

static const int kMicrosecondsPerMillisecond = 1000;

#define InputPacketMap(imagePacket, normalizedRectPacket) \
  {                                                       \
    {kImageInStreamName.cppString, imagePacket}, {        \
      kNormRectStreamName.cppString, normalizedRectPacket \
    }                                                     \
  }

#define ImageEmbedderResultWithOutputPacketMap(outputPacketMap)                             \
  ([MPPImageEmbedderResult                                                                  \
      imageEmbedderResultWithEmbeddingResultPacket:outputPacketMap[kEmbeddingsOutStreamName \
                                                                       .cppString]])

@interface MPPImageEmbedder () {
  /** iOS Vision Task Runner */
  MPPVisionTaskRunner *_visionTaskRunner;
  dispatch_queue_t _callbackQueue;
}
@property(nonatomic, weak) id<MPPImageEmbedderLiveStreamDelegate> imageEmbedderLiveStreamDelegate;
@end

@implementation MPPImageEmbedder

#pragma mark - Public

- (instancetype)initWithOptions:(MPPImageEmbedderOptions *)options error:(NSError **)error {
  self = [super init];
  if (self) {
    MPPTaskInfo *taskInfo = [[MPPTaskInfo alloc]
        initWithTaskGraphName:kTaskGraphName
                 inputStreams:@[
                   [NSString stringWithFormat:@"%@:%@", kImageTag, kImageInStreamName],
                   [NSString stringWithFormat:@"%@:%@", kNormRectTag, kNormRectStreamName]
                 ]
                outputStreams:@[
                  [NSString stringWithFormat:@"%@:%@", kEmbeddingsTag, kEmbeddingsOutStreamName],
                  [NSString stringWithFormat:@"%@:%@", kImageTag, kImageOutStreamName]
                ]
                  taskOptions:options
           enableFlowLimiting:options.runningMode == MPPRunningModeLiveStream
                        error:error];

    if (!taskInfo) {
      return nil;
    }

    PacketsCallback packetsCallback = nullptr;

    if (options.imageEmbedderLiveStreamDelegate) {
      _imageEmbedderLiveStreamDelegate = options.imageEmbedderLiveStreamDelegate;

      // Create a private serial dispatch queue in which the deleagte method will be called
      // asynchronously. This is to ensure that if the client performs a long running operation in
      // the delegate method, the queue on which the C++ callbacks is invoked is not blocked and is
      // freed up to continue with its operations.
      _callbackQueue = dispatch_queue_create(
          [MPPVisionTaskRunner uniqueDispatchQueueNameWithSuffix:kTaskName], nullptr);

      // Capturing `self` as weak in order to avoid `self` being kept in memory
      // and cause a retain cycle, after self is set to `nil`.
      MPPImageEmbedder *__weak weakSelf = self;
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
  MPPImageEmbedderOptions *options = [[MPPImageEmbedderOptions alloc] init];

  options.baseOptions.modelAssetPath = modelPath;

  return [self initWithOptions:options error:error];
}

- (nullable MPPImageEmbedderResult *)embedImage:(MPPImage *)image
                               regionOfInterest:(CGRect)roi
                                          error:(NSError **)error {
  std::optional<PacketMap> outputPacketMap = [_visionTaskRunner processImage:image
                                                            regionOfInterest:roi
                                                                       error:error];

  return [MPPImageEmbedder imageEmbedderResultWithOptionalOutputPacketMap:outputPacketMap];
}

- (nullable MPPImageEmbedderResult *)embedImage:(MPPImage *)image error:(NSError **)error {
  return [self embedImage:image regionOfInterest:CGRectZero error:error];
}

- (nullable MPPImageEmbedderResult *)embedVideoFrame:(MPPImage *)image
                             timestampInMilliseconds:(NSInteger)timestampInMilliseconds
                                    regionOfInterest:(CGRect)roi
                                               error:(NSError **)error {
  std::optional<PacketMap> outputPacketMap =
      [_visionTaskRunner processVideoFrame:image
                          regionOfInterest:roi
                   timestampInMilliseconds:timestampInMilliseconds
                                     error:error];

  return [MPPImageEmbedder imageEmbedderResultWithOptionalOutputPacketMap:outputPacketMap];
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
  return [_visionTaskRunner processLiveStreamImage:image
                                  regionOfInterest:roi
                           timestampInMilliseconds:timestampInMilliseconds
                                             error:error];
}

- (BOOL)embedAsyncImage:(MPPImage *)image
    timestampInMilliseconds:(NSInteger)timestampInMilliseconds
                      error:(NSError **)error {
  return [self embedAsyncImage:image
       timestampInMilliseconds:timestampInMilliseconds
              regionOfInterest:CGRectZero
                         error:error];
}

+ (nullable NSNumber *)cosineSimilarityBetweenEmbedding1:(MPPEmbedding *)embedding1
                                           andEmbedding2:(MPPEmbedding *)embedding2
                                                   error:(NSError **)error {
  return [MPPCosineSimilarity computeBetweenEmbedding1:embedding1
                                         andEmbedding2:embedding2
                                                 error:error];
}

#pragma mark - Private

- (void)processLiveStreamResult:(absl::StatusOr<PacketMap>)liveStreamResult {
  if (![self.imageEmbedderLiveStreamDelegate
          respondsToSelector:@selector(imageEmbedder:
                                 didFinishEmbeddingWithResult:timestampInMilliseconds:error:)]) {
    return;
  }

  NSError *callbackError = nil;
  if (![MPPCommonUtils checkCppError:liveStreamResult.status() toError:&callbackError]) {
    dispatch_async(_callbackQueue, ^{
      [self.imageEmbedderLiveStreamDelegate imageEmbedder:self
                             didFinishEmbeddingWithResult:nil
                                  timestampInMilliseconds:Timestamp::Unset().Value()
                                                    error:callbackError];
    });
    return;
  }

  PacketMap &outputPacketMap = liveStreamResult.value();
  if (outputPacketMap[kImageOutStreamName.cppString].IsEmpty()) {
    return;
  }

  MPPImageEmbedderResult *result = ImageEmbedderResultWithOutputPacketMap(outputPacketMap);

  NSInteger timestampInMilliseconds =
      outputPacketMap[kImageOutStreamName.cppString].Timestamp().Value() /
      kMicrosecondsPerMillisecond;
  dispatch_async(_callbackQueue, ^{
    [self.imageEmbedderLiveStreamDelegate imageEmbedder:self
                           didFinishEmbeddingWithResult:result
                                timestampInMilliseconds:timestampInMilliseconds
                                                  error:callbackError];
  });
}

+ (nullable MPPImageEmbedderResult *)imageEmbedderResultWithOptionalOutputPacketMap:
    (std::optional<PacketMap>)outputPacketMap {
  if (!outputPacketMap.has_value()) {
    return nil;
  }

  return ImageEmbedderResultWithOutputPacketMap(outputPacketMap.value());
}

@end
