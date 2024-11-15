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

#import "mediapipe/tasks/ios/audio/audio_embedder/sources/MPPAudioEmbedder.h"

#import "mediapipe/tasks/ios/audio/audio_embedder/utils/sources/MPPAudioEmbedderOptions+Helpers.h"
#import "mediapipe/tasks/ios/audio/audio_embedder/utils/sources/MPPAudioEmbedderResult+Helpers.h"
#import "mediapipe/tasks/ios/audio/core/sources/MPPAudioTaskRunner.h"
#import "mediapipe/tasks/ios/common/utils/sources/MPPCommonUtils.h"
#import "mediapipe/tasks/ios/common/utils/sources/NSString+Helpers.h"
#import "mediapipe/tasks/ios/core/sources/MPPTaskInfo.h"

namespace {
using ::mediapipe::Timestamp;
using ::mediapipe::tasks::core::PacketMap;
using ::mediapipe::tasks::core::PacketsCallback;
}  // namespace

static NSString *const kAudioTag = @"AUDIO";
static NSString *const kAudioInStreamName = @"audio_in";
static NSString *const kEmbeddingsOutStreamName = @"embeddings_out";
static NSString *const kEmbeddingsTag = @"EMBEDDINGS";
static NSString *const kSampleRateInStreamName = @"sample_rate_in";
static NSString *const kSampleRateTag = @"SAMPLE_RATE";
static NSString *const kTimestampedEmbeddingsOutStreamName = @"timestamped_embeddings_out";
static NSString *const kTimestampedEmbeddingsTag = @"TIMESTAMPED_EMBEDDINGS";

static NSString *const kTaskGraphName = @"mediapipe.tasks.audio.audio_embedder.AudioEmbedderGraph";
static NSString *const kTaskName = @"audioEmbedder";

static const int kMicrosecondsPerMillisecond = 1000;

#define AudioEmbedderResultWithOutputPacketMap(outputPacketMap, outputStreamName) \
  [MPPAudioEmbedderResult                                                         \
      audioEmbedderResultWithEmbeddingResultPacket:outputPacketMap[outputStreamName]]

@interface MPPAudioEmbedder () {
  /** iOS Audio Task Runner */
  MPPAudioTaskRunner *_audioTaskRunner;
  dispatch_queue_t _callbackQueue;
}
@property(nonatomic, weak) id<MPPAudioEmbedderStreamDelegate> audioEmbedderStreamDelegate;
@end

@implementation MPPAudioEmbedder

#pragma mark - Public

- (instancetype)initWithOptions:(MPPAudioEmbedderOptions *)options error:(NSError **)error {
  self = [super init];
  if (self) {
    MPPTaskInfo *taskInfo = [[MPPTaskInfo alloc]
        initWithTaskGraphName:kTaskGraphName
                 inputStreams:@[
                   [NSString stringWithFormat:@"%@:%@", kAudioTag, kAudioInStreamName],
                   [NSString stringWithFormat:@"%@:%@", kSampleRateTag, kSampleRateInStreamName]
                 ]
                outputStreams:@[
                  [NSString stringWithFormat:@"%@:%@", kEmbeddingsTag, kEmbeddingsOutStreamName],
                  [NSString stringWithFormat:@"%@:%@", kTimestampedEmbeddingsTag,
                                             kTimestampedEmbeddingsOutStreamName]
                ]
                  taskOptions:options
           enableFlowLimiting:NO
                        error:error];

    if (!taskInfo) {
      return nil;
    }

    PacketsCallback packetsCallback = nullptr;

    if (options.audioEmbedderStreamDelegate) {
      _audioEmbedderStreamDelegate = options.audioEmbedderStreamDelegate;

      // Create a private serial dispatch queue in which the deleagte method will be called
      // asynchronously. This is to ensure that if the client performs a long running operation in
      // the delegate method, the queue on which the C++ callbacks is invoked is not blocked and is
      // freed up to continue with its operations.
      _callbackQueue = dispatch_queue_create(
          [MPPAudioTaskRunner uniqueDispatchQueueNameWithSuffix:kTaskName], nullptr);

      // Capturing `self` as weak in order to avoid `self` being kept in memory
      // and cause a retain cycle, after self is set to `nil`.
      MPPAudioEmbedder *__weak weakSelf = self;
      packetsCallback = [=](absl::StatusOr<PacketMap> audioStreamResult) {
        [weakSelf processAudioStreamResult:audioStreamResult];
      };
    }

    _audioTaskRunner = [[MPPAudioTaskRunner alloc] initWithTaskInfo:taskInfo
                                                        runningMode:options.runningMode
                                                    packetsCallback:std::move(packetsCallback)
                                               audioInputStreamName:kAudioInStreamName
                                          sampleRateInputStreamName:kSampleRateInStreamName
                                                              error:error];

    if (!_audioTaskRunner) {
      return nil;
    }
  }
  return self;
}

- (instancetype)initWithModelPath:(NSString *)modelPath error:(NSError **)error {
  MPPAudioEmbedderOptions *options = [[MPPAudioEmbedderOptions alloc] init];

  options.baseOptions.modelAssetPath = modelPath;

  return [self initWithOptions:options error:error];
}

- (nullable MPPAudioEmbedderResult *)embedAudioClip:(MPPAudioData *)audioClip
                                              error:(NSError **)error {
  std::optional<PacketMap> outputPacketMap = [_audioTaskRunner processAudioClip:audioClip
                                                                          error:error];

  if (!outputPacketMap.has_value()) {
    return nil;
  }

  PacketMap &outputPacketMapValue = outputPacketMap.value();

  return AudioEmbedderResultWithOutputPacketMap(outputPacketMapValue,
                                                kTimestampedEmbeddingsOutStreamName.cppString);
}

- (BOOL)embedAsyncAudioBlock:(MPPAudioData *)audioBlock
     timestampInMilliseconds:(NSInteger)timestampInMilliseconds
                       error:(NSError **)error {
  return [_audioTaskRunner processStreamAudioClip:audioBlock
                          timestampInMilliseconds:timestampInMilliseconds
                                            error:error];
}

+ (MPPAudioRecord *)createAudioRecordWithChannelCount:(NSUInteger)channelCount
                                           sampleRate:(double)sampleRate
                                         bufferLength:(NSUInteger)bufferLength
                                                error:(NSError **)error {
  return [MPPAudioTaskRunner createAudioRecordWithChannelCount:channelCount
                                                    sampleRate:sampleRate
                                                  bufferLength:bufferLength
                                                         error:error];
}

- (BOOL)closeWithError:(NSError **)error {
  return [_audioTaskRunner closeWithError:error];
}

#pragma mark - Private

- (void)processAudioStreamResult:(absl::StatusOr<PacketMap>)audioStreamResult {
  if (![self.audioEmbedderStreamDelegate
          respondsToSelector:@selector(audioEmbedder:
                                 didFinishEmbeddingWithResult:timestampInMilliseconds:error:)]) {
    return;
  }

  NSError *callbackError = nil;
  if (![MPPCommonUtils checkCppError:audioStreamResult.status() toError:&callbackError]) {
    dispatch_async(_callbackQueue, ^{
      [self.audioEmbedderStreamDelegate audioEmbedder:self
                         didFinishEmbeddingWithResult:nil
                              timestampInMilliseconds:Timestamp::Unset().Value()
                                                error:callbackError];
    });
    return;
  }

  PacketMap &outputPacketMap = audioStreamResult.value();
  std::string cppEmbeddingsOutStreamName = kEmbeddingsOutStreamName.cppString;

  MPPAudioEmbedderResult *result =
      AudioEmbedderResultWithOutputPacketMap(outputPacketMap, cppEmbeddingsOutStreamName);

  NSInteger timestampInMilliseconds =
      outputPacketMap[cppEmbeddingsOutStreamName].Timestamp().Value() / kMicrosecondsPerMillisecond;
  dispatch_async(_callbackQueue, ^{
    [self.audioEmbedderStreamDelegate audioEmbedder:self
                       didFinishEmbeddingWithResult:result
                            timestampInMilliseconds:timestampInMilliseconds
                                              error:callbackError];
  });
}

@end
