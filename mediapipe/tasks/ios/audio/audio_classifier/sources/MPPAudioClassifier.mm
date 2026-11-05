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

#import "mediapipe/tasks/ios/audio/audio_classifier/sources/MPPAudioClassifier.h"

#import "mediapipe/tasks/ios/audio/audio_classifier/utils/sources/MPPAudioClassifierOptions+Helpers.h"
#import "mediapipe/tasks/ios/audio/audio_classifier/utils/sources/MPPAudioClassifierResult+Helpers.h"
#import "mediapipe/tasks/ios/audio/core/sources/MPPAudioRecord.h"
#import "mediapipe/tasks/ios/common/sources/MPPCommon.h"
#import "mediapipe/tasks/ios/common/utils/sources/MPPCommonUtils.h"

#include <array>
#include <cstdint>
#include <limits>
#include <mutex>
#include <utility>

#include "mediapipe/tasks/c/audio/audio_classifier/audio_classifier.h"

@interface MPPAudioClassifier () {
  MpAudioClassifierPtr _classifier;
  std::mutex _classifierMutex;
  MPPAudioRunningMode _runningMode;
  dispatch_queue_t _callbackQueue;
  int _slotIndex;
}

@property(nonatomic, weak) id<MPPAudioClassifierStreamDelegate> audioClassifierStreamDelegate;

- (void)handleAsyncResult:(MpAudioClassifierResult *)result status:(MpStatus)status;

@end

namespace {

constexpr size_t kMaxAudioClassifierSlots = 64;

struct AudioClassifierSlot {
  __weak MPPAudioClassifier *classifier;
  bool in_use = false;
};

struct AudioClassifierSlotPool {
  std::mutex mutex;
  std::array<AudioClassifierSlot, kMaxAudioClassifierSlots> slots;
};

AudioClassifierSlotPool &GetAudioClassifierSlotPool() {
  static auto *const kPool = new AudioClassifierSlotPool();
  return *kPool;
}

int AllocateCallbackSlot(MPPAudioClassifier *classifier) {
  if (classifier == nil) {
    return -1;
  }
  auto &pool = GetAudioClassifierSlotPool();
  std::lock_guard<std::mutex> lock(pool.mutex);
  for (size_t i = 0; i < kMaxAudioClassifierSlots; ++i) {
    if (!pool.slots[i].in_use && pool.slots[i].classifier == nil) {
      pool.slots[i].classifier = classifier;
      pool.slots[i].in_use = true;
      return static_cast<int>(i);
    }
  }
  return -1;
}

void FreeCallbackSlot(int slot) {
  if (slot < 0 || slot >= static_cast<int>(kMaxAudioClassifierSlots)) {
    return;
  }
  auto &pool = GetAudioClassifierSlotPool();
  std::lock_guard<std::mutex> lock(pool.mutex);
  pool.slots[slot].classifier = nil;
  pool.slots[slot].in_use = false;
}

void HandleAudioClassifierCallback(size_t slot_index, MpStatus status,
                                   MpAudioClassifierResult *result) {
  CFTypeRef retainedClassifier = nullptr;
  @autoreleasepool {
    MPPAudioClassifier *strongClassifier = nil;
    {
      auto &pool = GetAudioClassifierSlotPool();
      std::lock_guard<std::mutex> lock(pool.mutex);
      if (slot_index < kMaxAudioClassifierSlots && pool.slots[slot_index].in_use) {
        strongClassifier = pool.slots[slot_index].classifier;
      }
    }
    if (strongClassifier != nil) {
      retainedClassifier = (__bridge_retained CFTypeRef)strongClassifier;
      strongClassifier = nil;
      [(__bridge MPPAudioClassifier *)retainedClassifier handleAsyncResult:result status:status];
    }
  }
  if (retainedClassifier != nullptr) {
    dispatch_async(dispatch_get_global_queue(QOS_CLASS_DEFAULT, 0), ^{
      CFRelease(retainedClassifier);
    });
  }
}

template <size_t N>
struct AudioTrampoline {
  static void Callback(MpStatus status, MpAudioClassifierResult *result) {
    HandleAudioClassifierCallback(N, status, result);
  }
};

template <size_t... Is>
constexpr auto MakeTrampolineTable(std::index_sequence<Is...>) {
  return std::array<MpAudioClassifierOptions::result_callback_fn, sizeof...(Is)>{
      (&AudioTrampoline<Is>::Callback)...};
}

constexpr auto kAudioTrampolines =
    MakeTrampolineTable(std::make_index_sequence<kMaxAudioClassifierSlots>{});

MPPFloatBuffer *ExtractAudioBufferAndPopulateCAudioData(MPPAudioData *audioData,
                                                        NSString *paramName,
                                                        MpAudioData *outCAudioData,
                                                        NSError **error) {
  if (!audioData) {
    [MPPCommonUtils createCustomError:error
                             withCode:MPPTasksErrorCodeInvalidArgumentError
                          description:[NSString stringWithFormat:@"%@ cannot be nil.", paramName]];
    return nil;
  }

  if (audioData.format.channelCount == 0 ||
      audioData.format.channelCount > static_cast<NSUInteger>(std::numeric_limits<int>::max())) {
    [MPPCommonUtils
        createCustomError:error
                 withCode:MPPTasksErrorCodeInvalidArgumentError
              description:@"Audio format channel count must be between 1 and INT_MAX."];
    return nil;
  }
  if (audioData.bufferLength == 0 ||
      audioData.bufferLength > static_cast<NSUInteger>(std::numeric_limits<int>::max())) {
    [MPPCommonUtils
        createCustomError:error
                 withCode:MPPTasksErrorCodeInvalidArgumentError
              description:@"Audio data buffer length must be between 1 and INT_MAX."];
    return nil;
  }

  NS_VALID_UNTIL_END_OF_SCOPE MPPFloatBuffer *audioBuffer = audioData.buffer;
  if (!audioBuffer.data) {
    [MPPCommonUtils createCustomError:error
                             withCode:MPPTasksErrorCodeInvalidArgumentError
                          description:@"Audio data buffer cannot be nil."];
    return nil;
  }

  *outCAudioData = {
      .num_channels = static_cast<int>(audioData.format.channelCount),
      .sample_rate = audioData.format.sampleRate,
      .audio_data = audioBuffer.data,
      .audio_data_size = static_cast<size_t>(audioBuffer.length),
  };
  return audioBuffer;
}

}  // namespace

@implementation MPPAudioClassifier

#pragma mark - Public

- (nullable instancetype)initWithOptions:(MPPAudioClassifierOptions *)options
                                   error:(NSError **)error {
  if (!options) {
    [MPPCommonUtils createCustomError:error
                             withCode:MPPTasksErrorCodeInvalidArgumentError
                          description:@"options cannot be nil."];
    return nil;
  }

  switch (options.runningMode) {
    case MPPAudioRunningModeAudioClips: {
      if (options.audioClassifierStreamDelegate) {
        [MPPCommonUtils createCustomError:error
                                 withCode:MPPTasksErrorCodeInvalidArgumentError
                              description:@"The audio task is in audio clips mode. The "
                                          @"delegate must not be set in the task's options."];
        return nil;
      }
      break;
    }
    case MPPAudioRunningModeAudioStream: {
      if (!options.audioClassifierStreamDelegate) {
        [MPPCommonUtils
            createCustomError:error
                     withCode:MPPTasksErrorCodeInvalidArgumentError
                  description:
                      @"The audio task is in audio stream mode. An object must be set as the "
                      @"delegate of the task in its options to ensure asynchronous delivery of "
                      @"results."];
        return nil;
      }
      break;
    }
    default: {
      [MPPCommonUtils createCustomError:error
                               withCode:MPPTasksErrorCodeInvalidArgumentError
                            description:@"Unrecognized running mode"];
      return nil;
    }
  }

  self = [super init];
  if (!self) {
    return nil;
  }

  _runningMode = options.runningMode;
  _slotIndex = -1;

  MpAudioClassifierOptions cOptions = {};
  [options copyToCOptions:&cOptions];

  if (_runningMode == MPPAudioRunningModeAudioStream) {
    _audioClassifierStreamDelegate = options.audioClassifierStreamDelegate;

    _slotIndex = AllocateCallbackSlot(self);
    if (_slotIndex < 0) {
      [MPPAudioClassifierOptions freeCOptions:&cOptions];
      [MPPCommonUtils createCustomError:error
                               withCode:MPPTasksErrorCodeInternalError
                            description:@"Exceeded maximum number of concurrent AudioClassifier "
                                        @"stream instances (64)."];
      return nil;
    }

    _callbackQueue = dispatch_queue_create(
        [NSString stringWithFormat:@"com.mediapipe.tasks.audio.audioClassifier_%@",
                                   [NSUUID UUID].UUIDString]
            .UTF8String,
        nullptr);

    cOptions.result_callback = kAudioTrampolines[_slotIndex];
  }

  char *errorMsg = nullptr;
  MpStatus status = MpAudioClassifierCreate(&cOptions, &_classifier, &errorMsg);
  [MPPAudioClassifierOptions freeCOptions:&cOptions];

  if (![MPPCommonUtils checkCError:status errorMessage:errorMsg toError:error]) {
    if (_slotIndex >= 0) {
      FreeCallbackSlot(_slotIndex);
      _slotIndex = -1;
    }
    return nil;
  }

  return self;
}

- (nullable instancetype)initWithModelPath:(NSString *)modelPath error:(NSError **)error {
  MPPAudioClassifierOptions *options = [[MPPAudioClassifierOptions alloc] init];
  options.baseOptions.modelAssetPath = modelPath;
  return [self initWithOptions:options error:error];
}

- (nullable MPPAudioClassifierResult *)classifyAudioClip:(MPPAudioData *)audioClip
                                                   error:(NSError **)error {
  std::lock_guard<std::mutex> lock(_classifierMutex);
  if (!_classifier) {
    [MPPCommonUtils createCustomError:error
                             withCode:MPPTasksErrorCodeInvalidArgumentError
                          description:@"INVALID_ARGUMENT: Task runner is currently not running."];
    return nil;
  }

  if (_runningMode != MPPAudioRunningModeAudioClips) {
    [MPPCommonUtils
        createCustomError:error
                 withCode:MPPTasksErrorCodeInvalidArgumentError
              description:[NSString stringWithFormat:@"The audio task is not initialized with "
                                                     @"audio clips. Current Running Mode: %@",
                                                     MPPAudioRunningModeDisplayName(_runningMode)]];
    return nil;
  }

  MpAudioData cAudioData = {};
  NS_VALID_UNTIL_END_OF_SCOPE MPPFloatBuffer *audioBuffer =
      ExtractAudioBufferAndPopulateCAudioData(audioClip, @"audioClip", &cAudioData, error);
  if (!audioBuffer) {
    return nil;
  }

  MpAudioClassifierResult cResult = {};
  char *errorMsg = nullptr;
  MpStatus status = MpAudioClassifierClassify(_classifier, &cAudioData, &cResult, &errorMsg);
  if (![MPPCommonUtils checkCError:status errorMessage:errorMsg toError:error]) {
    return nil;
  }

  MPPAudioClassifierResult *result =
      [MPPAudioClassifierResult audioClassifierResultWithCResult:cResult timestampInMilliseconds:0];
  MpAudioClassifierCloseResult(&cResult);
  return result;
}

- (BOOL)classifyAsyncAudioBlock:(MPPAudioData *)audioBlock
        timestampInMilliseconds:(NSInteger)timestampInMilliseconds
                          error:(NSError **)error {
  std::lock_guard<std::mutex> lock(_classifierMutex);
  if (!_classifier) {
    [MPPCommonUtils createCustomError:error
                             withCode:MPPTasksErrorCodeInvalidArgumentError
                          description:@"INVALID_ARGUMENT: Task runner is currently not running."];
    return NO;
  }

  if (_runningMode != MPPAudioRunningModeAudioStream) {
    [MPPCommonUtils
        createCustomError:error
                 withCode:MPPTasksErrorCodeInvalidArgumentError
              description:[NSString stringWithFormat:@"The audio task is not initialized with "
                                                     @"audio stream mode. Current Running Mode: %@",
                                                     MPPAudioRunningModeDisplayName(_runningMode)]];
    return NO;
  }

  MpAudioData cAudioData = {};
  NS_VALID_UNTIL_END_OF_SCOPE MPPFloatBuffer *audioBuffer =
      ExtractAudioBufferAndPopulateCAudioData(audioBlock, @"audioBlock", &cAudioData, error);
  if (!audioBuffer) {
    return NO;
  }

  char *errorMsg = nullptr;
  MpStatus status = MpAudioClassifierClassifyAsync(
      _classifier, &cAudioData, static_cast<int64_t>(timestampInMilliseconds), &errorMsg);
  return [MPPCommonUtils checkCError:status errorMessage:errorMsg toError:error];
}

+ (MPPAudioRecord *)createAudioRecordWithChannelCount:(NSUInteger)channelCount
                                           sampleRate:(double)sampleRate
                                         bufferLength:(NSUInteger)bufferLength
                                                error:(NSError **)error {
  MPPAudioDataFormat *format = [[MPPAudioDataFormat alloc] initWithChannelCount:channelCount
                                                                     sampleRate:sampleRate];
  return [[MPPAudioRecord alloc] initWithAudioDataFormat:format
                                            bufferLength:bufferLength
                                                   error:error];
}

- (BOOL)closeWithError:(NSError **)error {
  std::lock_guard<std::mutex> lock(_classifierMutex);
  if (!_classifier) {
    return YES;
  }

  char *errorMsg = nullptr;
  MpStatus status = MpAudioClassifierClose(_classifier, &errorMsg);
  _classifier = nullptr;

  if (_slotIndex >= 0) {
    FreeCallbackSlot(_slotIndex);
    _slotIndex = -1;
  }

  return [MPPCommonUtils checkCError:status errorMessage:errorMsg toError:error];
}

- (void)dealloc {
  [self closeWithError:nil];
}

#pragma mark - Private

- (void)handleAsyncResult:(MpAudioClassifierResult *)result status:(MpStatus)status {
  if (![self.audioClassifierStreamDelegate
          respondsToSelector:@selector(audioClassifier:didFinishClassificationWithResult:
                                       timestampInMilliseconds:error:)]) {
    return;
  }

  if (status != kMpOk) {
    NSError *callbackError = nil;
    [MPPCommonUtils checkCError:status errorMessage:nullptr toError:&callbackError];
    dispatch_async(_callbackQueue, ^{
      [self.audioClassifierStreamDelegate
                            audioClassifier:self
          didFinishClassificationWithResult:nil
                    timestampInMilliseconds:std::numeric_limits<int64_t>::min()
                                      error:callbackError];
    });
    return;
  }

  NSInteger timestampInMilliseconds = 0;
  if (result && result->results_count > 0 && result->results[0].has_timestamp_ms) {
    timestampInMilliseconds = static_cast<NSInteger>(result->results[0].timestamp_ms);
  }

  MPPAudioClassifierResult *objcResult = nil;
  if (result) {
    objcResult =
        [MPPAudioClassifierResult audioClassifierResultWithCResult:*result
                                           timestampInMilliseconds:timestampInMilliseconds];
  }

  dispatch_async(_callbackQueue, ^{
    [self.audioClassifierStreamDelegate audioClassifier:self
                      didFinishClassificationWithResult:objcResult
                                timestampInMilliseconds:timestampInMilliseconds
                                                  error:nil];
  });
}

@end
