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

#import "mediapipe/tasks/ios/audio/core/sources/MPPAudioTaskRunner.h"

#import "mediapipe/tasks/ios/audio/core/sources/MPPAudioPacketCreator.h"
#import "mediapipe/tasks/ios/common/sources/MPPCommon.h"
#import "mediapipe/tasks/ios/common/utils/sources/MPPCommonUtils.h"
#import "mediapipe/tasks/ios/common/utils/sources/NSString+Helpers.h"
#import "mediapipe/tasks/ios/core/sources/MPPPacketCreator.h"
#import "mediapipe/tasks/ios/core/sources/MPPTaskInfo.h"

#include <optional>

namespace {
using ::mediapipe::Packet;
using ::mediapipe::Timestamp;
using ::mediapipe::tasks::core::PacketMap;
using ::mediapipe::tasks::core::PacketsCallback;
}  // namespace

static NSString *const kTaskPrefix = @"com.mediapipe.tasks.audio";
static const double kInitialDefaultSampleRate = -1.0f;

@interface MPPAudioTaskRunner () {
  MPPAudioRunningMode _runningMode;
  NSString *_audioInputStreamName;
  NSString *_sampleRateInputStreamName;
  double _sampleRate;
}
@end

@implementation MPPAudioTaskRunner

- (nullable instancetype)initWithTaskInfo:(MPPTaskInfo *)taskInfo
                              runningMode:(MPPAudioRunningMode)runningMode
                          packetsCallback:(mediapipe::tasks::core::PacketsCallback)packetsCallback
                     audioInputStreamName:(NSString *)audioInputStreamName
                sampleRateInputStreamName:(nullable NSString *)sampleRateInputStreamName
                                    error:(NSError **)error {
  if (!taskInfo) {
    [MPPCommonUtils createCustomError:error
                             withCode:MPPTasksErrorCodeInvalidArgumentError
                          description:@"`taskInfo` cannot be `nil`."];
    return nil;
  }

  if (!audioInputStreamName) {
    [MPPCommonUtils createCustomError:error
                             withCode:MPPTasksErrorCodeInvalidArgumentError
                          description:@"`audioInputStreamName` cannot be `nil.`"];
    return nil;
  }

  if (!sampleRateInputStreamName) {
    [MPPCommonUtils createCustomError:error
                             withCode:MPPTasksErrorCodeInvalidArgumentError
                          description:@"`sampleRateInputStreamName` cannot be `nil.`"];
    return nil;
  }

  _audioInputStreamName = audioInputStreamName;
  _sampleRateInputStreamName = sampleRateInputStreamName;

  switch (runningMode) {
    case MPPAudioRunningModeAudioClips: {
      if (packetsCallback) {
        [MPPCommonUtils createCustomError:error
                                 withCode:MPPTasksErrorCodeInvalidArgumentError
                              description:@"The audio task is in audio clips mode. The "
                                          @"delegate must not be set in the task's options."];
        return nil;
      }
      break;
    }
    case MPPAudioRunningModeAudioStream: {
      if (!packetsCallback) {
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

  _runningMode = runningMode;
  _sampleRate = kInitialDefaultSampleRate;

  self = [super initWithTaskInfo:taskInfo packetsCallback:packetsCallback error:error];
  return self;
}

- (std::optional<mediapipe::tasks::core::PacketMap>)processAudioClip:(MPPAudioData *)audioClip
                                                               error:(NSError **)error {
  if (_runningMode != MPPAudioRunningModeAudioClips) {
    [MPPCommonUtils
        createCustomError:error
                 withCode:MPPTasksErrorCodeInvalidArgumentError
              description:[NSString stringWithFormat:@"The audio task is not initialized with "
                                                     @"audio clips. Current Running Mode: %@",
                                                     MPPAudioRunningModeDisplayName(_runningMode)]];
    return std::nullopt;
  }

  std::optional<PacketMap> inputPacketMap = [self inputPacketMapWithMPPAudioData:audioClip
                                                                           error:error];

  return inputPacketMap.has_value() ? [self processPacketMap:inputPacketMap.value() error:error]
                                    : std::nullopt;
}

- (BOOL)processStreamAudioClip:(MPPAudioData *)audioClip
       timestampInMilliseconds:(NSInteger)timestampInMilliseconds
                         error:(NSError **)error {
  if (_runningMode != MPPAudioRunningModeAudioStream) {
    [MPPCommonUtils
        createCustomError:error
                 withCode:MPPTasksErrorCodeInvalidArgumentError
              description:[NSString stringWithFormat:@"The audio task is not initialized with "
                                                     @"audio stream mode. Current Running Mode: %@",
                                                     MPPAudioRunningModeDisplayName(_runningMode)]];
    return NO;
  }

  if (![self checkOrSetSampleRate:audioClip.format.sampleRate error:error]) {
    return NO;
  }

  Packet matrixPacket = [MPPAudioPacketCreator createPacketWithAudioData:audioClip
                                                 timestampInMilliseconds:timestampInMilliseconds
                                                                   error:error];
  if (matrixPacket.IsEmpty()) {
    return NO;
  }

  PacketMap inputPacketMap = {{_audioInputStreamName.cppString, matrixPacket}};
  return [self sendPacketMap:inputPacketMap error:error];
}

- (BOOL)checkOrSetSampleRate:(double)sampleRate error:(NSError **)error {
  if (_runningMode != MPPAudioRunningModeAudioStream) {
    [MPPCommonUtils
        createCustomError:error
                 withCode:MPPTasksErrorCodeInvalidArgumentError
              description:[NSString stringWithFormat:@"The audio task is not initialized with "
                                                     @"audio stream mode. Current Running Mode: %@",
                                                     MPPAudioRunningModeDisplayName(_runningMode)]];
    return NO;
  }

  if (_sampleRate == kInitialDefaultSampleRate) {
    // Sample rate hasn't been initialized yet
    PacketMap inputPacketMap = {
        {_sampleRateInputStreamName.cppString,
         [MPPPacketCreator createWithDouble:sampleRate].At(Timestamp::PreStream())}};
    BOOL sendStatus = [self sendPacketMap:inputPacketMap error:error];
    if (sendStatus) {
      _sampleRate = sampleRate;
    }
    return sendStatus;
  }

  if (sampleRate != _sampleRate) {
    [MPPCommonUtils
        createCustomError:error
                 withCode:MPPTasksErrorCodeInvalidArgumentError
              description:[NSString
                              stringWithFormat:@"The input audio sample rate: %f is inconsistent "
                                               @"with the previously provided: %f",
                                               sampleRate, _sampleRate]];
    return NO;
  }

  return YES;
}

- (std::optional<PacketMap>)inputPacketMapWithMPPAudioData:(MPPAudioData *)audioData
                                                     error:(NSError **)error {
  PacketMap inputPacketMap;

  Packet matrixPacket = [MPPAudioPacketCreator createPacketWithAudioData:audioData error:error];
  if (matrixPacket.IsEmpty()) {
    return std::nullopt;
  }

  inputPacketMap[_audioInputStreamName.cppString] = matrixPacket;

  inputPacketMap[_sampleRateInputStreamName.cppString] =
      [MPPPacketCreator createWithDouble:audioData.format.sampleRate];

  return inputPacketMap;
}

@end
