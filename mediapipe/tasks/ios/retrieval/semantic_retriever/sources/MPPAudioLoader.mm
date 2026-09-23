// Copyright 2026 The MediaPipe Authors. All Rights Reserved.
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

#import "mediapipe/tasks/ios/retrieval/semantic_retriever/sources/MPPAudioLoader.h"

#import <AVFoundation/AVFoundation.h>

#include <cstdint>
#include <vector>

#include "absl/strings/string_view.h"
#include "mediapipe/tasks/cc/retrieval/semantic_retriever/audio_decoder.h"
#import "mediapipe/tasks/ios/audio/core/sources/MPPAudioDataFormat.h"
#import "mediapipe/tasks/ios/common/sources/MPPCommon.h"
#import "mediapipe/tasks/ios/common/utils/sources/MPPCommonUtils.h"

@implementation MPPAudioLoader

+ (nullable MPPAudioData *)loadAudioFromFilePath:(NSString *)filePath error:(NSError **)error {
  if (!filePath || filePath.length == 0) {
    [MPPCommonUtils createCustomError:error
                             withCode:MPPTasksErrorCodeInvalidArgumentError
                          description:@"Audio file path cannot be nil or empty."];
    return nil;
  }
  NSURL *url = [NSURL fileURLWithPath:filePath];
  NSError *initError = nil;
  AVAudioFile *audioFile = [[AVAudioFile alloc] initForReading:url error:&initError];
  if (!audioFile) {
    if (error) {
      *error = initError;
    }
    return nil;
  }

  double targetSampleRate = audioFile.processingFormat.sampleRate;
  AVAudioFormat *targetFormat = [[AVAudioFormat alloc] initWithCommonFormat:AVAudioPCMFormatFloat32
                                                                 sampleRate:targetSampleRate
                                                                   channels:1
                                                                interleaved:NO];

  AVAudioPCMBuffer *inputBuffer =
      [[AVAudioPCMBuffer alloc] initWithPCMFormat:audioFile.processingFormat
                                    frameCapacity:(AVAudioFrameCount)audioFile.length];
  NSError *readError = nil;
  if (![audioFile readIntoBuffer:inputBuffer error:&readError]) {
    if (error) {
      *error = readError;
    }
    return nil;
  }

  if ([inputBuffer.format isEqual:targetFormat]) {
    float *data = inputBuffer.floatChannelData[0];
    MPPAudioDataFormat *format = [[MPPAudioDataFormat alloc] initWithChannelCount:1
                                                                       sampleRate:targetSampleRate];
    MPPAudioData *audioData = [[MPPAudioData alloc] initWithFormat:format
                                                       sampleCount:inputBuffer.frameLength];
    MPPFloatBuffer *floatBuffer = [[MPPFloatBuffer alloc] initWithData:data
                                                                length:inputBuffer.frameLength];
    [audioData loadBuffer:floatBuffer offset:0 length:inputBuffer.frameLength error:error];
    return audioData;
  }

  AVAudioConverter *converter = [[AVAudioConverter alloc] initFromFormat:inputBuffer.format
                                                                toFormat:targetFormat];
  if (!converter) {
    [MPPCommonUtils createCustomError:error
                             withCode:MPPTasksErrorCodeInternalError
                          description:@"Failed to create AVAudioConverter."];
    return nil;
  }

  AVAudioFrameCount capacity = ceil(inputBuffer.frameLength * converter.outputFormat.sampleRate /
                                    converter.inputFormat.sampleRate);
  AVAudioPCMBuffer *outputBuffer = [[AVAudioPCMBuffer alloc] initWithPCMFormat:targetFormat
                                                                 frameCapacity:capacity];

  AVAudioConverterInputBlock inputBlock = ^AVAudioBuffer *_Nullable(
      AVAudioPacketCount inNumberOfPackets, AVAudioConverterInputStatus *_Nonnull outStatus) {
    *outStatus = AVAudioConverterInputStatus_HaveData;
    return inputBuffer;
  };

  NSError *convertError = nil;
  AVAudioConverterOutputStatus status = [converter convertToBuffer:outputBuffer
                                                             error:&convertError
                                                withInputFromBlock:inputBlock];
  if (status == AVAudioConverterOutputStatus_Error || outputBuffer.frameLength == 0) {
    if (error) {
      *error = convertError;
    }
    return nil;
  }

  MPPAudioDataFormat *format = [[MPPAudioDataFormat alloc] initWithChannelCount:1
                                                                     sampleRate:targetSampleRate];
  MPPAudioData *audioData = [[MPPAudioData alloc] initWithFormat:format
                                                     sampleCount:outputBuffer.frameLength];
  MPPFloatBuffer *floatBuffer =
      [[MPPFloatBuffer alloc] initWithData:outputBuffer.floatChannelData[0]
                                    length:outputBuffer.frameLength];
  [audioData loadBuffer:floatBuffer offset:0 length:outputBuffer.frameLength error:error];
  return audioData;
}

+ (nullable MPPAudioData *)loadAudioFromData:(NSData *)data error:(NSError **)error {
  if (!data || data.length == 0) {
    [MPPCommonUtils createCustomError:error
                             withCode:MPPTasksErrorCodeInvalidArgumentError
                          description:@"Audio data cannot be nil or empty."];
    return nil;
  }

  uint32_t sampleRate = 0;
  absl::string_view wavBytes(static_cast<const char *>(data.bytes), data.length);
  auto decodedOr =
      mediapipe::tasks::retrieval::AudioDecoder::DecodeAudioBytes(wavBytes, &sampleRate);
  if (!decodedOr.ok() || decodedOr->empty() || sampleRate == 0) {
    NSString *errMsg = decodedOr.ok()
                           ? @"Decoded audio contains no frames."
                           : [NSString stringWithUTF8String:decodedOr.status().message().data()];
    [MPPCommonUtils createCustomError:error
                             withCode:MPPTasksErrorCodeInvalidArgumentError
                          description:errMsg];
    return nil;
  }

  const std::vector<float> &pcmSamples = *decodedOr;
  MPPAudioDataFormat *format = [[MPPAudioDataFormat alloc] initWithChannelCount:1
                                                                     sampleRate:sampleRate];
  MPPAudioData *audioData = [[MPPAudioData alloc] initWithFormat:format
                                                     sampleCount:pcmSamples.size()];
  MPPFloatBuffer *floatBuffer = [[MPPFloatBuffer alloc] initWithData:pcmSamples.data()
                                                              length:pcmSamples.size()];
  [audioData loadBuffer:floatBuffer offset:0 length:pcmSamples.size() error:error];
  return audioData;
}

@end
