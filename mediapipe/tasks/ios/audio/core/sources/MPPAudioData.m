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

#import "mediapipe/tasks/ios/audio/core/sources/MPPAudioData.h"
#import "mediapipe/tasks/ios/audio/core/sources/MPPFloatRingBuffer.h"
#import "mediapipe/tasks/ios/common/sources/MPPCommon.h"
#import "mediapipe/tasks/ios/common/utils/sources/MPPCommonUtils.h"

@implementation MPPAudioData {
  MPPFloatRingBuffer *_ringBuffer;
}

- (instancetype)initWithFormat:(MPPAudioDataFormat *)format sampleCount:(NSUInteger)sampleCount {
  self = [super init];
  if (self) {
    _format = format;

    const NSInteger length = sampleCount * format.channelCount;
    _ringBuffer = [[MPPFloatRingBuffer alloc] initWithLength:length];
  }
  return self;
}

- (BOOL)loadBuffer:(MPPFloatBuffer *)buffer
            offset:(NSUInteger)offset
            length:(NSUInteger)length
             error:(NSError **)error {
  return [_ringBuffer loadFloatBuffer:buffer offset:offset length:length error:error];
}

- (BOOL)loadAudioRecord:(MPPAudioRecord *)audioRecord error:(NSError **)error {
  if (![self isValidAudioRecordFormat:audioRecord.audioDataFormat error:error]) {
    return NO;
  }

  MPPFloatBuffer *audioRecordBuffer = [audioRecord readAtOffset:0
                                                     withLength:audioRecord.bufferLength
                                                          error:error];
  return [self loadRingBufferWithAudioRecordBuffer:audioRecordBuffer error:error];
}

- (BOOL)isValidAudioRecordFormat:(MPPAudioDataFormat *)format error:(NSError **)error {
  if (![format isEqual:self.format]) {
    [MPPCommonUtils createCustomError:error
                             withCode:MPPTasksErrorCodeInvalidArgumentError
                          description:@"The provided audio record has incompatible audio format"];
    return NO;
  }

  return YES;
}

- (BOOL)loadRingBufferWithAudioRecordBuffer:(MPPFloatBuffer *)audioRecordBuffer
                                      error:(NSError **)error {
  // Returns `NO` without populating an error since the function that created `audioRecordBuffer` is
  // expected to populate the error param of the caller (`loadAudioRecord`) which passed into this
  // function.
  // For ease of mocking the logic of `loadAudioRecord` in the tests.
  if (!audioRecordBuffer) {
    return NO;
  }
  return [_ringBuffer loadFloatBuffer:audioRecordBuffer
                               offset:0
                               length:audioRecordBuffer.length
                                error:error];
}

- (MPPFloatBuffer *)buffer {
  return _ringBuffer.floatBuffer;
}

- (NSUInteger)bufferLength {
  return _ringBuffer.length / _format.channelCount;
}

@end
