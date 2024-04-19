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

@implementation MPPAudioData {
  MPPFloatRingBuffer *_ringBuffer;
}

- (instancetype)initWithFormat:(MPPAudioDataFormat *)format sampleCount:(NSUInteger)sampleCount {
  self = [super init];
  if (self) {
    _audioFormat = format;

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

- (MPPFloatBuffer *)buffer {
  return _ringBuffer.floatBuffer;
}

- (NSUInteger)bufferLength {
  return _ringBuffer.length;
}

@end
