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

#import "mediapipe/tasks/ios/test/audio/core/utils/sources/AVAudioPCMBuffer+TestUtils.h"
#import "mediapipe/tasks/ios/test/audio/core/utils/sources/MPPAudioData+TestUtils.h"

@implementation MPPAudioData (TestUtils)

- (instancetype)initWithChannelCount:(NSUInteger)channelCount
                          sampleRate:(double)sampleRate
                         sampleCount:(NSUInteger)sampleCount {
  MPPAudioDataFormat *audioDataFormat =
      [[MPPAudioDataFormat alloc] initWithChannelCount:channelCount sampleRate:sampleRate];
  return [self initWithFormat:audioDataFormat sampleCount:sampleCount];
}

- (instancetype)initWithFileInfo:(MPPFileInfo *)fileInfo {
  // Load the samples from the audio file in `Float32` interleaved format to
  // an `AVAudioPCMBuffer`.
  AVAudioPCMBuffer *buffer =
      [AVAudioPCMBuffer interleavedFloat32BufferFromAudioFileWithInfo:fileInfo];

  // Create a float buffer from the `floatChannelData` of `AVAudioPCMBuffer`. This float buffer will
  // be used to load the audio data.
  MPPFloatBuffer *bufferData = [[MPPFloatBuffer alloc] initWithData:buffer.floatChannelData[0]
                                                             length:buffer.frameLength];

  MPPAudioData *audioData = [self initWithChannelCount:buffer.format.channelCount
                                            sampleRate:buffer.format.sampleRate
                                           sampleCount:buffer.frameLength];

  // Load all the samples in the audio file to the newly created audio data.
  [audioData loadBuffer:bufferData offset:0 length:bufferData.length error:nil];
  return audioData;
}

@end
