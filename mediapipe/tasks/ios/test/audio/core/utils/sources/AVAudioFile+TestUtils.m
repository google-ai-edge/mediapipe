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

#import "mediapipe/tasks/ios/test/audio/core/utils/sources/AVAudioFile+TestUtils.h"

#import "mediapipe/tasks/ios/test/audio/core/utils/sources/AVAudioPCMBuffer+TestUtils.h"

static const NSInteger kMilliSecondsPerSecond = 1000;

@implementation MPPTimestampedAudioData

- (instancetype)initWithAudioData:(MPPAudioData *)audioData
          timestampInMilliseconds:(NSInteger)timestampInMilliseconds {
  self = [super init];
  if (self) {
    _audioData = audioData;
    _timestampInMilliseconds = timestampInMilliseconds;
    ;
  }
  return self;
}
@end

@implementation AVAudioFile (TestUtils)

+ (NSArray<MPPTimestampedAudioData *> *)
    streamedAudioBlocksFromAudioFileWithInfo:(MPPFileInfo *)fileInfo
                            modelSampleCount:(NSInteger)modelSampleCount
                             modelSampleRate:(double)modelSampleRate {
  AVAudioPCMBuffer *audioPCMBuffer =
      [AVAudioPCMBuffer interleavedFloat32BufferFromAudioFileWithInfo:fileInfo];

  // Calculates the maximum no: of samples possible in any one input chunk of audio samples by
  // scaling the `modelSampleCount` to the sample rate of the input audio PCM Buffer. Each audio
  // data in the returned array will have `intervalSize` of samples except for the last one. The
  // last one will have all the samples at the end of the audio file that have not been read for
  // streaming until then.
  AVAudioFrameCount intervalSize = (AVAudioFrameCount)ceil(
      modelSampleCount * audioPCMBuffer.format.sampleRate / modelSampleRate);
  NSInteger currentPosition = 0;
  NSInteger audioListCount =
      (NSInteger)ceil((float)audioPCMBuffer.frameLength / (float)intervalSize);
  NSMutableArray<MPPTimestampedAudioData *> *timestampedAudioDataList =
      [NSMutableArray arrayWithCapacity:audioListCount];

  while (currentPosition < audioPCMBuffer.frameLength) {
    NSInteger lengthToBeLoaded = MIN(audioPCMBuffer.frameLength - currentPosition, intervalSize);

    MPPAudioDataFormat *audioDataFormat =
        [[MPPAudioDataFormat alloc] initWithChannelCount:audioPCMBuffer.format.channelCount
                                              sampleRate:audioPCMBuffer.format.sampleRate];
    MPPAudioData *audioData =
        [[MPPAudioData alloc] initWithFormat:audioDataFormat
                                 sampleCount:lengthToBeLoaded / audioDataFormat.channelCount];
    // Can safely access `floatChannelData[0]` since the input file is expected to have atleast 1
    // channel.
    MPPFloatBuffer *floatBuffer =
        [[MPPFloatBuffer alloc] initWithData:audioPCMBuffer.floatChannelData[0] + currentPosition
                                      length:lengthToBeLoaded];
    [audioData loadBuffer:floatBuffer offset:0 length:floatBuffer.length error:nil];

    NSInteger timestampInMilliseconds =
        currentPosition / audioPCMBuffer.format.sampleRate * kMilliSecondsPerSecond;
    MPPTimestampedAudioData *timestampedAudioData =
        [[MPPTimestampedAudioData alloc] initWithAudioData:audioData
                                   timestampInMilliseconds:timestampInMilliseconds];
    [timestampedAudioDataList addObject:timestampedAudioData];

    currentPosition += lengthToBeLoaded;
  }
  return timestampedAudioDataList;
}
@end
